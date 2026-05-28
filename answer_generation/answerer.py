from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import csv
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EFFECTIVITY_ROOT = ROOT / "data" / "preprocessed" / "effectivity"

INSUFFICIENT_CONTEXT_ANSWER = "Không tìm thấy căn cứ đủ rõ trong tài liệu được truy xuất."
PROMPT_VERSION = "extractive_multi_agent_v2"

DIRECT_SYSTEM_PROMPT = f"""Bạn là trợ lý pháp lý chuyên về giao thông đường bộ Việt Nam.

Nhiệm vụ của bạn là trả lời câu hỏi chỉ dựa trên phần CONTEXT được cung cấp.

Quy tắc bắt buộc:
1. Không sử dụng kiến thức ngoài CONTEXT.
2. Không tự suy diễn mức phạt, thời hạn, điều kiện hoặc căn cứ nếu CONTEXT không nêu rõ.
3. Nếu CONTEXT không có căn cứ đủ rõ, trả lời đúng câu: "{INSUFFICIENT_CONTEXT_ANSWER}"
4. Ưu tiên căn cứ chứa nội dung trả lời trực tiếp.
5. Nếu một passage chỉ dẫn chiếu kiểu "theo quy định tại..." mà không chứa nội dung trả lời, không dùng passage đó làm căn cứ chính nếu CONTEXT có passage đích trực tiếp.
6. Với câu hỏi về số liệu, mức phạt, thời hạn, tối đa/tối thiểu, phải giữ đúng con số và đơn vị trong CONTEXT.
7. Nếu câu hỏi có nhiều ý, trả lời đủ từng ý.
8. Không trích dẫn căn cứ không được dùng để trả lời.
9. Nếu câu hỏi hỏi về hiệu lực, phải dùng các dòng "Hiệu lực văn bản", "Hiệu lực riêng" hoặc "Hiệu lực riêng chưa xác định" trong CONTEXT. Không tự suy diễn ngày hiệu lực nếu metadata không có.
10. Nếu câu hỏi hỏi "còn hiệu lực không" hoặc "hiện nay có hiệu lực không", phải dùng dòng "Tình trạng hiệu lực văn bản theo ngày hiện tại" để trả lời trực tiếp có/không, rồi nêu ngày bắt đầu/hết hiệu lực nếu có.

Định dạng đầu ra:
Trả lời: <câu trả lời ngắn gọn, trực tiếp>
Dựa theo: <điều/khoản/điểm, văn bản hoặc đường dẫn pháp lý liên quan>
"""

EXTRACTIVE_MULTI_AGENT_SYSTEM_PROMPT = f"""Bạn là hệ thống trả lời pháp lý chuyên về giao thông đường bộ Việt Nam.

Bạn phải vận hành như 3 agent nội bộ, nhưng không được in quá trình làm việc.

Agent 1 - Query Decomposer:
- Tách câu hỏi thành từng ý cần trả lời.
- Nếu câu hỏi có "và", "đồng thời", "nếu... thì...", "mức phạt", "trừ điểm", "tước giấy phép", hoặc hỏi theo từng loại phương tiện, phải xem là có nhiều ý.

Agent 2 - Evidence Extractor:
- Chỉ dùng CONTEXT.
- Tìm cụm chứa đáp án trực tiếp cho từng ý.
- Ưu tiên passage có nội dung trực tiếp, không ưu tiên passage chỉ nói "theo quy định tại..." nếu passage đích có nội dung.
- Với số liệu, mức phạt, thời hạn, điều kiện, hình thức xử lý, phải giữ đầy đủ từ giới hạn và đơn vị như "không quá", "tối đa", "tối thiểu", "ít nhất", "từ ... đến ...", "trừ ... điểm", "tước ... từ ... đến ...".
- Không rút gọn "không quá 04 giờ" thành "04 giờ".
- Không rút gọn câu có/không thành chỉ "Có" hoặc "Không"; phải nêu điều kiện/hành vi đi kèm.

Agent 3 - Answer Composer:
- Viết câu trả lời ngắn, trực tiếp, nhưng phải chứa nguyên văn các cụm đáp án quan trọng tìm được trong CONTEXT.
- Nếu câu hỏi nhiều ý, trả lời bằng các bullet, mỗi bullet một ý.
- Nếu chỉ thiếu căn cứ cho một ý, ghi rõ ý đó không tìm thấy căn cứ; không phủ định toàn bộ câu hỏi nếu các ý khác có căn cứ.
- Không dùng kiến thức ngoài CONTEXT.
- Không trích dẫn căn cứ không được dùng.
- Nếu câu hỏi hỏi về hiệu lực, phải ưu tiên các dòng "Hiệu lực văn bản", "Hiệu lực riêng" hoặc "Hiệu lực riêng chưa xác định" trong CONTEXT.
- Nếu câu hỏi hỏi "còn hiệu lực không" hoặc "hiện nay có hiệu lực không", phải dùng dòng "Tình trạng hiệu lực văn bản theo ngày hiện tại" để trả lời trực tiếp có/không.
- Nếu một điều/khoản/điểm có hiệu lực riêng khác hiệu lực chung của văn bản, phải nêu rõ hiệu lực riêng đó.

Nếu CONTEXT không chứa bất kỳ căn cứ đủ rõ nào để trả lời, trả lời đúng câu:
"{INSUFFICIENT_CONTEXT_ANSWER}"

Định dạng đầu ra bắt buộc:
Trả lời:
- <ý 1, chứa nguyên văn cụm đáp án trực tiếp>
- <ý 2 nếu có>
Dựa theo:
- <điều/khoản/điểm, văn bản hoặc đường dẫn pháp lý liên quan>
- <căn cứ tiếp theo nếu có>
"""

QUERY_ROUTER_SYSTEM_PROMPT = """Bạn là bộ tiền xử lý truy vấn cho hệ thống RAG pháp luật giao thông đường bộ Việt Nam.

Nhiệm vụ:
1. Phân loại câu hỏi vào đúng một route:
   - "traffic_law": câu hỏi liên quan giao thông đường bộ Việt Nam, xử phạt vi phạm giao thông, giấy phép lái xe, đăng kiểm, vận tải đường bộ, hạ tầng đường bộ, phương tiện, người điều khiển phương tiện, hoặc quy định của địa phương về tổ chức giao thông, hạn chế phương tiện, vùng phát thải thấp, khu vực hạn chế phương tiện giao thông gây ô nhiễm môi trường.
   - "general_chat": câu hỏi chào hỏi, trò chuyện thông thường, hoặc nội dung không thuộc phạm vi pháp luật giao thông đường bộ Việt Nam.
2. Nếu route là "traffic_law", viết lại câu hỏi thành truy vấn truy xuất dùng thuật ngữ gần với văn bản pháp luật.
3. Nếu route là "general_chat", trả lời ngắn gọn như chatbot thông thường, không nhắc đến CONTEXT hay retrieval.

Quy tắc rewrite cho traffic_law:
- "say rượu", "uống rượu", "có cồn", "hơi men" -> "trong máu hoặc hơi thở có nồng độ cồn".
- "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
- "bằng lái", "GPLX" -> "giấy phép lái xe".
- "không đội mũ" -> "không đội mũ bảo hiểm".
- "chạy quá tốc độ", "vượt tốc độ" -> "điều khiển xe chạy quá tốc độ quy định".
- Giữ lại loại phương tiện nếu người dùng nêu: ô tô, mô tô, xe gắn máy, xe máy chuyên dùng, xe đạp.
- Nếu hỏi "từng loại phương tiện", rewrite phải nêu rõ nhu cầu so sánh theo từng loại phương tiện.

Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON:
{
  "route": "traffic_law" hoặc "general_chat",
  "rewritten_query": "truy vấn đã viết lại, hoặc chuỗi rỗng nếu general_chat",
  "reason": "lý do rất ngắn",
  "chat_answer": "câu trả lời nếu general_chat, hoặc chuỗi rỗng nếu traffic_law"
}
"""

GENERAL_CHAT_SYSTEM_PROMPT = """Bạn là trợ lý tiếng Việt thân thiện và ngắn gọn.

Hãy trả lời trực tiếp câu hỏi của người dùng như chatbot thông thường. Nếu người dùng hỏi về năng lực hệ thống, có thể nói hệ thống này chủ yếu được thiết kế để demo hỏi đáp pháp luật giao thông đường bộ Việt Nam.
"""

ROUTE_TRAFFIC_LAW = "traffic_law"
ROUTE_GENERAL_CHAT = "general_chat"

LEGAL_RAG_HINT_PATTERN = re.compile(
    r"("
    r"\b(nghị\s*định|thông\s*tư|luật|bộ\s*luật|quyết\s*định|nghị\s*quyết|điều|khoản|điểm|hiệu\s*lực|hết\s*hiệu\s*lực)\b"
    r"|"
    r"\b(quy\s*định\s*hiện\s*hành|giao\s*thông|đường\s*bộ|phương\s*tiện|vận\s*tải|đăng\s*kiểm|giấy\s*phép\s*lái\s*xe|vùng\s*phát\s*thải\s*thấp|khu\s*vực\s*hạn\s*chế\s*phương\s*tiện|hạn\s*chế\s*phương\s*tiện|phương\s*tiện\s*giao\s*thông\s*gây\s*ô\s*nhiễm|ô\s*nhiễm\s*môi\s*trường)\b"
    r"|"
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐa-zđ-]+"
    r"|"
    r"\b(NĐ-CP|ND-CP|TT-BGTVT|TT-BCA|QH\d+|UBTVQH\d+)\b"
    r")",
    flags=re.IGNORECASE,
)
DOC_NUMBER_PATTERN = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐa-zđ-]+", flags=re.IGNORECASE)
EFFECTIVITY_QUERY_PATTERN = re.compile(
    r"\b(hiệu\s*lực|hết\s*hiệu\s*lực|còn\s*hiệu\s*lực|thi\s*hành|áp\s*dụng)\b",
    flags=re.IGNORECASE,
)
ALCOHOL_QUERY_PATTERN = re.compile(
    r"\b(say|say\s*rượu|uống\s*rượu|rượu\s*bia|bia\s*rượu|có\s*cồn|hơi\s*men|nồng\s*độ\s*cồn)\b",
    flags=re.IGNORECASE,
)
PENALTY_QUERY_PATTERN = re.compile(
    r"\b(mức\s*phạt|bị\s*phạt|phạt\s*bao\s*nhiêu|xử\s*phạt|phạt\s*tiền)\b",
    flags=re.IGNORECASE,
)
CURRENT_TIME_QUERY_PATTERN = re.compile(r"\b(hiện\s*tại|hiện\s*nay|bây\s*giờ|ngày\s*nay)\b", flags=re.IGNORECASE)
YEAR_TIME_QUERY_PATTERN = re.compile(r"\b(?:năm|nam)\s*(20\d{2}|19\d{2})\b", flags=re.IGNORECASE)
SLASH_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2}|19\d{2})\b")


def _mojibake_score(text: str) -> int:
    markers = ["Ã", "Â", "Ä", "Æ", "Ð", "ð", "â€", "ï¿½"]
    return sum(text.count(marker) for marker in markers)


def repair_mojibake_text(value: str) -> str:
    if not isinstance(value, str) or not value:
        return value
    if _mojibake_score(value) == 0:
        return value

    best = value
    best_score = _mojibake_score(value)
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        score = _mojibake_score(repaired)
        if score < best_score:
            best = repaired
            best_score = score
    return best


def repair_mojibake(value: Any) -> Any:
    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, list):
        return [repair_mojibake(v) for v in value]
    if isinstance(value, dict):
        return {k: repair_mojibake(v) for k, v in value.items()}
    return value


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _first_env(names: list[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def get_api_key(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    load_dotenv()
    provider = provider.lower()
    if provider == "openrouter":
        key = _first_env(["OPENROUTER_API_KEY", "openrouter_api_key", "OPEN_ROUTER_API", "open_router_api"])
    elif provider == "openai":
        key = _first_env(["OPENAI_API_KEY", "openai_api_key"])
    else:
        raise ValueError(f"Unsupported API provider: {provider}")
    if not key:
        raise RuntimeError(
            f"Missing API key for {provider}. Put OPENROUTER_API_KEY/openrouter_api_key "
            "or OPENAI_API_KEY/openai_api_key in .env, or pass --api-key."
        )
    return key


def provider_base_url(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    load_dotenv()
    provider = provider.lower()
    if provider == "openrouter":
        return (_first_env(["OPENROUTER_BASE_URL", "openrouter_base_url"]) or "https://openrouter.ai/api/v1").rstrip("/")
    if provider == "openai":
        return (_first_env(["OPENAI_BASE_URL", "openai_base_url"]) or "https://api.openai.com/v1").rstrip("/")
    raise ValueError(f"Unsupported API provider: {provider}")


def _empty_cell(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "null", "none", "nan"}


def _cell(value: Any) -> str | None:
    if _empty_cell(value):
        return None
    return str(value).strip()


def _normalize_doc_key(value: str | None) -> str:
    if not value:
        return ""
    value = repair_mojibake_text(str(value)).lower()
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _normalize_for_match(value: str | None) -> str:
    if not value:
        return ""
    value = repair_mojibake_text(str(value)).lower()
    value = value.replace("đ", "d")
    import unicodedata

    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _format_date(value: str | None) -> str:
    value = _cell(value)
    if not value:
        return "chưa xác định"
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        year, month, day = match.groups()
        return f"{day}/{month}/{year}"
    return value


def _parse_iso_date(value: str | None) -> date | None:
    value = _cell(value)
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@lru_cache(maxsize=4)
def load_effectivity_metadata(effectivity_root: str = str(EFFECTIVITY_ROOT)) -> dict[str, Any]:
    root = Path(effectivity_root)
    by_doc_id: dict[str, dict[str, str]] = {}
    by_doc_number: dict[str, dict[str, str]] = {}
    unit_overrides: dict[str, list[dict[str, str]]] = {}
    unit_unresolved: dict[str, list[dict[str, str]]] = {}

    for row in _read_csv_dicts(root / "effectivity_index.csv"):
        row = {key: (_cell(value) or "") for key, value in row.items()}
        doc_id_key = _normalize_doc_key(row.get("document_id"))
        doc_number_key = _normalize_doc_key(row.get("document_number"))
        if doc_id_key:
            by_doc_id[doc_id_key] = row
        if doc_number_key:
            by_doc_number[doc_number_key] = row

    for row in _read_csv_dicts(root / "effectivity_unit_overrides.csv"):
        row = {key: (_cell(value) or "") for key, value in row.items()}
        doc_id_key = _normalize_doc_key(row.get("document_id") or row.get("document_number"))
        if doc_id_key:
            unit_overrides.setdefault(doc_id_key, []).append(row)

    for row in _read_csv_dicts(root / "effectivity_unresolved.csv"):
        row = {key: (_cell(value) or "") for key, value in row.items()}
        doc_id_key = _normalize_doc_key(row.get("document_id") or row.get("document_number"))
        if doc_id_key:
            unit_unresolved.setdefault(doc_id_key, []).append(row)

    return {
        "by_doc_id": by_doc_id,
        "by_doc_number": by_doc_number,
        "unit_overrides": unit_overrides,
        "unit_unresolved": unit_unresolved,
    }


def _doc_effectivity(result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str] | None:
    doc_id = _normalize_doc_key(result.get("document_id"))
    doc_number = _normalize_doc_key(result.get("document_number"))
    return metadata["by_doc_id"].get(doc_id) or metadata["by_doc_number"].get(doc_number)


def _doc_effectivity_key(result: dict[str, Any], metadata: dict[str, Any]) -> str:
    doc_meta = _doc_effectivity(result, metadata)
    if doc_meta:
        return _normalize_doc_key(doc_meta.get("document_id") or doc_meta.get("document_number"))
    return _normalize_doc_key(result.get("document_id") or result.get("document_number"))


def _point_aliases(value: str | None) -> set[str]:
    value = _cell(value)
    if not value:
        return set()
    value = value.lower().strip(". ")
    aliases = {value, value.replace("đ", "d")}
    if value in {"đ", "d", "dd"}:
        aliases.update({"đ", "d", "dd"})
    return aliases


def _extract_unit_selector(result: dict[str, Any]) -> dict[str, str]:
    raw = " ".join(
        str(value or "")
        for value in [
            result.get("unit_id"),
            result.get("passage_id"),
            result.get("path_text"),
        ]
    )
    selector: dict[str, str] = {}
    article_match = re.search(r"(?:^|[._\s>])dieu[_\s.-]*(\d+)|Điều\s+(\d+)", raw, flags=re.IGNORECASE)
    clause_match = re.search(r"(?:^|[._\s>])khoan[_\s.-]*(\d+)|Khoản\s+(\d+)", raw, flags=re.IGNORECASE)
    point_match = re.search(r"(?:^|[._\s>])diem[_\s.-]*([a-z]+|dd|\d+)|Điểm\s+([a-zA-ZđĐ]+)", raw, flags=re.IGNORECASE)
    if article_match:
        selector["article"] = article_match.group(1) or article_match.group(2)
    if clause_match:
        selector["clause"] = clause_match.group(1) or clause_match.group(2)
    if point_match:
        selector["point"] = (point_match.group(1) or point_match.group(2)).lower()
    return selector


def _selector_matches(row: dict[str, str], selector: dict[str, str]) -> bool:
    target_article = _cell(row.get("target_article"))
    target_clause = _cell(row.get("target_clause"))
    target_point = _cell(row.get("target_point"))

    if target_article and selector.get("article") != target_article:
        return False
    if target_clause and selector.get("clause") != target_clause:
        return False
    if target_point:
        point = selector.get("point")
        if not point or not (_point_aliases(point) & _point_aliases(target_point)):
            return False
    return bool(target_article or target_clause or target_point)


def _effectivity_lines_for_result(result: dict[str, Any]) -> list[str]:
    metadata = load_effectivity_metadata()
    lines: list[str] = []

    doc_meta = _doc_effectivity(result, metadata)
    if doc_meta:
        effective_from = _format_date(doc_meta.get("effective_from"))
        effective_to_raw = _cell(doc_meta.get("effective_to"))
        effective_to = _format_date(effective_to_raw) if effective_to_raw else "chưa có ngày hết hiệu lực trong dữ liệu"
        source = _cell(doc_meta.get("effective_to_source_document_number"))
        source_text = f"; văn bản làm hết hiệu lực: {source}" if source else ""
        lines.append(f"Hiệu lực văn bản: từ {effective_from}; đến {effective_to}{source_text}.")

        today = date.today()
        start_date = _parse_iso_date(doc_meta.get("effective_from"))
        end_date = _parse_iso_date(doc_meta.get("effective_to"))
        if start_date and today < start_date:
            status = f"chưa có hiệu lực, bắt đầu từ {_format_date(doc_meta.get('effective_from'))}"
        elif end_date and today >= end_date:
            status = f"đã hết hiệu lực từ {_format_date(doc_meta.get('effective_to'))}"
        elif start_date:
            if end_date:
                status = f"đang có hiệu lực đến trước {_format_date(doc_meta.get('effective_to'))}"
            else:
                status = "đang có hiệu lực theo dữ liệu hiện có; chưa ghi nhận ngày hết hiệu lực"
        else:
            status = "chưa xác định được ngày bắt đầu hiệu lực"
        lines.append(f"Tình trạng hiệu lực văn bản theo ngày hiện tại ({today.isoformat()}): {status}.")

    doc_key = _doc_effectivity_key(result, metadata)
    selector = _extract_unit_selector(result)
    override_rows = [
        row for row in metadata["unit_overrides"].get(doc_key, [])
        if _selector_matches(row, selector)
    ]
    unresolved_rows = [
        row for row in metadata["unit_unresolved"].get(doc_key, [])
        if _selector_matches(row, selector)
    ]

    for row in override_rows[:3]:
        selector_text = row.get("target_selector_raw") or "quy định này"
        date_text = _format_date(row.get("effective_from"))
        raw_text = row.get("raw_text") or ""
        source_path = row.get("source_path_text") or ""
        evidence = raw_text or source_path
        suffix = f" Căn cứ: {evidence}" if evidence else ""
        lines.append(f"Hiệu lực riêng: {selector_text} có hiệu lực từ {date_text}.{suffix}")

    for row in unresolved_rows[:3]:
        selector_text = row.get("target_selector_raw") or "quy định này"
        raw_text = row.get("raw_text") or ""
        notes = row.get("notes") or "chưa quy đổi được thành ngày cụ thể"
        evidence = f" Căn cứ: {raw_text}" if raw_text else ""
        lines.append(f"Hiệu lực riêng chưa xác định ngày cụ thể: {selector_text}; {notes}.{evidence}")

    return lines


def _extract_doc_number_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen = set()
    for match in DOC_NUMBER_PATTERN.finditer(repair_mojibake_text(text or "")):
        value = match.group(0).replace(" ", "")
        key = _normalize_doc_key(value)
        if key and key not in seen:
            seen.add(key)
            candidates.append(value)
    return candidates


def _is_effectivity_query(query: str) -> bool:
    return bool(EFFECTIVITY_QUERY_PATTERN.search(repair_mojibake_text(query or "")))


def _doc_meta_from_query(query: str, metadata: dict[str, Any]) -> dict[str, str] | None:
    for doc_number in _extract_doc_number_candidates(query):
        doc_key = _normalize_doc_key(doc_number)
        doc_meta = metadata["by_doc_number"].get(doc_key) or metadata["by_doc_id"].get(doc_key)
        if doc_meta:
            return doc_meta
    return None


def _effectivity_status(doc_meta: dict[str, str]) -> str:
    today = date.today()
    start_date = _parse_iso_date(doc_meta.get("effective_from"))
    end_date = _parse_iso_date(doc_meta.get("effective_to"))
    if start_date and today < start_date:
        return f"chưa có hiệu lực, bắt đầu từ {_format_date(doc_meta.get('effective_from'))}"
    if end_date and today >= end_date:
        return f"đã hết hiệu lực từ {_format_date(doc_meta.get('effective_to'))}"
    if start_date:
        if end_date:
            return f"đang có hiệu lực đến trước {_format_date(doc_meta.get('effective_to'))}"
        return "đang có hiệu lực theo dữ liệu hiện có; chưa ghi nhận ngày hết hiệu lực"
    return "chưa xác định được ngày bắt đầu hiệu lực"


def _direct_effectivity_answer(query: str) -> str | None:
    if not _is_effectivity_query(query):
        return None

    metadata = load_effectivity_metadata()
    doc_meta = _doc_meta_from_query(query, metadata)
    if not doc_meta:
        return None

    doc_number = doc_meta.get("document_number") or doc_meta.get("document_id") or "văn bản này"
    doc_key = _normalize_doc_key(doc_meta.get("document_id") or doc_meta.get("document_number"))
    selector = _extract_unit_selector({"path_text": query, "passage_id": query, "unit_id": query})
    override_rows = [
        row for row in metadata["unit_overrides"].get(doc_key, [])
        if _selector_matches(row, selector)
    ]
    unresolved_rows = [
        row for row in metadata["unit_unresolved"].get(doc_key, [])
        if _selector_matches(row, selector)
    ]

    asks_current_status = bool(re.search(r"(còn\s*hiệu\s*lực|hiện\s*nay|bây\s*giờ|đang\s*có\s*hiệu\s*lực)", query, flags=re.IGNORECASE))
    asks_end_date = bool(re.search(r"(hết\s*hiệu\s*lực|ngày\s*kết\s*thúc|đến\s*ngày\s*nào)", query, flags=re.IGNORECASE))

    if override_rows:
        row = override_rows[0]
        selector_text = row.get("target_selector_raw") or "quy định này"
        answer = (
            "Trả lời:\n"
            f"- {selector_text} của {doc_number} có hiệu lực từ {_format_date(row.get('effective_from'))}."
        )
        if asks_current_status:
            start_date = _parse_iso_date(row.get("effective_from"))
            status = "đang có hiệu lực theo dữ liệu hiện có" if start_date and date.today() >= start_date else "chưa có hiệu lực"
            answer = (
                "Trả lời:\n"
                f"- {selector_text} của {doc_number} {status}; ngày bắt đầu hiệu lực riêng là {_format_date(row.get('effective_from'))}."
            )
        evidence = row.get("source_path_text") or row.get("raw_text") or "effectivity_unit_overrides.csv"
        return f"{answer}\nDựa theo:\n- {evidence}"

    if unresolved_rows:
        row = unresolved_rows[0]
        selector_text = row.get("target_selector_raw") or "quy định này"
        evidence = row.get("raw_text") or row.get("source_path_text") or "effectivity_unresolved.csv"
        return (
            "Trả lời:\n"
            f"- {selector_text} của {doc_number} có hiệu lực theo một căn cứ gián tiếp, nhưng dữ liệu hiện chưa quy đổi được thành ngày cụ thể.\n"
            "Dựa theo:\n"
            f"- {evidence}"
        )

    if asks_end_date:
        effective_to = _cell(doc_meta.get("effective_to"))
        if effective_to:
            answer_line = f"{doc_number} hết hiệu lực từ {_format_date(effective_to)}."
        else:
            answer_line = f"{doc_number} chưa ghi nhận ngày hết hiệu lực trong dữ liệu."
    elif asks_current_status:
        answer_line = f"{doc_number} {_effectivity_status(doc_meta)}."
    else:
        answer_line = f"{doc_number} có hiệu lực từ {_format_date(doc_meta.get('effective_from'))}."

    source = doc_meta.get("effective_to_source_document_number")
    source_text = f"; văn bản làm hết hiệu lực: {source}" if source else ""
    return (
        "Trả lời:\n"
        f"- {answer_line}\n"
        "Dựa theo:\n"
        f"- effectivity_index.csv: effective_from={doc_meta.get('effective_from') or 'null'}, "
        f"effective_to={doc_meta.get('effective_to') or 'null'}{source_text}"
    )


def apply_rule_based_query_rewrite(query: str) -> str:
    query = repair_mojibake_text(query).strip()
    rewritten_parts = [query]
    normalized = _normalize_for_match(query)

    if "say" in normalized or "ruou" in normalized or "bia" in normalized or "nong do con" in normalized:
        alcohol_phrase = "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn"
        if "nong do con" not in normalized:
            rewritten_parts.append(alcohol_phrase)
        if "phat" in normalized or "xu phat" in normalized or "bao nhieu" in normalized:
            rewritten_parts.append("mức phạt phạt tiền")

    if "vuot den do" in normalized:
        rewritten_parts.append("không chấp hành hiệu lệnh của đèn tín hiệu giao thông")
        if "phat" in normalized or "bao nhieu" in normalized:
            rewritten_parts.append("mức phạt phạt tiền")

    if "bang lai" in normalized or "gplx" in normalized:
        rewritten_parts.append("giấy phép lái xe")

    if (
        "vung phat thai thap" in normalized
        or "han che phuong tien" in normalized
        or ("phuong tien" in normalized and "o nhiem" in normalized)
    ):
        rewritten_parts.append(
            "vùng phát thải thấp khu vực hạn chế phương tiện giao thông gây ô nhiễm môi trường"
        )
        if "ha noi" in normalized:
            rewritten_parts.append("thành phố Hà Nội lộ trình thực hiện theo giai đoạn")

    deduped: list[str] = []
    seen = set()
    for part in rewritten_parts:
        key = _normalize_for_match(part)
        if key and key not in seen:
            seen.add(key)
            deduped.append(part)
    return " ".join(deduped)


def detect_temporal_scope(query: str) -> dict[str, Any] | None:
    query = repair_mojibake_text(query)
    if CURRENT_TIME_QUERY_PATTERN.search(query):
        today = date.today()
        return {
            "kind": "point",
            "label": f"hiện tại ({today.isoformat()})",
            "start": today,
            "end": today,
        }

    slash_match = SLASH_DATE_PATTERN.search(query)
    if slash_match:
        day, month, year = map(int, slash_match.groups())
        try:
            point = date(year, month, day)
        except ValueError:
            return None
        return {
            "kind": "point",
            "label": point.isoformat(),
            "start": point,
            "end": point,
        }

    year_match = YEAR_TIME_QUERY_PATTERN.search(query)
    if year_match:
        year = int(year_match.group(1))
        return {
            "kind": "year",
            "label": f"năm {year}",
            "start": date(year, 1, 1),
            "end": date(year + 1, 1, 1),
        }
    return None


def _doc_active_for_scope(result: dict[str, Any], temporal_scope: dict[str, Any] | None) -> bool | None:
    if not temporal_scope:
        return None
    metadata = load_effectivity_metadata()
    doc_meta = _doc_effectivity(result, metadata)
    if not doc_meta:
        return None

    start_date = _parse_iso_date(doc_meta.get("effective_from"))
    end_date = _parse_iso_date(doc_meta.get("effective_to"))
    scope_start = temporal_scope["start"]
    scope_end = temporal_scope["end"]

    if temporal_scope["kind"] == "point":
        if start_date and scope_start < start_date:
            return False
        if end_date and scope_start >= end_date:
            return False
        return True

    if start_date and start_date >= scope_end:
        return False
    if end_date and end_date <= scope_start:
        return False
    return True


def _result_match_text(result: dict[str, Any]) -> str:
    return _normalize_for_match(
        " ".join(
            str(value or "")
            for value in [
                result.get("document_number"),
                result.get("document_title"),
                result.get("path_text"),
                result.get("text"),
            ]
        )
    )


def _query_profile(query: str) -> dict[str, bool]:
    normalized = _normalize_for_match(query)
    penalty = bool(
        PENALTY_QUERY_PATTERN.search(query)
        or re.search(r"\b(muc\s+phat|bi\s+phat|phat\s+bao\s+nhieu|xu\s+phat|phat\s+tien|tien\s+phat)\b", normalized)
    )
    criminal_intent = any(token in normalized for token in ["hinh su", "phat tu", "toi pham", "truy cuu", "trach nhiem hinh su"])
    return {
        "alcohol": bool(ALCOHOL_QUERY_PATTERN.search(query)) or any(token in normalized for token in ["say", "ruou", "bia", "nong do con"]),
        "penalty": penalty,
        "criminal_intent": criminal_intent,
        "administrative_penalty": penalty and not criminal_intent,
        "low_emission_stage": (
            "vung phat thai thap" in normalized
            and ("lo trinh" in normalized or "giai doan" in normalized or "trien khai" in normalized)
        ),
        "vehicle_unspecified": not any(token in normalized for token in ["o to", "mo to", "xe may", "xe dap", "xe may chuyen dung"]),
    }


def _domain_relevance_score(result: dict[str, Any], profile: dict[str, bool]) -> tuple[float, list[str]]:
    text = _result_match_text(result)
    score = 0.0
    notes: list[str] = []

    if profile["alcohol"]:
        if "nong do con" in text or "ruou bia" in text or "ruou" in text:
            score += 3.0
            notes.append("alcohol_match")
        else:
            score -= 6.0
            notes.append("missing_alcohol")

    if profile["penalty"]:
        if "phat tien tu" in text or ("phat tien" in text and "dong" in text):
            score += 3.0
            notes.append("fine_range_match")
        else:
            score -= 4.0
            notes.append("missing_fine")

        if "bo luat hinh su" in text or "phat tu" in text or "toi " in text:
            score -= 6.0 if profile["administrative_penalty"] else 2.0
            notes.append("criminal_context_penalty")

    if profile["alcohol"] and profile["penalty"] and "nghi dinh quy dinh xu phat vi pham hanh chinh" in text:
        score += 2.0
        notes.append("admin_penalty_decree")

    if profile["vehicle_unspecified"]:
        article_match = re.search(r"dieu\s+(6|7|8|9)\b", text)
        if article_match:
            score += {"6": 0.4, "7": 0.3, "8": 0.2, "9": 0.1}[article_match.group(1)]

    if profile["low_emission_stage"]:
        stage_path = _normalize_for_match(result.get("path_text") or "")
        if "dieu 11" in stage_path or "lo trinh thuc hien vung phat thai thap" in stage_path:
            score += 5.0
            notes.append("low_emission_stage_match")
            if "diem a" in stage_path:
                score += 0.5
            elif "diem b" in stage_path:
                score += 0.4
            elif "diem c" in stage_path:
                score += 0.3
            elif "khoan 2" in stage_path:
                score += 0.2
        elif "chuong iv" in stage_path and "lo trinh thuc hien" in stage_path:
            score += 2.0
            notes.append("low_emission_chapter_match")
        if "ke tu ngay" in text or "truoc ngay" in text:
            score += 1.0
            notes.append("date_stage_match")
    return score, notes


def postprocess_retrieval_for_query(
    retrieval: dict[str, Any],
    original_query: str,
    retrieval_query: str,
    top_k: int,
) -> dict[str, Any]:
    results = list(retrieval.get("results") or [])
    if not results:
        return retrieval

    profile = _query_profile(retrieval_query)
    temporal_scope = detect_temporal_scope(original_query)
    if not temporal_scope and profile["administrative_penalty"]:
        today = date.today()
        temporal_scope = {
            "kind": "point",
            "label": f"hiện tại mặc định ({today.isoformat()})",
            "start": today,
            "end": today,
        }
    scored: list[tuple[float, dict[str, Any]]] = []
    active_count = 0

    for result in results:
        result = dict(result)
        original_score = float(result.get("score") or 0.0)
        active = _doc_active_for_scope(result, temporal_scope)
        domain_score, notes = _domain_relevance_score(result, profile)
        temporal_score = 0.0
        if active is True:
            temporal_score = 2.5
            active_count += 1
        elif active is False:
            temporal_score = -8.0
        composite = original_score + domain_score + temporal_score
        result["temporal_match"] = active
        result["domain_rerank_notes"] = notes
        result["original_score"] = original_score
        result["rerank_score"] = round(composite, 6)
        scored.append((composite, result))

    if temporal_scope and active_count:
        scored = [(score, result) for score, result in scored if result.get("temporal_match") is not False]

    if profile["alcohol"]:
        alcohol_scored = [
            (score, result)
            for score, result in scored
            if "missing_alcohol" not in result.get("domain_rerank_notes", [])
        ]
        if alcohol_scored:
            scored = alcohol_scored

    if profile["penalty"]:
        fine_scored = [
            (score, result)
            for score, result in scored
            if "missing_fine" not in result.get("domain_rerank_notes", [])
        ]
        if fine_scored:
            scored = fine_scored

    if profile["administrative_penalty"]:
        non_criminal_scored = [
            (score, result)
            for score, result in scored
            if "criminal_context_penalty" not in result.get("domain_rerank_notes", [])
        ]
        scored = non_criminal_scored

    scored.sort(key=lambda item: item[0], reverse=True)
    retrieval = dict(retrieval)
    retrieval["results"] = [result for _, result in scored[:top_k]]
    retrieval.setdefault("debug", {})
    retrieval["debug"]["temporal_scope"] = (
        {
            "kind": temporal_scope["kind"],
            "label": temporal_scope["label"],
            "start": temporal_scope["start"].isoformat(),
            "end": temporal_scope["end"].isoformat(),
        }
        if temporal_scope
        else None
    )
    retrieval["debug"]["query_profile"] = profile
    retrieval["debug"]["postprocess"] = {
        "input_results": len(results),
        "output_results": len(retrieval["results"]),
        "active_results_before_filter": active_count,
        "rule_based_retrieval_query": retrieval_query,
    }
    return retrieval


def needs_retrieval_postprocess(original_query: str, retrieval_query: str) -> bool:
    profile = _query_profile(retrieval_query)
    return bool(detect_temporal_scope(original_query) or profile["alcohol"] or profile["penalty"] or profile["low_emission_stage"])


def format_context(
    results: list[dict[str, Any]],
    max_passages: int = 5,
    max_chars_per_passage: int = 1800,
    include_effectivity: bool = True,
) -> str:
    blocks = []
    for i, raw_result in enumerate(results[:max_passages], start=1):
        result = repair_mojibake(raw_result)
        doc_number = result.get("document_number") or result.get("document_id") or "Không rõ số hiệu"
        doc_title = result.get("document_title") or ""
        path = result.get("path_text") or result.get("passage_id") or "Không rõ đường dẫn"
        text = (result.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rstrip() + "..."

        title_line = f"Tên văn bản: {doc_title}\n" if doc_title else ""
        effectivity_lines = _effectivity_lines_for_result(result) if include_effectivity else []
        effectivity_text = "".join(f"{line}\n" for line in effectivity_lines)
        blocks.append(
            f"[{i}]\n"
            f"Số hiệu: {doc_number}\n"
            f"{title_line}"
            f"{effectivity_text}"
            f"Đường dẫn: {path}\n"
            f"Nội dung: {text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str, answer_mode: str = "extractive_multi_agent") -> list[dict[str, str]]:
    question = repair_mojibake_text(question)
    today = date.today().isoformat()
    if answer_mode == "direct":
        user_prompt = f"""Câu hỏi:
{question}

Ngày hiện tại dùng để đánh giá hiệu lực: {today}

CONTEXT:
{context}

Hãy trả lời câu hỏi dựa trên các căn cứ trong CONTEXT.
"""
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    if answer_mode != "extractive_multi_agent":
        raise ValueError(f"Unsupported answer_mode: {answer_mode}")

    user_prompt = f"""Câu hỏi:
{question}

Ngày hiện tại dùng để đánh giá hiệu lực: {today}

CONTEXT:
{context}

Yêu cầu:
1. Tự tách câu hỏi thành từng ý.
2. Tự tìm cụm đáp án trực tiếp trong CONTEXT.
3. Câu trả lời cuối cùng phải chứa nguyên văn cụm đáp án quan trọng, đặc biệt là số liệu, mức phạt, thời hạn, điều kiện, hành vi bị cấm.
4. Không in phân tích nội bộ.
5. Chỉ in đúng định dạng đã yêu cầu.
"""
    return [
        {"role": "system", "content": EXTRACTIVE_MULTI_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def run_retriever(
    retriever_script: Path,
    index_dir: Path,
    gazetteer_root: Path,
    query: str,
    top_k: int = 10,
    candidate_k: int = 300,
    dense_weight: float = 0.25,
    bm25_weight: float = 0.25,
    graph_weight: float = 0.20,
    reference_weight: float = 0.30,
    use_reference_expansion: bool = True,
    semantic_entity_top_k: int = 20,
    semantic_entity_min_score: float = 0.45,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(retriever_script),
        "--index-dir",
        str(index_dir),
        "--gazetteer-root",
        str(gazetteer_root),
        "--query",
        query,
        "--top-k",
        str(top_k),
        "--candidate-k",
        str(candidate_k),
        "--semantic-entity-top-k",
        str(semantic_entity_top_k),
        "--semantic-entity-min-score",
        str(semantic_entity_min_score),
        "--dense-weight",
        str(dense_weight),
        "--bm25-weight",
        str(bm25_weight),
        "--graph-weight",
        str(graph_weight),
        "--reference-weight",
        str(reference_weight),
    ]
    if not use_reference_expansion:
        cmd.append("--no-reference-expansion")

    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Retriever failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    try:
        return repair_mojibake(json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot parse retriever JSON output:\n{result.stdout[:2000]}") from exc


def _resolve_torch_dtype(torch_module, dtype: str):
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"float16", "fp16"}:
        return torch_module.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(
    model_name: str,
    load_4bit: bool = False,
    dtype: str = "auto",
    device_map: str = "auto",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = _resolve_torch_dtype(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def _model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generate_answer(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    inputs = tokenizer(text, return_tensors="pt")
    input_device = _model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_answer_api(
    messages: list[dict[str, str]],
    model_name: str,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    timeout_seconds: int = 120,
) -> str:
    provider = provider.lower()
    key = get_api_key(provider, api_key)
    url = provider_base_url(provider, base_url) + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://local.traffic-bot"
        headers["X-Title"] = "Traffic Bot RAG"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider} API error {exc.code}: {body}") from exc

    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected {provider} API response: {json.dumps(data, ensure_ascii=False)[:2000]}") from exc


def generate_answer_with_backend(
    messages: list[dict[str, str]],
    model_name: str,
    mode: str,
    tokenizer=None,
    model=None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    mode = mode.lower()
    if mode == "local":
        if tokenizer is None or model is None:
            raise ValueError("Local generation requires tokenizer and model.")
        return generate_answer(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    if mode in {"openai", "openrouter"}:
        return generate_answer_api(
            messages=messages,
            model_name=model_name,
            provider=mode,
            api_key=api_key,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    raise ValueError(f"Unsupported generation mode: {mode}")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _fallback_query_preprocessing(query: str, reason: str = "fallback") -> dict[str, str]:
    return {
        "route": ROUTE_TRAFFIC_LAW,
        "rewritten_query": repair_mojibake_text(query),
        "reason": reason,
        "chat_answer": "",
        "raw_response": "",
    }


def _looks_like_legal_rag_query(query: str) -> bool:
    return bool(LEGAL_RAG_HINT_PATTERN.search(repair_mojibake_text(query or "")))


def preprocess_user_query(
    query: str,
    model_name: str,
    mode: str,
    tokenizer=None,
    model=None,
    api_key: str | None = None,
    base_url: str | None = None,
    enabled: bool = True,
) -> dict[str, str]:
    query = repair_mojibake_text(query).strip()
    if not enabled:
        return _fallback_query_preprocessing(query, reason="query router disabled")

    messages = [
        {"role": "system", "content": QUERY_ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Câu hỏi người dùng:\n{query}"},
    ]
    try:
        raw_response = generate_answer_with_backend(
            messages=messages,
            model_name=model_name,
            mode=mode,
            tokenizer=tokenizer,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as exc:
        return _fallback_query_preprocessing(query, reason=f"query router failed: {exc}")

    parsed = _extract_json_object(raw_response)
    if not parsed:
        fallback = _fallback_query_preprocessing(query, reason="query router returned non-json")
        fallback["raw_response"] = raw_response
        return fallback

    route = str(parsed.get("route") or "").strip().lower()
    if route not in {ROUTE_TRAFFIC_LAW, ROUTE_GENERAL_CHAT}:
        route = ROUTE_TRAFFIC_LAW
    if route == ROUTE_GENERAL_CHAT and _looks_like_legal_rag_query(query):
        route = ROUTE_TRAFFIC_LAW

    rewritten_query = str(parsed.get("rewritten_query") or "").strip()
    if route == ROUTE_TRAFFIC_LAW and not rewritten_query:
        rewritten_query = query
    if route == ROUTE_GENERAL_CHAT:
        rewritten_query = ""

    return {
        "route": route,
        "rewritten_query": repair_mojibake_text(rewritten_query),
        "reason": (
            "legal citation/effectivity heuristic override"
            if route == ROUTE_TRAFFIC_LAW and _looks_like_legal_rag_query(query) and parsed.get("route") == ROUTE_GENERAL_CHAT
            else str(parsed.get("reason") or "").strip()
        ),
        "chat_answer": "" if route == ROUTE_TRAFFIC_LAW else str(parsed.get("chat_answer") or "").strip(),
        "raw_response": raw_response,
    }


def build_general_chat_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": repair_mojibake_text(query)},
    ]


def answer_one(
    query: str,
    model_name: str,
    mode: str,
    retriever_script: Path,
    index_dir: Path,
    gazetteer_root: Path,
    top_k: int,
    max_context_passages: int,
    candidate_k: int = 300,
    dense_weight: float = 0.25,
    bm25_weight: float = 0.25,
    graph_weight: float = 0.20,
    reference_weight: float = 0.30,
    use_reference_expansion: bool = True,
    semantic_entity_top_k: int = 20,
    semantic_entity_min_score: float = 0.45,
    load_4bit: bool = False,
    dtype: str = "auto",
    device_map: str = "auto",
    answer_mode: str = "extractive_multi_agent",
    enable_query_rewrite: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    max_chars_per_passage: int = 1800,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> dict[str, Any]:
    query = repair_mojibake_text(query).strip()
    tokenizer = None
    model = None

    early_effectivity_answer = _direct_effectivity_answer(query)
    if early_effectivity_answer:
        return {
            "query": query,
            "rewritten_query": query,
            "route": ROUTE_TRAFFIC_LAW,
            "route_reason": "structured effectivity fast-path",
            "query_preprocessing": {
                "route": ROUTE_TRAFFIC_LAW,
                "rewritten_query": query,
                "reason": "structured effectivity fast-path",
                "chat_answer": "",
                "raw_response": "",
            },
            "mode": mode,
            "model": model_name,
            "answer_mode": "structured_effectivity",
            "prompt_version": PROMPT_VERSION,
            "answer": early_effectivity_answer,
            "context_used": "Structured effectivity metadata from data/preprocessed/effectivity.",
            "retrieval": None,
        }

    if mode == "local" and enable_query_rewrite:
        tokenizer, model = load_model(
            model_name,
            load_4bit=load_4bit,
            dtype=dtype,
            device_map=device_map,
        )

    query_preprocessing = preprocess_user_query(
        query=query,
        model_name=model_name,
        mode=mode,
        tokenizer=tokenizer,
        model=model,
        api_key=api_key,
        base_url=base_url,
        enabled=enable_query_rewrite,
    )
    route = query_preprocessing["route"]

    if route == ROUTE_GENERAL_CHAT:
        answer = query_preprocessing.get("chat_answer") or ""
        if not answer:
            if mode == "local" and (tokenizer is None or model is None):
                tokenizer, model = load_model(
                    model_name,
                    load_4bit=load_4bit,
                    dtype=dtype,
                    device_map=device_map,
                )
            answer = generate_answer_with_backend(
                messages=build_general_chat_messages(query),
                model_name=model_name,
                mode=mode,
                tokenizer=tokenizer,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        return {
            "query": query,
            "rewritten_query": "",
            "route": route,
            "route_reason": query_preprocessing.get("reason", ""),
            "query_preprocessing": query_preprocessing,
            "mode": mode,
            "model": model_name,
            "answer_mode": "general_chat",
            "prompt_version": PROMPT_VERSION,
            "answer": answer,
            "context_used": "",
            "retrieval": None,
        }

    direct_effectivity_answer = _direct_effectivity_answer(query)
    if direct_effectivity_answer:
        return {
            "query": query,
            "rewritten_query": query_preprocessing.get("rewritten_query") or query,
            "route": route,
            "route_reason": query_preprocessing.get("reason", ""),
            "query_preprocessing": query_preprocessing,
            "mode": mode,
            "model": model_name,
            "answer_mode": "structured_effectivity",
            "prompt_version": PROMPT_VERSION,
            "answer": direct_effectivity_answer,
            "context_used": "Structured effectivity metadata from data/preprocessed/effectivity.",
            "retrieval": None,
        }

    retrieval_query = apply_rule_based_query_rewrite(query_preprocessing.get("rewritten_query") or query)
    retrieval_top_k = max(top_k, 40) if needs_retrieval_postprocess(query, retrieval_query) else top_k
    retrieval = run_retriever(
        retriever_script=retriever_script,
        index_dir=index_dir,
        gazetteer_root=gazetteer_root,
        query=retrieval_query,
        top_k=retrieval_top_k,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        graph_weight=graph_weight,
        reference_weight=reference_weight,
        use_reference_expansion=use_reference_expansion,
        semantic_entity_top_k=semantic_entity_top_k,
        semantic_entity_min_score=semantic_entity_min_score,
    )
    retrieval = postprocess_retrieval_for_query(retrieval, query, retrieval_query, top_k=top_k)
    context = format_context(
        retrieval.get("results", []),
        max_passages=max_context_passages,
        max_chars_per_passage=max_chars_per_passage,
    )
    if not context.strip():
        answer = INSUFFICIENT_CONTEXT_ANSWER
    else:
        messages = build_prompt(query, context, answer_mode=answer_mode)
        if mode == "local":
            if tokenizer is None or model is None:
                tokenizer, model = load_model(
                    model_name,
                    load_4bit=load_4bit,
                    dtype=dtype,
                    device_map=device_map,
                )
            answer = generate_answer_with_backend(
                messages=messages,
                model_name=model_name,
                mode=mode,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            answer = generate_answer_with_backend(
                messages=messages,
                model_name=model_name,
                mode=mode,
                api_key=api_key,
                base_url=base_url,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    return {
        "query": query,
        "rewritten_query": retrieval_query,
        "route": route,
        "route_reason": query_preprocessing.get("reason", ""),
        "query_preprocessing": query_preprocessing,
        "mode": mode,
        "model": model_name,
        "answer_mode": answer_mode,
        "prompt_version": PROMPT_VERSION,
        "answer": answer,
        "context_used": context,
        "retrieval": retrieval,
    }
