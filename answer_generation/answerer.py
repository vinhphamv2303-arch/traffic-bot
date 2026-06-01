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
from typing import Any, Callable

from answer_generation.conversation_memory import (
    ConversationMemory,
    coerce_memory,
    empty_memory,
    is_reset_query,
    resolve_query_with_memory,
    update_memory_after_answer,
)


ROOT = Path(__file__).resolve().parents[1]
EFFECTIVITY_ROOT = ROOT / "data" / "preprocessed" / "effectivity"

ProgressCallback = Callable[[str, dict[str, Any]], None]

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
   - "traffic_law": câu hỏi liên quan giao thông đường bộ Việt Nam, xử phạt vi phạm giao thông, giấy phép lái xe, đăng kiểm, vận tải đường bộ, hạ tầng đường bộ, phương tiện, người điều khiển phương tiện, đăng ký xe, cấp/thu hồi đăng ký và biển số xe, biển kiểm soát, ký hiệu biển số xe theo tỉnh/thành phố, hoặc quy định của địa phương về tổ chức giao thông, hạn chế phương tiện, vùng phát thải thấp, khu vực hạn chế phương tiện giao thông gây ô nhiễm môi trường.
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

- Các câu hỏi như "tỉnh X mang biển số mấy", "biển số xe của tỉnh X", "ký hiệu biển số tỉnh/thành phố", "biển kiểm soát của địa phương X" phải route là "traffic_law".
- Nếu không chắc câu hỏi có thuộc corpus pháp luật giao thông hay không, ưu tiên route "traffic_law" để truy xuất tài liệu.
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
    r"\b(quy\s*định\s*hiện\s*hành|giao\s*thông|đường\s*bộ|phương\s*tiện|vận\s*tải|đăng\s*kiểm|giấy\s*phép\s*lái\s*xe|vùng\s*phát\s*thải\s*thấp|khu\s*vực\s*hạn\s*chế\s*phương\s*tiện|hạn\s*chế\s*phương\s*tiện|phương\s*tiện\s*giao\s*thông\s*gây\s*ô\s*nhiễm|ô\s*nhiễm\s*môi\s*trường|biển\s*số|biển\s*kiểm\s*soát|bảng\s*số|ký\s*hiệu\s*biển|mã\s*biển|mang\s*biển\s*số|đăng\s*ký\s*xe|cấp\s*biển\s*số|thu\s*hồi\s*biển\s*số)\b"
    r"|"
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐa-zđ-]+"
    r"|"
    r"\b(NĐ-CP|ND-CP|TT-BGTVT|TT-BCA|QH\d+|UBTVQH\d+)\b"
    r")",
    flags=re.IGNORECASE,
)

LICENSE_PLATE_QUERY_PATTERN = re.compile(
    r"\b("
    r"biển\s*số|biển\s*kiểm\s*soát|bảng\s*số|"
    r"ký\s*hiệu\s*biển|mã\s*biển|mang\s*biển\s*số|"
    r"đăng\s*ký\s*xe|cấp\s*biển\s*số|thu\s*hồi\s*biển\s*số"
    r")\b",
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
TIME_DRIVING_QUERY_PATTERN = re.compile(
    r"\b("
    r"thời\s*gian\s*lái\s*xe|"
    r"lái\s*xe\s*liên\s*tục|"
    r"thời\s*gian\s*làm\s*việc\s*của\s*người\s*lái\s*xe|"
    r"không\s*quá\s*04\s*giờ|"
    r"không\s*quá\s*4\s*giờ|"
    r"vượt\s*quá\s*4\s*giờ|"
    r"vượt\s*quá\s*04\s*giờ"
    r")\b",
    flags=re.IGNORECASE,
)
TEMPORAL_APPLICABILITY_PATTERN = re.compile(
    r"\b("
    r"phạt\s*nguội|"
    r"mức\s*cũ|mức\s*mới|quy\s*định\s*cũ|quy\s*định\s*mới|"
    r"cuối\s*tháng|đầu\s*tháng|trước\s*ngày|sau\s*ngày|"
    r"thời\s*điểm\s*vi\s*phạm|thời\s*điểm\s*xử\s*phạt|"
    r"bị\s*xử\s*phạt\s*sau|vi\s*phạm\s*từ|vi\s*phạm\s*vào"
    r")\b",
    flags=re.IGNORECASE,
)
CURRENT_TIME_QUERY_PATTERN = re.compile(r"\b(hiện\s*tại|hiện\s*nay|bây\s*giờ|ngày\s*nay)\b", flags=re.IGNORECASE)
YEAR_TIME_QUERY_PATTERN = re.compile(r"\b(?:năm|nam)\s*(20\d{2}|19\d{2})\b", flags=re.IGNORECASE)
MONTH_YEAR_TIME_PATTERN = re.compile(
    r"\b(?:cuối|cuoi|đầu|dau|giữa|giua)?\s*(?:tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)\s*(20\d{2}|19\d{2})\b",
    flags=re.IGNORECASE,
)
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


def _coerce_conversation_memory(memory: ConversationMemory | dict[str, Any] | None) -> ConversationMemory | None:
    return coerce_memory(memory)


def _memory_dict(memory: ConversationMemory | None) -> dict[str, Any]:
    return (memory or empty_memory()).to_dict()


def _emit_progress(progress_callback: ProgressCallback | None, stage: str, **payload: Any) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, payload)
    except Exception:
        return


def _make_conversation_resolver_llm_call(
    model_name: str,
    mode: str,
    tokenizer=None,
    model=None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    def llm_call(messages: list[dict[str, str]]) -> str:
        return generate_answer_with_backend(
            messages=messages,
            model_name=model_name,
            mode=mode,
            tokenizer=tokenizer,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_new_tokens=700,
            temperature=0.0,
            top_p=1.0,
        )

    return llm_call


def _map_resolver_route(route: str) -> str:
    route = (route or "").strip().lower()
    if route in {"normal_chat", ROUTE_GENERAL_CHAT}:
        return ROUTE_GENERAL_CHAT
    if route == "effectivity_index":
        return "effectivity_index"
    return ROUTE_TRAFFIC_LAW


def _query_preprocessing_from_resolution(
    original_query: str,
    processing_query: str,
    resolution: dict[str, Any],
) -> dict[str, Any] | None:
    if not resolution.get("accepted"):
        return None

    route = _map_resolver_route(str(resolution.get("route") or ""))
    force_retrieval = _should_force_retrieval(original_query) or _should_force_retrieval(processing_query)
    if route == ROUTE_GENERAL_CHAT and force_retrieval:
        route = ROUTE_TRAFFIC_LAW

    rewritten_query = "" if route == ROUTE_GENERAL_CHAT else processing_query
    return {
        "route": route,
        "rewritten_query": repair_mojibake_text(rewritten_query),
        "reason": (
            "traffic-domain heuristic override"
            if route == ROUTE_TRAFFIC_LAW and force_retrieval and resolution.get("route") in {"normal_chat", ROUTE_GENERAL_CHAT}
            else f"conversation resolver: {resolution.get('relation') or ''}; {resolution.get('reason') or ''}".strip()
        ),
        "chat_answer": "",
        "raw_response": resolution.get("raw_response") or "",
        "conversation_resolution": resolution,
    }


GENERAL_CHAT_HINT_PATTERN = re.compile(
    r"\b("
    r"xin\s*chào|chào\s*bạn|hello|hi|hey|"
    r"cảm\s*ơn|thank\s*you|thanks|"
    r"bạn\s*là\s*ai|giới\s*thiệu\s*về\s*bạn|"
    r"hôm\s*nay|thời\s*tiết|kể\s*chuyện|đùa"
    r")\b",
    flags=re.IGNORECASE,
)


def _fallback_route_for_query(query: str, processing_query: str, memory_context: str = "") -> str:
    query = repair_mojibake_text(query or "")
    processing_query = repair_mojibake_text(processing_query or "")

    if EFFECTIVITY_QUERY_PATTERN.search(processing_query):
        return "effectivity_index"

    if memory_context:
        return ROUTE_TRAFFIC_LAW

    if (
        _should_force_retrieval(query)
        or _should_force_retrieval(processing_query)
        or ALCOHOL_QUERY_PATTERN.search(processing_query)
        or PENALTY_QUERY_PATTERN.search(processing_query)
        or TIME_DRIVING_QUERY_PATTERN.search(processing_query)
        or re.search(r"\b(vượt\s*đèn\s*đỏ|đèn\s*tín\s*hiệu|mũ\s*bảo\s*hiểm|bằng\s*lái|gplx)\b", processing_query, re.IGNORECASE)
    ):
        return ROUTE_TRAFFIC_LAW

    if GENERAL_CHAT_HINT_PATTERN.search(query):
        return ROUTE_GENERAL_CHAT

    return ROUTE_TRAFFIC_LAW


def _fallback_query_preprocessing_from_query(
    original_query: str,
    processing_query: str,
    conversation_resolution: dict[str, Any],
    memory_context: str = "",
) -> dict[str, Any]:
    route = _fallback_route_for_query(original_query, processing_query, memory_context=memory_context)
    return {
        "route": route,
        "rewritten_query": "" if route == ROUTE_GENERAL_CHAT else repair_mojibake_text(processing_query),
        "reason": str(conversation_resolution.get("reason") or "conversation resolver fallback"),
        "chat_answer": "",
        "raw_response": str(conversation_resolution.get("raw_response") or ""),
        "conversation_resolution": conversation_resolution,
    }


def _display_resolved_query(original_query: str, processing_query: str) -> str:
    original_query = repair_mojibake_text(original_query or "").strip()
    processing_query = repair_mojibake_text(processing_query or "").strip()
    if processing_query and processing_query != original_query:
        return processing_query
    return ""


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


def _result_consequence_tags(result: dict[str, Any]) -> list[str]:
    text = _result_body_match_text(result)
    tags: list[str] = []
    if "phat tien tu" in text or ("phat tien" in text and "dong" in text):
        tags.append("phat_tien")
    if "bi tru diem giay phep lai xe" in text or "tru diem giay phep lai xe" in text:
        tags.append("tru_diem_gplx")
    if "tuoc quyen su dung giay phep lai xe" in text:
        tags.append("tuoc_gplx")
    if "hinh thuc xu phat bo sung" in text:
        tags.append("xu_phat_bo_sung")
    return tags


def _result_references_selector(result: dict[str, Any], selector: dict[str, str]) -> bool:
    if not selector:
        return False
    text = _result_body_match_text(result, tail_depth=6)
    article = selector.get("article")
    clause = selector.get("clause")
    point = selector.get("point")
    if article and f"dieu {article}" not in text:
        return False
    if clause and f"khoan {clause}" not in text:
        return False
    if point:
        aliases = _point_aliases(point)
        if not any(f"diem {alias.replace('đ', 'd')}" in text for alias in aliases):
            return False
    return bool(article or clause or point)


def _is_direct_penalty_anchor(result: dict[str, Any], profile: dict[str, Any]) -> bool:
    selector = _extract_unit_selector(result)
    if not selector.get("article") or not selector.get("clause"):
        return False
    text = _result_body_match_text(result)
    notes = set(result.get("domain_rerank_notes") or [])
    consequence_tags = set(_result_consequence_tags(result))
    if "phat_tien" not in consequence_tags:
        return False
    if "tru_diem_gplx" in consequence_tags or "tuoc_gplx" in consequence_tags:
        return False
    if "missing_fine" in notes:
        return False
    if profile.get("traffic_light_signal") and "missing_traffic_light" in notes:
        return False
    if profile.get("vehicle_target") and "vehicle_article_mismatch" in notes:
        return False
    return "phat tien tu" in text or ("phat tien" in text and "dong" in text)


def _anchor_selectors_for_consequences(
    scored: list[tuple[float, dict[str, Any]]],
    profile: dict[str, Any],
    limit: int = 2,
) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for _, result in scored:
        if not _is_direct_penalty_anchor(result, profile):
            continue
        selector = _extract_unit_selector(result)
        key = (
            selector.get("article", ""),
            selector.get("clause", ""),
            selector.get("point", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        anchors.append(selector)
        if len(anchors) >= limit:
            break
    return anchors


def _boost_related_consequence_results(
    scored: list[tuple[float, dict[str, Any]]],
    profile: dict[str, Any],
) -> tuple[list[tuple[float, dict[str, Any]]], list[dict[str, str]], int]:
    if not profile.get("administrative_penalty"):
        return scored, [], 0

    anchors = _anchor_selectors_for_consequences(scored, profile)
    if not anchors:
        return scored, [], 0

    boosted_count = 0
    updated: list[tuple[float, dict[str, Any]]] = []
    for score, raw_result in scored:
        result = dict(raw_result)
        tags = set(_result_consequence_tags(result))
        notes = list(result.get("domain_rerank_notes") or [])
        bonus = 0.0
        for selector in anchors:
            if not _result_references_selector(result, selector):
                continue
            if "tru_diem_gplx" in tags:
                bonus = max(bonus, 8.5)
                if "point_deduction_selector_match" not in notes:
                    notes.append("point_deduction_selector_match")
            if "tuoc_gplx" in tags:
                bonus = max(bonus, 7.0)
                if "license_revocation_selector_match" not in notes:
                    notes.append("license_revocation_selector_match")
        if bonus > 0:
            boosted_count += 1
            score += bonus
            result["domain_rerank_notes"] = notes
            result["rerank_score"] = round(score, 6)
        updated.append((score, result))
    updated.sort(key=lambda item: item[0], reverse=True)
    return updated, anchors, boosted_count


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

    license_plate_pattern = globals().get("LICENSE_PLATE_QUERY_PATTERN")
    if (
        (license_plate_pattern is not None and license_plate_pattern.search(query))
        or any(
            term in normalized
            for term in (
                "bien so",
                "bien kiem soat",
                "bang so",
                "ky hieu bien",
                "ma bien",
                "mang bien so",
                "dang ky xe",
                "cap bien so",
                "thu hoi bien so",
            )
        )
    ):
        rewritten_parts.append("biển số xe biển kiểm soát ký hiệu biển số đăng ký xe")
        rewritten_parts.append("ký hiệu biển số xe của tỉnh thành phố trực thuộc trung ương")
        rewritten_parts.append("phụ lục ký hiệu biển số xe")

    if "say" in normalized or "ruou" in normalized or "bia" in normalized or "nong do con" in normalized:
        alcohol_phrase = "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn"
        if "nong do con" not in normalized:
            rewritten_parts.append(alcohol_phrase)
        if "phat" in normalized or "xu phat" in normalized or "bao nhieu" in normalized:
            rewritten_parts.append("mức phạt tiền")

    if "vuot den do" in normalized:
        if "nguoi di bo" in normalized:
            rewritten_parts.append("người đi bộ không chấp hành hiệu lệnh hoặc chỉ dẫn của đèn tín hiệu")
        else:
            rewritten_parts.append("không chấp hành hiệu lệnh của đèn tín hiệu giao thông")
        if "phat" in normalized or "bao nhieu" in normalized:
            rewritten_parts.append("mức phạt tiền")

    if "bang lai" in normalized or "gplx" in normalized:
        rewritten_parts.append("giấy phép lái xe")

    if (
            "thoi gian lai xe" in normalized
            or "lai xe lien tuc" in normalized
            or "thoi gian lam viec cua nguoi lai xe" in normalized
            or "vuot qua 4 gio" in normalized
            or "vuot qua 04 gio" in normalized
            or "khong qua 4 gio" in normalized
            or "khong qua 04 gio" in normalized
    ):
        rewritten_parts.append(
            "thời gian làm việc của người lái xe ô tô kinh doanh vận tải và vận tải nội bộ"
        )
        rewritten_parts.append(
            "lái xe liên tục không quá 04 giờ thời gian lái xe không quá 10 giờ trong một ngày không quá 48 giờ trong một tuần"
        )
        if (
                "phat" in normalized
                or "xu phat" in normalized
                or "bi phat" in normalized
                or "vuot qua" in normalized
        ):
            rewritten_parts.append(
                "xử phạt vi phạm quy định về thời gian lái xe liên tục thời gian làm việc của người lái xe"
            )

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


def _query_temporal_window(query: str) -> dict[str, date] | None:
    query = repair_mojibake_text(query or "")
    starts: list[date] = []
    ends: list[date] = []

    for match in SLASH_DATE_PATTERN.finditer(query):
        day, month, year = map(int, match.groups())
        try:
            point = date(year, month, day)
        except ValueError:
            continue
        starts.append(point)
        ends.append(point)

    for match in MONTH_YEAR_TIME_PATTERN.finditer(query):
        month = int(match.group(1))
        year = int(match.group(2))
        if month < 1 or month > 12:
            continue
        start = date(year, month, 1)
        end = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
        starts.append(start)
        ends.append(end)

    for match in YEAR_TIME_QUERY_PATTERN.finditer(query):
        year = int(match.group(1))
        starts.append(date(year, 1, 1))
        ends.append(date(year + 1, 1, 1))

    if not starts:
        return None
    return {"start": min(starts), "end": max(ends)}


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


def _result_primary_text(result: dict[str, Any]) -> str:
    text = repair_mojibake_text(str(result.get("text") or ""))
    for marker in ("Nội dung:", "Noi dung:"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for marker in ("Tham chiếu đã giải:", "Resolved references:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def _result_body_match_text(result: dict[str, Any], tail_depth: int = 2) -> str:
    path = str(result.get("path_text") or "")
    path_parts = [part.strip() for part in path.split(">") if part.strip()]
    path_tail = " > ".join(path_parts[-tail_depth:])
    return _normalize_for_match(
        " ".join(
            str(value or "")
            for value in [
                path_tail,
                _result_primary_text(result),
            ]
        )
    )


def _has_time_driving_terms(text: str) -> bool:
    return any(
        term in text
        for term in [
            "thoi gian lai xe",
            "lai xe lien tuc",
            "thoi gian lam viec cua nguoi lai xe",
            "nguoi lai xe o to kinh doanh van tai",
            "van tai noi bo",
            "khong qua 04 gio",
            "khong qua 4 gio",
            "04 gio",
            "4 gio",
            "10 gio",
            "48 gio",
        ]
    )


def _has_temporal_applicability_terms(text: str) -> bool:
    return any(
        term in text
        for term in [
            "quy dinh chuyen tiep",
            "dieu khoan thi hanh",
            "thoi hieu xu phat",
            "hanh vi vi pham hanh chinh da ket thuc",
            "hanh vi vi pham hanh chinh dang thuc hien",
            "hieu luc thi hanh",
            "co hieu luc thi hanh",
            "ap dung",
            "truoc ngay",
            "sau ngay",
            "truoc khi",
            "sau khi",
        ]
    )


def _vehicle_target_from_query(normalized: str) -> str | None:
    if re.search(r"\bnguoi\s+di\s+bo\b", normalized):
        return "pedestrian"
    if re.search(r"\bxe\s+may\s+chuyen\s+dung\b", normalized):
        return "special_machine"
    if re.search(r"\b(xe\s+may|mo\s+to|xe\s+mo\s+to|xe\s+gan\s+may)\b", normalized):
        return "motorcycle"
    if re.search(r"\b(xe\s+o\s+to|o\s+to|oto)\b", normalized):
        return "car"
    if re.search(r"\b(xe\s+dap|xe\s+dap\s+may|xe\s+dap\s+dien)\b", normalized):
        return "bicycle"
    return None


def _query_profile(query: str) -> dict[str, Any]:
    normalized = _normalize_for_match(query)
    vehicle_target = _vehicle_target_from_query(normalized)
    penalty = bool(
        PENALTY_QUERY_PATTERN.search(query)
        or re.search(r"\b(muc\s+phat|bi\s+phat|phat\s+bao\s+nhieu|xu\s+phat|phat\s+tien|tien\s+phat)\b", normalized)
    )
    criminal_intent = any(token in normalized for token in ["hinh su", "phat tu", "toi pham", "truy cuu", "trach nhiem hinh su"])
    time_driving = (
        bool(TIME_DRIVING_QUERY_PATTERN.search(query))
        or any(
            token in normalized
            for token in [
                "thoi gian lai xe",
                "lai xe lien tuc",
                "thoi gian lam viec cua nguoi lai xe",
                "nguoi lai xe o to kinh doanh van tai",
                "van tai noi bo",
                "vuot qua 4 gio",
                "vuot qua 04 gio",
                "khong qua 4 gio",
                "khong qua 04 gio",
                "04 gio",
                "4 gio",
            ]
        )
    )
    license_point_deduction = any(
        token in normalized
        for token in [
            "tru diem",
            "bi tru diem",
            "tru bao nhieu diem",
            "giay phep lai xe con bao nhieu diem",
            "diem giay phep lai xe",
            "gplx bi tru",
        ]
    )
    temporal_applicability = (
        bool(TEMPORAL_APPLICABILITY_PATTERN.search(query))
        or (
            bool(YEAR_TIME_QUERY_PATTERN.search(query) or SLASH_DATE_PATTERN.search(query))
            and any(
                token in normalized
                for token in [
                    "muc cu",
                    "muc moi",
                    "quy dinh cu",
                    "quy dinh moi",
                    "phat nguoi",
                    "bi xu phat sau",
                    "ap dung",
                ]
            )
        )
    )
    return {
        "alcohol": bool(ALCOHOL_QUERY_PATTERN.search(query)) or any(token in normalized for token in ["say", "ruou", "bia", "nong do con"]),
        "penalty": penalty,
        "criminal_intent": criminal_intent,
        "administrative_penalty": penalty and not criminal_intent,
        "license_point_deduction": license_point_deduction,
        "time_driving": time_driving,
        "temporal_applicability": temporal_applicability,
        "traffic_light_signal": (
            "den tin hieu giao thong" in normalized
            or "den tin hieu dieu khien giao thong" in normalized
            or "vuot den do" in normalized
            or "khong chap hanh hieu lenh cua den tin hieu" in normalized
        ),
        "low_emission_stage": (
            "vung phat thai thap" in normalized
            and ("lo trinh" in normalized or "giai doan" in normalized or "trien khai" in normalized)
        ),
        "vehicle_target": vehicle_target,
        "vehicle_unspecified": vehicle_target is None,
    }


def _domain_relevance_score(result: dict[str, Any], profile: dict[str, Any]) -> tuple[float, list[str]]:
    full_text = _result_match_text(result)
    text = _result_body_match_text(result)
    path_text = _normalize_for_match(result.get("path_text") or "")
    score = 0.0
    notes: list[str] = []
    consequence_tags = set(_result_consequence_tags(result))
    point_deduction_match = "tru_diem_gplx" in consequence_tags

    if profile.get("temporal_applicability"):
        if _has_temporal_applicability_terms(full_text):
            score += 7.0
            notes.append("temporal_applicability_match")
        else:
            score -= 8.0
            notes.append("missing_temporal_applicability")
        if "dieu khoan chuyen tiep" in full_text or "quy dinh chuyen tiep" in full_text:
            score += 4.0
            notes.append("transition_clause_match")
        if (
            "tai thoi diem thuc hien hanh vi vi pham" in full_text
            or "xay ra va ket thuc truoc ngay" in full_text
            or "sau do moi bi phat hien" in full_text
        ):
            score += 4.0
            notes.append("temporal_rule_statement_match")
        if "phat tien tu" in text and not _has_temporal_applicability_terms(full_text):
            score -= 6.0
            notes.append("specific_penalty_not_principle")

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

    if profile.get("license_point_deduction"):
        if point_deduction_match:
            score += 4.0
            notes.append("point_deduction_match")
        else:
            score -= 3.0
            notes.append("missing_point_deduction")
    elif profile["penalty"] and point_deduction_match:
        score += 1.5
        notes.append("point_deduction_related")

    if profile.get("time_driving"):
        if _has_time_driving_terms(text):
            score += 6.0
            notes.append("time_driving_match")
        else:
            score -= 8.0
            notes.append("missing_time_driving")

    if profile["alcohol"] and profile["penalty"] and "nghi dinh quy dinh xu phat vi pham hanh chinh" in full_text:
        score += 2.0
        notes.append("admin_penalty_decree")

    if profile.get("traffic_light_signal"):
        if (
            "khong chap hanh hieu lenh cua den tin hieu giao thong" in text
            or "khong chap hanh hieu lenh hoac chi dan cua den tin hieu" in text
            or "den tin hieu" in text
            or "den tin hieu giao thong" in text
            or "den tin hieu dieu khien giao thong" in text
        ):
            score += 4.0
            notes.append("traffic_light_match")
        else:
            score -= 2.0
            notes.append("missing_traffic_light")

    vehicle_target = profile.get("vehicle_target")
    if vehicle_target:
        target_article = {
            "car": "6",
            "motorcycle": "7",
            "special_machine": "8",
            "bicycle": "9",
            "pedestrian": "10",
        }.get(vehicle_target)
        article_match = re.search(r"dieu\s+(6|7|8|9|10)\b", path_text)
        if target_article and article_match:
            if article_match.group(1) == target_article:
                score += 5.0
                notes.append("vehicle_article_match")
            else:
                score -= 4.0
                notes.append("vehicle_article_mismatch")
        if vehicle_target == "pedestrian":
            if "nguoi di bo" in full_text or "nguoi di bo" in path_text:
                score += 3.0
                notes.append("pedestrian_actor_match")
            else:
                score -= 4.0
                notes.append("pedestrian_actor_mismatch")

    if profile["vehicle_unspecified"]:
        article_match = re.search(r"dieu\s+(6|7|8|9|10)\b", path_text)
        if article_match:
            score += {"6": 0.4, "7": 0.3, "8": 0.2, "9": 0.1, "10": 0.1}[article_match.group(1)]
        if profile.get("temporal_applicability") and article_match:
            score -= 3.5
            notes.append("specific_vehicle_penalty_demote")

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


def _prioritize_by_note_absence(
    scored: list[tuple[float, dict[str, Any]]],
    note: str,
) -> tuple[list[tuple[float, dict[str, Any]]], int]:
    preferred: list[tuple[float, dict[str, Any]]] = []
    fallback: list[tuple[float, dict[str, Any]]] = []
    for item in scored:
        result = item[1]
        if note in result.get("domain_rerank_notes", []):
            fallback.append(item)
        else:
            preferred.append(item)
    return preferred + fallback, len(preferred)


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
    query_temporal_window = _query_temporal_window(original_query) if profile.get("temporal_applicability") else None
    if profile.get("temporal_applicability"):
        temporal_scope = None
    elif not temporal_scope and profile["administrative_penalty"]:
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
        if query_temporal_window:
            metadata = load_effectivity_metadata()
            doc_meta = _doc_effectivity(result, metadata)
            start_date = _parse_iso_date((doc_meta or {}).get("effective_from"))
            if start_date:
                if start_date >= query_temporal_window["end"]:
                    temporal_score -= 8.0
                    notes.append("effective_after_query_window")
                else:
                    temporal_score += 2.0
                    notes.append("effective_within_query_window")
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

    scored.sort(key=lambda item: item[0], reverse=True)
    priority_counts: dict[str, int] = {}

    if profile["alcohol"]:
        scored, preferred_count = _prioritize_by_note_absence(scored, "missing_alcohol")
        priority_counts["missing_alcohol"] = preferred_count

    if profile["penalty"]:
        scored, preferred_count = _prioritize_by_note_absence(scored, "missing_fine")
        priority_counts["missing_fine"] = preferred_count

    if profile.get("traffic_light_signal"):
        scored, preferred_count = _prioritize_by_note_absence(scored, "missing_traffic_light")
        priority_counts["missing_traffic_light"] = preferred_count

    if profile.get("time_driving"):
        scored, preferred_count = _prioritize_by_note_absence(scored, "missing_time_driving")
        priority_counts["missing_time_driving"] = preferred_count

    if profile.get("temporal_applicability"):
        scored, preferred_count = _prioritize_by_note_absence(scored, "missing_temporal_applicability")
        priority_counts["missing_temporal_applicability"] = preferred_count
        scored, preferred_count = _prioritize_by_note_absence(scored, "specific_penalty_not_principle")
        priority_counts["specific_penalty_not_principle"] = preferred_count
        scored, preferred_count = _prioritize_by_note_absence(scored, "effective_after_query_window")
        priority_counts["effective_after_query_window"] = preferred_count

    if profile.get("license_point_deduction"):
        scored, preferred_count = _prioritize_by_note_absence(scored, "missing_point_deduction")
        priority_counts["missing_point_deduction"] = preferred_count

    if profile["administrative_penalty"]:
        scored, preferred_count = _prioritize_by_note_absence(scored, "criminal_context_penalty")
        priority_counts["criminal_context_penalty"] = preferred_count

    scored, consequence_anchors, boosted_consequences = _boost_related_consequence_results(scored, profile)

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
    retrieval["debug"]["query_temporal_window"] = (
        {
            "start": query_temporal_window["start"].isoformat(),
            "end": query_temporal_window["end"].isoformat(),
        }
        if query_temporal_window
        else None
    )
    retrieval["debug"]["query_profile"] = profile
    retrieval["debug"]["postprocess"] = {
        "input_results": len(results),
        "output_results": len(retrieval["results"]),
        "active_results_before_filter": active_count,
        "priority_counts": priority_counts,
        "rule_based_retrieval_query": retrieval_query,
        "consequence_anchor_selectors": consequence_anchors,
        "boosted_consequence_results": boosted_consequences,
    }
    return retrieval


def needs_retrieval_postprocess(original_query: str, retrieval_query: str) -> bool:
    profile = _query_profile(retrieval_query)
    return bool(
        detect_temporal_scope(original_query)
        or profile["alcohol"]
        or profile["penalty"]
        or profile["low_emission_stage"]
        or profile.get("time_driving")
        or profile.get("vehicle_target")
    )


def _answer_focus_instructions(question: str) -> list[str]:
    profile = _query_profile(question)
    normalized = _normalize_for_match(question)
    instructions: list[str] = []
    if profile.get("temporal_applicability"):
        instructions.append(
            "Neu cau hoi dang hoi nguyen tac ap dung muc phat theo thoi diem xay ra vi pham va thoi diem xu phat, chi duoc ket luan 'muc cu' hoac 'muc moi' neu CONTEXT neu ro nguyen tac ap dung theo thoi diem."
        )
        instructions.append(
            "Neu CONTEXT chi co cac dieu khoan muc phat cu the, thoi hieu xu phat, hieu luc van ban, hoac quy dinh lien quan nhung khong noi ro nguyen tac ap dung theo thoi diem, phai tra loi dung cau: \"Khong tim thay can cu du ro trong tai lieu duoc truy xuat.\""
        )
    if profile.get("administrative_penalty"):
        instructions.append(
            "Neu CONTEXT cho thay cung mot hanh vi va cung loai phuong tien co nhieu hau qua phap ly, phai tong hop day du theo thu tu: phat tien; tru diem giay phep lai xe; tuoc quyen su dung giay phep lai xe; hinh thuc xu phat bo sung khac neu co."
        )
        instructions.append(
            "Khong duoc bo qua tru diem hoac tuoc GPLX chi vi cau hoi dung cum 'bi phat bao nhieu'."
        )
    if profile.get("license_point_deduction"):
        instructions.append(
            "Neu CONTEXT co quy dinh tru diem GPLX, phai neu ro so diem bi tru va can cu tuong ung."
        )
    if (
        ("quang duong" in normalized or re.search(r"\bkm\b", normalized))
        and ("thoi gian" in normalized or re.search(r"\bgio\b", normalized))
    ):
        instructions.append(
            "Neu cau hoi dong thoi hoi quang duong va thoi gian, phai tra loi day du ca hai dai luong; khong duoc chi tra loi mot ve."
        )
    return instructions


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
        consequence_tags = _result_consequence_tags(result)
        consequence_line = f"Loai thong tin: {', '.join(consequence_tags)}\n" if consequence_tags else ""
        blocks.append(
            f"[{i}]\n"
            f"Số hiệu: {doc_number}\n"
            f"{title_line}"
            f"{effectivity_text}"
            f"{consequence_line}"
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

    focus_instructions = _answer_focus_instructions(question)
    focus_text = ""
    if focus_instructions:
        focus_text = "\n".join(f"- {instruction}" for instruction in focus_instructions)

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
    if focus_text:
        user_prompt += f"\n{focus_text}"
    return [
        {"role": "system", "content": EXTRACTIVE_MULTI_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _format_result_citation(result: dict[str, Any]) -> str:
    selector = _extract_unit_selector(result)
    parts: list[str] = []
    if selector.get("article"):
        parts.append(f"Điều {selector['article']}")
    if selector.get("clause"):
        parts.append(f"Khoản {selector['clause']}")
    if selector.get("point"):
        point = selector["point"]
        if point in {"d", "dd"}:
            point = "đ"
        parts.append(f"Điểm {point}")
    doc = result.get("document_number") or result.get("document_id")
    if doc:
        doc_text = str(doc)
        title_text = _normalize_for_match(result.get("document_title") or "")
        if "nghi dinh" in title_text:
            doc_text = f"Nghị định {doc_text}"
        elif "thong tu" in title_text:
            doc_text = f"Thông tư {doc_text}"
        elif "luat" in title_text:
            doc_text = f"Luật {doc_text}"
        parts.append(doc_text)
    return ", ".join(parts) if parts else str(doc or result.get("path_text") or "")


def _extract_point_deduction_summary(result: dict[str, Any]) -> str | None:
    text = _result_body_match_text(result, tail_depth=6)
    match = re.search(r"bi tru diem giay phep lai xe\s+(\d+)\s+diem", text)
    if not match:
        return None
    return f"Ngoài ra, người điều khiển xe thực hiện hành vi này bị trừ điểm giấy phép lái xe {match.group(1)} điểm."


def _extract_license_revocation_summary(result: dict[str, Any]) -> str | None:
    text = _result_body_match_text(result, tail_depth=6)
    match = re.search(
        r"bi tuoc quyen su dung giay phep lai xe tu\s+([0-9]+\s+thang)\s+den\s+([0-9]+\s+thang)",
        text,
    )
    if not match:
        return None
    return (
        "Ngoài ra, người điều khiển xe thực hiện hành vi này bị tước quyền sử dụng giấy phép lái xe "
        f"từ {match.group(1)} đến {match.group(2)}."
    )


def _append_bullets(section_text: str, items: list[str], header: str) -> str:
    section_text = (section_text or "").strip()
    if not items:
        return section_text or header

    if section_text.startswith(header):
        content = section_text[len(header):].strip()
    else:
        content = section_text

    bullets: list[str] = []
    if content:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines and all(line.startswith("- ") for line in lines):
            bullets.extend(lines)
        else:
            bullets.append(f"- {content.lstrip('- ').strip()}")

    normalized_existing = {_normalize_for_match(line) for line in bullets}
    for item in items:
        bullet = f"- {item.strip()}"
        if _normalize_for_match(bullet) not in normalized_existing:
            bullets.append(bullet)
            normalized_existing.add(_normalize_for_match(bullet))

    return f"{header}\n" + "\n".join(bullets)


def _answer_mentions_point_deduction(normalized_answer: str) -> bool:
    return bool(
        re.search(
            r"\b(tru\s+\d+\s+diem\s+giay\s+phep\s+lai\s+xe|tru\s+diem\s+giay\s+phep\s+lai\s+xe|"
            r"bi\s+tru\s+diem\s+giay\s+phep\s+lai\s+xe\s+\d+\s+diem)\b",
            normalized_answer,
        )
    )


def _answer_mentions_license_revocation(normalized_answer: str) -> bool:
    return bool(
        re.search(
            r"\b(tuoc\s+quyen\s+su\s+dung\s+giay\s+phep\s+lai\s+xe|"
            r"bi\s+tuoc\s+quyen\s+su\s+dung\s+giay\s+phep\s+lai\s+xe)\b",
            normalized_answer,
        )
    )


def _augment_answer_with_related_consequences(
    answer: str,
    retrieval: dict[str, Any] | None,
    question: str,
) -> str:
    answer = answer or ""
    retrieval_results = (retrieval or {}).get("results") or []
    if not answer or not retrieval_results:
        return answer

    profile = _query_profile(question)
    if not profile.get("administrative_penalty"):
        return answer
    if profile.get("temporal_applicability"):
        return answer

    scored = [
        (float(result.get("rerank_score") or result.get("score") or 0.0), result)
        for result in retrieval_results
    ]
    anchors = _anchor_selectors_for_consequences(scored, profile)
    if not anchors:
        return answer

    normalized_answer = _normalize_for_match(answer)
    extra_answer_items: list[str] = []
    extra_citations: list[str] = []

    for result in retrieval_results:
        if not any(_result_references_selector(result, selector) for selector in anchors):
            continue
        tags = set(_result_consequence_tags(result))
        if "tru_diem_gplx" in tags and not _answer_mentions_point_deduction(normalized_answer):
            summary = _extract_point_deduction_summary(result)
            if summary:
                extra_answer_items.append(summary)
                extra_citations.append(_format_result_citation(result))
                normalized_answer += " tru diem giay phep lai xe "
        if "tuoc_gplx" in tags and not _answer_mentions_license_revocation(normalized_answer):
            summary = _extract_license_revocation_summary(result)
            if summary:
                extra_answer_items.append(summary)
                extra_citations.append(_format_result_citation(result))
                normalized_answer += " tuoc quyen su dung giay phep lai xe "

    if not extra_answer_items:
        return answer

    citation_label = "Dựa vào:"
    answer_part = answer.strip()
    citation_part = ""
    for label in ("Dựa vào:", "Dựa theo:"):
        if label in answer:
            answer_part, citation_part = answer.split(label, 1)
            citation_label = label
            answer_part = answer_part.strip()
            citation_part = citation_part.strip()
            break

    answer_part = _append_bullets(answer_part, extra_answer_items, "Trả lời:")
    if extra_citations:
        citation_part = _append_bullets(citation_part, extra_citations, citation_label)
        return f"{answer_part}\n{citation_part}".strip()
    return answer_part.strip()


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


def run_retrieval_stage(
    original_query: str,
    retrieval_seed_query: str,
    retriever_script: Path,
    index_dir: Path,
    gazetteer_root: Path,
    top_k: int,
    candidate_k: int = 300,
    dense_weight: float = 0.25,
    bm25_weight: float = 0.25,
    graph_weight: float = 0.20,
    reference_weight: float = 0.30,
    use_reference_expansion: bool = True,
    semantic_entity_top_k: int = 20,
    semantic_entity_min_score: float = 0.45,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, dict[str, Any]]:
    retrieval_query = apply_rule_based_query_rewrite(retrieval_seed_query or original_query)
    retrieval_top_k = max(top_k, 40) if needs_retrieval_postprocess(original_query, retrieval_query) else top_k
    _emit_progress(
        progress_callback,
        "retrieval_started",
        retrieval_query=retrieval_query,
        requested_top_k=top_k,
        retrieval_top_k=retrieval_top_k,
        candidate_k=candidate_k,
    )
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
    _emit_progress(
        progress_callback,
        "retrieval_raw_done",
        raw_result_count=len(retrieval.get("results") or []),
        activated_entity_count=len(retrieval.get("activated_entities") or []),
    )
    retrieval = postprocess_retrieval_for_query(retrieval, original_query, retrieval_query, top_k=top_k)
    _emit_progress(
        progress_callback,
        "retrieval_done",
        retrieval_query=retrieval_query,
        result_count=len(retrieval.get("results") or []),
        activated_entity_count=len(retrieval.get("activated_entities") or []),
    )
    return retrieval_query, retrieval


def retrieve_passages_for_query(
    query: str,
    retriever_script: Path,
    index_dir: Path,
    gazetteer_root: Path,
    top_k: int,
    candidate_k: int = 300,
    dense_weight: float = 0.25,
    bm25_weight: float = 0.25,
    graph_weight: float = 0.20,
    reference_weight: float = 0.30,
    use_reference_expansion: bool = True,
    semantic_entity_top_k: int = 20,
    semantic_entity_min_score: float = 0.45,
    conversation_memory: ConversationMemory | dict[str, Any] | None = None,
    model_name: str = "gpt-4o-mini",
    mode: str = "openai",
    enable_query_rewrite: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    load_4bit: bool = False,
    dtype: str = "auto",
    device_map: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    original_query = repair_mojibake_text(query).strip()
    memory = _coerce_conversation_memory(conversation_memory)
    display_resolved_query = ""
    tokenizer = None
    model = None

    if mode == "local" and enable_query_rewrite:
        _emit_progress(progress_callback, "loading_local_model", purpose="resolver", model_name=model_name)
        tokenizer, model = load_model(
            model_name,
            load_4bit=load_4bit,
            dtype=dtype,
            device_map=device_map,
        )

    _emit_progress(progress_callback, "resolver_started", query=original_query)
    resolver_llm_call = (
        _make_conversation_resolver_llm_call(
            model_name=model_name,
            mode=mode,
            tokenizer=tokenizer,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        if enable_query_rewrite
        else None
    )
    processing_query, memory_context, conversation_resolution = resolve_query_with_memory(
        original_query,
        memory,
        llm_call=resolver_llm_call,
        enable_llm=enable_query_rewrite,
    )
    processing_query = repair_mojibake_text(processing_query).strip()
    display_resolved_query = _display_resolved_query(original_query, processing_query)
    _emit_progress(
        progress_callback,
        "resolver_done",
        processing_query=processing_query,
        expanded_query=display_resolved_query,
        used_memory=bool(memory_context),
        relation=(conversation_resolution or {}).get("relation"),
        confidence=(conversation_resolution or {}).get("confidence"),
    )

    query_preprocessing = _query_preprocessing_from_resolution(
        original_query,
        processing_query,
        conversation_resolution,
    )
    if query_preprocessing is None:
        query_preprocessing = _fallback_query_preprocessing_from_query(
            original_query,
            processing_query,
            conversation_resolution,
            memory_context=memory_context,
        )
    route = query_preprocessing["route"]
    _emit_progress(
        progress_callback,
        "route_decided",
        route=route,
        reason=query_preprocessing.get("reason", ""),
        memory_context=memory_context,
    )
    if route == ROUTE_GENERAL_CHAT and memory_context:
        route = ROUTE_TRAFFIC_LAW
        query_preprocessing = dict(query_preprocessing)
        query_preprocessing["route"] = ROUTE_TRAFFIC_LAW
        query_preprocessing["rewritten_query"] = processing_query
        query_preprocessing["chat_answer"] = ""
        query_preprocessing["reason"] = "conversation memory follow-up override"

    
    if route == ROUTE_GENERAL_CHAT:
        updated_memory = empty_memory() if is_reset_query(original_query) else memory
        _emit_progress(progress_callback, "retrieval_skipped", reason="general_chat_route")
        _emit_progress(progress_callback, "completed", route=route)
        return {
            "query": original_query,
            "expanded_query": display_resolved_query,
            "memory_context": memory_context,
            "rewritten_query": "",
            "route": route,
            "route_reason": query_preprocessing.get("reason", ""),
            "query_preprocessing": query_preprocessing,
            "conversation_resolution": conversation_resolution,
            "retrieval": {"results": [], "activated_entities": [], "debug": {"route": route}},
            "conversation_memory": _memory_dict(updated_memory),
        }

    if memory_context:
        # Với câu hỏi nối tiếp, dùng câu hỏi gốc + memory context.
        # Không để router rewrite mơ hồ ghi đè chủ đề cũ.
        retrieval_seed_query = processing_query
    else:
        retrieval_seed_query = query_preprocessing.get("rewritten_query") or processing_query

    retrieval_query, retrieval = run_retrieval_stage(
        original_query=original_query,
        retrieval_seed_query=retrieval_seed_query,
        retriever_script=retriever_script,
        index_dir=index_dir,
        gazetteer_root=gazetteer_root,
        top_k=top_k,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        graph_weight=graph_weight,
        reference_weight=reference_weight,
        use_reference_expansion=use_reference_expansion,
        semantic_entity_top_k=semantic_entity_top_k,
        semantic_entity_min_score=semantic_entity_min_score,
        progress_callback=progress_callback,
    )
    updated_memory = update_memory_after_answer(memory, original_query, processing_query, retrieval)
    _emit_progress(progress_callback, "completed", route=route)
    return {
        "query": original_query,
        "expanded_query": display_resolved_query,
        "memory_context": memory_context,
        "rewritten_query": retrieval_query,
        "route": route,
        "route_reason": query_preprocessing.get("reason", ""),
        "query_preprocessing": query_preprocessing,
        "conversation_resolution": conversation_resolution,
        "retrieval": retrieval,
        "conversation_memory": _memory_dict(updated_memory),
    }


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


def _should_force_retrieval(query: str) -> bool:
    query = repair_mojibake_text(query or "")
    normalized = _normalize_for_match(query)

    if _looks_like_legal_rag_query(query):
        return True

    if LICENSE_PLATE_QUERY_PATTERN.search(query):
        return True

    if (
        "bien so" in normalized
        or "bien kiem soat" in normalized
        or "bang so" in normalized
        or "ky hieu bien" in normalized
        or "ma bien" in normalized
        or "dang ky xe" in normalized
        or "cap bien so" in normalized
        or "thu hoi bien so" in normalized
    ):
        return True

    return False



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
    force_retrieval = _should_force_retrieval(query)
    if route == ROUTE_GENERAL_CHAT and force_retrieval:
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
            "traffic-domain heuristic override"
            if route == ROUTE_TRAFFIC_LAW and force_retrieval and parsed.get("route") == ROUTE_GENERAL_CHAT
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
    conversation_memory: ConversationMemory | dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    original_query = repair_mojibake_text(query).strip()
    memory = _coerce_conversation_memory(conversation_memory)
    display_resolved_query = ""
    tokenizer = None
    model = None

    if mode == "local" and enable_query_rewrite:
        _emit_progress(progress_callback, "loading_local_model", purpose="resolver", model_name=model_name)
        tokenizer, model = load_model(
            model_name,
            load_4bit=load_4bit,
            dtype=dtype,
            device_map=device_map,
        )

    _emit_progress(progress_callback, "resolver_started", query=original_query)
    resolver_llm_call = (
        _make_conversation_resolver_llm_call(
            model_name=model_name,
            mode=mode,
            tokenizer=tokenizer,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        if enable_query_rewrite
        else None
    )
    processing_query, memory_context, conversation_resolution = resolve_query_with_memory(
        original_query,
        memory,
        llm_call=resolver_llm_call,
        enable_llm=enable_query_rewrite,
    )
    processing_query = repair_mojibake_text(processing_query).strip()
    display_resolved_query = _display_resolved_query(original_query, processing_query)
    _emit_progress(
        progress_callback,
        "resolver_done",
        processing_query=processing_query,
        expanded_query=display_resolved_query,
        used_memory=bool(memory_context),
        relation=(conversation_resolution or {}).get("relation"),
        confidence=(conversation_resolution or {}).get("confidence"),
    )

    early_effectivity_answer = _direct_effectivity_answer(processing_query)
    if early_effectivity_answer:
        updated_memory = update_memory_after_answer(memory, original_query, processing_query, None, answer=early_effectivity_answer)
        _emit_progress(progress_callback, "route_decided", route=ROUTE_TRAFFIC_LAW, reason="structured effectivity fast-path", memory_context=memory_context)
        _emit_progress(progress_callback, "effectivity_fast_path", answer_mode="structured_effectivity")
        _emit_progress(progress_callback, "completed", route=ROUTE_TRAFFIC_LAW)
        return {
            "query": original_query,
            "expanded_query": display_resolved_query,
            "memory_context": memory_context,
            "rewritten_query": processing_query,
            "route": ROUTE_TRAFFIC_LAW,
            "route_reason": "structured effectivity fast-path",
            "query_preprocessing": {
                "route": ROUTE_TRAFFIC_LAW,
                "rewritten_query": processing_query,
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
            "conversation_resolution": conversation_resolution,
            "conversation_memory": _memory_dict(updated_memory),
        }

    query_preprocessing = _query_preprocessing_from_resolution(
        original_query,
        processing_query,
        conversation_resolution,
    )
    if query_preprocessing is None:
        query_preprocessing = _fallback_query_preprocessing_from_query(
            original_query,
            processing_query,
            conversation_resolution,
            memory_context=memory_context,
        )
    route = query_preprocessing["route"]
    _emit_progress(
        progress_callback,
        "route_decided",
        route=route,
        reason=query_preprocessing.get("reason", ""),
        memory_context=memory_context,
    )
    if route == ROUTE_GENERAL_CHAT and memory_context:
        route = ROUTE_TRAFFIC_LAW
        query_preprocessing = dict(query_preprocessing)
        query_preprocessing["route"] = ROUTE_TRAFFIC_LAW
        query_preprocessing["rewritten_query"] = processing_query
        query_preprocessing["chat_answer"] = ""
        query_preprocessing["reason"] = "conversation memory follow-up override"
        _emit_progress(
            progress_callback,
            "route_decided",
            route=ROUTE_TRAFFIC_LAW,
            reason=query_preprocessing["reason"],
            memory_context=memory_context,
        )
        _emit_progress(
            progress_callback,
            "route_decided",
            route=ROUTE_TRAFFIC_LAW,
            reason=query_preprocessing["reason"],
            memory_context=memory_context,
        )

    if route == ROUTE_GENERAL_CHAT:
        answer = query_preprocessing.get("chat_answer") or ""
        if not answer:
            if mode == "local" and (tokenizer is None or model is None):
                _emit_progress(progress_callback, "loading_local_model", purpose="answer", model_name=model_name)
                tokenizer, model = load_model(
                    model_name,
                    load_4bit=load_4bit,
                    dtype=dtype,
                    device_map=device_map,
                )
            _emit_progress(progress_callback, "retrieval_skipped", reason="general_chat_route")
            _emit_progress(progress_callback, "context_skipped", reason="general_chat_route")
            _emit_progress(progress_callback, "generation_started", answer_mode="general_chat", model_name=model_name)
            answer = generate_answer_with_backend(
                messages=build_general_chat_messages(original_query),
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
            _emit_progress(progress_callback, "generation_done", answer_chars=len(answer or ""))
        else:
            _emit_progress(progress_callback, "retrieval_skipped", reason="general_chat_route")
            _emit_progress(progress_callback, "context_skipped", reason="general_chat_route")
            _emit_progress(progress_callback, "generation_done", answer_chars=len(answer or ""))
        updated_memory = empty_memory() if is_reset_query(original_query) else memory
        _emit_progress(progress_callback, "completed", route=route)
        return {
            "query": original_query,
            "expanded_query": display_resolved_query,
            "memory_context": memory_context,
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
            "conversation_resolution": conversation_resolution,
            "conversation_memory": _memory_dict(updated_memory),
        }

    direct_effectivity_answer = _direct_effectivity_answer(processing_query)
    if direct_effectivity_answer:
        retrieval_query = query_preprocessing.get("rewritten_query") or processing_query
        updated_memory = update_memory_after_answer(memory, original_query, processing_query, None, answer=direct_effectivity_answer)
        _emit_progress(progress_callback, "effectivity_fast_path", answer_mode="structured_effectivity")
        _emit_progress(progress_callback, "completed", route=route)
        return {
            "query": original_query,
            "expanded_query": display_resolved_query,
            "memory_context": memory_context,
            "rewritten_query": retrieval_query,
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
            "conversation_resolution": conversation_resolution,
            "conversation_memory": _memory_dict(updated_memory),
        }

    if memory_context:
        # Với câu hỏi nối tiếp, dùng câu hỏi gốc + memory context.
        # Không để router rewrite mơ hồ ghi đè chủ đề cũ.
        retrieval_seed_query = processing_query
    else:
        retrieval_seed_query = query_preprocessing.get("rewritten_query") or processing_query

    retrieval_query, retrieval = run_retrieval_stage(
        original_query=original_query,
        retrieval_seed_query=retrieval_seed_query,
        retriever_script=retriever_script,
        index_dir=index_dir,
        gazetteer_root=gazetteer_root,
        top_k=top_k,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        graph_weight=graph_weight,
        reference_weight=reference_weight,
        use_reference_expansion=use_reference_expansion,
        semantic_entity_top_k=semantic_entity_top_k,
        semantic_entity_min_score=semantic_entity_min_score,
        progress_callback=progress_callback,
    )
    context = format_context(
        retrieval.get("results", []),
        max_passages=max_context_passages,
        max_chars_per_passage=max_chars_per_passage,
    )
    context_passages = min(len(retrieval.get("results", [])), max_context_passages)
    if not context.strip():
        _emit_progress(progress_callback, "context_skipped", reason="empty_context", context_passages=0)
    if not context.strip():
        answer = INSUFFICIENT_CONTEXT_ANSWER
        _emit_progress(progress_callback, "generation_skipped", reason="empty_context")
    else:
        _emit_progress(
            progress_callback,
            "context_ready",
            context_chars=len(context),
            context_passages=context_passages,
        )
        prompt_query = processing_query or original_query
        messages = build_prompt(prompt_query, context, answer_mode=answer_mode)
        if mode == "local":
            if tokenizer is None or model is None:
                _emit_progress(progress_callback, "loading_local_model", purpose="answer", model_name=model_name)
                tokenizer, model = load_model(
                    model_name,
                    load_4bit=load_4bit,
                    dtype=dtype,
                    device_map=device_map,
                )
            _emit_progress(progress_callback, "generation_started", answer_mode=answer_mode, model_name=model_name)
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
            _emit_progress(progress_callback, "generation_started", answer_mode=answer_mode, model_name=model_name)
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
        _emit_progress(progress_callback, "generation_done", answer_chars=len(answer or ""))

    answer = _augment_answer_with_related_consequences(
        answer=answer,
        retrieval=retrieval,
        question=prompt_query if context.strip() else (processing_query or original_query),
    )

    updated_memory = update_memory_after_answer(memory, original_query, processing_query, retrieval, answer=answer)
    _emit_progress(progress_callback, "completed", route=route)
    return {
        "query": original_query,
        "expanded_query": display_resolved_query,
        "memory_context": memory_context,
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
        "conversation_resolution": conversation_resolution,
        "conversation_memory": _memory_dict(updated_memory),
    }
