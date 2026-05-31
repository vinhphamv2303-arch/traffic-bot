from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


FOLLOWUP_PATTERNS = re.compile(
    r"\b("
    r"trường\s*hợp\s*này|quy\s*định\s*(này|đó|trên)|văn\s*bản\s*(này|đó)|"
    r"điều\s*(này|đó|trên)|khoản\s*(này|đó|trên)|điểm\s*(này|đó|trên)|"
    r"hành\s*vi\s*(này|đó)|mức\s*(này|đó)|thời\s*gian\s*(này|đó)|"
    r"nếu\s*vậy|như\s*vậy|như\s*thế|vậy\s*thì|thế\s*thì|"
    r"nếu\s+là|còn\s*nếu|thế\s*còn|hiện\s*nay|còn\s*áp\s*dụng|còn\s*hiệu\s*lực|"
    r"cụ\s*thể|cụ\s*thể\s*là|ra\s*sao|như\s*thế\s*nào|thế\s*nào|"
    r"có\s*bị\s*phạt\s*không|bị\s*phạt\s*thế\s*nào|mức\s*phạt|phạt\s*bao\s*nhiêu|"
    r"nếu\s*vượt\s*quá|vượt\s*quá\s*thời\s*gian|quá\s*thời\s*gian\s*quy\s*định"
    r")\b"
    r"|"
    r"\b(còn|thế\s*còn)\s+.{1,80}?\s+thì\s+sao\b"
    r"|"
    r"\bthì\s+sao\b"
    r"trong\s*trường\s*hợp\s*(này|đó|trên)|"
    r"trường\s*hợp\s*(này|đó|trên)|"
    r"quy\s*định\s*(này|đó|trên)|"
    r"thời\s*gian\s*quy\s*định\s*(này|đó|trên)|"
    r"vượt\s*quá\s*thời\s*gian\s*quy\s*định|"
    r"có\s*bị\s*xử\s*phạt\s*hay\s*không|"
    r"có\s*bị\s*xử\s*phạt\s*không|"
    r"bị\s*xử\s*phạt\s*thế\s*nào|",
    flags=re.IGNORECASE,
)

RESET_PATTERNS = re.compile(
    r"\b(chủ\s*đề\s*khác|hỏi\s*câu\s*khác|bỏ\s*qua|không\s*liên\s*quan|quay\s*lại\s*từ\s*đầu|reset)\b",
    flags=re.IGNORECASE,
)

VEHICLE_PATTERNS = {
    "ô tô": re.compile(r"\b(ô\s*tô|xe\s*ô\s*tô|oto|ôto)\b", re.IGNORECASE),
    "xe máy": re.compile(r"\b(xe\s*máy|mô\s*tô|xe\s*mô\s*tô|xe\s*gắn\s*máy)\b", re.IGNORECASE),
    "xe đạp": re.compile(r"\b(xe\s*đạp|xe\s*đạp\s*điện)\b", re.IGNORECASE),
    "xe máy chuyên dùng": re.compile(r"\b(xe\s*máy\s*chuyên\s*dùng)\b", re.IGNORECASE),
}

PENALTY_PATTERNS = re.compile(
    r"\b(mức\s*phạt|bị\s*phạt|xử\s*phạt|phạt\s*bao\s*nhiêu|phạt\s*thế\s*nào|"
    r"có\s*bị\s*phạt|vượt\s*quá|quá\s*thời\s*gian)\b",
    flags=re.IGNORECASE,
)

EFFECTIVITY_PATTERNS = re.compile(
    r"\b(còn\s*hiệu\s*lực|hết\s*hiệu\s*lực|còn\s*áp\s*dụng|hiện\s*nay|hiệu\s*lực)\b",
    flags=re.IGNORECASE,
)

TIME_DRIVING_PATTERNS = re.compile(
    r"\b(thời\s*gian\s*lái\s*xe|lái\s*xe\s*liên\s*tục|thời\s*gian\s*làm\s*việc\s*của\s*người\s*lái\s*xe)\b",
    flags=re.IGNORECASE,
)

GENERIC_ENTITY_SURFACES = {
    "người",
    "phương tiện",
    "đường bộ",
    "văn bản",
    "quy định",
    "hành vi",
}


@dataclass
class ConversationMemory:
    topic: str = ""
    last_rewritten_query: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_memory() -> ConversationMemory:
    return ConversationMemory()


def is_reset_query(query: str) -> bool:
    return bool(RESET_PATTERNS.search(query or ""))


def is_contextual_followup(query: str) -> bool:
    query = query or ""
    return bool(FOLLOWUP_PATTERNS.search(query))


def detect_vehicle(query: str) -> str | None:
    for vehicle, pattern in VEHICLE_PATTERNS.items():
        if pattern.search(query or ""):
            return vehicle
    return None


def detect_intent(query: str) -> str | None:
    query = query or ""
    if PENALTY_PATTERNS.search(query):
        return "penalty"
    if EFFECTIVITY_PATTERNS.search(query):
        return "effectivity"
    if TIME_DRIVING_PATTERNS.search(query):
        return "time_driving"
    return None


def _dedupe_by_key(items: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        value = item.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _compact_text(text: str | None, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _surface(entity: dict[str, Any]) -> str:
    return str(entity.get("canonical") or entity.get("surface") or entity.get("text") or "").strip()


def _is_useful_entity(entity: dict[str, Any]) -> bool:
    surface = _surface(entity).lower()
    if not surface or surface in GENERIC_ENTITY_SURFACES:
        return False
    return entity.get("label") in {
        "BEHAVIOR",
        "CONDITION",
        "VEHICLE",
        "VEHICLE_CONDITION_OR_EQUIPMENT",
        "DOCUMENT",
        "ACTOR",
        "INFRASTRUCTURE",
    }


def build_memory_context(memory: ConversationMemory, current_query: str) -> str:
    if not memory or memory.turn_count == 0:
        return ""

    if not is_contextual_followup(current_query):
        return ""

    parts: list[str] = []

    if memory.topic:
        parts.append(f"Chủ đề gốc cần giữ: {memory.topic}")

    current_intent = detect_intent(current_query)
    if current_intent == "penalty":
        parts.append(
            "Ý định lượt hiện tại: hỏi mức xử phạt/hậu quả đối với hành vi hoặc quy định đã nêu ở lượt trước."
        )
    elif current_intent == "effectivity":
        parts.append("Ý định lượt hiện tại: hỏi hiệu lực/tình trạng áp dụng của quy định hoặc văn bản đã nêu.")
    elif current_intent == "time_driving":
        parts.append("Ý định lượt hiện tại: hỏi quy định về thời gian lái xe liên tục/thời gian làm việc của người lái xe.")

    constraints = dict(memory.constraints)

    current_vehicle = detect_vehicle(current_query)
    if current_vehicle:
        constraints["vehicle"] = current_vehicle

    if current_intent:
        constraints["intent"] = current_intent

    if constraints:
        constraint_text = "; ".join(f"{k}: {v}" for k, v in constraints.items() if v)
        if constraint_text:
            parts.append(f"Ràng buộc đã biết: {constraint_text}")

    useful_entities = [e for e in memory.entities if _is_useful_entity(e)]
    if useful_entities:
        ents = []
        for e in useful_entities[:8]:
            surface = _surface(e)
            label = e.get("label")
            ents.append(f"{surface} ({label})" if label else surface)
        if ents:
            parts.append("Thực thể trọng tâm: " + "; ".join(ents))

    if memory.documents:
        docs = []
        for d in memory.documents[:3]:
            number = d.get("document_number") or d.get("document_id")
            title = d.get("document_title")
            if number and title:
                docs.append(f"{number} - {title}")
            elif number:
                docs.append(str(number))
        if docs:
            parts.append("Văn bản đang tham chiếu: " + "; ".join(docs))

    if memory.passages:
        refs = []
        for p in memory.passages[:3]:
            path = p.get("path_text")
            doc = p.get("document_number")
            text_sample = p.get("text_sample")
            ref = ""
            if path and doc:
                ref = f"{path}, {doc}"
            elif path:
                ref = str(path)
            elif doc:
                ref = str(doc)
            if text_sample:
                ref = f"{ref}: {_compact_text(text_sample, 260)}" if ref else _compact_text(text_sample, 260)
            if ref:
                refs.append(ref)
        if refs:
            parts.append("Passage/căn cứ gần nhất: " + " | ".join(refs))

    parts.append(
        "Quy tắc dùng memory: hiểu câu hỏi hiện tại trong chủ đề trên; không tự chuyển sang chủ đề khác nếu người dùng chỉ nói 'quy định này', 'cụ thể', 'vượt quá thời gian này'."
    )

    return "\n".join(parts)


def expand_query_with_memory(query: str, memory: ConversationMemory | None) -> tuple[str, str]:
    if not memory:
        return query, ""

    if is_reset_query(query):
        return query, ""

    memory_context = build_memory_context(memory, query)
    if not memory_context:
        return query, ""

    expanded = (
        f"Câu hỏi nối tiếp: {query}\n\n"
        f"Ngữ cảnh hội thoại đã chuẩn hóa:\n{memory_context}"
    )
    return expanded, memory_context


def update_memory_after_answer(
    memory: ConversationMemory | None,
    original_query: str,
    retrieval_query: str,
    retrieval: dict[str, Any] | None,
) -> ConversationMemory:
    if memory is None or is_reset_query(original_query):
        memory = empty_memory()

    followup = is_contextual_followup(original_query)
    memory.turn_count += 1
    memory.last_rewritten_query = retrieval_query or original_query

    vehicle = detect_vehicle(original_query) or detect_vehicle(retrieval_query)
    if vehicle:
        memory.constraints["vehicle"] = vehicle

    intent = detect_intent(original_query) or detect_intent(retrieval_query)
    if intent:
        memory.constraints["intent"] = intent

    # Không để các lượt follow-up mơ hồ như "cụ thể là bị phạt thế nào"
    # ghi đè topic gốc. Đây là nguyên nhân làm memory trôi chủ đề.
    if retrieval_query and (not followup or not memory.topic):
        memory.topic = _compact_text(retrieval_query, 300)

    if not retrieval:
        return memory

    activated = [e for e in (retrieval.get("activated_entities") or []) if _is_useful_entity(e)]
    if followup:
        memory.entities = _dedupe_by_key(memory.entities + activated, key="entity_id", limit=12)
    else:
        memory.entities = _dedupe_by_key(activated + memory.entities, key="entity_id", limit=12)

    results = retrieval.get("results") or []

    new_docs = []
    for r in results[:5]:
        new_docs.append({
            "document_id": r.get("document_id"),
            "document_number": r.get("document_number"),
            "document_title": r.get("document_title"),
        })

    if followup:
        memory.documents = _dedupe_by_key(memory.documents + new_docs, key="document_number", limit=5)
    else:
        memory.documents = _dedupe_by_key(new_docs + memory.documents, key="document_number", limit=5)

    new_passages = []
    for r in results[:5]:
        new_passages.append({
            "passage_id": r.get("passage_id"),
            "document_number": r.get("document_number"),
            "document_title": r.get("document_title"),
            "path_text": r.get("path_text"),
            "text_sample": _compact_text(r.get("text"), 500),
            "score": r.get("score"),
        })

    if followup:
        # Với follow-up, ưu tiên giữ passage cũ trước để không bị kết quả truy xuất sai ghi đè.
        memory.passages = _dedupe_by_key(memory.passages + new_passages, key="passage_id", limit=8)
    else:
        memory.passages = _dedupe_by_key(new_passages + memory.passages, key="passage_id", limit=8)

    return memory
