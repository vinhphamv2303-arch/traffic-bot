from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


FOLLOWUP_PATTERNS = re.compile(
    r"\b(trường hợp này|quy định đó|văn bản đó|điều đó|khoản đó|nếu vậy|như vậy|"
    r"nếu là|còn nếu|thế còn|hiện nay|còn áp dụng|còn hiệu lực)\b"
    r"|"
    r"\b(còn|thế còn)\s+.{1,80}?\s+thì\s+sao\b"
    r"|"
    r"\bthì\s+sao\b",
    flags=re.IGNORECASE,
)

RESET_PATTERNS = re.compile(
    r"\b(chủ đề khác|hỏi câu khác|bỏ qua|không liên quan|quay lại từ đầu)\b",
    flags=re.IGNORECASE,
)

VEHICLE_PATTERNS = {
    "ô tô": re.compile(r"\b(ô\s*tô|xe\s*ô\s*tô|oto|ôto)\b", re.IGNORECASE),
    "xe máy": re.compile(r"\b(xe\s*máy|mô\s*tô|xe\s*mô\s*tô|xe\s*gắn\s*máy)\b", re.IGNORECASE),
    "xe đạp": re.compile(r"\b(xe\s*đạp|xe\s*đạp\s*điện)\b", re.IGNORECASE),
    "xe máy chuyên dùng": re.compile(r"\b(xe\s*máy\s*chuyên\s*dùng)\b", re.IGNORECASE),
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
    return bool(FOLLOWUP_PATTERNS.search(query or ""))


def detect_vehicle(query: str) -> str | None:
    for vehicle, pattern in VEHICLE_PATTERNS.items():
        if pattern.search(query or ""):
            return vehicle
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


def build_memory_context(memory: ConversationMemory, current_query: str) -> str:
    if not memory or memory.turn_count == 0:
        return ""

    if not is_contextual_followup(current_query):
        return ""

    parts = []

    if memory.topic:
        parts.append(f"Chủ đề đang hỏi: {memory.topic}")

    constraints = dict(memory.constraints)

    # Nếu lượt hiện tại đã nêu phương tiện mới, ưu tiên phương tiện mới,
    # không ép dùng phương tiện cũ.
    current_vehicle = detect_vehicle(current_query)
    if current_vehicle:
        constraints["vehicle"] = current_vehicle

    if constraints:
        constraint_text = "; ".join(f"{k}: {v}" for k, v in constraints.items() if v)
        if constraint_text:
            parts.append(f"Ràng buộc đã biết: {constraint_text}")

    if memory.entities:
        ents = []
        for e in memory.entities[:8]:
            canonical = e.get("canonical")
            label = e.get("label")
            if canonical:
                ents.append(f"{canonical} ({label})" if label else canonical)
        if ents:
            parts.append("Thực thể liên quan: " + "; ".join(ents))

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
            if path and doc:
                refs.append(f"{path}, {doc}")
            elif path:
                refs.append(path)
        if refs:
            parts.append("Điều khoản/passage gần nhất: " + "; ".join(refs))

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
        f"{query}\n\n"
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

    memory.turn_count += 1
    memory.last_rewritten_query = retrieval_query or original_query

    vehicle = detect_vehicle(original_query) or detect_vehicle(retrieval_query)
    if vehicle:
        memory.constraints["vehicle"] = vehicle

    # Topic đơn giản: dùng retrieval query gần nhất.
    if retrieval_query:
        memory.topic = retrieval_query[:300]

    if not retrieval:
        return memory

    activated = retrieval.get("activated_entities") or []
    memory.entities = _dedupe_by_key(
        activated + memory.entities,
        key="entity_id",
        limit=12,
    )

    results = retrieval.get("results") or []

    new_docs = []
    for r in results[:5]:
        new_docs.append({
            "document_id": r.get("document_id"),
            "document_number": r.get("document_number"),
            "document_title": r.get("document_title"),
        })
    memory.documents = _dedupe_by_key(
        new_docs + memory.documents,
        key="document_number",
        limit=5,
    )

    new_passages = []
    for r in results[:5]:
        new_passages.append({
            "passage_id": r.get("passage_id"),
            "document_number": r.get("document_number"),
            "document_title": r.get("document_title"),
            "path_text": r.get("path_text"),
        })
    memory.passages = _dedupe_by_key(
        new_passages + memory.passages,
        key="passage_id",
        limit=8,
    )

    return memory
