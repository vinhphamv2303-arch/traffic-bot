from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .models import ConversationState, MemoryDocument, MemoryEntity, QueryPlan
from .utils import unique_keep_order

DOC_ID_PATTERN = re.compile(
    r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+[A-ZĐ0-9\-]*\b",
    re.IGNORECASE,
)

DEFAULT_IMPORTANT_PHRASES = [
    "vùng phát thải thấp",
    "nồng độ cồn",
    "mũ bảo hiểm",
    "giấy phép lái xe",
    "đăng kiểm",
    "xe máy",
    "xe mô tô",
    "ô tô",
    "xe ô tô",
    "cao tốc",
    "vượt đèn đỏ",
    "vượt tốc độ",
    "tốc độ",
    "thời gian lái xe",
    "lái xe liên tục",
    "tai nạn giao thông",
    "phạt tiền",
    "trừ điểm",
    "tước giấy phép",
]


def _passage_text(p: dict[str, Any]) -> str:
    fields = [
        "doc_id",
        "document_number",
        "law_id",
        "title",
        "document_title",
        "path",
        "breadcrumb",
        "content",
        "text",
    ]
    return " ".join(str(p.get(k, "")) for k in fields)


def extract_doc_ids_from_passages(passages: list[dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    for p in passages[:5]:
        text = _passage_text(p)
        # First prefer explicit metadata values.
        for key in ["doc_id", "document_number", "law_id"]:
            value = str(p.get(key, "")).strip()
            if value:
                docs.append(value.upper())
        # Then regex fallback.
        for m in DOC_ID_PATTERN.findall(text):
            docs.append(m.upper())
    return unique_keep_order(docs)[:5]


def extract_entities_simple(question: str, passages: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    all_text = " ".join([question] + [_passage_text(p)[:1000] for p in passages[:3]]).lower()
    for phrase in DEFAULT_IMPORTANT_PHRASES:
        if phrase in all_text:
            candidates.append(phrase)
    return unique_keep_order(candidates)[:8]


def summarize_answer_for_memory(answer: str, max_chars: int = 350) -> str:
    text = re.sub(r"\s+", " ", answer or "").strip()
    return text[:max_chars]


def extract_citations_simple(answer: str) -> list[str]:
    if not answer:
        return []
    citations = DOC_ID_PATTERN.findall(answer)
    return unique_keep_order([c.upper() for c in citations])[:8]


def update_state_after_answer(
    state: ConversationState,
    plan: QueryPlan,
    answer: str,
    retrieved_passages: list[dict[str, Any]],
    entity_extractor: Callable[[str, list[dict[str, Any]]], list[str]] | None = None,
) -> ConversationState:
    """Update state only from query plan + retrieved evidence + final answer."""
    entity_extractor = entity_extractor or extract_entities_simple

    docs = extract_doc_ids_from_passages(retrieved_passages)
    entities = entity_extractor(plan.raw_question, retrieved_passages)
    citations = extract_citations_simple(answer)

    # Topic should be the clean standalone question, not a long prompt.
    state.active_topic = plan.answer_question
    state.last_user_question = plan.raw_question
    state.last_standalone_question = plan.answer_question
    state.last_answer_summary = summarize_answer_for_memory(answer)
    state.last_intent = plan.intent
    state.last_citations = citations

    if docs:
        state.focus_docs = [MemoryDocument(doc_id=d) for d in docs]
    elif plan.doc_filter:
        # Preserve previous doc focus if no new docs were extracted.
        state.focus_docs = [MemoryDocument(doc_id=d) for d in plan.doc_filter[:3]]

    if entities:
        state.focus_entities = [MemoryEntity(text=e) for e in entities]

    state.recent_turns.append({
        "user": plan.raw_question,
        "standalone": plan.answer_question,
        "intent": plan.intent,
        "route": plan.route,
        "answer_summary": state.last_answer_summary,
        "docs": docs,
        "entities": entities,
        "use_memory": plan.use_memory,
        "memory_score": round(plan.memory_score, 4),
    })
    state.recent_turns = state.recent_turns[-4:]
    state.turn_count += 1

    return state
