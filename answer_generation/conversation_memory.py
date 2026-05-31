from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from conversation_memory.models import (
    ConversationResolveResult,
    ConversationState,
    MemoryDocument,
    MemoryEntity,
    QueryPlan,
)
from conversation_memory.planner import build_plan
from conversation_memory.resolver import (
    resolve_with_llm,
    should_call_conversation_resolver,
)
from conversation_memory.updater import (
    extract_entities_simple,
    update_state_after_answer,
)
from conversation_memory.utils import unique_keep_order


"""Compatibility layer between answer_generation and the conversation_memory package."""

ConversationMemory = ConversationState
DEFAULT_SESSION_ID = "default"

MEMORY_MARKER_PATTERN = re.compile(
    r"\s*(Thực\s*thể\s*liên\s*quan|Văn\s*bản\s*liên\s*quan|Chủ\s*đề\s*đang\s*theo\s*dõi)\s*:",
    flags=re.IGNORECASE,
)

VEHICLE_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "special_machine",
        "xe máy chuyên dùng",
        re.compile(r"\bxe\s*máy\s*chuyên\s*dùng\b", flags=re.IGNORECASE),
    ),
    (
        "car",
        "xe ô tô",
        re.compile(r"\b(ô\s*tô|xe\s*ô\s*tô|oto|ôto)\b", flags=re.IGNORECASE),
    ),
    (
        "motorcycle",
        "xe máy, xe mô tô, xe gắn máy",
        re.compile(r"\b(xe\s*máy|mô\s*tô|xe\s*mô\s*tô|xe\s*gắn\s*máy)\b", flags=re.IGNORECASE),
    ),
    (
        "bicycle",
        "xe đạp, xe đạp máy",
        re.compile(r"\b(xe\s*đạp|xe\s*đạp\s*máy|xe\s*đạp\s*điện)\b", flags=re.IGNORECASE),
    ),
]

FOLLOWUP_PREFIX_PATTERN = re.compile(
    r"^\s*(vậy|thế|còn|vậy\s*còn|thế\s*còn|nếu\s+vậy|nếu\s+thế)\b[,:\s]*",
    flags=re.IGNORECASE,
)

RESET_PATTERNS = re.compile(
    r"\b("
    r"chủ\s*đề\s*khác|hỏi\s*câu\s*khác|bỏ\s*qua|không\s*liên\s*quan|"
    r"quay\s*lại\s*từ\s*đầu|reset|chat\s*mới"
    r")\b",
    flags=re.IGNORECASE,
)


def empty_memory(session_id: str = DEFAULT_SESSION_ID) -> ConversationMemory:
    return ConversationState(session_id=session_id)


def is_reset_query(query: str) -> bool:
    return bool(RESET_PATTERNS.search(query or ""))


def _entity_text(entity: dict[str, Any] | MemoryEntity) -> str:
    if isinstance(entity, MemoryEntity):
        return entity.text
    if isinstance(entity, str):
        return entity.strip()
    if not isinstance(entity, dict):
        return ""
    return str(
        entity.get("canonical")
        or entity.get("surface")
        or entity.get("text")
        or entity.get("entity_id")
        or ""
    ).strip()


def _entity_label(entity: dict[str, Any] | MemoryEntity) -> str | None:
    if isinstance(entity, MemoryEntity):
        return entity.label
    if not isinstance(entity, dict):
        return None
    value = entity.get("label")
    return str(value).strip() if value else None


def _doc_id(document: dict[str, Any] | MemoryDocument) -> str:
    if isinstance(document, MemoryDocument):
        return document.doc_id
    if isinstance(document, str):
        return document.strip()
    if not isinstance(document, dict):
        return ""
    return str(
        document.get("document_number")
        or document.get("doc_id")
        or document.get("document_id")
        or ""
    ).strip()


def _doc_title(document: dict[str, Any] | MemoryDocument) -> str | None:
    if isinstance(document, MemoryDocument):
        return document.title
    if not isinstance(document, dict):
        return None
    value = document.get("document_title") or document.get("title")
    return str(value).strip() if value else None


def _coerce_entities(raw_entities: list[Any], limit: int = 8) -> list[MemoryEntity]:
    entities: list[MemoryEntity] = []
    seen = set()
    for item in raw_entities or []:
        text = _entity_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(MemoryEntity(text=text, label=_entity_label(item)))
        if len(entities) >= limit:
            break
    return entities


def _coerce_documents(raw_documents: list[Any], limit: int = 5) -> list[MemoryDocument]:
    documents: list[MemoryDocument] = []
    seen = set()
    for item in raw_documents or []:
        doc_id = _doc_id(item)
        if not doc_id:
            continue
        key = doc_id.lower()
        if key in seen:
            continue
        seen.add(key)
        documents.append(MemoryDocument(doc_id=doc_id, title=_doc_title(item)))
        if len(documents) >= limit:
            break
    return documents


def coerce_memory(memory: ConversationMemory | dict[str, Any] | None) -> ConversationMemory | None:
    if memory is None:
        return None
    if isinstance(memory, ConversationState):
        return memory
    if not isinstance(memory, dict):
        return None

    session_id = str(memory.get("session_id") or DEFAULT_SESSION_ID)

    # New memory schema.
    if "active_topic" in memory or "focus_entities" in memory or "focus_docs" in memory:
        state = ConversationState(session_id=session_id)
        state.active_topic = memory.get("active_topic") or None
        state.last_user_question = memory.get("last_user_question") or None
        state.last_standalone_question = memory.get("last_standalone_question") or None
        state.last_answer_summary = memory.get("last_answer_summary") or None
        state.last_intent = memory.get("last_intent") or None
        state.last_citations = list(memory.get("last_citations") or [])
        state.recent_turns = list(memory.get("recent_turns") or [])
        try:
            state.turn_count = int(memory.get("turn_count") or 0)
        except (TypeError, ValueError):
            state.turn_count = 0
        state.focus_entities = _coerce_entities(memory.get("focus_entities") or [])
        state.focus_docs = _coerce_documents(memory.get("focus_docs") or [])
        return state

    # Legacy schema used by the previous answer_generation.conversation_memory.
    state = ConversationState(session_id=session_id)
    state.active_topic = str(memory.get("topic") or memory.get("last_rewritten_query") or "") or None
    state.last_user_question = str(memory.get("last_rewritten_query") or "") or None
    state.last_standalone_question = str(memory.get("last_rewritten_query") or "") or None
    try:
        state.turn_count = int(memory.get("turn_count") or 0)
    except (TypeError, ValueError):
        state.turn_count = 0
    state.focus_entities = _coerce_entities(memory.get("entities") or [])
    state.focus_docs = _coerce_documents(memory.get("documents") or [])
    return state


def _state_for_plan(memory: ConversationMemory | None) -> ConversationMemory | None:
    if memory and memory.turn_count > 0:
        return memory
    return None


def _clean_memory_base(text: str) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    marker_match = MEMORY_MARKER_PATTERN.search(value)
    if marker_match:
        value = value[: marker_match.start()].strip()
    return value.strip(" .")


def _detect_vehicle(text: str) -> tuple[str, str] | None:
    for vehicle_key, replacement, pattern in VEHICLE_PATTERNS:
        if pattern.search(text or ""):
            return vehicle_key, replacement
    return None


def _replace_vehicle(text: str, replacement: str) -> tuple[str, bool]:
    for _, _, pattern in VEHICLE_PATTERNS:
        if pattern.search(text or ""):
            return pattern.sub(replacement, text, count=1).strip(), True
    return text, False


def _append_doc_hint(text: str, docs: list[str]) -> str:
    if not docs:
        return text
    doc_hint = "; ".join(docs[:2])
    if not doc_hint or doc_hint.lower() in text.lower():
        return text
    return f"{text.strip(' .')} theo {doc_hint}"


def _rule_rewrite_followup(question: str, state: ConversationState | None, intent: str) -> str:
    if not state:
        return question

    base = _clean_memory_base(state.last_standalone_question or state.active_topic or "")
    focus_docs = [d.doc_id for d in state.focus_docs[:2] if d.doc_id]
    current_question = re.sub(r"\s+", " ", question or "").strip()

    target_vehicle = _detect_vehicle(current_question)
    if base and target_vehicle:
        _, replacement = target_vehicle
        rewritten, replaced = _replace_vehicle(base, replacement)
        if replaced:
            return _append_doc_hint(rewritten, focus_docs)
        return _append_doc_hint(f"{base} đối với {replacement}", focus_docs)

    if not base:
        return current_question

    cleaned_followup = FOLLOWUP_PREFIX_PATTERN.sub("", current_question).strip(" ,.;")
    if not cleaned_followup:
        return _append_doc_hint(base, focus_docs)

    rewritten = f"{base}. {cleaned_followup}"
    return _append_doc_hint(rewritten, focus_docs)


def prepare_memory_plan(query: str, memory: ConversationMemory | dict[str, Any] | None) -> QueryPlan:
    state = _state_for_plan(coerce_memory(memory))
    session_id = state.session_id if state else DEFAULT_SESSION_ID
    return build_plan(
        session_id=session_id,
        question=query,
        state=state,
        rewrite_fn=_rule_rewrite_followup,
    )


def build_memory_context(memory: ConversationMemory, current_query: str) -> str:
    if is_reset_query(current_query):
        return ""
    plan = prepare_memory_plan(current_query, memory)
    return plan.answer_memory_context if plan.use_memory else ""


def expand_query_with_memory(query: str, memory: ConversationMemory | None) -> tuple[str, str]:
    if not memory or is_reset_query(query):
        return query, ""

    plan = prepare_memory_plan(query, memory)
    if not plan.use_memory:
        return query, ""

    memory_context = plan.answer_memory_context
    expanded_query = plan.primary_query or query
    return expanded_query, memory_context


def _context_from_resolution(state: ConversationState | None, result: ConversationResolveResult) -> str:
    if not state or not result.use_memory:
        return ""
    docs = "; ".join(d.doc_id for d in state.focus_docs[:3] if d.doc_id)
    entities = "; ".join(e.text for e in state.focus_entities[:5] if e.text)
    parts = [
        f"Trọng tâm lượt hiện tại: {result.current_focus or result.standalone_question}",
        f"Quan hệ với lượt trước: {result.relation}",
    ]
    if result.changed_constraints:
        parts.append(f"Ràng buộc thay đổi: {result.changed_constraints}")
    if entities:
        parts.append(f"Thực thể lượt trước liên quan: {entities}")
    if docs:
        parts.append(f"Văn bản liên quan gần nhất: {docs}")
    if state.last_answer_summary:
        parts.append(f"Tóm tắt lượt trước: {state.last_answer_summary}")
    return "\n".join(part for part in parts if part)


def resolve_query_with_memory(
    query: str,
    memory: ConversationMemory | dict[str, Any] | None,
    llm_call: Callable[[list[dict[str, str]]], str] | None = None,
    enable_llm: bool = True,
    min_confidence: float = 0.55,
) -> tuple[str, str, dict[str, Any]]:
    if is_reset_query(query):
        return query, "", {
            "accepted": False,
            "used_llm": False,
            "reason": "reset query",
        }

    state = _state_for_plan(coerce_memory(memory))
    if not state:
        return query, "", {
            "accepted": False,
            "used_llm": False,
            "reason": "no usable memory",
        }

    fallback_plan = prepare_memory_plan(query, state)
    fallback_query = fallback_plan.primary_query if fallback_plan.use_memory else query
    fallback_context = fallback_plan.answer_memory_context if fallback_plan.use_memory else ""

    debug: dict[str, Any] = {
        "accepted": False,
        "used_llm": False,
        "reason": "rule fallback",
        "fallback_plan": fallback_plan.debug,
    }

    if not (enable_llm and llm_call and should_call_conversation_resolver(query, state)):
        return fallback_query, fallback_context, debug

    try:
        result = resolve_with_llm(query, state, llm_call)
    except Exception as exc:
        debug["used_llm"] = True
        debug["reason"] = f"resolver failed: {exc}"
        return fallback_query, fallback_context, debug

    if not result:
        debug["used_llm"] = True
        debug["reason"] = "resolver returned empty result"
        return fallback_query, fallback_context, debug

    result_dict = result.to_dict()
    result_dict["accepted"] = False

    resolved_query = (result.retrieval_query or result.standalone_question or query).strip()
    if not resolved_query:
        result_dict["reason"] = result.reason or "resolver returned empty retrieval query"
        return fallback_query, fallback_context, result_dict

    if result.error or result.confidence < min_confidence:
        result_dict["reason"] = result.reason or result.error or "resolver confidence below threshold"
        return fallback_query, fallback_context, result_dict

    result_dict["accepted"] = True
    memory_context = _context_from_resolution(state, result)
    return resolved_query, memory_context, result_dict


def _compact_text(text: str | None, limit: int = 350) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _entities_from_retrieval(retrieval: dict[str, Any] | None, passages: list[dict[str, Any]]) -> list[str]:
    entities: list[str] = []
    if retrieval:
        for item in retrieval.get("activated_entities") or []:
            text = _entity_text(item)
            if text:
                entities.append(text)
    entities.extend(extract_entities_simple("", passages))
    return unique_keep_order(entities)[:8]


def update_memory_after_answer(
    memory: ConversationMemory | None,
    original_query: str,
    retrieval_query: str,
    retrieval: dict[str, Any] | None,
    answer: str = "",
) -> ConversationMemory:
    if memory is None or is_reset_query(original_query):
        memory = empty_memory()
    else:
        memory = coerce_memory(memory) or empty_memory()

    passages = list((retrieval or {}).get("results") or [])
    plan = prepare_memory_plan(original_query, memory)
    if retrieval_query:
        plan.primary_query = retrieval_query
        plan.answer_question = retrieval_query
        if plan.retrieval_queries:
            plan.retrieval_queries = unique_keep_order([*plan.retrieval_queries, retrieval_query])
        else:
            plan.retrieval_queries = unique_keep_order([original_query, retrieval_query])

    updated = update_state_after_answer(
        state=memory,
        plan=plan,
        answer=answer or "",
        retrieved_passages=passages,
        entity_extractor=lambda question, found_passages: _entities_from_retrieval(retrieval, found_passages),
    )

    if not updated.last_answer_summary and answer:
        updated.last_answer_summary = _compact_text(answer)

    return updated
