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
from conversation_memory.resolver import resolve_with_llm
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
        "pedestrian",
        "người đi bộ",
        re.compile(r"\bngười\s*đi\s*bộ\b", flags=re.IGNORECASE),
    ),
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
        "xe máy",
        re.compile(r"\b(xe\s*máy|mô\s*tô|xe\s*mô\s*tô|xe\s*gắn\s*máy)\b", flags=re.IGNORECASE),
    ),
    (
        "bicycle",
        "xe đạp",
        re.compile(r"\b(xe\s*đạp|xe\s*đạp\s*máy|xe\s*đạp\s*điện)\b", flags=re.IGNORECASE),
    ),
]

VEHICLE_BUCKET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bxe\s*máy\s*,\s*xe\s*mô\s*tô\s*,\s*xe\s*gắn\s*máy\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bxe\s*mô\s*tô\s*,\s*xe\s*gắn\s*máy\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bxe\s*đạp\s*,\s*xe\s*đạp\s*máy\b",
        flags=re.IGNORECASE,
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
ENTITY_ID_PATTERN = re.compile(r"^ent_[0-9a-f]{8,}$", flags=re.IGNORECASE)

DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"\b("
    r"văn\s*bản\s*này|văn\s*bản\s*đó|nghị\s*định\s*này|nghị\s*định\s*đó|"
    r"luật\s*này|luật\s*đó|thông\s*tư\s*này|thông\s*tư\s*đó|"
    r"quyết\s*định\s*này|quyết\s*định\s*đó|nghị\s*quyết\s*này|nghị\s*quyết\s*đó|"
    r"nó"
    r")\b",
    flags=re.IGNORECASE,
)

EFFECTIVITY_PATTERN = re.compile(
    r"\b("
    r"hiệu\s*lực|còn\s*hiệu\s*lực|hết\s*hiệu\s*lực|ngày\s*hiệu\s*lực|"
    r"có\s*hiệu\s*lực|áp\s*dụng\s*từ|ngày\s*áp\s*dụng|bãi\s*bỏ|thay\s*thế"
    r")\b",
    flags=re.IGNORECASE,
)

EVIDENCE_PATTERN = re.compile(
    r"\b("
    r"căn\s*cứ|dựa\s*vào|ở\s*đâu|điều\s*nào|khoản\s*nào|điểm\s*nào|"
    r"quy\s*định\s*trong\s*văn\s*bản\s*nào|văn\s*bản\s*nào"
    r")\b",
    flags=re.IGNORECASE,
)

SHORT_FOLLOWUP_PATTERN = re.compile(
    r"\b(còn|vậy|thế|thì\s*sao|đối\s*với|trường\s*hợp\s*này|trường\s*hợp\s*đó)\b",
    flags=re.IGNORECASE,
)

COMPARE_STOPWORDS = {
    "bi",
    "bị",
    "bao",
    "bao_nhieu",
    "bao_nhiêu",
    "cac",
    "các",
    "cho",
    "cua",
    "của",
    "doi",
    "đối",
    "duoc",
    "được",
    "giao",
    "giao_thong",
    "giao_thông",
    "giu",
    "giữ",
    "hoi",
    "hỏi",
    "khong",
    "không",
    "la",
    "là",
    "nao",
    "nào",
    "nhu",
    "như",
    "nguoi",
    "người",
    "oto",
    "phat",
    "phạt",
    "ra",
    "ra_sao",
    "sao",
    "the",
    "thế",
    "thi",
    "thì",
    "vi",
    "vi_pham",
    "vi_phạm",
    "voi",
    "với",
    "vuot",
    "vượt",
    "xe",
    "xu",
    "xử",
    "va",
    "và",
    "vay",
    "vậy",
}


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
        if ENTITY_ID_PATTERN.fullmatch(text):
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
    value = re.sub(
        r"\b(mức\s*xử\s*phạt|mức\s*phạt(?:\s*phạt)?\s*tiền)(?:\s+\1\b)+",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(không\s*chấp\s*hành\s*hiệu\s*lệnh\s*của\s*đèn\s*tín\s*hiệu\s*giao\s*thông)(?:\s+\1\b)+",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )
    if re.search(
        r"\bkhông\s*chấp\s*hành\s*hiệu\s*lệnh\s*của\s*đèn\s*tín\s*hiệu\s*giao\s*thông\b",
        value,
        flags=re.IGNORECASE,
    ):
        value = re.sub(r"\bvượt\s*đèn\s*đỏ\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    value = _canonicalize_memory_base(value)
    return value.strip(" .")


def _normalize_vehicle_surface(vehicle_key: str, surface: str) -> str:
    value = re.sub(r"\s+", " ", surface or "").strip().lower()
    if vehicle_key == "car":
        if value in {"oto", "ôto", "ô tô", "xe ô tô"}:
            return "xe ô tô" if value == "xe ô tô" else "ô tô"
    if vehicle_key == "motorcycle":
        if "gắn máy" in value:
            return "xe gắn máy"
        if "mô tô" in value:
            return "xe mô tô"
        return "xe máy"
    if vehicle_key == "bicycle":
        if "đạp máy" in value:
            return "xe đạp máy"
        if "đạp điện" in value:
            return "xe đạp điện"
        return "xe đạp"
    if vehicle_key == "pedestrian":
        return "người đi bộ"
    return re.sub(r"\s+", " ", surface or "").strip()


def _actor_phrase_for_vehicle(vehicle_key: str, vehicle_surface: str) -> str:
    if vehicle_key == "pedestrian":
        return "Người đi bộ"
    return f"Người điều khiển {vehicle_surface}"


def _detect_vehicle(text: str) -> tuple[str, str] | None:
    for vehicle_key, _, pattern in VEHICLE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return vehicle_key, _normalize_vehicle_surface(vehicle_key, match.group(0))
    return None


def _replace_vehicle(text: str, replacement: str) -> tuple[str, bool]:
    for pattern in VEHICLE_BUCKET_PATTERNS:
        if pattern.search(text or ""):
            rewritten = pattern.sub(replacement, text, count=1).strip()
            rewritten = re.sub(r"\s+,", ",", rewritten)
            rewritten = re.sub(r"\s{2,}", " ", rewritten).strip(" ,.;")
            return rewritten, True
    for _, _, pattern in VEHICLE_PATTERNS:
        if pattern.search(text or ""):
            rewritten = pattern.sub(replacement, text, count=1).strip()
            rewritten = re.sub(r"\s+,", ",", rewritten)
            rewritten = re.sub(r"\s{2,}", " ", rewritten).strip(" ,.;")
            return rewritten, True
    return text, False


def _append_doc_hint(text: str, docs: list[str]) -> str:
    if not docs:
        return text
    doc_hint = "; ".join(docs[:2])
    if not doc_hint or doc_hint.lower() in text.lower():
        return text
    return f"{text.strip(' .')} theo {doc_hint}"


def _should_attach_doc_hint(question: str, intent: str) -> bool:
    value = question or ""
    if DOCUMENT_REFERENCE_PATTERN.search(value):
        return True
    if EFFECTIVITY_PATTERN.search(value):
        return True
    if EVIDENCE_PATTERN.search(value):
        return True
    return intent in {"effectivity"}


def _canonicalize_memory_base(text: str) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    normalized = _normalize_compare(value)
    vehicle = _detect_vehicle(value)
    if not vehicle:
        return value
    vehicle_key, vehicle_surface = vehicle
    if (
        "khong chap hanh hieu lenh cua den tin hieu giao thong" in normalized
        or "vuot den do" in normalized
    ):
        return (
            f"{_actor_phrase_for_vehicle(vehicle_key, vehicle_surface)} không chấp hành hiệu lệnh của đèn tín hiệu giao thông "
            "bị xử phạt như thế nào?"
        )
    return value


def _normalize_compare(text: str) -> str:
    value = (text or "").lower().replace("đ", "d")
    import unicodedata

    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _normalize_compare(text).split():
        if len(token) <= 1:
            continue
        if token in COMPARE_STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.add(token)
    return tokens


def _vehicle_keys_in_text(text: str) -> set[str]:
    keys: set[str] = set()
    for vehicle_key, _, pattern in VEHICLE_PATTERNS:
        if pattern.search(text or ""):
            keys.add(vehicle_key)
    return keys


def _followup_payload(question: str) -> str:
    value = FOLLOWUP_PREFIX_PATTERN.sub("", re.sub(r"\s+", " ", question or "").strip())
    return value.strip(" ,.;:?!")


def _is_under_specified_followup(question: str) -> bool:
    payload = _followup_payload(question)
    token_count = len(payload.split())
    return token_count <= 6 or bool(SHORT_FOLLOWUP_PATTERN.search(question or ""))


def _should_prefer_rule_followup(
    question: str,
    fallback_query: str,
    resolved_query: str,
    relation: str,
) -> bool:
    if relation not in {"continue_same_topic", "replace_constraint", "add_constraint", "clarify_previous"}:
        return False
    if not fallback_query or not resolved_query:
        return False
    if not _is_under_specified_followup(question):
        return False

    fallback_terms = _content_tokens(fallback_query)
    resolved_terms = _content_tokens(resolved_query)
    if not fallback_terms or not resolved_terms:
        return False

    target_vehicle = _detect_vehicle(question)
    if target_vehicle:
        target_key = target_vehicle[0]
        resolved_vehicle_keys = _vehicle_keys_in_text(resolved_query)
        if resolved_vehicle_keys and resolved_vehicle_keys != {target_key}:
            return True

    missing_terms = fallback_terms - resolved_terms
    if len(missing_terms) >= 3 and len(fallback_terms) >= len(resolved_terms) + 2:
        return True

    resolved_norm = _normalize_compare(resolved_query)
    if "vi pham giao thong" in resolved_norm and len(missing_terms) >= 2:
        return True

    if len(resolved_terms) >= len(fallback_terms) + 4 and len(resolved_query) > len(fallback_query) * 1.35:
        return True

    return False


def _should_override_new_topic_followup(
    question: str,
    state: ConversationState | None,
    fallback_plan: QueryPlan | None,
    relation: str,
) -> bool:
    if relation != "new_topic":
        return False
    if not state or not fallback_plan or not fallback_plan.use_memory:
        return False
    if not _is_under_specified_followup(question):
        return False
    if DOCUMENT_REFERENCE_PATTERN.search(question or ""):
        return False
    if EFFECTIVITY_PATTERN.search(question or "") or EVIDENCE_PATTERN.search(question or ""):
        return False
    if not _detect_vehicle(question):
        return False
    base = _clean_memory_base(state.last_standalone_question or state.active_topic or "")
    if not base:
        return False
    return True


def _rule_rewrite_followup(question: str, state: ConversationState | None, intent: str) -> str:
    if not state:
        return question

    base = _clean_memory_base(state.last_standalone_question or state.active_topic or "")
    focus_docs = [d.doc_id for d in state.focus_docs[:2] if d.doc_id]
    current_question = re.sub(r"\s+", " ", question or "").strip()
    allow_doc_hint = _should_attach_doc_hint(current_question, intent)

    target_vehicle = _detect_vehicle(current_question)
    if base and target_vehicle:
        vehicle_key, replacement = target_vehicle
        rewritten, replaced = _replace_vehicle(base, replacement)
        if replaced:
            rewritten = _canonicalize_memory_base(rewritten)
            return _append_doc_hint(rewritten, focus_docs) if allow_doc_hint else rewritten
        rewritten = f"{base} đối với {replacement}"
        if vehicle_key == "pedestrian":
            rewritten = _canonicalize_memory_base(
                f"Người đi bộ {FOLLOWUP_PREFIX_PATTERN.sub('', current_question).strip(' ,.;')}"
            ) or "Người đi bộ bị xử phạt như thế nào?"
        return _append_doc_hint(rewritten, focus_docs) if allow_doc_hint else rewritten

    if not base:
        return current_question

    cleaned_followup = FOLLOWUP_PREFIX_PATTERN.sub("", current_question).strip(" ,.;")
    if not cleaned_followup:
        return _append_doc_hint(base, focus_docs) if allow_doc_hint else base

    rewritten = f"{base}. {cleaned_followup}"
    return _append_doc_hint(rewritten, focus_docs) if allow_doc_hint else rewritten


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
            "route": "traffic_law",
        }

    state = _state_for_plan(coerce_memory(memory))
    fallback_plan = prepare_memory_plan(query, state) if state else None
    fallback_query = fallback_plan.primary_query if fallback_plan and fallback_plan.use_memory else query
    fallback_context = fallback_plan.answer_memory_context if fallback_plan and fallback_plan.use_memory else ""

    debug: dict[str, Any] = {
        "accepted": False,
        "used_llm": False,
        "reason": "rule fallback" if fallback_plan else "raw fallback",
        "route": "traffic_law",
    }
    if fallback_plan:
        debug["fallback_plan"] = fallback_plan.debug

    if not (enable_llm and llm_call):
        if not state:
            debug["reason"] = "resolver disabled and no usable memory"
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

    if _should_override_new_topic_followup(
        question=query,
        state=state,
        fallback_plan=fallback_plan,
        relation=result.relation,
    ):
        result_dict["repair_applied"] = "short_followup_cannot_start_new_topic"
        result_dict["resolver_query_before_repair"] = resolved_query
        result_dict["reason"] = (
            (result.reason or "resolver accepted")
            + "; repaired because short vehicle follow-up must inherit previous legal act"
        )
        result_dict["relation"] = "replace_constraint"
        resolved_query = fallback_query

    if fallback_plan and fallback_plan.use_memory and _should_prefer_rule_followup(
        question=query,
        fallback_query=fallback_query,
        resolved_query=resolved_query,
        relation=result.relation,
    ):
        result_dict["repair_applied"] = "rule_followup_specificity_guard"
        result_dict["resolver_query_before_repair"] = resolved_query
        result_dict["reason"] = (
            (result.reason or "resolver accepted")
            + "; repaired with rule follow-up because resolver query lost previous constraints"
        )
        resolved_query = fallback_query

    result_dict["accepted"] = True
    memory_context = _context_from_resolution(state, result)
    return resolved_query, memory_context, result_dict


def _compact_text(text: str | None, limit: int = 350) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _entities_from_retrieval(
    question: str,
    retrieval: dict[str, Any] | None,
    passages: list[dict[str, Any]],
) -> list[str]:
    entities: list[str] = []
    question_text = question or ""
    if retrieval:
        question_text = str(retrieval.get("query") or retrieval.get("rewritten_query") or question_text)
    normalized_question = _normalize_compare(question_text)
    if retrieval:
        for item in retrieval.get("activated_entities") or []:
            text = _entity_text(item)
            if text:
                if ENTITY_ID_PATTERN.fullmatch(text):
                    continue
                if normalized_question and _normalize_compare(text) not in normalized_question:
                    continue
                entities.append(text)
    entities.extend(extract_entities_simple(question_text, []))
    if not entities:
        entities.extend(extract_entities_simple(question_text, passages))
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
        entity_extractor=lambda question, found_passages: _entities_from_retrieval(question, retrieval, found_passages),
    )

    if not updated.last_answer_summary and answer:
        updated.last_answer_summary = _compact_text(answer)

    return updated
