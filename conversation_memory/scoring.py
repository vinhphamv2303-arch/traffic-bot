from __future__ import annotations

from .models import ConversationState
from .utils import norm

FOLLOWUP_MARKERS = [
    "vậy",
    "thế",
    "trường hợp đó",
    "trường hợp này",
    "quy định này",
    "quy định đó",
    "hành vi này",
    "hành vi đó",
    "văn bản này",
    "văn bản đó",
    "điều này",
    "điều đó",
    "khoản này",
    "điểm này",
    "nếu vậy",
    "nếu thế",
    "cụ thể",
    "thì sao",
    "như trên",
    "nêu trên",
    "đó",
    "này",
]

CORRECTION_MARKERS = [
    "ý tôi là",
    "ý mình là",
    "không phải",
    "nhầm",
    "sửa lại",
    "tức là",
]

NEW_TOPIC_HINTS = [
    "theo quy định hiện hành",
    "cho tôi hỏi về",
    "tôi muốn hỏi về",
    "quy định về",
]


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def followup_marker_score(question: str) -> float:
    q = norm(question)
    return 1.0 if _contains_any(q, FOLLOWUP_MARKERS) else 0.0


def correction_score(question: str) -> float:
    q = norm(question)
    return 1.0 if _contains_any(q, CORRECTION_MARKERS) else 0.0


def entity_overlap_score(question: str, state: ConversationState | None) -> float:
    if not state or not state.focus_entities:
        return 0.0

    q = norm(question)
    entities = [norm(e.text) for e in state.focus_entities if e.text]
    if not entities:
        return 0.0

    hit_weight = 0.0
    total_weight = 0.0
    for e in state.focus_entities:
        text = norm(e.text)
        if not text:
            continue
        w = max(float(e.weight), 0.0)
        total_weight += w
        if text in q:
            hit_weight += w

    if total_weight <= 0:
        return 0.0
    return min(1.0, hit_weight / total_weight)


def doc_overlap_score(question: str, state: ConversationState | None) -> float:
    if not state or not state.focus_docs:
        return 0.0

    q = norm(question)
    docs = [norm(d.doc_id) for d in state.focus_docs if d.doc_id]
    if not docs:
        return 0.0

    hit_weight = 0.0
    total_weight = 0.0
    for d in state.focus_docs:
        doc_id = norm(d.doc_id)
        if not doc_id:
            continue
        w = max(float(d.weight), 0.0)
        total_weight += w
        if doc_id in q:
            hit_weight += w

    if total_weight <= 0:
        return 0.0
    return min(1.0, hit_weight / total_weight)


def new_topic_score(question: str, state: ConversationState | None) -> float:
    """Score high when the current question likely opens a new topic."""
    if not state or state.turn_count == 0:
        return 1.0

    q = norm(question)
    token_count = len(q.split())

    # If it mentions old entities/docs, it is probably not a new topic.
    overlap = entity_overlap_score(q, state) + doc_overlap_score(q, state)
    if overlap > 0:
        return 0.0

    # Corrections and short ellipsis are usually follow-up.
    if correction_score(q) > 0:
        return 0.0
    if token_count <= 5 and followup_marker_score(q) > 0:
        return 0.1

    # Long, self-contained legal question without overlap is likely a new topic.
    if token_count >= 10:
        return 0.85

    if _contains_any(q, NEW_TOPIC_HINTS) and token_count >= 7:
        return 0.75

    return 0.5


def memory_score(question: str, state: ConversationState | None) -> float:
    if not state or state.turn_count == 0:
        return 0.0

    marker = followup_marker_score(question)
    correction = correction_score(question)
    entity = entity_overlap_score(question, state)
    doc = doc_overlap_score(question, state)
    novelty = new_topic_score(question, state)

    score = 0.40 * marker + 0.20 * correction + 0.30 * entity + 0.10 * doc

    # Strongly down-rank long self-contained questions with no overlap.
    if novelty >= 0.75 and marker == 0 and correction == 0:
        score -= 0.45

    return max(0.0, min(1.0, score))
