from __future__ import annotations

from .intent import detect_intent
from .models import ConversationState, QueryPlan, Route, TopicAction
from .scoring import (
    correction_score,
    entity_overlap_score,
    memory_score,
    new_topic_score,
)
from .utils import unique_keep_order


def choose_topic_action(question: str, mem_score: float, novelty_score: float) -> TopicAction:
    if correction_score(question) > 0:
        return "clarify_previous"
    if mem_score >= 0.35 and novelty_score < 0.75:
        return "continue"
    if novelty_score >= 0.75:
        return "start_new"
    return "switch_topic"


def choose_route(intent: str) -> Route:
    # Only effectivity questions can use the effectivity fast path.
    if intent == "effectivity":
        return "effectivity_index"
    if intent == "chitchat":
        return "normal_chat"
    return "traffic_law"


def build_answer_memory_context(state: ConversationState | None, use_memory: bool) -> str:
    if not state or not use_memory:
        return ""

    docs = "; ".join(d.doc_id for d in state.focus_docs[:3])
    entities = "; ".join(e.text for e in state.focus_entities[:5])

    parts: list[str] = []
    if state.active_topic:
        parts.append(f"Chủ đề đang theo dõi: {state.active_topic}")
    if entities:
        parts.append(f"Thực thể liên quan: {entities}")
    if docs:
        parts.append(f"Văn bản liên quan gần nhất: {docs}")
    if state.last_answer_summary:
        parts.append(f"Tóm tắt lượt trước: {state.last_answer_summary}")

    return "\n".join(parts)


def build_retrieval_queries(raw_question: str, standalone_question: str) -> list[str]:
    queries = [raw_question.strip()]
    if standalone_question.strip().lower() != raw_question.strip().lower():
        queries.append(standalone_question.strip())
    return unique_keep_order(queries)


def build_plan(
    session_id: str,
    question: str,
    state: ConversationState | None,
    rewrite_fn=None,
) -> QueryPlan:
    intent = detect_intent(question)
    mem_score = memory_score(question, state)
    novelty = new_topic_score(question, state)
    ent_overlap = entity_overlap_score(question, state)

    correction = correction_score(question)
    intent_needs_previous_subject = intent in {"penalty", "effectivity"} and state is not None
    use_memory = (mem_score >= 0.35 or correction > 0 or intent_needs_previous_subject) and novelty < 0.75

    # Important exception: definition question that repeats a previous entity.
    # Example: after asking about "vùng phát thải thấp", user asks "vùng phát thải thấp nghĩa là gì".
    if intent == "definition" and ent_overlap > 0:
        use_memory = True

    # Chitchat should not pull legal memory into retrieval.
    if intent == "chitchat":
        use_memory = False

    topic_action = choose_topic_action(question, mem_score, novelty)

    if not use_memory:
        standalone = question
    else:
        standalone = rewrite_fn(question=question, state=state, intent=intent) if rewrite_fn else question

    route = choose_route(intent)
    retrieval_queries = build_retrieval_queries(question, standalone)

    boost_terms: list[str] = []
    doc_filter: list[str] = []

    if state and use_memory:
        boost_terms = [e.text for e in state.focus_entities[:5]]

        # Soft doc filter only. If your current retriever does not support it, ignore it.
        if intent in ["definition", "effectivity", "legal_qa", "penalty", "procedure", "condition", "roadmap"]:
            doc_filter = [d.doc_id for d in state.focus_docs[:2]]

    answer_context = build_answer_memory_context(state, use_memory)

    debug = {
        "intent": intent,
        "route": route,
        "memory_score": round(mem_score, 4),
        "new_topic_score": round(novelty, 4),
        "entity_overlap": round(ent_overlap, 4),
        "use_memory": use_memory,
        "topic_action": topic_action,
        "primary_query": standalone,
        "retrieval_queries": retrieval_queries,
        "boost_terms": boost_terms,
        "doc_filter": doc_filter,
    }

    return QueryPlan(
        session_id=session_id,
        raw_question=question,
        primary_query=standalone,
        answer_question=standalone,
        intent=intent,
        route=route,
        topic_action=topic_action,
        use_memory=use_memory,
        memory_score=mem_score,
        new_topic_score=novelty,
        retrieval_queries=retrieval_queries,
        boost_terms=boost_terms,
        doc_filter=doc_filter,
        answer_memory_context=answer_context,
        debug=debug,
    )
