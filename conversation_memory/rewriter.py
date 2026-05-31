from __future__ import annotations

from .models import ConversationState
from .prompts import REWRITE_PROMPT
from .utils import extract_json_object


def render_memory_for_rewrite(state: ConversationState | None) -> str:
    if not state:
        return "Không có memory."

    docs = ", ".join(d.doc_id for d in state.focus_docs[:3])
    entities = ", ".join(e.text for e in state.focus_entities[:6])

    lines = [
        f"active_topic: {state.active_topic or ''}",
        f"last_question: {state.last_user_question or ''}",
        f"last_standalone_question: {state.last_standalone_question or ''}",
        f"last_intent: {state.last_intent or ''}",
        f"focus_entities: {entities}",
        f"focus_docs: {docs}",
        f"last_answer_summary: {state.last_answer_summary or ''}",
    ]
    return "\n".join(lines)


def rewrite_with_llm(question: str, state: ConversationState, intent: str, llm_call) -> str:
    """Rewrite a follow-up question into a standalone question.

    llm_call is expected to accept a prompt string and return text.
    If parsing fails, the original question is returned.
    """
    if not llm_call:
        return question

    prompt = REWRITE_PROMPT.format(
        intent=intent,
        memory=render_memory_for_rewrite(state),
        question=question,
    )

    raw = llm_call(prompt)
    data = extract_json_object(raw or "")
    if not data:
        return question

    standalone = str(data.get("standalone_question", "")).strip()
    if not standalone:
        return question

    return standalone
