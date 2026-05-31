from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .planner import build_plan
from .rewriter import rewrite_with_llm
from .store import InMemoryConversationStore
from .updater import update_state_after_answer


class ConversationMemoryManager:
    """High-level memory middleware.

    prepare() is called before retrieval.
    commit() is called after the answer is generated.
    """

    def __init__(
        self,
        llm_call: Callable[[str], str] | None = None,
        store=None,
        entity_extractor: Callable[[str, list[dict[str, Any]]], list[str]] | None = None,
    ):
        self.llm_call = llm_call
        self.store = store or InMemoryConversationStore()
        self.entity_extractor = entity_extractor

    def _rewrite_fn(self, question, state, intent):
        if not self.llm_call:
            return question
        return rewrite_with_llm(
            question=question,
            state=state,
            intent=intent,
            llm_call=self.llm_call,
        )

    def prepare(self, session_id: str, question: str):
        state = self.store.get(session_id)
        return build_plan(
            session_id=session_id,
            question=question,
            state=state,
            rewrite_fn=self._rewrite_fn,
        )

    def commit(
        self,
        session_id: str,
        plan,
        answer: str,
        retrieved_passages: list[dict[str, Any]],
    ) -> None:
        state = self.store.get(session_id)
        new_state = update_state_after_answer(
            state=state,
            plan=plan,
            answer=answer,
            retrieved_passages=retrieved_passages,
            entity_extractor=self.entity_extractor,
        )
        self.store.set(session_id, new_state)

    def get_state(self, session_id: str):
        return self.store.get(session_id)

    def reset(self, session_id: str) -> None:
        self.store.reset(session_id)
