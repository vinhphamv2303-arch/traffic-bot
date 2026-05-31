from __future__ import annotations

import json
from pathlib import Path

from .models import ConversationState, MemoryDocument, MemoryEntity


class InMemoryConversationStore:
    """Simple per-process memory store for demos/tests."""

    def __init__(self):
        self._states: dict[str, ConversationState] = {}

    def get(self, session_id: str) -> ConversationState:
        if session_id not in self._states:
            self._states[session_id] = ConversationState(session_id=session_id)
        return self._states[session_id]

    def set(self, session_id: str, state: ConversationState) -> None:
        self._states[session_id] = state

    def reset(self, session_id: str) -> None:
        self._states[session_id] = ConversationState(session_id=session_id)


class JsonFileConversationStore:
    """Tiny persistent store. Useful for demo UI restarts.

    Not designed for concurrent writes. Use Redis/DB for production.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_all(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, session_id: str) -> ConversationState:
        data = self._read_all()
        raw = data.get(session_id)
        if not raw:
            return ConversationState(session_id=session_id)

        state = ConversationState(session_id=session_id)
        state.active_topic = raw.get("active_topic")
        state.last_user_question = raw.get("last_user_question")
        state.last_standalone_question = raw.get("last_standalone_question")
        state.last_answer_summary = raw.get("last_answer_summary")
        state.last_intent = raw.get("last_intent")
        state.last_citations = raw.get("last_citations", [])
        state.recent_turns = raw.get("recent_turns", [])
        state.turn_count = raw.get("turn_count", 0)
        state.focus_entities = [MemoryEntity(**x) for x in raw.get("focus_entities", [])]
        state.focus_docs = [MemoryDocument(**x) for x in raw.get("focus_docs", [])]
        return state

    def set(self, session_id: str, state: ConversationState) -> None:
        data = self._read_all()
        data[session_id] = state.to_dict()
        self._write_all(data)

    def reset(self, session_id: str) -> None:
        data = self._read_all()
        data.pop(session_id, None)
        self._write_all(data)
