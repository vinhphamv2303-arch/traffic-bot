from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Intent = Literal[
    "definition",
    "effectivity",
    "penalty",
    "comparison",
    "roadmap",
    "procedure",
    "condition",
    "legal_qa",
    "chitchat",
    "unknown",
]

TopicAction = Literal[
    "continue",
    "start_new",
    "switch_topic",
    "clarify_previous",
]

Route = Literal[
    "traffic_law",
    "effectivity_index",
    "normal_chat",
]

Relation = Literal[
    "new_topic",
    "continue_same_topic",
    "replace_constraint",
    "add_constraint",
    "clarify_previous",
    "ask_evidence",
    "ask_effectivity",
]


@dataclass
class MemoryEntity:
    text: str
    label: str | None = None
    weight: float = 1.0


@dataclass
class MemoryDocument:
    doc_id: str
    title: str | None = None
    weight: float = 1.0


@dataclass
class ConversationState:
    """Short structured state for one chat session.

    Do not store long rewritten prompts here. Store only compact, verified state.
    """

    session_id: str
    active_topic: str | None = None

    last_user_question: str | None = None
    last_standalone_question: str | None = None
    last_answer_summary: str | None = None
    last_intent: Intent | None = None

    focus_entities: list[MemoryEntity] = field(default_factory=list)
    focus_docs: list[MemoryDocument] = field(default_factory=list)
    last_citations: list[str] = field(default_factory=list)

    # Keep only a few latest turns for debugging/rewrite. Do not use as legal evidence.
    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    session_id: str
    raw_question: str
    primary_query: str
    answer_question: str

    intent: Intent
    route: Route
    topic_action: TopicAction
    use_memory: bool
    memory_score: float
    new_topic_score: float

    retrieval_queries: list[str] = field(default_factory=list)
    boost_terms: list[str] = field(default_factory=list)
    doc_filter: list[str] = field(default_factory=list)

    # Context for answer prompt only. This is not legal evidence.
    answer_memory_context: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationResolveResult:
    relation: Relation
    use_memory: bool
    reason: str
    current_focus: str
    dropped_answered_content: list[str] = field(default_factory=list)
    changed_constraints: dict[str, Any] = field(default_factory=dict)
    standalone_question: str = ""
    retrieval_query: str = ""
    route: Route = "traffic_law"
    confidence: float = 0.0
    raw_response: str = ""
    used_llm: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
