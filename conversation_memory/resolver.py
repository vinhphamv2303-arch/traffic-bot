from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .models import ConversationResolveResult, ConversationState, Relation, Route
from .scoring import followup_marker_score
from .utils import extract_json_object


VALID_RELATIONS: set[str] = {
    "new_topic",
    "continue_same_topic",
    "replace_constraint",
    "add_constraint",
    "clarify_previous",
    "ask_evidence",
    "ask_effectivity",
}

VALID_ROUTES: set[str] = {"traffic_law", "effectivity_index", "normal_chat"}

AMBIGUOUS_FOLLOWUP_PATTERN = re.compile(
    r"\b("
    r"vậy|thế|còn|thì\s*sao|trường\s*hợp\s*này|trường\s*hợp\s*đó|"
    r"quy\s*định\s*này|quy\s*định\s*đó|hành\s*vi\s*này|hành\s*vi\s*đó|"
    r"văn\s*bản\s*này|văn\s*bản\s*đó|điều\s*này|điều\s*đó|"
    r"khoản\s*này|khoản\s*đó|điểm\s*này|điểm\s*đó|"
    r"đối\s*với|ý\s*tôi\s*là|ý\s*mình\s*là|không\s*phải"
    r")\b",
    flags=re.IGNORECASE,
)


RESOLVER_SYSTEM_PROMPT = """
Bạn là Conversation Resolver cho hệ thống RAG pháp luật giao thông Việt Nam.

Nhiệm vụ:
- Xác định câu hỏi trọng tâm của lượt hiện tại.
- Nếu câu hỏi hiện tại là follow-up, dùng MEMORY để viết lại thành câu hỏi độc lập.
- Nếu câu hỏi hiện tại thay đổi một điều kiện so với lượt trước, thay điều kiện cũ bằng điều kiện mới.
- Nếu câu hỏi hiện tại mở chủ đề mới, không dùng memory cũ.
- Bỏ các phần đã được trả lời ở lượt trước, trừ khi cần giữ để hiểu ngữ cảnh.
- Không trả lời câu hỏi pháp luật.
- Không tự bịa Điều/Khoản/Điểm.
- Không tự bịa mức phạt, thời hạn, ngày hiệu lực.
- Output bắt buộc là JSON hợp lệ, không markdown, không giải thích ngoài JSON.

Các relation hợp lệ:
- new_topic
- continue_same_topic
- replace_constraint
- add_constraint
- clarify_previous
- ask_evidence
- ask_effectivity

Route hợp lệ:
- traffic_law
- effectivity_index
- normal_chat

JSON schema bắt buộc:
{
  "relation": "new_topic | continue_same_topic | replace_constraint | add_constraint | clarify_previous | ask_evidence | ask_effectivity",
  "use_memory": true,
  "reason": "lý do ngắn",
  "current_focus": "người dùng đang hỏi gì ở lượt này",
  "dropped_answered_content": ["phần đã trả lời ở lượt trước nếu có"],
  "changed_constraints": {},
  "standalone_question": "câu hỏi độc lập sạch để người đọc hiểu được",
  "retrieval_query": "truy vấn sạch cho RAG, không chứa mô tả memory dài",
  "route": "traffic_law | effectivity_index | normal_chat",
  "confidence": 0.0
}
""".strip()


FEW_SHOTS: list[dict[str, str]] = [
    {
        "role": "user",
        "content": """
CURRENT_QUESTION:
vậy còn xe máy thì sao

MEMORY:
{
  "last_user_question": "Phạt người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông như thế nào?",
  "last_standalone_question": "Phạt người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông như thế nào?",
  "last_answer_summary": "Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
  "last_intent": "penalty",
  "focus_entities": ["người điều khiển xe ô tô", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"],
  "focus_docs": ["168/2024/NĐ-CP"]
}
""".strip(),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "relation": "replace_constraint",
                "use_memory": True,
                "reason": "Người dùng hỏi tiếp cùng hành vi nhưng đổi phương tiện từ ô tô sang xe máy.",
                "current_focus": "mức xử phạt đối với người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "dropped_answered_content": ["mức phạt đối với xe ô tô đã được trả lời"],
                "changed_constraints": {"vehicle": {"from": "xe ô tô", "to": "xe máy"}},
                "standalone_question": "Người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào?",
                "retrieval_query": "người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào 168/2024/NĐ-CP",
                "route": "traffic_law",
                "confidence": 0.93,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION:
văn bản này còn hiệu lực không

MEMORY:
{
  "last_user_question": "Nghị định 168/2024/NĐ-CP quy định gì về xử phạt vi phạm giao thông?",
  "last_standalone_question": "Nghị định 168/2024/NĐ-CP quy định gì về xử phạt vi phạm giao thông?",
  "last_answer_summary": "Văn bản quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ.",
  "last_intent": "legal_qa",
  "focus_entities": ["xử phạt vi phạm giao thông"],
  "focus_docs": ["168/2024/NĐ-CP"]
}
""".strip(),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "relation": "ask_effectivity",
                "use_memory": True,
                "reason": "Cụm 'văn bản này' tham chiếu văn bản ở lượt trước.",
                "current_focus": "tình trạng hiệu lực của Nghị định 168/2024/NĐ-CP",
                "dropped_answered_content": ["nội dung xử phạt chung đã được trả lời"],
                "changed_constraints": {},
                "standalone_question": "Nghị định 168/2024/NĐ-CP còn hiệu lực không?",
                "retrieval_query": "Nghị định 168/2024/NĐ-CP còn hiệu lực không",
                "route": "effectivity_index",
                "confidence": 0.9,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION:
Tỉnh Đồng Tháp và tỉnh Kiên Giang mang biển số mấy?

MEMORY:
{
  "last_user_question": "Lái xe khi có nồng độ cồn bị xử phạt thế nào?",
  "last_standalone_question": "Lái xe khi có nồng độ cồn bị xử phạt thế nào?",
  "last_answer_summary": "Đã trả lời về xử phạt nồng độ cồn.",
  "last_intent": "penalty",
  "focus_entities": ["nồng độ cồn"],
  "focus_docs": ["168/2024/NĐ-CP"]
}
""".strip(),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "relation": "new_topic",
                "use_memory": False,
                "reason": "Câu hỏi mới về ký hiệu biển số địa phương, không phụ thuộc chủ đề nồng độ cồn.",
                "current_focus": "ký hiệu biển số xe của tỉnh Đồng Tháp và tỉnh Kiên Giang",
                "dropped_answered_content": [],
                "changed_constraints": {},
                "standalone_question": "Tỉnh Đồng Tháp và tỉnh Kiên Giang có ký hiệu biển số xe là gì?",
                "retrieval_query": "ký hiệu biển số xe tỉnh Đồng Tháp tỉnh Kiên Giang",
                "route": "traffic_law",
                "confidence": 0.92,
            },
            ensure_ascii=False,
        ),
    },
]


def should_call_conversation_resolver(question: str, state: ConversationState | None) -> bool:
    if not state or state.turn_count <= 0:
        return False
    q = (question or "").strip()
    if not q:
        return False
    token_count = len(q.split())
    return (
        token_count <= 14
        or bool(AMBIGUOUS_FOLLOWUP_PATTERN.search(q))
        or followup_marker_score(q) > 0
    )


def render_memory_for_resolver(state: ConversationState | None) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "last_user_question": state.last_user_question or "",
        "last_standalone_question": state.last_standalone_question or "",
        "last_answer_summary": state.last_answer_summary or "",
        "last_intent": state.last_intent or "",
        "focus_entities": [entity.text for entity in state.focus_entities[:8] if entity.text],
        "focus_docs": [doc.doc_id for doc in state.focus_docs[:4] if doc.doc_id],
        "last_citations": list(state.last_citations or [])[:6],
        "recent_turns": list(state.recent_turns or [])[-3:],
    }


def build_resolver_messages(current_question: str, state: ConversationState | None) -> list[dict[str, str]]:
    memory_json = json.dumps(render_memory_for_resolver(state), ensure_ascii=False, indent=2)
    current_prompt = f"""
CURRENT_QUESTION:
{current_question}

MEMORY:
{memory_json}
""".strip()
    return [
        {"role": "system", "content": RESOLVER_SYSTEM_PROMPT},
        *FEW_SHOTS,
        {"role": "user", "content": current_prompt},
    ]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "có", "co"}
    return bool(value)


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, confidence))


def parse_resolver_output(raw_response: str) -> ConversationResolveResult | None:
    data = extract_json_object(raw_response)
    if not data:
        return None

    relation = str(data.get("relation") or "").strip()
    if relation not in VALID_RELATIONS:
        relation = "new_topic"

    route = str(data.get("route") or "").strip()
    if route == "general_chat":
        route = "normal_chat"
    if route not in VALID_ROUTES:
        route = "traffic_law"

    dropped = data.get("dropped_answered_content") or []
    if not isinstance(dropped, list):
        dropped = [str(dropped)]

    changed = data.get("changed_constraints") or {}
    if not isinstance(changed, dict):
        changed = {}

    standalone = str(data.get("standalone_question") or "").strip()
    retrieval_query = str(data.get("retrieval_query") or standalone).strip()
    use_memory = _coerce_bool(data.get("use_memory"))
    if relation == "new_topic":
        use_memory = False

    return ConversationResolveResult(
        relation=relation,  # type: ignore[arg-type]
        use_memory=use_memory,
        reason=str(data.get("reason") or "").strip(),
        current_focus=str(data.get("current_focus") or "").strip(),
        dropped_answered_content=[str(item).strip() for item in dropped if str(item).strip()],
        changed_constraints=changed,
        standalone_question=standalone,
        retrieval_query=retrieval_query,
        route=route,  # type: ignore[arg-type]
        confidence=_coerce_confidence(data.get("confidence")),
        raw_response=raw_response,
        used_llm=True,
    )


def resolve_with_llm(
    current_question: str,
    state: ConversationState | None,
    llm_call: Callable[[list[dict[str, str]]], str],
) -> ConversationResolveResult | None:
    raw_response = llm_call(build_resolver_messages(current_question, state))
    result = parse_resolver_output(raw_response)
    if result:
        return result
    return ConversationResolveResult(
        relation="new_topic",
        use_memory=False,
        reason="resolver returned non-json",
        current_focus=current_question,
        standalone_question=current_question,
        retrieval_query=current_question,
        route="traffic_law",
        confidence=0.0,
        raw_response=raw_response,
        used_llm=True,
        error="resolver returned non-json",
    )
