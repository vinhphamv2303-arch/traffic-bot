from __future__ import annotations

import json
import re
import unicodedata
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
    r"văn\s*bản\s*này|văn\s*bản\s*đó|nghị\s*định\s*này|nghị\s*định\s*đó|"
    r"luật\s*này|luật\s*đó|thông\s*tư\s*này|thông\s*tư\s*đó|"
    r"điều\s*này|điều\s*đó|khoản\s*này|khoản\s*đó|điểm\s*này|điểm\s*đó|"
    r"đối\s*với|ý\s*tôi\s*là|ý\s*mình\s*là|không\s*phải"
    r")\b",
    flags=re.IGNORECASE,
)

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

# Số hiệu văn bản phổ biến trong corpus giao thông: 168/2024/NĐ-CP, 36/2024/QH15,
# 57/2025/NQ-HĐND, 41/2024/TT-BGTVT, 65/2024/TT-BCA, ...
DOC_ID_RE = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*(?:"
    r"NĐ-CP|ND-CP|QH\d*|NQ-[A-ZÀ-ỸĐ]+|TT-[A-ZÀ-ỸĐ]+|"
    r"QĐ-[A-ZÀ-ỸĐ]+|QD-[A-ZÀ-ỸĐ]+|TTBCA|TTBGTVT"
    r")\b",
    flags=re.IGNORECASE,
)
ENTITY_ID_PATTERN = re.compile(r"^ent_[0-9a-f]{8,}$", flags=re.IGNORECASE)

RESOLVER_SYSTEM_PROMPT = """
Bạn là Conversation Resolver kiêm Legal Query Rewriter cho hệ thống RAG pháp luật giao thông Việt Nam.

Nhiệm vụ duy nhất:
Biến CURRENT_QUESTION thành một câu hỏi/truy vấn pháp lý độc lập, sạch, phù hợp để đưa vào retriever/answerer.

Bạn phải làm đồng thời:
- Xác định câu hỏi trọng tâm của lượt hiện tại.
- Nếu CURRENT_QUESTION là follow-up, chỉ kế thừa phần thật sự cần thiết từ MEMORY.
- Nếu CURRENT_QUESTION thay đổi điều kiện so với lượt trước, hãy thay điều kiện cũ bằng điều kiện mới.
  Ví dụ: ô tô -> xe máy; dưới 18 tuổi -> dưới 16 tuổi; Hà Nội -> TP.HCM.
- Nếu CURRENT_QUESTION mở chủ đề mới, không dùng MEMORY cũ.
- Nếu CURRENT_QUESTION là follow-up rất ngắn như "còn ô tô thì sao", "với xe đạp thì sao", "thế xe máy", thì mặc định phải kế thừa hành vi/vấn đề pháp lý ở lượt trước, trừ khi CURRENT_QUESTION tự nêu rõ một chủ đề pháp lý mới.
- Bỏ phần đã được trả lời ở lượt trước, trừ khi cần giữ để hiểu câu hỏi hiện tại.
- Chuẩn hóa ngôn ngữ đời thường thành ngôn ngữ pháp luật vừa đủ cho retrieval.
- Nếu câu hỏi hỏi về việc áp dụng quy định theo thời gian, ví dụ vi phạm xảy ra trước một mốc nhưng bị xử phạt sau mốc đó, thì phải giữ đây là câu hỏi về nguyên tắc áp dụng theo thời điểm, không được tự đổi thành truy vấn mức phạt của một hành vi cụ thể.
- Nếu câu hỏi hỏi về đào tạo lái xe thực hành ngoài đường, quãng đường, thời gian, số km, số giờ, thì phải rewrite về thuật ngữ đào tạo thực hành lái xe trên đường giao thông; không được trôi sang đăng kiểm, thực tập đăng kiểm viên hay biểu mẫu báo cáo nếu câu hỏi đang hỏi yêu cầu đào tạo.
- Không trả lời câu hỏi pháp luật.
- Không tự bịa Điều/Khoản/Điểm.
- Không tự bịa mức phạt, thời hạn, ngày hiệu lực.
- Output bắt buộc là JSON hợp lệ, không markdown, không giải thích ngoài JSON.

QUY TẮC RẤT QUAN TRỌNG VỀ MEMORY VĂN BẢN:
- MEMORY có thể chứa focus_docs, last_citations hoặc số hiệu văn bản của lượt trước.
- Không được chép focus_docs, last_citations, Điều/Khoản/Điểm hoặc số hiệu văn bản từ MEMORY vào standalone_question/retrieval_query chỉ vì MEMORY có chúng.
- Với câu hỏi thông thường về hành vi, đối tượng, phương tiện, mức phạt, điều kiện..., MEMORY chỉ dùng để hiểu ngữ cảnh, không dùng để giới hạn văn bản.
- Chỉ được dùng số hiệu văn bản từ MEMORY khi CURRENT_QUESTION tham chiếu trực tiếp đến văn bản đó, ví dụ:
  "văn bản này", "văn bản đó", "nghị định này", "luật này", "thông tư này", "nghị quyết này", "nó".
- Hoặc khi CURRENT_QUESTION hỏi trực tiếp về hiệu lực/còn hiệu lực/hết hiệu lực/bãi bỏ/thay thế/căn cứ/văn bản nào.
- Nếu người dùng không tự nêu số văn bản trong CURRENT_QUESTION và câu hỏi không tham chiếu trực tiếp tới văn bản, retrieval_query không được chứa số văn bản.
- Không được làm rộng một hành vi cụ thể thành cụm chung kiểu "vi phạm giao thông", "xử phạt giao thông", "quy định giao thông" nếu lượt trước đã có hành vi cụ thể.
- Nếu CURRENT_QUESTION chỉ đổi phương tiện/đối tượng, standalone_question phải giữ nguyên hành vi pháp lý của lượt trước và chỉ thay đúng ràng buộc đó.
- Không được mở rộng một phương tiện thành danh sách nhiều phương tiện tương tự, trừ khi CURRENT_QUESTION tự hỏi nhiều loại phương tiện. Ví dụ:
  - "xe đạp" thì chỉ giữ "xe đạp"
  - "ô tô" thì chỉ giữ "ô tô"
  - "xe máy" thì chỉ giữ "xe máy" hoặc "xe mô tô" nếu CURRENT_QUESTION dùng đúng cách gọi đó
- Với follow-up ngắn, chỉ chọn relation "new_topic" khi CURRENT_QUESTION thật sự nêu một chủ đề pháp lý mới, không phụ thuộc hành vi/vấn đề ở lượt trước.

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
  "retrieval_query": "truy vấn sạch cho RAG, không chứa mô tả memory dài, không tự thêm số văn bản cũ",
  "route": "traffic_law | effectivity_index | normal_chat",
  "confidence": 0.0
}
""".strip()

FEW_SHOTS: list[dict[str, str]] = [
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: vậy còn xe máy thì sao
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
                "retrieval_query": "người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào",
                "route": "traffic_law",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: với xe đạp thì sao
MEMORY:
{
  "last_user_question": "Còn xe máy thì sao?",
  "last_standalone_question": "Người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào?",
  "last_answer_summary": "Đã trả lời về mức xử phạt đối với xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
  "last_intent": "penalty",
  "focus_entities": ["xe máy", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"],
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
                "reason": "Người dùng hỏi tiếp cùng hành vi nhưng đổi phương tiện từ xe máy sang xe đạp.",
                "current_focus": "mức xử phạt đối với người điều khiển xe đạp không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "dropped_answered_content": ["mức phạt đối với xe máy đã được trả lời"],
                "changed_constraints": {"vehicle": {"from": "xe máy", "to": "xe đạp"}},
                "standalone_question": "Người điều khiển xe đạp không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào?",
                "retrieval_query": "người điều khiển xe đạp không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào",
                "route": "traffic_law",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: đối với người dưới 16 thì sao?
MEMORY:
{
  "last_user_question": "Người dưới 18 tuổi vi phạm giao thông có bị phạt không?",
  "last_standalone_question": "Người dưới 18 tuổi vi phạm giao thông có bị xử phạt không?",
  "last_answer_summary": "Đã trả lời về trách nhiệm xử phạt đối với người dưới 18 tuổi vi phạm giao thông.",
  "last_intent": "penalty",
  "focus_entities": ["người dưới 18 tuổi", "vi phạm giao thông"],
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
                "reason": "Người dùng hỏi tiếp cùng chủ đề nhưng thay điều kiện độ tuổi từ dưới 18 tuổi sang dưới 16 tuổi.",
                "current_focus": "người dưới 16 tuổi vi phạm giao thông có bị xử phạt không",
                "dropped_answered_content": ["nội dung đã trả lời về người dưới 18 tuổi"],
                "changed_constraints": {"age": {"from": "dưới 18 tuổi", "to": "dưới 16 tuổi"}},
                "standalone_question": "Người dưới 16 tuổi vi phạm giao thông có bị xử phạt không?",
                "retrieval_query": "người dưới 16 tuổi vi phạm giao thông có bị xử phạt không",
                "route": "traffic_law",
                "confidence": 0.92,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: Nếu tôi vi phạm giao thông vào cuối tháng 12 năm 2024 nhưng đến tháng 2 năm 2025 mới bị phạt nguội thì áp dụng mức phạt cũ hay mức mới?
MEMORY:
{}
""".strip(),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "relation": "new_topic",
                "use_memory": False,
                "reason": "Câu hỏi mới về nguyên tắc áp dụng mức xử phạt theo thời điểm xảy ra vi phạm và thời điểm xử phạt.",
                "current_focus": "áp dụng mức phạt cũ hay mức phạt mới khi hành vi vi phạm xảy ra trước thời điểm quy định mới có hiệu lực nhưng bị xử phạt sau thời điểm đó",
                "dropped_answered_content": [],
                "changed_constraints": {},
                "standalone_question": "Nếu hành vi vi phạm giao thông xảy ra trước khi quy định xử phạt mới có hiệu lực nhưng việc xử phạt diễn ra sau thời điểm quy định mới có hiệu lực thì áp dụng mức phạt cũ hay mức phạt mới?",
                "retrieval_query": "áp dụng mức phạt cũ hay mức phạt mới khi hành vi vi phạm giao thông xảy ra trước khi quy định mới có hiệu lực nhưng bị xử phạt sau thời điểm quy định mới có hiệu lực phạt nguội",
                "route": "traffic_law",
                "confidence": 0.93,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: học viên sẽ phải ôm vô lăng chạy thực tế ngoài đường giao thông (không tính sân tập) tổng cộng bao nhiêu km và trong mấy tiếng?
MEMORY:
{}
""".strip(),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "relation": "new_topic",
                "use_memory": False,
                "reason": "Câu hỏi mới về yêu cầu đào tạo lái xe thực hành trên đường giao thông.",
                "current_focus": "thời gian và quãng đường đào tạo thực hành lái xe trên đường giao thông của học viên, không tính phần sân tập",
                "dropped_answered_content": [],
                "changed_constraints": {},
                "standalone_question": "Thời gian và quãng đường đào tạo thực hành lái xe trên đường giao thông của học viên là bao nhiêu, không tính phần thực hành trên sân tập lái?",
                "retrieval_query": "đào tạo lái xe thời gian thực hành trên đường giao thông quãng đường thực hành trên đường giao thông của học viên không tính sân tập là bao nhiêu km bao nhiêu giờ",
                "route": "traffic_law",
                "confidence": 0.92,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: văn bản này còn hiệu lực không
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
                "reason": "Cụm 'văn bản này' tham chiếu trực tiếp văn bản ở lượt trước.",
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
CURRENT_QUESTION: căn cứ ở đâu?
MEMORY:
{
  "last_user_question": "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
  "last_standalone_question": "Người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào?",
  "last_answer_summary": "Người điều khiển xe máy bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.",
  "last_intent": "penalty",
  "focus_entities": ["xe máy", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"],
  "focus_docs": ["168/2024/NĐ-CP"],
  "last_citations": ["Điều 7, Khoản 7, Điểm c, Nghị định 168/2024/NĐ-CP"]
}
""".strip(),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "relation": "ask_evidence",
                "use_memory": True,
                "reason": "Người dùng hỏi căn cứ của câu trả lời ngay trước đó.",
                "current_focus": "căn cứ pháp lý cho mức xử phạt xe máy không chấp hành đèn tín hiệu giao thông",
                "dropped_answered_content": ["mức phạt đã được trả lời"],
                "changed_constraints": {},
                "standalone_question": "Căn cứ pháp lý về xử phạt người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông nằm ở đâu?",
                "retrieval_query": "căn cứ pháp lý xử phạt người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "route": "traffic_law",
                "confidence": 0.88,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """
CURRENT_QUESTION: Tỉnh Đồng Tháp và tỉnh Kiên Giang mang biển số mấy?
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


def _normalize_overlap_text(text: str) -> str:
    text = _strip_accents(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _filtered_focus_entities(state: ConversationState) -> list[str]:
    reference_text = " ".join(
        part for part in [state.last_standalone_question or "", state.last_user_question or ""] if part
    )
    normalized_reference = _normalize_overlap_text(reference_text)
    filtered: list[str] = []
    for entity in state.focus_entities[:8]:
        text = (entity.text or "").strip()
        if not text:
            continue
        if ENTITY_ID_PATTERN.fullmatch(text):
            continue
        normalized_entity = _normalize_overlap_text(text)
        if normalized_reference and normalized_entity and normalized_entity not in normalized_reference:
            continue
        filtered.append(text)
    return filtered[:6]


def render_memory_for_resolver(state: ConversationState | None) -> dict[str, Any]:
    if not state:
        return {}

    # Giữ focus_docs trong MEMORY để resolver hiểu được các cụm "văn bản này",
    # nhưng prompt + post-validation sẽ không cho chép doc vào query thường.
    return {
        "last_user_question": state.last_user_question or "",
        "last_standalone_question": state.last_standalone_question or "",
        "last_answer_summary": state.last_answer_summary or "",
        "last_intent": state.last_intent or "",
        "focus_entities": _filtered_focus_entities(state),
        "focus_docs": [doc.doc_id for doc in state.focus_docs[:4] if doc.doc_id],
        "last_citations": list(state.last_citations or [])[:6],
        "recent_turns": list(state.recent_turns or [])[-3:],
    }


def build_resolver_messages(
    current_question: str,
    state: ConversationState | None,
) -> list[dict[str, str]]:
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


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def _doc_key(doc_id: str) -> str:
    # So sánh mềm giữa NĐ/ND, có/không dấu, có/không khoảng trắng.
    return re.sub(r"\s+", "", _strip_accents(doc_id).lower())


def _doc_ids_in_text(text: str) -> set[str]:
    return {_doc_key(match.group(0)) for match in DOC_ID_RE.finditer(text or "")}


def _allows_memory_document_in_query(
    current_question: str,
    relation: str,
    route: str,
) -> bool:
    q = current_question or ""

    if DOCUMENT_REFERENCE_PATTERN.search(q):
        return True

    if relation in {"ask_effectivity"} or route == "effectivity_index":
        return True

    # Hỏi căn cứ/văn bản nào có thể cần citation/document từ lượt trước để hiểu "căn cứ ở đâu".
    if relation == "ask_evidence" and EVIDENCE_PATTERN.search(q):
        return True

    if EFFECTIVITY_PATTERN.search(q):
        return True

    return False


def _clean_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:]){2,}", r"\1", text)
    return text.strip(" .,-;:")


def _strip_disallowed_doc_ids(
    text: str,
    current_question: str,
    relation: str,
    route: str,
) -> str:
    """Remove doc ids copied from MEMORY unless the current question really needs them.

    Keep a doc id if:
    - the user explicitly typed it in CURRENT_QUESTION; or
    - CURRENT_QUESTION refers to the document itself ("văn bản này", effectivity, evidence).

    This prevents normal follow-ups like "vậy còn xe máy thì sao" from becoming
    "... 168/2024/NĐ-CP", while preserving cases like "văn bản này còn hiệu lực không".
    """
    if not text:
        return text

    current_doc_keys = _doc_ids_in_text(current_question)
    allow_memory_doc = _allows_memory_document_in_query(current_question, relation, route)

    def repl(match: re.Match[str]) -> str:
        doc_id = match.group(0)
        if _doc_key(doc_id) in current_doc_keys:
            return doc_id
        if allow_memory_doc:
            return doc_id
        return ""

    return _clean_spacing(DOC_ID_RE.sub(repl, text))


def _force_route_by_current_question(current_question: str, route: str) -> str:
    q = current_question or ""

    if EFFECTIVITY_PATTERN.search(q):
        return "effectivity_index"

    # Câu định nghĩa không được nhảy vào fast-path hiệu lực chỉ vì memory có document.
    if re.search(r"\b(là\s*gì|nghĩa\s*là\s*gì|định\s*nghĩa|được\s*hiểu\s*là)\b", q, re.I):
        return "traffic_law"

    return route


def parse_resolver_output(
    raw_response: str,
    current_question: str = "",
) -> ConversationResolveResult | None:
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
    route = _force_route_by_current_question(current_question, route)

    dropped = data.get("dropped_answered_content") or []
    if not isinstance(dropped, list):
        dropped = [str(dropped)]

    changed = data.get("changed_constraints") or {}
    if not isinstance(changed, dict):
        changed = {}

    standalone = str(data.get("standalone_question") or "").strip()
    retrieval_query = str(data.get("retrieval_query") or standalone).strip()

    standalone = _strip_disallowed_doc_ids(
        standalone,
        current_question=current_question,
        relation=relation,
        route=route,
    )
    retrieval_query = _strip_disallowed_doc_ids(
        retrieval_query,
        current_question=current_question,
        relation=relation,
        route=route,
    )

    if not standalone:
        standalone = current_question.strip()
    if not retrieval_query:
        retrieval_query = standalone

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
    result = parse_resolver_output(raw_response, current_question=current_question)
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
