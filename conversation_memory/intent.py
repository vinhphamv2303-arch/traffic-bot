from __future__ import annotations

from .models import Intent
from .utils import norm

# Rule-first intent classifier. Keep this explicit and easy to debug.
INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    ("definition", [
        "là gì",
        "nghĩa là gì",
        "định nghĩa",
        "được hiểu là",
        "khái niệm",
        "giải thích",
    ]),
    ("effectivity", [
        "hiệu lực",
        "có hiệu lực",
        "hết hiệu lực",
        "còn hiệu lực",
        "ngưng hiệu lực",
        "bãi bỏ",
        "thay thế",
        "áp dụng từ ngày",
        "ngày áp dụng",
    ]),
    ("penalty", [
        "mức phạt",
        "phạt bao nhiêu",
        "bị phạt",
        "xử phạt",
        "phạt tiền",
        "tước giấy phép",
        "tước quyền sử dụng",
        "trừ điểm",
        "có bị phạt không",
        "bị xử lý",
    ]),
    ("comparison", [
        "so sánh",
        "khác nhau",
        "giống nhau",
        "điểm mới",
        "thay đổi gì",
        "so với",
    ]),
    ("roadmap", [
        "lộ trình",
        "giai đoạn",
        "triển khai",
        "thực hiện theo các giai đoạn",
        "từ ngày nào đến ngày nào",
    ]),
    ("procedure", [
        "thủ tục",
        "quy trình",
        "trình tự",
        "hồ sơ",
        "cần làm gì",
        "phải làm gì",
    ]),
    ("condition", [
        "điều kiện",
        "yêu cầu",
        "tiêu chí",
        "phải đáp ứng",
        "được phép khi nào",
    ]),
]

CHITCHAT_PATTERNS = [
    "xin chào",
    "chào bạn",
    "hello",
    "hi ",
    "cảm ơn",
    "thank",
]


def detect_intent(question: str) -> Intent:
    q = norm(question)

    if any(p in q for p in CHITCHAT_PATTERNS):
        return "chitchat"

    # Definition must win before effectivity to avoid routing "nghĩa là gì" into effectivity fast path.
    for intent, patterns in INTENT_PATTERNS:
        if any(p in q for p in patterns):
            return intent

    return "legal_qa"
