from __future__ import annotations
import json
from .schema import ENTITY_SCHEMA
SYSTEM_PROMPT = """Bạn là bộ trích xuất entity ngữ nghĩa phục vụ truy xuất văn bản pháp luật giao thông Việt Nam.

Mục tiêu:
- Chỉ trích các cụm giúp nối câu/passage theo ý nghĩa truy vấn.
- Chỉ trích entity xuất hiện trong trường text.
- Trường context chỉ để hiểu nghĩa, tuyệt đối không trích entity chỉ xuất hiện trong context.
- Không trích số hiệu văn bản, Điều, khoản, điểm, Phụ lục, Mẫu số, QCVN, hoặc tên văn bản quy phạm.

7 nhãn hợp lệ:
{schema}

Nguyên tắc chọn span:
1. Chỉ chọn span cụ thể, giàu thông tin. Không chọn từ chung chung.
2. Nếu cụm đầy đủ là "người điều khiển xe gắn máy", không trả "người điều khiển".
3. Nếu cụm đầy đủ là "xe ô tô tải", không trả "xe" hoặc "xe ô tô" nếu mất nghĩa.
4. Nếu cụm là "không có gương chiếu hậu", nhãn là VEHICLE_CONDITION_OR_EQUIPMENT.
5. DOCUMENT chỉ là giấy tờ/giấy phép/chứng nhận/hồ sơ nghiệp vụ; không phải văn bản pháp luật hay phụ lục/mẫu.
6. CONDITION chỉ là điều kiện cụ thể như "đủ tuổi lái xe", "có giấy phép lái xe phù hợp"; không trả riêng "điều kiện", "yêu cầu", "tiêu chuẩn".
7. Nếu không có entity tốt, trả mảng rỗng.

Phân biệt:
- BEHAVIOR = hành động/tình huống vi phạm hoặc sử dụng phương tiện.
- VEHICLE_CONDITION_OR_EQUIPMENT = tình trạng/trang bị/bộ phận/yêu cầu của phương tiện.
- INFRASTRUCTURE = hạ tầng, tín hiệu, biển báo, thiết bị/hệ thống/cơ sở vật chất giao thông.
"""
USER_PROMPT = """Trích xuất entity từ các câu sau.

Input JSON array:
- id: sentence id
- text: câu gốc. Chỉ được trích entity từ text này.
- context: ngữ cảnh pháp lý ngắn. Chỉ dùng để hiểu nghĩa, không trích entity từ context.

Output JSON object:
{{
  "results": [
    {{"id": "sentence id", "entities": [{{"text": "span xuất hiện nguyên văn trong text", "label": "LABEL"}}]}}
  ]
}}

Câu cần xử lý:
{items_json}
"""
def build_system_prompt():
    schema="\n".join([f"- {k}: {v}" for k,v in ENTITY_SCHEMA.items()])
    return SYSTEM_PROMPT.format(schema=schema)
def build_user_prompt(items):
    return USER_PROMPT.format(items_json=json.dumps(items, ensure_ascii=False, indent=2))
