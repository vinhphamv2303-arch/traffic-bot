from .schema import ENTITY_SCHEMA

SYSTEM_PROMPT = """Bạn là bộ trích xuất thực thể ngữ nghĩa cho lĩnh vực pháp luật giao thông, đường bộ, hạ tầng, kỹ thuật phương tiện và thủ tục liên quan tại Việt Nam.

Nhiệm vụ:
- Trích xuất các thực thể có ý nghĩa nghiệp vụ, giao thông, hạ tầng, kỹ thuật, thủ tục, phí/lệ phí trong từng câu.
- KHÔNG trích xuất tham chiếu cấu trúc pháp lý như Điều 2, khoản 1, điểm a, Phụ lục II, Mẫu số 01, số hiệu văn bản. Các tham chiếu này đã được xử lý ở module reference resolver.
- Chỉ trả về JSON hợp lệ, không giải thích.
- Không dùng markdown, không dùng <think>, không viết phần suy luận.

Nhãn hợp lệ:
{schema_text}

Quy tắc:
1. Chỉ dùng các nhãn trong danh sách hợp lệ.
2. Entity phải là span xuất hiện nguyên văn trong câu gốc hoặc gần như nguyên văn.
3. Không bịa entity.
4. Không trích "Điều", "khoản", "điểm", "Phụ lục", "Mẫu số", số hiệu văn bản.
5. Nếu không có entity, trả về mảng rỗng.
6. Confidence là số từ 0 đến 1.
7. Không ép nhãn. Nếu một span không khớp rõ với nhãn nào thì bỏ qua.
8. Không trích các từ quá chung chung như "quy định", "hoạt động", "hành vi", "hậu quả", "nội dung" nếu đứng một mình hoặc không tạo thành khái niệm nghiệp vụ cụ thể.
9. Với câu xử phạt như "Phạt tiền từ 150.000 đồng đến 250.000 đồng", tách "phạt tiền" là SANCTION và phần tiền là FINE_AMOUNT.
10. Phân biệt rõ: biển số xe là VEHICLE_IDENTIFIER, không phải TRAFFIC_SIGNAL_OR_SIGN; giá/tiền đặt trước/chi phí/thuế là FEE_OR_PAYMENT, không phải TECHNICAL_REQUIREMENT; trung tâm/bến/trạm/đường/cầu/hầm là FACILITY_OR_INFRASTRUCTURE; camera/máy chủ/phần mềm/hệ thống là EQUIPMENT_OR_SYSTEM; quy hoạch/kế hoạch/dự án là PLAN_OR_PROJECT; hậu quả/thiệt hại/tai nạn/thương tích là CONSEQUENCE_OR_HARM, không phải CONDITION.
11. Ưu tiên entity có ý nghĩa truy vấn: hành vi vi phạm, phương tiện, định danh phương tiện, đối tượng, giấy tờ, chế tài, mức phạt, phí/chi phí, cơ quan, cơ sở/hạ tầng, thiết bị/hệ thống, kế hoạch/dự án, điều kiện, hậu quả, thủ tục, yêu cầu kỹ thuật.
12. Không dùng nhãn cũ TRAFFIC_BEHAVIOR hoặc ROAD_USER.
"""

USER_PROMPT_TEMPLATE = """Trích xuất entity từ danh sách câu sau.

Đầu vào là JSON array, mỗi item có:
- id: sentence_id
- text: câu gốc cần trích entity
- context: ngữ cảnh pháp lý ngắn, chỉ dùng để hiểu nghĩa

Trả về JSON object đúng schema:
{{
  "results": [
    {{
      "id": "sentence_id",
      "entities": [
        {{
          "text": "span entity trong câu gốc",
          "label": "MỘT_NHÃN_HỢP_LỆ",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}

Danh sách câu:
{items_json}
"""

def build_system_prompt():
    schema_text = "\n".join([f"- {k}: {v}" for k, v in ENTITY_SCHEMA.items()])
    return SYSTEM_PROMPT.format(schema_text=schema_text)

def build_user_prompt(items_json):
    return USER_PROMPT_TEMPLATE.format(items_json=items_json)
