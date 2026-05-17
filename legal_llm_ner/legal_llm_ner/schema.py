ENTITY_SCHEMA = {
    "VIOLATION_OR_BEHAVIOR": "Hành vi vi phạm hoặc hành vi nghiệp vụ cụ thể được điều chỉnh, ví dụ: không đội mũ bảo hiểm, vượt đèn đỏ, đi ngược chiều, chở hàng quá tải. Không gán cho các cụm chung chung như 'hành vi', 'hoạt động', 'hậu quả'.",
    "VEHICLE_TYPE": "Loại phương tiện, ví dụ: xe mô tô, xe gắn máy, xe ô tô, xe máy chuyên dùng.",
    "VEHICLE_IDENTIFIER": "Định danh phương tiện, ví dụ: biển số xe, số khung, số máy, mã định danh phương tiện.",
    "REGULATED_SUBJECT": "Đối tượng/cá nhân/tổ chức chịu điều chỉnh hoặc tham gia quy trình, ví dụ: người điều khiển xe, người đi bộ, chủ xe, người trúng đấu giá, tổ chức hành nghề đấu giá tài sản.",
    "DOCUMENT_OR_PERMIT": "Giấy tờ/giấy phép/chứng nhận, ví dụ: giấy phép lái xe, giấy đăng ký xe, chứng nhận kiểm định.",
    "LICENSE_CLASS": "Hạng giấy phép/chứng chỉ, ví dụ: hạng A1, hạng B, hạng C1E.",
    "SANCTION": "Hình thức xử phạt/chế tài, ví dụ: phạt tiền, tước quyền sử dụng GPLX, trừ điểm, tạm giữ phương tiện.",
    "FINE_AMOUNT": "Mức tiền phạt hoặc khoảng tiền phạt.",
    "FEE_OR_PAYMENT": "Khoản tiền không phải tiền phạt, ví dụ: phí, lệ phí, giá khởi điểm, tiền đặt trước, chi phí đấu giá, thuế giá trị gia tăng.",
    "AUTHORITY": "Cơ quan/người có thẩm quyền, ví dụ: Cảnh sát giao thông, cơ quan đăng ký xe, Bộ Công an.",
    "FACILITY_OR_INFRASTRUCTURE": "Công trình/cơ sở/hạ tầng giao thông hoặc cơ sở nghiệp vụ, ví dụ: đường bộ, quốc lộ, cầu, hầm, bến xe, trạm dừng nghỉ, trung tâm sát hạch, cơ sở đăng kiểm.",
    "EQUIPMENT_OR_SYSTEM": "Thiết bị/hệ thống/phần mềm phục vụ quản lý, vận hành, kiểm tra, ví dụ: camera, máy chủ, phần mềm quản lý, thiết bị giám sát hành trình, hệ thống thu phí.",
    "PLAN_OR_PROJECT": "Kế hoạch/quy hoạch/dự án/chương trình, ví dụ: quy hoạch mạng lưới đường bộ, dự án đầu tư xây dựng đường bộ, kế hoạch tổ chức đấu giá.",
    "LOCATION_OR_ROAD_CONTEXT": "Bối cảnh đường/vị trí giao thông, ví dụ: đường cao tốc, làn đường, phần đường, giao lộ.",
    "TRAFFIC_SIGNAL_OR_SIGN": "Tín hiệu/biển báo/vạch kẻ đường/hiệu lệnh giao thông, ví dụ: đèn tín hiệu, biển báo, vạch phân làn. Không dùng cho biển số xe.",
    "TECHNICAL_REQUIREMENT": "Yêu cầu/chỉ tiêu kỹ thuật, ví dụ: tốc độ tối đa, tải trọng, kích thước, nồng độ cồn, khí thải.",
    "PROCEDURE": "Thủ tục/quy trình/nghiệp vụ, ví dụ: đăng ký xe, sát hạch lái xe, cấp GPLX, kiểm định, đấu giá biển số xe, khám sức khỏe.",
    "CONDITION": "Điều kiện/ngoại lệ/yêu cầu áp dụng, ví dụ: đủ tuổi, đủ sức khỏe, có giấy phép phù hợp.",
    "CONSEQUENCE_OR_HARM": "Hậu quả/thiệt hại/tổn hại/tai nạn/thương tích có ý nghĩa pháp lý, ví dụ: gây tai nạn giao thông, thiệt hại tài sản, tổn thương cơ thể.",
    "TIME_OR_DURATION": "Thời hạn/thời gian/thời lượng, ví dụ: 07 ngày làm việc, 12 tháng, 24 tháng.",
}

BLOCKED_STRUCTURAL_LABELS = {
    "ARTICLE",
    "CLAUSE",
    "POINT",
    "APPENDIX",
    "FORM",
    "LEGAL_DOC",
    "LEGAL_DOC_NUMBER",
    "SECTION",
    "CHAPTER",
    "PART",
}

ALLOWED_LABELS = set(ENTITY_SCHEMA.keys())

LEGACY_LABEL_ALIASES = {
    "TRAFFIC_BEHAVIOR": "VIOLATION_OR_BEHAVIOR",
    "ROAD_USER": "REGULATED_SUBJECT",
}
