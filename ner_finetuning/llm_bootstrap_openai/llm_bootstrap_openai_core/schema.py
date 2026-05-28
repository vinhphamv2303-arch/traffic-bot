ENTITY_SCHEMA = {
    "BEHAVIOR": "Hành vi/tình huống vi phạm hoặc hành vi sử dụng phương tiện cụ thể: vượt đèn đỏ, đi ngược chiều, không đội mũ bảo hiểm.",
    "VEHICLE": "Loại phương tiện cụ thể: xe mô tô, xe gắn máy, xe ô tô tải, xe khách, xe máy chuyên dùng.",
    "ACTOR": "Chủ thể cụ thể trong giao thông/nghiệp vụ: người điều khiển xe mô tô, chủ xe, Cảnh sát giao thông, cơ quan đăng ký xe.",
    "INFRASTRUCTURE": "Hạ tầng, thiết bị, hệ thống, biển báo, tín hiệu, bối cảnh đường/cơ sở vật chất giao thông.",
    "DOCUMENT": "Giấy tờ, giấy phép, chứng nhận, hồ sơ nghiệp vụ cụ thể; không phải văn bản pháp luật/reference.",
    "VEHICLE_CONDITION_OR_EQUIPMENT": "Tình trạng, kết cấu, bộ phận, trang bị, thiết bị hoặc đặc điểm kỹ thuật của phương tiện.",
    "CONDITION": "Điều kiện/yêu cầu áp dụng cụ thể: đủ tuổi lái xe, đủ sức khỏe, có GPLX phù hợp, chưa đủ tuổi điều khiển xe.",
}
ALLOWED_LABELS = set(ENTITY_SCHEMA.keys())
REFERENCE_LIKE_PATTERNS = [
    r"^điều\s+\d+[a-z]?$", r"^khoản\s+\d+$", r"^điểm\s+[a-zđ]$",
    r"^phụ\s+lục\s+([ivxlcdm]+|\d+|[a-z])$", r"^mẫu\s+(số\s+)?[0-9a-z_.\-\/]+$",
    r"^chương\s+([ivxlcdm]+|\d+)$", r"^mục\s+([ivxlcdm]+|\d+)$",
    r"^\d+/\d{4}/[a-zđ]+(-[a-zđ0-9]+)?$",
    r"^(thông tư|nghị định|luật|quyết định|nghị quyết)\s+(này|số\s+.+)$", r"^qcvn\s+.+$",
]
