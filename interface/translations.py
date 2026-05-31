"""Bilingual UI strings for the Traffic Law RAG chatbot."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── Sidebar ──────────────────────────────────────────────
    "sidebar_title":        {"vi": "Cấu hình",              "en": "Settings"},
    "mode_label":           {"vi": "Chế độ",                "en": "Mode"},
    "retrieval_section":    {"vi": "Truy xuất",             "en": "Retrieval"},
    "pipeline_label":       {"vi": "Pipeline",              "en": "Pipeline"},
    "top_k_label":          {"vi": "Số passage top-k",      "en": "Top-k passages"},
    "candidate_k_label":    {"vi": "Ứng viên (Candidate-k)","en": "Candidate-k"},
    "context_llm_section":  {"vi": "Context đưa vào LLM",  "en": "Context for LLM"},
    "max_passages_label":   {"vi": "Số passage",            "en": "Max passages"},
    "max_chars_label":      {"vi": "Ký tự tối đa/passage", "en": "Max chars/passage"},
    "conv_section":         {"vi": "Hội thoại",             "en": "Conversation"},
    "memory_checkbox":      {"vi": "Ghi nhớ ngữ cảnh",     "en": "Enable memory"},
    "route_checkbox":       {"vi": "Route/rewrite câu hỏi","en": "Route/rewrite query"},
    "new_chat_btn":         {"vi": "Chat mới",              "en": "New chat"},
    "clear_memory_btn":     {"vi": "Xoá memory",            "en": "Clear memory"},
    "current_memory":       {"vi": "Memory hiện tại",       "en": "Current memory"},

    # ── Model settings ───────────────────────────────────────
    "model_section":        {"vi": "Cài đặt Model",         "en": "Model Settings"},
    "backend_label":        {"vi": "Backend",               "en": "Backend"},
    "preset_label":         {"vi": "Preset",                "en": "Preset"},
    "model_label":          {"vi": "Model",                 "en": "Model"},
    "max_tokens_label":     {"vi": "Token tối đa",          "en": "Max tokens"},
    "temperature_label":    {"vi": "Temperature",           "en": "Temperature"},
    "api_section":          {"vi": "Cài đặt API",           "en": "API Settings"},
    "api_key_label":        {"vi": "API key",               "en": "API key"},
    "base_url_label":       {"vi": "Base URL",              "en": "Base URL"},
    "ner_mode_label":       {"vi": "Chế độ NER",            "en": "NER Mode"},
    "threshold_label":      {"vi": "Ngưỡng (Threshold)",    "en": "Threshold"},
    "device_label":         {"vi": "Thiết bị",              "en": "Device"},

    # ── Header ───────────────────────────────────────────────
    "app_title":            {"vi": "Traffic Bot",           "en": "Traffic Bot"},
    "app_subtitle":         {"vi": "Trợ lý pháp luật giao thông Việt Nam",
                             "en": "Vietnamese Traffic Law Assistant"},

    # ── Mode labels ──────────────────────────────────────────
    "mode_answer":          {"vi": "Hỏi đáp",              "en": "Q&A"},
    "mode_retriever":       {"vi": "Test retriever",        "en": "Test Retriever"},
    "mode_ner":             {"vi": "Test NER",              "en": "Test NER"},

    # ── Suggestions ──────────────────────────────────────────
    "suggestions_label":    {"vi": "Gợi ý câu hỏi",        "en": "Suggestions"},

    # ── Chat ─────────────────────────────────────────────────
    "processing_spinner":   {"vi": "Đang xử lý...",         "en": "Processing..."},
    "step_analyzing":       {"vi": "Đang phân tích câu hỏi...", "en": "Analyzing query..."},
    "step_retrieving":      {"vi": "Đang truy xuất văn bản pháp luật...", "en": "Retrieving legal documents..."},
    "step_generating":      {"vi": "Đang tạo câu trả lời...", "en": "Generating answer..."},
    "step_done":            {"vi": "Hoàn tất",               "en": "Complete"},
    "step_ner_processing":  {"vi": "Đang nhận diện thực thể...", "en": "Recognizing entities..."},
    "no_answer":            {"vi": "Không có câu trả lời.", "en": "No answer returned."},
    "retrieved_count":      {"vi": "Đã truy xuất được **{n} passage**.",
                             "en": "Retrieved **{n} passages**."},
    "entity_count":         {"vi": "Phát hiện được **{n} entity**.",
                             "en": "Found **{n} entities**."},
    "error_prefix":         {"vi": "⚠️ Lỗi",               "en": "⚠️ Error"},

    # ── Result panels ────────────────────────────────────────
    "query_memory_expander":{"vi": "Truy vấn và bộ nhớ",   "en": "Query & Memory"},
    "route_label":          {"vi": "Route",                 "en": "Route"},
    "rewrite_label":        {"vi": "Query sau rewrite",     "en": "Rewritten query"},
    "memory_context_label": {"vi": "Ngữ cảnh hội thoại",   "en": "Conversation context"},
    "expanded_query_label": {"vi": "Query sau ghép memory", "en": "Memory-expanded query"},
    "memory_after_label":   {"vi": "Memory sau lượt này",   "en": "Memory after turn"},
    "retrieved_passages":   {"vi": "Passages được truy xuất","en": "Retrieved passages"},
    "context_expander":     {"vi": "Context đưa vào LLM",  "en": "Context used for LLM"},
    "no_entity":            {"vi": "Không phát hiện entity.","en": "No entities detected."},
    "raw_ner_output":       {"vi": "Kết quả NER thô",      "en": "Raw NER output"},
    "activated_entities":   {"vi": "Thực thể kích hoạt",   "en": "Activated entities"},
    "general_chat_info":    {"vi": "Câu hỏi được phân loại ngoài phạm vi RAG, hệ thống không truy xuất corpus.",
                             "en": "Question classified outside RAG scope, corpus not queried."},
    "passages_count":       {"vi": "Passages",              "en": "Passages"},
    "pipeline_info":        {"vi": "Pipeline",              "en": "Pipeline"},

    # ── Placeholders ─────────────────────────────────────────
    "placeholder_answer":   {"vi": "Nhập câu hỏi pháp luật giao thông...",
                             "en": "Ask about Vietnamese traffic law..."},
    "placeholder_retriever":{"vi": "Nhập truy vấn để test retriever...",
                             "en": "Enter query to test retriever..."},
    "placeholder_ner":      {"vi": "Nhập câu/đoạn để test NER...",
                             "en": "Enter text to test NER..."},

    # ── Settings ─────────────────────────────────────────────
    "lang_label":           {"vi": "Ngôn ngữ",              "en": "Language"},
    "settings_section":     {"vi": "Cài đặt",               "en": "Settings"},

    # ── Misc ─────────────────────────────────────────────────
    "unknown_doc":          {"vi": "Không rõ văn bản",      "en": "Unknown document"},
    "gliner_missing":       {"vi": "Env hiện tại chưa có package 'gliner'. Cài bằng: conda run -n kltn pip install gliner",
                             "en": "Package 'gliner' not found. Install with: conda run -n kltn pip install gliner"},
    "model_not_found":      {"vi": "Không tìm thấy GLiNER model",
                             "en": "GLiNER model not found"},
    "missing_artifact":     {"vi": "Thiếu artifact để chạy demo",
                             "en": "Missing artifact(s) for demo"},
}


def t(key: str, lang: str = "vi", **kwargs: str) -> str:
    """Return the translated string for *key* in *lang*, with optional formatting."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("vi", key))
    if kwargs:
        text = text.format(**kwargs)
    return text
