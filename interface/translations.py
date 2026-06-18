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
    "context_llm_section":  {"vi": "Ngữ cảnh đưa vào LLM",  "en": "Context for LLM"},
    "max_passages_label":   {"vi": "Số passage",            "en": "Max passages"},
    "max_chars_label":      {"vi": "Ký tự tối đa/passage",  "en": "Max chars/passage"},
    "conv_section":         {"vi": "Hội thoại",             "en": "Conversation"},
    "memory_checkbox":      {"vi": "Ghi nhớ ngữ cảnh",     "en": "Enable memory"},
    "route_checkbox":       {"vi": "Bộ xử lý truy vấn",     "en": "Query resolver"},
    "new_chat_btn":         {"vi": "Chat mới",              "en": "New chat"},
    "clear_memory_btn":     {"vi": "Xoá memory",            "en": "Clear memory"},
    "current_memory":       {"vi": "Memory hiện tại",       "en": "Current memory"},

    # ── Model settings ───────────────────────────────────────
    "model_section":        {"vi": "Cài đặt Model",         "en": "Model Settings"},
    "backend_label":        {"vi": "Hệ thống (Backend)",    "en": "Backend"},
    "preset_label":         {"vi": "Cấu hình sẵn (Preset)", "en": "Preset"},
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
    "mode_retriever":       {"vi": "Kiểm thử truy xuất (Retriever)", "en": "Test Retriever"},
    "mode_ner":             {"vi": "Kiểm thử NER",          "en": "Test NER"},

    # ── Suggestions ──────────────────────────────────────────
    "suggestions_label":    {"vi": "Gợi ý câu hỏi",        "en": "Suggestions"},

    # ── Chat ─────────────────────────────────────────────────
    "processing_spinner":   {"vi": "Đang xử lý...",         "en": "Processing..."},
    "step_analyzing":       {"vi": "Đang phân tích câu hỏi...", "en": "Analyzing query..."},
    "step_retrieving":      {"vi": "Đang truy xuất văn bản pháp luật...", "en": "Retrieving legal documents..."},
    "step_generating":      {"vi": "Đang tạo câu trả lời...", "en": "Generating answer..."},
    "step_done":            {"vi": "Hoàn tất",               "en": "Complete"},
    "step_ner_processing":  {"vi": "Đang nhận diện thực thể...", "en": "Recognizing entities..."},
    "step_routing":         {"vi": "Xác định luồng xử lý...", "en": "Determining route..."},
    "step_context":         {"vi": "Chuẩn bị ngữ cảnh...",   "en": "Preparing context..."},
    "step_effectivity":     {"vi": "Đang tra cứu hiệu lực...","en": "Checking effectivity..."},
    "route_traffic":        {"vi": "Tra cứu luật GT",        "en": "Traffic law RAG"},
    "route_general":        {"vi": "Hỏi đáp thông thường",   "en": "General chat"},
    "route_effectivity":    {"vi": "Tra cứu hiệu lực",       "en": "Effectivity lookup"},
    "reasoning_process":    {"vi": "Quá trình suy luận",     "en": "Reasoning process"},
    "no_answer":            {"vi": "Không có câu trả lời.", "en": "No answer returned."},
    "retrieved_count":      {"vi": "Đã truy xuất được **{n} passage**.",
                             "en": "Retrieved **{n} passages**."},
    "entity_count":         {"vi": "Phát hiện được **{n} thực thể**.",
                             "en": "Found **{n} entities**."},
    "error_prefix":         {"vi": "⚠️ Lỗi",               "en": "⚠️ Error"},

    # ── Result panels ────────────────────────────────────────
    "query_memory_expander":{"vi": "Truy vấn và bộ nhớ",   "en": "Query & Memory"},
    "route_label":          {"vi": "Route",                 "en": "Route"},
    "rewrite_label":        {"vi": "Standalone query",        "en": "Standalone query"},
    "memory_context_label": {"vi": "Ngữ cảnh hội thoại",   "en": "Conversation context"},
    "expanded_query_label": {"vi": "Query sau resolver",      "en": "Resolver query"},
    "resolver_label":       {"vi": "Bộ xử lý truy vấn",     "en": "Query resolver"},
    "memory_after_label":   {"vi": "Bộ nhớ sau lượt này",   "en": "Memory after turn"},
    "retrieved_passages":   {"vi": "Passages được truy xuất","en": "Retrieved passages"},
    "context_expander":     {"vi": "Ngữ cảnh đưa vào LLM",  "en": "Context used for LLM"},
    "no_entity":            {"vi": "Không phát hiện thực thể nào.","en": "No entities detected."},
    "raw_ner_output":       {"vi": "Kết quả NER thô",      "en": "Raw NER output"},
    "activated_entities":   {"vi": "Thực thể kích hoạt",   "en": "Activated entities"},
    "general_chat_info":    {"vi": "Câu hỏi được phân loại ngoài phạm vi RAG, hệ thống không truy xuất corpus.",
                             "en": "Question classified outside RAG scope, corpus not queried."},
    "passages_count":       {"vi": "Passage",               "en": "Passages"},
    "pipeline_info":        {"vi": "Luồng xử lý (Pipeline)","en": "Pipeline"},

    # -- Added labels --
    "score_label":          {"vi": "Score",                 "en": "Score"},
    "entities_count":       {"vi": "Thực thể",              "en": "Entities"},
    "memory_badge":         {"vi": "có bộ nhớ",             "en": "memory"},
    "loading_model":        {"vi": "Đang tải model cục bộ: {model_name}", "en": "Loading local model: {model_name}"},
    "skip_retrieval_fast":  {"vi": "Tra cứu trực tiếp từ metadata", "en": "Used structured metadata"},
    "skip_context_fast":    {"vi": "Không cần truy xuất",   "en": "No retrieval context needed"},
    "skip_generation_fast": {"vi": "Không cần gọi LLM",     "en": "No LLM generation needed"},
    "note_fast_path":       {"vi": "Đã dùng fast-path hiệu lực.", "en": "Used structured effectivity fast-path."},
    "skip_retrieval":       {"vi": "Bỏ qua truy xuất",      "en": "Retrieval skipped"},
    "skip_context":         {"vi": "Không có ngữ cảnh đủ rõ", "en": "Context unavailable"},
    "skip_generation":      {"vi": "Không gọi LLM do thiếu ngữ cảnh", "en": "Skipped due to missing context"},

    # ── Placeholders ─────────────────────────────────────────
    "placeholder_answer":   {"vi": "Nhập câu hỏi...",
                             "en": "Ask a question..."},
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
