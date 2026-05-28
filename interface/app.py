from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_generation.conversation_memory import empty_memory  # noqa: E402
from interface.rag_pipeline import DEFAULT_MODELS, PIPELINES, run_demo_answer  # noqa: E402


st.set_page_config(
    page_title="Traffic Law RAG Demo",
    page_icon="TL",
    layout="wide",
)


if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = empty_memory().to_dict()


def reset_conversation_memory() -> None:
    st.session_state.conversation_memory = empty_memory().to_dict()


def render_retrieved_passages(result: dict) -> None:
    retrieval = result.get("retrieval") or {}
    passages = retrieval.get("results") or []
    if not passages:
        st.caption("Không có passage retrieval.")
        return

    st.caption(f"{len(passages)} passages")
    for idx, item in enumerate(passages, start=1):
        doc = item.get("document_number") or item.get("document_id") or "Không rõ văn bản"
        title = item.get("document_title") or ""
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else score
        heading = f"{idx}. {doc}"
        if title:
            heading += f" | {title}"
        if score_text is not None:
            heading += f" | score={score_text}"

        with st.expander(heading, expanded=idx <= 2):
            st.caption(item.get("path_text") or item.get("passage_id") or "")
            st.write(item.get("text") or "")
            components = item.get("score_components") or {}
            if components:
                st.json(components, expanded=False)


st.title("Demo hỏi đáp pháp luật giao thông")
st.caption("Hệ thống tự phân loại câu hỏi, viết lại truy vấn theo thuật ngữ pháp lý, retrieve passage và sinh câu trả lời.")

with st.sidebar:
    st.header("Cấu hình")
    pipeline_key = st.selectbox(
        "Pipeline retrieval",
        options=list(PIPELINES.keys()),
        format_func=lambda key: PIPELINES[key].display_name,
        index=0,
    )
    st.caption(PIPELINES[pipeline_key].description)

    mode = st.radio(
        "LLM backend",
        options=["openai", "openrouter", "local"],
        format_func=lambda value: {
            "openai": "OpenAI API",
            "openrouter": "OpenRouter API",
            "local": "Local Hugging Face",
        }[value],
        horizontal=False,
    )
    model_name = st.text_input("Model", value=DEFAULT_MODELS[mode])
    api_key = st.text_input(
        "API key",
        value="",
        type="password",
        help="Có thể bỏ trống nếu đã đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env.",
        disabled=mode == "local",
    )
    base_url = st.text_input(
        "Base URL tuỳ chọn",
        value="",
        help="Chỉ cần điền nếu dùng endpoint tương thích OpenAI khác mặc định.",
        disabled=mode == "local",
    )

    enable_query_router = st.checkbox(
        "Bật route/rewrite câu hỏi",
        value=True,
        help="Dùng LLM để nhận diện câu hỏi ngoài phạm vi và rewrite cách hỏi đời thường trước retrieval.",
    )
    enable_memory = st.checkbox(
        "Ghi nhớ ngữ cảnh hội thoại",
        value=True,
        help="Dùng văn bản, điều khoản và ràng buộc ở lượt trước để hiểu các câu hỏi nối tiếp như 'trường hợp này', 'còn xe máy thì sao'.",
    )
    if st.button("Xoá ngữ cảnh hội thoại"):
        reset_conversation_memory()
        st.rerun()
    with st.expander("Memory hiện tại", expanded=False):
        st.json(st.session_state.conversation_memory, expanded=False)

    top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=5)
    candidate_k = st.slider("Candidate-k", min_value=50, max_value=500, value=300, step=50)
    max_context_passages = st.slider("Số passage đưa vào LLM", min_value=1, max_value=10, value=5)
    max_chars_per_passage = st.slider("Ký tự tối đa mỗi passage", min_value=600, max_value=2500, value=1800, step=100)
    max_new_tokens = st.slider("Max new tokens", min_value=128, max_value=1024, value=512, step=64)

examples = [
    "Mức xử phạt khi lái xe sau khi uống rượu đối với từng loại phương tiện là gì?",
    "Ô tô vượt đèn đỏ bị phạt bao nhiêu?",
    "Thời gian lái xe liên tục tối đa của người lái xe là bao nhiêu?",
    "Xin chào, bạn có thể làm gì?",
]

example = st.selectbox("Ví dụ", options=examples, index=0)
question = st.text_area("Câu hỏi", value=example, height=110)

run_button = st.button("Chạy demo", type="primary")

if run_button:
    question = question.strip()
    if not question:
        st.warning("Vui lòng nhập câu hỏi.")
        st.stop()

    try:
        with st.spinner("Đang xử lý câu hỏi..."):
            result = run_demo_answer(
                question=question,
                pipeline_key=pipeline_key,
                mode=mode,
                model_name=model_name.strip() or DEFAULT_MODELS[mode],
                api_key=api_key.strip() or None,
                base_url=base_url.strip() or None,
                top_k=top_k,
                candidate_k=candidate_k,
                max_context_passages=max_context_passages,
                max_chars_per_passage=max_chars_per_passage,
                enable_query_router=enable_query_router,
                max_new_tokens=max_new_tokens,
                conversation_memory=st.session_state.conversation_memory if enable_memory else None,
            )
            if enable_memory:
                st.session_state.conversation_memory = result.get("conversation_memory") or st.session_state.conversation_memory
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([0.95, 1.05])
    with left:
        st.subheader("Truy vấn")
        st.write(f"Route: `{result.get('route')}`")
        if result.get("route_reason"):
            st.caption(result["route_reason"])
        if result.get("rewritten_query"):
            st.write("Query sau rewrite:")
            st.code(result["rewritten_query"], language="text")
        if result.get("memory_context"):
            st.write("Ngữ cảnh hội thoại được dùng:")
            st.code(result["memory_context"], language="text")
        if result.get("expanded_query"):
            with st.expander("Query sau khi ghép memory", expanded=False):
                st.code(result["expanded_query"], language="text")
        with st.expander("Chi tiết route/rewrite", expanded=False):
            st.json(result.get("query_preprocessing") or {}, expanded=False)
        with st.expander("Memory sau lượt này", expanded=False):
            st.json(result.get("conversation_memory") or {}, expanded=False)

        st.subheader("Passage retrieval")
        render_retrieved_passages(result)

    with right:
        st.subheader("Câu trả lời")
        st.markdown(result.get("answer") or "")
        if result.get("route") == "general_chat":
            st.info("Câu hỏi được phân loại ngoài phạm vi RAG, nên hệ thống không truy xuất corpus.")
        with st.expander("Context đưa vào LLM", expanded=False):
            st.text(result.get("context_used") or "")
else:
    st.info("Nhập câu hỏi rồi bấm **Chạy demo**.")
