from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.rag_pipeline import (  # noqa: E402
    GAZETTEER_ROOT,
    PIPELINES,
    generate_answer_openai,
    load_retriever,
    retrieve_multi_query,
)


st.set_page_config(
    page_title="Traffic Law RAG Demo",
    page_icon="TL",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_retriever(pipeline_key: str):
    config = PIPELINES[pipeline_key]
    return load_retriever(config, gazetteer_root=GAZETTEER_ROOT)


def render_retrieved_passages(results: list[dict]):
    for idx, item in enumerate(results, start=1):
        doc = item.get("document_number") or item.get("document_id") or "Không rõ văn bản"
        score = item.get("score")
        title = f"{idx}. {doc} | score={score}"
        with st.expander(title, expanded=idx <= 2):
            st.caption(item.get("path_text") or item.get("passage_id") or "")
            st.write(item.get("text") or "")
            comps = item.get("score_components") or {}
            if comps:
                st.json(comps, expanded=False)


st.title("Demo hỏi đáp pháp luật giao thông")
st.caption("Retrieve từ corpus pháp luật giao thông và sinh câu trả lời bằng GPT-4o mini.")

with st.sidebar:
    st.header("Cấu hình")
    pipeline_key = st.selectbox(
        "Pipeline truy xuất",
        options=list(PIPELINES.keys()),
        format_func=lambda key: PIPELINES[key].display_name,
        index=0,
    )
    st.caption(PIPELINES[pipeline_key].description)

    provider = st.radio(
        "LLM provider",
        options=["openai", "openrouter"],
        format_func=lambda x: "OpenAI API" if x == "openai" else "OpenRouter API",
        horizontal=True,
    )
    default_model = "gpt-4o-mini" if provider == "openai" else "openai/gpt-4o-mini"
    model = st.text_input("Model", value=default_model)
    api_key = st.text_input(
        "API key",
        value="",
        type="password",
        help="Có thể bỏ trống nếu đã có OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env ở repo root.",
    )
    top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=5)
    candidate_k = st.slider("Candidate-k", min_value=50, max_value=500, value=300, step=50)
    max_chars_per_passage = st.slider("Số ký tự tối đa mỗi passage", min_value=600, max_value=2500, value=1800, step=100)
    enable_expansion = st.checkbox(
        "Mở rộng truy vấn pháp lý",
        value=True,
        help="Tự thêm cụm pháp lý chuẩn như 'nồng độ cồn trong máu hoặc hơi thở' khi người dùng hỏi bằng ngôn ngữ đời thường.",
    )
    max_queries = st.slider("Số query tối đa sau mở rộng", min_value=1, max_value=10, value=8)

examples = [
    "Thời gian lái xe liên tục tối đa của người lái xe ô tô là bao nhiêu?",
    "Người điều khiển xe mô tô không đội mũ bảo hiểm bị xử lý như thế nào?",
    "Giấy phép lái xe phải đáp ứng điều kiện gì?",
]

question = st.text_area(
    "Câu hỏi",
    value=examples[0],
    height=110,
)

cols = st.columns([1, 1, 3])
run_button = cols[0].button("Chạy demo", type="primary", use_container_width=True)
clear_button = cols[1].button("Xóa cache retriever", use_container_width=True)

if clear_button:
    st.cache_resource.clear()
    st.success("Đã xóa cache retriever.")

if run_button:
    question = question.strip()
    if not question:
        st.warning("Vui lòng nhập câu hỏi.")
        st.stop()

    config = PIPELINES[pipeline_key]
    with st.spinner("Đang load retriever và truy xuất passage..."):
        retriever = cached_retriever(pipeline_key)
        retrieval_result = retrieve_multi_query(
            retriever,
            question=question,
            config=config,
            top_k=top_k,
            candidate_k=candidate_k,
            enable_expansion=enable_expansion,
            max_queries=max_queries,
        )

    results = retrieval_result.get("results") or []
    left, right = st.columns([0.92, 1.08])

    with left:
        st.subheader("Kết quả truy xuất")
        st.caption(f"{len(results)} passages")
        with st.expander("Các query đã dùng để retrieve", expanded=True):
            for idx, query in enumerate(retrieval_result.get("expanded_queries") or [question], start=1):
                st.write(f"{idx}. {query}")
        render_retrieved_passages(results)

        with st.expander("Entity được kích hoạt", expanded=False):
            st.json(retrieval_result.get("activated_entities") or [], expanded=False)

    with right:
        st.subheader("Câu trả lời")
        try:
            with st.spinner("Đang gọi GPT-4o mini..."):
                generation = generate_answer_openai(
                    question=question,
                    retrieval_result=retrieval_result,
                    api_key=api_key or None,
                    model=model.strip() or "gpt-4o-mini",
                    provider=provider,
                    max_passages=top_k,
                    max_chars_per_passage=max_chars_per_passage,
                )
            st.markdown(generation["answer"])
            if generation.get("usage"):
                st.caption(f"Token usage: {generation['usage']}")
            with st.expander("Context gửi vào model", expanded=False):
                st.text(generation.get("context") or "")
        except Exception as exc:
            st.error(str(exc))
            with st.expander("Vẫn hiển thị passages đã retrieve", expanded=False):
                render_retrieved_passages(results)
else:
    st.info("Nhập câu hỏi rồi bấm **Chạy demo**.")
