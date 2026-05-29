from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_generation.conversation_memory import empty_memory  # noqa: E402
from interface.rag_pipeline import (  # noqa: E402
    DEFAULT_MODELS,
    GAZETTEER_ROOT,
    PIPELINES,
    run_demo_answer,
    run_demo_retrieval,
)

NER_DIR = ROOT / "ner_finetuning"
GLINER_MODEL_DIR = NER_DIR / "data" / "models" / "gliner_traffic_ner" / "final_model"
GAZETTEER_BUILDING_DIR = NER_DIR / "gazetteer_building"

LABELS = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]

MODE_LABELS = {
    "answer": "Hỏi đáp",
    "retriever": "Test retriever",
    "ner": "Test NER",
}

BACKEND_LABELS = {
    "openai": "OpenAI API",
    "openrouter": "OpenRouter API",
    "local": "Local Hugging Face",
}

MODEL_PRESETS = {
    "openai": {
        "GPT-4o mini": "gpt-4o-mini",
    },
    "openrouter": {
        "Qwen2.5 7B Instruct": "qwen/qwen-2.5-7b-instruct",
        "GPT-4o mini": "openai/gpt-4o-mini",
    },
    "local": {
        "Qwen2.5 7B Instruct": "Qwen/Qwen2.5-7B-Instruct",
    },
}

NER_MODE_LABELS = {
    "gazetteer": "Gazetteer",
    "model": "GLiNER",
    "hybrid": "Hybrid",
}


st.set_page_config(
    page_title="Traffic Law RAG",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root, .stApp, [data-testid="stAppViewContainer"] {
            --primary-color: #3b82f6 !important;
            --primary-color-background: rgba(59, 130, 246, 0.18) !important;
            --primary-color-dark: #2563eb !important;
            --text-color: #e5e7eb !important;
            --background-color: #05070d !important;
            --secondary-background-color: #0b1220 !important;
            --sidebar-width: 300px;
            --chat-width: min(900px, calc(100vw - var(--sidebar-width) - 2rem));
            --chat-left: calc(var(--sidebar-width) + (100vw - var(--sidebar-width) - var(--chat-width)) / 2);
        }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], main {
            background: #05070d !important;
            color: #e5e7eb !important;
        }
        .block-container {
            padding-top: 0.85rem;
            padding-bottom: 6rem;
            max-width: 1080px;
            background: #05070d !important;
        }
        h1, h2, h3, h4, h5, h6,
        p, label, span, div, small,
        [data-testid="stMarkdownContainer"] {
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: #05070d !important;
            border-right: 1px solid #1f2937;
        }
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background: #05070d !important;
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stNumberInput label {
            color: #e5e7eb !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] hr {
            border-color: #1f2937 !important;
        }
        .hero {
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            background: #0b1220;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.34);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .hero h1 {
            color: #f8fafc !important;
            font-size: 1.35rem;
            line-height: 1.2;
            margin: 0;
            font-weight: 750;
            letter-spacing: 0;
        }
        .mode-chip {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.18);
            color: #bfdbfe !important;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .composer-tools {
            display: flex;
            justify-content: flex-start;
            width: auto;
            margin: 0;
            padding: 0;
        }
        [class*="st-key-composer_tools"] {
            position: fixed;
            left: calc(var(--chat-left) + 1rem);
            bottom: 4.05rem;
            z-index: 10000;
            width: auto !important;
            background: transparent !important;
            padding: 0 !important;
        }
        [class*="st-key-composer_tools"] > div {
            background: transparent !important;
            padding: 0 !important;
        }
        [class*="st-key-composer_tools"] [data-testid="stPopover"] button {
            min-height: 2.25rem !important;
            height: 2.25rem !important;
            width: 2.25rem !important;
            min-width: 2.25rem !important;
            padding: 0 !important;
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.055) !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
            font-size: 1.35rem !important;
            line-height: 1 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [class*="st-key-composer_tools"] [data-testid="stPopover"] button svg {
            display: none !important;
        }
        [class*="st-key-composer_tools"] [data-testid="stPopoverButton"] [aria-hidden="true"],
        [class*="st-key-composer_tools"] [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
            display: none !important;
        }
        [class*="st-key-composer_tools"] [data-testid="stPopoverButton"] > div {
            gap: 0 !important;
        }
        [class*="st-key-composer_tools"] [data-testid="stPopover"] button p {
            font-size: 1.35rem !important;
            line-height: 1 !important;
            margin: 0 !important;
            transform: translateY(-1px);
        }
        @media (max-width: 900px) {
            :root, .stApp, [data-testid="stAppViewContainer"] {
                --sidebar-width: 0px;
                --chat-width: calc(100vw - 2rem);
            }
            [class*="st-key-composer_tools"] {
                left: calc(var(--chat-left) + 0.75rem);
                bottom: 3.95rem;
            }
        }
        .metric-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.35rem;
        }
        .pill {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: #111827;
            color: #dbeafe;
            font-size: 0.78rem;
            border: 1px solid #1f2937;
        }
        .small-muted { color: #94a3b8; font-size: 0.86rem; }
        div[data-testid="stChatMessage"] {
            border-radius: 18px;
            background: #0b1220 !important;
        }
        div[data-testid="stExpander"] {
            border-radius: 14px;
            border-color: #1f2937 !important;
            background: #0b1220 !important;
        }
        [data-testid="stPopoverBody"] {
            min-width: 520px;
            background: #0b1220 !important;
            border: 1px solid #1f2937 !important;
            color: #e5e7eb !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            background-color: #111827 !important;
            color: #f8fafc !important;
            border-color: #1f2937 !important;
        }
        div[data-baseweb="select"] svg {
            color: #bfdbfe !important;
        }
        .stButton > button,
        [data-testid="stPopover"] button,
        [data-testid="stChatInput"] button {
            background: #111827 !important;
            color: #f8fafc !important;
            border: 1px solid #1f2937 !important;
            border-radius: 12px !important;
        }
        .stButton > button:hover,
        [data-testid="stPopover"] button:hover,
        [data-testid="stChatInput"] button:hover {
            border-color: #3b82f6 !important;
            color: #bfdbfe !important;
        }
        [data-testid="stChatInput"] {
            background: transparent !important;
        }
        [data-testid="stChatInput"] > div {
            background: #202123 !important;
            border: 1px solid #30343b !important;
            border-radius: 26px !important;
            box-shadow: 0 16px 45px rgba(0, 0, 0, 0.38) !important;
            min-height: 4.75rem !important;
            width: var(--chat-width) !important;
            max-width: var(--chat-width) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 0.75rem 1rem 0.75rem 1rem !important;
        }
        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] div[data-baseweb="base-input"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: #f8fafc !important;
            border: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
            min-height: 2.55rem !important;
            padding: 0.55rem 3.25rem 0.55rem 3.25rem !important;
            resize: none !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            border-radius: 999px !important;
            background: #3b82f6 !important;
            border-color: #3b82f6 !important;
            color: #ffffff !important;
            width: 2.25rem !important;
            height: 2.25rem !important;
        }
        [data-testid="stChatInputSubmitButton"]:disabled {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(255, 255, 255, 0.10) !important;
            color: #9ca3af !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background-color: #3b82f6 !important;
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.18) !important;
        }
        [data-testid="stSlider"] div[data-baseweb="slider"] div {
            color: #bfdbfe !important;
        }
        [data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {
            background-color: #3b82f6 !important;
        }
        [data-testid="stCheckbox"] svg,
        [data-testid="stRadio"] svg {
            color: #3b82f6 !important;
            fill: #3b82f6 !important;
        }
        [data-testid="stCheckbox"] label p,
        [data-testid="stRadio"] label p {
            background: transparent !important;
            color: #e5e7eb !important;
            box-shadow: none !important;
        }
        [data-testid="stCheckbox"] label p::selection,
        [data-testid="stRadio"] label p::selection {
            background: transparent !important;
            color: #e5e7eb !important;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
            background: transparent !important;
        }
        [data-testid="stCheckbox"] label > input + div,
        [data-testid="stRadio"] label > input + div {
            background: transparent !important;
        }
        [data-testid="stDataFrame"] {
            background: #0b1220 !important;
        }
        code, pre {
            background: #111827 !important;
            color: #dbeafe !important;
            border-color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = empty_memory().to_dict()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_example" not in st.session_state:
        st.session_state.pending_example = ""


def reset_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = empty_memory().to_dict()


def reset_memory_only() -> None:
    st.session_state.conversation_memory = empty_memory().to_dict()


def render_header(mode: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>Traffic Law RAG</h1>
          <div class="mode-chip">{MODE_LABELS[mode]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.title("Cấu hình")
        mode = st.radio(
            "Chế độ",
            options=["answer", "retriever", "ner"],
            format_func=lambda key: MODE_LABELS[key],
            horizontal=False,
        )

        st.divider()
        st.subheader("Retrieval")
        pipeline_key = st.selectbox(
            "Pipeline",
            options=list(PIPELINES.keys()),
            format_func=lambda key: PIPELINES[key].display_name,
            index=0,
        )

        top_k = st.slider("Top-k passages", min_value=1, max_value=20, value=5)
        candidate_k = st.slider("Candidate-k", min_value=50, max_value=1000, value=300, step=50)

        with st.expander("Context đưa vào LLM", expanded=False):
            max_context_passages = st.slider(
                "Số passage",
                min_value=1,
                max_value=12,
                value=min(5, top_k),
                key="max_context_passages",
            )
            max_chars_per_passage = st.slider(
                "Ký tự tối đa mỗi passage",
                min_value=600,
                max_value=3000,
                value=1800,
                step=100,
            )

        st.divider()
        st.subheader("Hội thoại")
        enable_memory = st.checkbox(
            "Ghi nhớ ngữ cảnh",
            value=True,
        )
        enable_query_router = st.checkbox(
            "Route/rewrite câu hỏi",
            value=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Chat mới", use_container_width=True):
                reset_chat()
                st.rerun()
        with col_b:
            if st.button("Xoá memory", use_container_width=True):
                reset_memory_only()
                st.rerun()

        with st.expander("Memory hiện tại", expanded=False):
            st.json(st.session_state.conversation_memory, expanded=False)

        return {
            "mode": mode,
            "pipeline_key": pipeline_key,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "max_context_passages": max_context_passages,
            "max_chars_per_passage": max_chars_per_passage,
            "enable_memory": enable_memory,
            "enable_query_router": enable_query_router,
        }


def settings_menu(label: str):
    if hasattr(st, "popover"):
        return st.popover(label)
    return st.expander(label, expanded=False)


def render_model_settings(mode: str) -> dict[str, Any]:
    label = "+" if mode in {"answer", "retriever"} else "+"

    if mode in {"answer", "retriever"}:
        with settings_menu(label):
            cols = st.columns([1.05, 1.35, 1.8])
            with cols[0]:
                backend = st.selectbox(
                    "Backend",
                    options=["openai", "openrouter", "local"],
                    format_func=lambda key: BACKEND_LABELS[key],
                    key="backend",
                )
            with cols[1]:
                preset_options = [*MODEL_PRESETS[backend].keys(), "Custom"]
                model_preset = st.selectbox(
                    "Preset",
                    options=preset_options,
                    key=f"model_preset_{backend}",
                )
            with cols[2]:
                if model_preset == "Custom":
                    model_name = st.text_input(
                        "Model",
                        value=DEFAULT_MODELS[backend],
                        key=f"model_name_{backend}",
                    )
                else:
                    model_name = MODEL_PRESETS[backend][model_preset]
                    st.text_input(
                        "Model",
                        value=model_name,
                        disabled=True,
                        key=f"selected_model_name_{backend}",
                    )

            cols2 = st.columns([1, 1])
            with cols2[0]:
                max_new_tokens = st.number_input(
                    "Max tokens",
                    min_value=128,
                    max_value=4096,
                    value=512,
                    step=64,
                )
            with cols2[1]:
                temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=0.0,
                    step=0.1,
                )

            with st.expander("API", expanded=False):
                api_key = st.text_input(
                    "API key",
                    value="",
                    type="password",
                    disabled=backend == "local",
                )
                base_url = st.text_input(
                    "Base URL",
                    value="",
                    disabled=backend == "local",
                )

        return {
            "backend": backend,
            "model_name": model_name.strip() or DEFAULT_MODELS[backend],
            "api_key": api_key.strip() or None,
            "base_url": base_url.strip() or None,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }

    if mode == "ner":
        with settings_menu(label):
            cols = st.columns([1, 1, 1])
            with cols[0]:
                ner_mode = st.selectbox(
                    "Mode",
                    options=["gazetteer", "model", "hybrid"],
                    format_func=lambda key: NER_MODE_LABELS[key],
                )
            with cols[1]:
                threshold = st.number_input(
                    "Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.70,
                    step=0.05,
                )
            with cols[2]:
                device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
        return {"ner_mode": ner_mode, "threshold": float(threshold), "device": device}

    return {}


def render_examples(mode: str) -> None:
    examples = {
        "answer": [
            "Ô tô vượt đèn đỏ bị phạt bao nhiêu?",
            "Nếu là xe máy thì sao?",
            "Hiện nay quy định đó còn áp dụng không?",
        ],
        "retriever": [
            "không đội mũ bảo hiểm bị phạt bao nhiêu",
            "điều khiển xe có nồng độ cồn",
            "thời gian lái xe liên tục tối đa",
        ],
        "ner": [
            "Người điều khiển xe mô tô không đội mũ bảo hiểm khi tham gia giao thông.",
            "Xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
            "Giấy phép lái xe phải còn thời hạn sử dụng.",
        ],
    }
    with st.expander("Gợi ý", expanded=False):
        cols = st.columns(len(examples[mode]))
        for col, text in zip(cols, examples[mode]):
            with col:
                if st.button(text, use_container_width=True):
                    st.session_state.pending_example = text
                    st.rerun()


@st.cache_resource(show_spinner=False)
def load_gazetteer_aliases() -> list[dict[str, Any]]:
    if str(GAZETTEER_BUILDING_DIR) not in sys.path:
        sys.path.insert(0, str(GAZETTEER_BUILDING_DIR))
    from gazetteer_building_core.matcher import load_aliases
    return load_aliases(GAZETTEER_ROOT)


def predict_gazetteer_entities(text: str) -> list[dict[str, Any]]:
    if str(GAZETTEER_BUILDING_DIR) not in sys.path:
        sys.path.insert(0, str(GAZETTEER_BUILDING_DIR))
    from gazetteer_building_core.matcher import find_matches
    return find_matches(text, load_gazetteer_aliases())


@st.cache_resource(show_spinner=False)
def load_gliner_model(device: str):
    if importlib.util.find_spec("gliner") is None:
        raise RuntimeError("Env hiện tại chưa có package 'gliner'. Cài bằng: conda run -n kltn pip install gliner")
    if not GLINER_MODEL_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy GLiNER model: {GLINER_MODEL_DIR}")
    from gliner import GLiNER
    return GLiNER.from_pretrained(str(GLINER_MODEL_DIR)).to(device)


def predict_gliner_entities(text: str, threshold: float, device: str) -> list[dict[str, Any]]:
    model = load_gliner_model(device)
    try:
        preds = model.predict_entities(text, LABELS, threshold=threshold)
    except Exception:
        preds = model.batch_predict_entities([text], LABELS, threshold=threshold)[0]

    entities = []
    for pred in preds:
        surface = pred.get("text")
        entities.append({
            "text": surface,
            "surface": surface,
            "canonical": surface,
            "label": pred.get("label"),
            "start": pred.get("start"),
            "end": pred.get("end"),
            "confidence": float(pred.get("score", 0.0)),
            "source": "gliner",
        })
    return entities


def merge_ner_entities(gazetteer_entities: list[dict[str, Any]], model_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen = set()
    for entity in [*gazetteer_entities, *model_entities]:
        key = (
            entity.get("start"),
            entity.get("end"),
            entity.get("label"),
            (entity.get("canonical") or entity.get("surface") or entity.get("text") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(entity)
    return sorted(merged, key=lambda e: (e.get("start") is None, e.get("start") or 0, e.get("end") or 0))


def run_ner_test(text: str, mode: str, threshold: float, device: str) -> dict[str, Any]:
    gazetteer_entities = predict_gazetteer_entities(text) if mode in {"gazetteer", "hybrid"} else []
    model_entities = predict_gliner_entities(text, threshold, device) if mode in {"model", "hybrid"} else []

    if mode == "gazetteer":
        entities = gazetteer_entities
    elif mode == "model":
        entities = model_entities
    elif mode == "hybrid":
        entities = merge_ner_entities(gazetteer_entities, model_entities)
    else:
        raise ValueError(f"Unknown NER mode: {mode}")

    return {
        "text": text,
        "mode": mode,
        "entities": entities,
        "gazetteer_entities": gazetteer_entities,
        "model_entities": model_entities,
    }


def render_retrieval_summary(result: dict[str, Any]) -> None:
    retrieval = result.get("retrieval") or result
    results = retrieval.get("results") or []
    activated = retrieval.get("activated_entities") or []

    st.markdown(
        f"""
        <div class="metric-row">
          <span class="pill">Passages: {len(results)}</span>
          <span class="pill">Activated entities: {len(activated)}</span>
          <span class="pill">Pipeline: {result.get('pipeline_display_name') or result.get('pipeline') or 'N/A'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if activated:
        with st.expander("Activated entities", expanded=False):
            st.dataframe(activated, use_container_width=True, hide_index=True)

    for idx, item in enumerate(results, start=1):
        doc = item.get("document_number") or item.get("document_id") or "Không rõ văn bản"
        title = item.get("document_title") or ""
        path_text = item.get("path_text") or item.get("passage_id") or ""
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"

        with st.expander(f"{idx}. {doc} | score={score_text}", expanded=idx <= 3):
            if title:
                st.markdown(f"**{title}**")
            st.caption(path_text)
            st.write(item.get("text") or "")
            components = item.get("score_components") or {}
            if components:
                st.json(components, expanded=False)


def render_ner_result(result: dict[str, Any]) -> None:
    entities = result.get("entities") or []
    st.markdown(
        f"""
        <div class="metric-row">
          <span class="pill">Mode: {NER_MODE_LABELS.get(result.get('mode'), result.get('mode'))}</span>
          <span class="pill">Entities: {len(entities)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not entities:
        st.info("Không phát hiện entity.")
        return

    rows = []
    for entity in entities:
        rows.append({
            "surface": entity.get("surface") or entity.get("text") or entity.get("canonical"),
            "canonical": entity.get("canonical") or entity.get("text") or entity.get("surface"),
            "label": entity.get("label"),
            "start": entity.get("start"),
            "end": entity.get("end"),
            "confidence": entity.get("confidence"),
            "source": entity.get("source"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Raw NER output", expanded=False):
        st.json(result, expanded=False)


def render_answer_result(result: dict[str, Any]) -> None:
    if result.get("route") == "general_chat":
        st.info("Câu hỏi được phân loại ngoài phạm vi RAG, hệ thống không truy xuất corpus.")

    with st.expander("Truy vấn và bộ nhớ", expanded=False):
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
            st.write("Query sau khi ghép memory:")
            st.code(result["expanded_query"], language="text")
        st.write("Memory sau lượt này:")
        st.json(result.get("conversation_memory") or {}, expanded=False)

    retrieval = result.get("retrieval") or {}
    if retrieval.get("results"):
        with st.expander("Passages được truy xuất", expanded=False):
            render_retrieval_summary(result)

    with st.expander("Context đưa vào LLM", expanded=False):
        st.text(result.get("context_used") or "")


def render_chat_history() -> None:
    for item in st.session_state.chat_history:
        role = item.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(item.get("content") or "")
            payload = item.get("payload")
            kind = item.get("kind")
            if role == "assistant" and payload:
                if kind == "answer":
                    render_answer_result(payload)
                elif kind == "retriever":
                    render_retrieval_summary(payload)
                elif kind == "ner":
                    render_ner_result(payload)


def handle_prompt(prompt: str, sidebar_settings: dict[str, Any], model_settings: dict[str, Any]) -> None:
    mode = sidebar_settings["mode"]
    st.session_state.chat_history.append({"role": "user", "content": prompt, "kind": mode})

    try:
        with st.spinner("Đang xử lý..."):
            if mode == "answer":
                result = run_demo_answer(
                    question=prompt,
                    pipeline_key=sidebar_settings["pipeline_key"],
                    mode=model_settings["backend"],
                    model_name=model_settings["model_name"],
                    api_key=model_settings["api_key"],
                    base_url=model_settings["base_url"],
                    top_k=sidebar_settings["top_k"],
                    candidate_k=sidebar_settings["candidate_k"],
                    max_context_passages=sidebar_settings["max_context_passages"],
                    max_chars_per_passage=sidebar_settings["max_chars_per_passage"],
                    enable_query_router=sidebar_settings["enable_query_router"],
                    max_new_tokens=model_settings["max_new_tokens"],
                    temperature=model_settings["temperature"],
                    conversation_memory=(st.session_state.conversation_memory if sidebar_settings["enable_memory"] else None),
                )
                if sidebar_settings["enable_memory"]:
                    st.session_state.conversation_memory = result.get("conversation_memory") or st.session_state.conversation_memory
                assistant_content = result.get("answer") or "Không có câu trả lời."
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content, "kind": "answer", "payload": result})

            elif mode == "retriever":
                result = run_demo_retrieval(
                    question=prompt,
                    pipeline_key=sidebar_settings["pipeline_key"],
                    mode=model_settings["backend"],
                    model_name=model_settings["model_name"],
                    api_key=model_settings["api_key"],
                    base_url=model_settings["base_url"],
                    top_k=sidebar_settings["top_k"],
                    candidate_k=sidebar_settings["candidate_k"],
                    enable_query_router=sidebar_settings["enable_query_router"],
                    conversation_memory=(st.session_state.conversation_memory if sidebar_settings["enable_memory"] else None),
                )
                if sidebar_settings["enable_memory"]:
                    st.session_state.conversation_memory = result.get("conversation_memory") or st.session_state.conversation_memory
                count = len((result.get("retrieval") or result).get("results") or [])
                assistant_content = f"Đã truy xuất được **{count} passage**."
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content, "kind": "retriever", "payload": result})

            elif mode == "ner":
                result = run_ner_test(
                    text=prompt,
                    mode=model_settings["ner_mode"],
                    threshold=model_settings["threshold"],
                    device=model_settings["device"],
                )
                count = len(result.get("entities") or [])
                assistant_content = f"Phát hiện được **{count} entity**."
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content, "kind": "ner", "payload": result})

    except Exception as exc:
        st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Lỗi: `{exc}`", "kind": mode, "payload": None})

    st.rerun()


def main() -> None:
    inject_css()
    init_state()

    sidebar_settings = render_sidebar()
    mode = sidebar_settings["mode"]

    render_header(mode)
    render_examples(mode)
    render_chat_history()

    with st.container(key="composer_tools"):
        st.markdown('<div class="composer-tools">', unsafe_allow_html=True)
        model_settings = render_model_settings(mode)
        st.markdown("</div>", unsafe_allow_html=True)

    default_prompt = st.session_state.pending_example
    if default_prompt:
        st.session_state.pending_example = ""

    placeholder = {
        "answer": "Nhập câu hỏi pháp luật giao thông...",
        "retriever": "Nhập truy vấn để test retriever...",
        "ner": "Nhập một câu/đoạn ngắn để test NER...",
    }[mode]

    prompt = st.chat_input(placeholder)
    if prompt:
        handle_prompt(prompt, sidebar_settings, model_settings)
    elif default_prompt:
        handle_prompt(default_prompt, sidebar_settings, model_settings)


if __name__ == "__main__":
    main()
