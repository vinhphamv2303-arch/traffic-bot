from __future__ import annotations

import importlib.util
import sys
import time
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
from interface.translations import t  # noqa: E402

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


def get_lang() -> str:
    return st.session_state.get("lang", "vi")


def mode_label(key: str) -> str:
    return {"answer": t("mode_answer", get_lang()),
            "retriever": t("mode_retriever", get_lang()),
            "ner": t("mode_ner", get_lang())}.get(key, key)


import os
icon_path = os.path.join(os.path.dirname(__file__), "icon.png")

st.set_page_config(
    page_title="Traffic Bot",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root, .stApp, [data-testid="stAppViewContainer"] {
            --primary: #1841AF; --primary-glow: rgba(24,65,175,0.25);
            --accent: #22d3ee; --accent-glow: rgba(34,211,238,0.18);
            --bg: #000000; --bg-card: #12121a; --bg-elevated: #1a1a2e;
            --border: rgba(255,255,255,0.07); --border-hover: rgba(255,255,255,0.15);
            --text: #e4e4ed; --text-muted: #8b8ba3; --text-bright: #f8f8ff;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], main {
            background: var(--bg) !important; color: var(--text) !important;
        }
        .block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 920px; }
        h1,h2,h3,h4,h5,h6,p,label,span,div,small,[data-testid="stMarkdownContainer"] { color: var(--text); }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] { background: var(--bg) !important; border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] { background: var(--bg) !important; }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stCheckbox label, [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stTextInput label, [data-testid="stSidebar"] .stNumberInput label {
            color: var(--text-muted) !important; font-weight: 500; font-size: 0.82rem;
        }
        [data-testid="stSidebar"] hr { border-color: var(--border) !important; }
        .sidebar-section-title {
            font-size: 0.75rem; font-weight: 600;
            color: var(--text-muted) !important; margin: 1rem 0 0.4rem 0; padding: 0;
            display: flex; align-items: center; gap: 0.4rem;
        }
        .element-container:has(.sidebar-spacer) {
            flex-grow: 1 !important;
        }
        .sidebar-logo {
            display: flex; align-items: center; gap: 0.6rem; padding: 0.3rem 0 0.8rem 0; border-bottom: 1px solid var(--border); margin-bottom: 0.6rem;
        }
        .sidebar-logo-icon {
            width: 32px; height: 32px; border-radius: 8px;
            background: transparent;
            border: 1.5px solid var(--text-muted);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.85rem; flex-shrink: 0; color: var(--text-muted) !important;
        }
        .sidebar-logo-text { font-size: 1rem; font-weight: 700; color: var(--text-bright) !important; line-height: 1.2; }
        .sidebar-logo-sub { font-size: 0.7rem; color: var(--text-muted) !important; }

        /* ── Hero Header ── */
        .hero {
            border: 1px solid rgba(24, 65, 175, 0.4);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            background: linear-gradient(135deg, rgba(24, 65, 175, 0.15) 0%, rgba(10, 10, 15, 0.9) 100%);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            margin-bottom: 1rem;
            display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;
        }
        .hero-left { display: flex; align-items: center; gap: 0.8rem; }
        .hero h1 {
            font-size: 1.4rem; line-height: 1.2;
            margin: 0; font-weight: 800;
            background: linear-gradient(90deg, #ffffff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #22c55e; box-shadow: 0 0 12px rgba(34,197,94,0.8);
            flex-shrink: 0;
            animation: pulse-dot 2s ease-in-out infinite;
        }
        @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }
        .hero-subtitle { font-size: 0.8rem; color: #a5b4fc; margin: 0.2rem 0 0 0; font-weight: 500; letter-spacing: 0.3px; }
        .mode-chip {
            display: inline-flex; align-items: center;
            padding: 0.25rem 0.65rem; border-radius: 6px;
            background: var(--primary-glow); border: 1px solid rgba(99,102,241,0.3);
            color: #c7d2fe !important; font-size: 0.75rem; font-weight: 500;
        }

        /* ── Suggestion Chips ── */
        [class*="st-key-suggestion_"] button {
            background: var(--bg-card) !important; color: var(--text) !important;
            border: 1px solid var(--border) !important; border-radius: 14px !important;
            padding: 0.45rem 0.85rem !important; font-size: 0.8rem !important;
            transition: all 0.25s ease !important;
            text-align: left !important;
            line-height: 1.35 !important; min-height: auto !important;
            display: block !important; max-width: 100% !important;
            overflow: hidden !important;
        }
        [class*="st-key-suggestion_"] button div, [class*="st-key-suggestion_"] button p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            width: 100% !important;
            margin: 0 !important;
        }
        [class*="st-key-suggestion_"] button:hover {
            border-color: var(--primary) !important; background: var(--primary-glow) !important;
            color: #c7d2fe !important; transform: translateY(-1px) !important;
            position: relative; z-index: 10;
        }
        [class*="st-key-suggestion_"] button:hover div, [class*="st-key-suggestion_"] button:hover p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        /* ── Chat Messages ── */
        div[data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 0 !important; margin-bottom: 1.5rem !important;
            border: none !important;
        }
        
        /* User message bubble (Right aligned) */
        div[data-testid="stChatMessage"]:has(.is-user) > div:first-child {
            display: none !important;
        }
        div[data-testid="stChatMessage"]:has(.is-user) > div:last-child {
            background-color: var(--primary) !important; color: #ffffff !important;
            border-radius: 18px 18px 0 18px !important;
            padding: 0.6rem 1rem !important; /* Tighter padding */
            width: fit-content !important; max-width: 85% !important;
            margin-left: auto !important; margin-right: 0 !important;
            flex-grow: 0 !important;
        }
        div[data-testid="stChatMessage"]:has(.is-user) > div:last-child p,
        div[data-testid="stChatMessage"]:has(.is-user) > div:last-child div {
            margin-top: 0 !important; margin-bottom: 0 !important;
        }
        
        /* Bot message (Full width, transparent) */
        div[data-testid="stChatMessage"]:has(.is-bot) > div:first-child {
            display: none !important;
        }
        div[data-testid="stChatMessage"]:has(.is-bot) > div:last-child {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            width: 100% !important; max-width: 100% !important;
            margin: 0 !important; flex-grow: 1 !important;
        }

        /* Suy luận (Expanders/Status) inside Bot Message -> Left aligned, borderless */
        div[data-testid="stChatMessage"]:has(.is-bot) .stExpander,
        div[data-testid="stChatMessage"]:has(.is-bot) .stStatusWidget,
        div[data-testid="stChatMessage"]:has(.is-bot) [data-testid="stExpander"],
        div[data-testid="stChatMessage"]:has(.is-bot) [data-testid="stStatusWidget"],
        div[data-testid="stChatMessage"]:has(.is-bot) details {
            border: none !important;
            border-color: transparent !important;
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            margin-left: 0 !important; margin-right: auto !important;
            width: fit-content !important; min-width: 60% !important; max-width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stChatMessage"]:has(.is-bot) [data-testid="stExpander"] > summary,
        div[data-testid="stChatMessage"]:has(.is-bot) [data-testid="stStatusWidget"] > summary,
        div[data-testid="stChatMessage"]:has(.is-bot) details > summary {
            background: transparent !important;
            background-color: transparent !important;
            color: var(--text-muted) !important;
            border: none !important;
            padding-left: 0 !important;
        }

        /* ── Metric pills ── */
        .metric-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.3rem; }
        .pill {
            display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
            background: var(--bg-elevated); color: #c7d2fe; font-size: 0.75rem;
            border: 1px solid var(--border);
        }
        .small-muted { color: var(--text-muted); font-size: 0.84rem; }

        /* ── Expanders ── */
        div[data-testid="stExpander"] {
            border-radius: 12px; border-color: var(--border) !important;
            background: var(--bg-card) !important;
            transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
        }
        div[data-testid="stExpander"]:hover {
            border-color: var(--primary) !important;
            box-shadow: 0 4px 16px var(--primary-glow) !important;
            transform: translateY(-2px) !important;
        }

        /* ── Inputs & Selects ── */
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea, input {
            background-color: var(--bg-elevated) !important; color: var(--text-bright) !important;
            border-color: var(--border) !important; border-radius: 10px !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease !important;
        }
        div[data-baseweb="select"]:hover > div, div[data-baseweb="input"]:hover > div,
        div[data-testid="stSelectbox"]:hover div[data-baseweb="select"] > div {
            border-color: var(--primary) !important;
            background-color: var(--bg-card) !important;
            box-shadow: 0 0 0 1px var(--primary-glow), 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        div[data-baseweb="select"]:focus-within > div, div[data-baseweb="input"]:focus-within > div,
        div[data-testid="stSelectbox"]:focus-within div[data-baseweb="select"] > div {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px var(--primary-glow), 0 4px 12px rgba(0,0,0,0.2) !important;
        }
        div[data-baseweb="select"] svg { color: var(--text-muted) !important; transition: transform 0.2s ease, color 0.2s ease !important; }
        div[data-baseweb="select"]:hover svg, div[data-testid="stSelectbox"]:hover div[data-baseweb="select"] svg {
            color: var(--primary) !important; transform: scale(1.1);
        }

        /* ── Buttons ── */
        .stButton > button {
            background: var(--bg-elevated) !important; color: var(--text) !important;
            border: 1px solid var(--border) !important; border-radius: 10px !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            border-color: var(--primary) !important; color: #c7d2fe !important;
            background: var(--primary-glow) !important; transform: translateY(-1px) !important;
        }

        /* ── Chat Input ── */
        [data-testid="stChatInput"] { background: transparent !important; }
        [data-testid="stChatInput"] > div {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important; border-radius: 24px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
            margin: 0 auto !important;
            width: 100% !important;
            max-width: 340px !important;
            transition: max-width 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.3s ease, box-shadow 0.3s ease !important;
        }
        [data-testid="stChatInput"] > div:hover {
            border-color: var(--primary) !important;
        }
        [data-testid="stChatInput"] > div:focus-within {
            max-width: 600px !important;
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-glow), 0 8px 32px rgba(0,0,0,0.5) !important;
        }
        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] div[data-baseweb="base-input"] {
            background: transparent !important; border: 0 !important; box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea {
            background: transparent !important; color: var(--text-bright) !important;
            border: 0 !important; box-shadow: none !important; outline: 0 !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            border-radius: 999px !important;
            background: var(--primary) !important;
            border: none !important; color: #fff !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stChatInputSubmitButton"]:hover {
            background: #818cf8 !important;
        }
        [data-testid="stChatInputSubmitButton"]:disabled {
            background: rgba(255,255,255,0.06) !important; color: #6b7280 !important;
        }

        /* ── Sliders ── */
        [data-testid="stSlider"] [role="slider"] {
            background-color: var(--primary) !important; border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-glow) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        [data-testid="stSlider"] [role="slider"]:hover {
            transform: scale(1.15) !important;
            box-shadow: 0 0 0 6px var(--primary-glow) !important;
        }
        [data-testid="stSlider"] div[data-baseweb="slider"] div { color: var(--text) !important; }
        [data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {
            background-color: var(--primary) !important;
        }

        /* ── Checkboxes & Radios ── */
        [data-testid="stCheckbox"] svg, [data-testid="stRadio"] svg { color: var(--primary) !important; fill: var(--primary) !important; }
        [data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p {
            background: transparent !important; color: var(--text) !important; box-shadow: none !important;
            transition: color 0.2s ease !important;
        }
        [data-testid="stCheckbox"]:hover label p, [data-testid="stRadio"]:hover label p {
            color: var(--text-bright) !important;
        }
        [data-testid="stCheckbox"] label, [data-testid="stRadio"] label { background: transparent !important; }

        [data-testid="stDataFrame"] { background: var(--bg-card) !important; }
        code, pre { background: var(--bg-elevated) !important; color: var(--text) !important; border-color: var(--border) !important; }

        /* ── Hide old composer_tools ── */
        [class*="st-key-composer_tools"] { display: none !important; }

        /* ── Hide Streamlit material icon text leaks ── */
        [data-testid="stSidebar"] [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] {
            font-size: 0 !important;
        }
        [data-testid="collapsedControl"] svg { font-size: 1.5rem !important; }
        button[kind="header"] span[data-testid="stIconMaterial"] {
            font-size: 0 !important; overflow: hidden !important;
        }
        /* Hide any stray material icon text rendering */
        .st-emotion-cache-1rtdyuf, .st-emotion-cache-eczf16 {
            font-size: 0 !important;
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
    lang = get_lang()
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-left">
            <div class="hero-dot"></div>
            <div>
              <h1>{t("app_title", lang)}</h1>
              <p class="hero-subtitle">{t("app_subtitle", lang)}</p>
            </div>
          </div>
          <div class="mode-chip">{mode_label(mode)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        # ── Language State (Read first for logo) ──
        lang_options = {"VI  Tiếng Việt": "vi", "EN  English": "en"}
        if "lang_selector" in st.session_state:
            st.session_state["lang"] = lang_options[st.session_state["lang_selector"]]
        lang = get_lang()
        lang_display = [k for k, v in lang_options.items() if v == lang][0]

        # ── Logo ──
        st.markdown(
            f"""
            <div class="sidebar-logo">
              <div>
                <div class="sidebar-logo-text">{t("app_title", lang)}</div>
                <div class="sidebar-logo-sub">{t("app_subtitle", lang)}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Mode ──
        _icon_mode = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
        st.markdown(f'<p class="sidebar-section-title">{_icon_mode} {t("mode_label", lang)}</p>', unsafe_allow_html=True)
        mode = st.radio(
            t("mode_label", lang),
            options=["answer", "retriever", "ner"],
            format_func=mode_label,
            horizontal=False,
            label_visibility="collapsed",
        )

        st.divider()
        # ── Retrieval ──
        _icon_search = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
        st.markdown(f'<p class="sidebar-section-title">{_icon_search} {t("retrieval_section", lang)}</p>', unsafe_allow_html=True)
        pipeline_key = st.selectbox(
            t("pipeline_label", lang),
            options=list(PIPELINES.keys()),
            format_func=lambda key: PIPELINES[key].display_name,
            index=0,
        )
        top_k = st.slider(t("top_k_label", lang), min_value=1, max_value=20, value=5)
        candidate_k = st.slider(t("candidate_k_label", lang), min_value=50, max_value=1000, value=300, step=50)

        with st.expander(t("context_llm_section", lang), expanded=False):
            max_context_passages = st.slider(
                t("max_passages_label", lang),
                min_value=1, max_value=12, value=min(5, top_k),
                key="max_context_passages",
            )
            max_chars_per_passage = st.slider(
                t("max_chars_label", lang),
                min_value=600, max_value=3000, value=1800, step=100,
            )

        st.divider()
        # ── Conversation ──
        _icon_chat = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
        st.markdown(f'<p class="sidebar-section-title">{_icon_chat} {t("conv_section", lang)}</p>', unsafe_allow_html=True)
        enable_memory = st.checkbox(t("memory_checkbox", lang), value=True)
        enable_query_router = st.checkbox(t("route_checkbox", lang), value=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(t("new_chat_btn", lang), use_container_width=True):
                reset_chat()
                st.rerun()
        with col_b:
            if st.button(t("clear_memory_btn", lang), use_container_width=True):
                reset_memory_only()
                st.rerun()

        with st.expander(t("current_memory", lang), expanded=False):
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
    return st.expander(label, expanded=False)


def render_model_settings(mode: str) -> dict[str, Any]:
    lang = get_lang()

    with st.sidebar:
        st.divider()
        _icon_gear = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
        st.markdown(f'<p class="sidebar-section-title">{_icon_gear} {t("model_section", lang)}</p>', unsafe_allow_html=True)

        if mode in {"answer", "retriever"}:
            backend = st.selectbox(
                t("backend_label", lang),
                options=["openai", "openrouter", "local"],
                format_func=lambda key: BACKEND_LABELS[key],
                key="backend",
            )
            preset_options = [*MODEL_PRESETS[backend].keys(), "Custom"]
            model_preset = st.selectbox(
                t("preset_label", lang),
                options=preset_options,
                key=f"model_preset_{backend}",
            )
            if model_preset == "Custom":
                model_name = st.text_input(
                    t("model_label", lang),
                    value=DEFAULT_MODELS[backend],
                    key=f"model_name_{backend}",
                )
            else:
                model_name = MODEL_PRESETS[backend][model_preset]
                st.text_input(
                    t("model_label", lang),
                    value=model_name,
                    disabled=True,
                    key=f"selected_model_name_{backend}_{model_preset}",
                )

            cols2 = st.columns(2)
            with cols2[0]:
                max_new_tokens = st.number_input(
                    t("max_tokens_label", lang),
                    min_value=128, max_value=4096, value=512, step=64,
                )
            with cols2[1]:
                temperature = st.number_input(
                    t("temperature_label", lang),
                    min_value=0.0, max_value=1.5, value=0.0, step=0.1,
                )

            with st.expander(t("api_section", lang), expanded=False):
                api_key = st.text_input(
                    t("api_key_label", lang), value="", type="password",
                    disabled=backend == "local",
                )
                base_url = st.text_input(
                    t("base_url_label", lang), value="",
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
            ner_mode = st.selectbox(
                t("ner_mode_label", lang),
                options=["gazetteer", "model", "hybrid"],
                format_func=lambda key: NER_MODE_LABELS[key],
            )
            threshold = st.number_input(
                t("threshold_label", lang),
                min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            )
            device = st.selectbox(t("device_label", lang), options=["cpu", "cuda"], index=0)
            return {"ner_mode": ner_mode, "threshold": float(threshold), "device": device}

    return {}


def render_sidebar_footer() -> None:
    lang = get_lang()
    lang_options = {"VI  Tiếng Việt": "vi", "EN  English": "en"}
    lang_display = [k for k, v in lang_options.items() if v == lang][0]
    
    with st.sidebar:
        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        st.divider()
        # ── Language Settings ──
        _icon_lang = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
        st.markdown(f'<p class="sidebar-section-title">{_icon_lang} {t("lang_label", lang)}</p>', unsafe_allow_html=True)
        st.selectbox(
            t("lang_label", lang),
            options=list(lang_options.keys()),
            index=list(lang_options.keys()).index(lang_display),
            key="lang_selector",
            label_visibility="collapsed",
        )


def render_examples(mode: str) -> None:
    examples = {
        "answer": [
            "Ô tô vượt đèn đỏ bị phạt bao nhiêu?",
            "Nếu là xe máy thì sao?",
            "Hiện nay quy định đó còn áp dụng không?",
            "Xe tải nhỏ dưới 2 tấn ở Hà Nội có được chạy thoải mái trong giờ cao điểm không?"
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
    # Only show suggestions when chat is empty (like Gemini)
    if st.session_state.chat_history or st.session_state.pending_example:
        return
    lang = get_lang()
    st.markdown(f'<p class="small-muted">{t("suggestions_label", lang)}</p>', unsafe_allow_html=True)
    cols = st.columns(len(examples[mode]))
    for idx, (col, text) in enumerate(zip(cols, examples[mode])):
        with col:
            if st.button(text, use_container_width=True, key=f"suggestion_{mode}_{idx}"):
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
    lang = get_lang()
    retrieval = result.get("retrieval") or result
    results = retrieval.get("results") or []
    activated = retrieval.get("activated_entities") or []

    st.markdown(
        f"""
        <div class="metric-row">
          <span class="pill">{t("passages_count", lang)}: {len(results)}</span>
          <span class="pill">{t("activated_entities", lang)}: {len(activated)}</span>
          <span class="pill">{t("pipeline_info", lang)}: {result.get('pipeline_display_name') or result.get('pipeline') or 'N/A'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if activated:
        with st.expander(t("activated_entities", lang), expanded=False):
            st.dataframe(activated, use_container_width=True, hide_index=True)

    for idx, item in enumerate(results, start=1):
        doc = item.get("document_number") or item.get("document_id") or t("unknown_doc", lang)
        title = item.get("document_title") or ""
        path_text = item.get("path_text") or item.get("passage_id") or ""
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"

        with st.expander(f"{idx}. {doc} | {t('score_label', lang)}={score_text}", expanded=idx <= 3):
            if title:
                st.markdown(f"**{title}**")
            st.caption(path_text)
            st.write(item.get("text") or "")
            components = item.get("score_components") or {}
            if components:
                st.json(components, expanded=False)


def render_ner_result(result: dict[str, Any]) -> None:
    lang = get_lang()
    entities = result.get("entities") or []
    st.markdown(
        f"""
        <div class="metric-row">
          <span class="pill">{t("ner_mode_label", lang)}: {NER_MODE_LABELS.get(result.get('mode'), result.get('mode'))}</span>
          <span class="pill">{t("entities_count", lang)}: {len(entities)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not entities:
        st.info(t("no_entity", lang))
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

    with st.expander(t("raw_ner_output", lang), expanded=False):
        st.json(result, expanded=False)


def render_answer_result(result: dict[str, Any]) -> None:
    lang = get_lang()
    with st.container():
        if result.get("route") == "general_chat":
            st.info(t("general_chat_info", lang))

        with st.expander(t("query_memory_expander", lang), expanded=False):
            st.write(f"{t('route_label', lang)}: `{result.get('route')}`")
            if result.get("route_reason"):
                st.caption(result["route_reason"])
            if result.get("rewritten_query"):
                st.write(f"{t('rewrite_label', lang)}:")
                st.code(result["rewritten_query"], language="text")
            if result.get("memory_context"):
                st.write(f"{t('memory_context_label', lang)}:")
                st.code(result["memory_context"], language="text")
            if result.get("expanded_query"):
                st.write(f"{t('expanded_query_label', lang)}:")
                st.code(result["expanded_query"], language="text")
            if result.get("conversation_resolution"):
                resolution = result["conversation_resolution"]
                st.write(f"{t('resolver_label', lang)}:")
                st.json(
                    {
                        "accepted": resolution.get("accepted"),
                        "used_llm": resolution.get("used_llm"),
                        "relation": resolution.get("relation"),
                        "route": resolution.get("route"),
                        "confidence": resolution.get("confidence"),
                        "reason": resolution.get("reason"),
                        "changed_constraints": resolution.get("changed_constraints"),
                        "dropped_answered_content": resolution.get("dropped_answered_content"),
                    },
                    expanded=False,
                )
            st.write(f"{t('memory_after_label', lang)}:")
            st.json(result.get("conversation_memory") or {}, expanded=False)

        retrieval = result.get("retrieval") or {}
        if retrieval.get("results"):
            with st.expander(t("retrieved_passages", lang), expanded=False):
                render_retrieval_summary(result)

        with st.expander(t("context_expander", lang), expanded=False):
            st.text(result.get("context_used") or "")


def render_chat_history() -> None:
    for item in st.session_state.chat_history:
        role = item.get("role", "assistant")
        with st.chat_message(role):
            flag = '<div class="is-user" style="display:none;">u</div>' if role == "user" else '<div class="is-bot" style="display:none;">b</div>'
            st.markdown(flag + (item.get("content") or ""), unsafe_allow_html=True)

            payload = item.get("payload")
            kind = item.get("kind")
            if role == "assistant" and payload:
                if kind == "answer":
                    render_answer_result(payload)
                elif kind == "retriever":
                    render_retrieval_summary(payload)
                elif kind == "ner":
                    render_ner_result(payload)


def _typewriter(text: str, delay: float = 0.018):
    """Yield text word-by-word for st.write_stream typewriter effect."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def _short_text(value: Any, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _progress_step_definitions(mode: str, lang: str) -> list[tuple[str, str]]:
    if mode == "answer":
        return [
            ("resolver", t("step_analyzing", lang)),
            ("route", "Xac dinh luong xu ly" if lang == "vi" else "Determine route"),
            ("retrieval", t("step_retrieving", lang)),
            ("context", "Chuan bi ngu canh" if lang == "vi" else "Prepare context"),
            ("generation", t("step_generating", lang)),
        ]
    if mode == "retriever":
        return [
            ("resolver", t("step_analyzing", lang)),
            ("route", "Xac dinh luong xu ly" if lang == "vi" else "Determine route"),
            ("retrieval", t("step_retrieving", lang)),
        ]
    return []


def _route_badge_text(route: str, lang: str) -> str:
    mapping = {
        "traffic_law": "Tra cuu phap luat giao thong" if lang == "vi" else "Traffic law RAG",
        "general_chat": "Hoi dap thong thuong" if lang == "vi" else "General chat",
        "effectivity_index": "Tra cuu hieu luc" if lang == "vi" else "Effectivity lookup",
    }
    return mapping.get(route or "", route or "unknown")


def _create_progress_tracker(status, mode: str, lang: str, pipeline_name: str = "", model_name: str = ""):
    step_defs = _progress_step_definitions(mode, lang)
    step_order = [key for key, _ in step_defs]
    step_labels = dict(step_defs)
    step_state = {key: {"state": "pending", "detail": ""} for key in step_order}
    meta_placeholder = st.empty()
    steps_placeholder = st.empty()
    note_placeholder = st.empty()
    started_at = time.perf_counter()
    current_note = ""

    def render() -> None:
        elapsed = time.perf_counter() - started_at
        meta = []
        if pipeline_name:
            meta.append(f"Pipeline: **{pipeline_name}**")
        if model_name and mode in {"answer", "retriever"}:
            meta.append(f"Model: **{model_name}**")
        meta.append(f"Elapsed: **{elapsed:.1f}s**")
        meta_placeholder.markdown("  \n".join(meta))

        icon_map = {
            "pending": "WAIT",
            "running": "RUN",
            "done": "DONE",
            "skipped": "SKIP",
            "error": "ERR",
        }
        lines = []
        for key in step_order:
            state = step_state[key]["state"]
            detail = step_state[key]["detail"]
            line = f"- `{icon_map.get(state, 'WAIT')}` {step_labels[key]}"
            if detail:
                line += f"  \n  {_short_text(detail, 220)}"
            lines.append(line)
        steps_placeholder.markdown("\n".join(lines))

        if current_note:
            note_placeholder.caption(current_note)
        else:
            note_placeholder.empty()

    def set_step(step_key: str, state: str | None = None, detail: str | None = None) -> None:
        if step_key not in step_state:
            return
        if state is not None:
            step_state[step_key]["state"] = state
        if detail is not None:
            step_state[step_key]["detail"] = detail
        render()

    def mark_skipped(step_key: str, detail: str) -> None:
        if step_key in step_state and step_state[step_key]["state"] == "pending":
            set_step(step_key, "skipped", detail)

    def running_step() -> str | None:
        for key in step_order:
            if step_state[key]["state"] == "running":
                return key
        return None

    def callback(stage: str, payload: dict[str, Any]) -> None:
        nonlocal current_note

        if stage == "loading_local_model":
            current_note = f"Loading local model: {payload.get('model_name')}"
            render()
            return

        if stage == "resolver_started":
            status.update(label=t("step_analyzing", lang), state="running", expanded=True)
            current_note = "Dang phan tich cau hoi..." if lang == "vi" else "Analyzing query..."
            set_step("resolver", "running", "")
            return

        if stage == "resolver_done":
            detail = payload.get("expanded_query") or payload.get("processing_query") or ""
            if payload.get("used_memory"):
                detail = f"{_short_text(detail)} | memory"
            else:
                detail = _short_text(detail)
            set_step("resolver", "done", detail)
            current_note = ""
            return

        if stage == "route_decided":
            status.update(
                label="Xac dinh luong xu ly..." if lang == "vi" else "Determining route...",
                state="running",
                expanded=True,
            )
            route_text = _route_badge_text(str(payload.get("route") or ""), lang)
            reason = _short_text(payload.get("reason") or "", 120)
            detail = route_text if not reason else f"{route_text} | {reason}"
            set_step("route", "done", detail)
            return

        if stage == "effectivity_fast_path":
            status.update(
                label="Dang tra cuu hieu luc..." if lang == "vi" else "Checking structured effectivity...",
                state="running",
                expanded=True,
            )
            mark_skipped("retrieval", "Tra cuu truc tiep tu metadata" if lang == "vi" else "Used structured metadata")
            mark_skipped("context", "Khong can truy xuat" if lang == "vi" else "No retrieval context needed")
            mark_skipped("generation", "Khong can goi LLM" if lang == "vi" else "No LLM generation needed")
            current_note = "Da dung fast-path hieu luc." if lang == "vi" else "Used structured effectivity fast-path."
            return

        if stage == "retrieval_started":
            status.update(label=t("step_retrieving", lang), state="running", expanded=True)
            detail = payload.get("retrieval_query") or ""
            set_step("retrieval", "running", _short_text(detail))
            current_note = (
                f"candidate_k={payload.get('candidate_k')} | top_k={payload.get('requested_top_k')}"
                if payload.get("candidate_k")
                else ""
            )
            return

        if stage == "retrieval_raw_done":
            current_note = (
                f"{payload.get('raw_result_count', 0)} raw passages | "
                f"{payload.get('activated_entity_count', 0)} activated entities"
            )
            render()
            return

        if stage == "retrieval_done":
            detail = (
                f"{payload.get('result_count', 0)} passages | "
                f"{payload.get('activated_entity_count', 0)} activated entities"
            )
            set_step("retrieval", "done", detail)
            current_note = ""
            return

        if stage == "retrieval_skipped":
            mark_skipped("retrieval", "Bo qua retrieval" if lang == "vi" else "Retrieval skipped")
            return

        if stage == "context_ready":
            status.update(
                label="Dang chuan bi ngu canh..." if lang == "vi" else "Preparing context...",
                state="running",
                expanded=True,
            )
            detail = (
                f"{payload.get('context_passages', 0)} passages | "
                f"{payload.get('context_chars', 0)} chars"
            )
            set_step("context", "done", detail)
            return

        if stage == "context_skipped":
            mark_skipped("context", "Khong co ngu canh du ro" if lang == "vi" else "Context unavailable")
            return

        if stage == "generation_started":
            status.update(label=t("step_generating", lang), state="running", expanded=True)
            detail = payload.get("answer_mode") or payload.get("model_name") or ""
            set_step("generation", "running", _short_text(detail))
            current_note = "Dang doi mo hinh phan hoi..." if lang == "vi" else "Waiting for model response..."
            return

        if stage == "generation_done":
            set_step("generation", "done", f"{payload.get('answer_chars', 0)} chars")
            current_note = ""
            return

        if stage == "generation_skipped":
            mark_skipped("generation", "Khong goi LLM do thieu ngu canh" if lang == "vi" else "Skipped due to missing context")
            return

        if stage == "completed":
            current_note = ""
            render()
            status.update(label=t("step_done", lang), state="complete", expanded=False)

    def fail(exc: Exception) -> None:
        nonlocal current_note
        key = running_step()
        if key:
            set_step(key, "error", str(exc))
        current_note = str(exc)
        render()
        status.update(label=f"{t('error_prefix', lang)}", state="error", expanded=True)

    render()
    return callback, fail


def handle_prompt(prompt: str, sidebar_settings: dict[str, Any], model_settings: dict[str, Any]) -> None:
    mode = sidebar_settings["mode"]
    lang = get_lang()
    fail_progress = None

    # ── 1. Show user message immediately ──
    st.session_state.chat_history.append({"role": "user", "content": prompt, "kind": mode})
    with st.chat_message("user"):
        st.markdown(f'<div class="is-user" style="display:none;">u</div>{prompt}', unsafe_allow_html=True)

    # ── 2. Process with visible status + stream answer ──
    with st.chat_message("assistant"):
        st.markdown('<div class="is-bot" style="display:none;">b</div>', unsafe_allow_html=True)
        try:
            if mode == "answer":
                with st.status(t("step_analyzing", lang), expanded=True) as status:
                    progress_callback, fail_progress = _create_progress_tracker(
                        status=status,
                        mode="answer",
                        lang=lang,
                        pipeline_name=PIPELINES[sidebar_settings["pipeline_key"]].display_name,
                        model_name=model_settings["model_name"],
                    )

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
                        progress_callback=progress_callback,
                    )

                if sidebar_settings["enable_memory"]:
                    st.session_state.conversation_memory = result.get("conversation_memory") or st.session_state.conversation_memory

                assistant_content = result.get("answer") or t("no_answer", lang)
                st.write_stream(_typewriter(assistant_content))
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content, "kind": "answer", "payload": result})
                st.rerun()

            elif mode == "retriever":
                with st.status(t("step_retrieving", lang), expanded=True) as status:
                    progress_callback, fail_progress = _create_progress_tracker(
                        status=status,
                        mode="retriever",
                        lang=lang,
                        pipeline_name=PIPELINES[sidebar_settings["pipeline_key"]].display_name,
                        model_name=model_settings["model_name"],
                    )

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
                        progress_callback=progress_callback,
                    )

                if sidebar_settings["enable_memory"]:
                    st.session_state.conversation_memory = result.get("conversation_memory") or st.session_state.conversation_memory
                count = len((result.get("retrieval") or result).get("results") or [])
                assistant_content = t("retrieved_count", lang, n=str(count))
                st.markdown(assistant_content)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content, "kind": "retriever", "payload": result})
                st.rerun()

            elif mode == "ner":
                with st.status(t("step_ner_processing", lang), expanded=True) as status:
                    result = run_ner_test(
                        text=prompt,
                        mode=model_settings["ner_mode"],
                        threshold=model_settings["threshold"],
                        device=model_settings["device"],
                    )
                    status.update(label=f"{t('step_done', lang)}", state="complete", expanded=False)

                count = len(result.get("entities") or [])
                assistant_content = t("entity_count", lang, n=str(count))
                st.markdown(assistant_content)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content, "kind": "ner", "payload": result})
                st.rerun()

        except Exception as exc:
            if fail_progress is not None:
                fail_progress(exc)
            error_content = f"{t('error_prefix', lang)}: `{exc}`"
            st.error(error_content)
            st.session_state.chat_history.append({"role": "assistant", "content": error_content, "kind": mode, "payload": None})


def main() -> None:
    inject_css()
    init_state()

    sidebar_settings = render_sidebar()
    mode = sidebar_settings["mode"]
    model_settings = render_model_settings(mode)
    render_sidebar_footer()

    render_header(mode)
    render_examples(mode)
    render_chat_history()

    default_prompt = st.session_state.pending_example
    if default_prompt:
        st.session_state.pending_example = ""

    lang = get_lang()
    placeholder = {
        "answer": t("placeholder_answer", lang),
        "retriever": t("placeholder_retriever", lang),
        "ner": t("placeholder_ner", lang),
    }[mode]

    prompt = st.chat_input(placeholder)
    if prompt:
        handle_prompt(prompt, sidebar_settings, model_settings)
    elif default_prompt:
        handle_prompt(default_prompt, sidebar_settings, model_settings)


if __name__ == "__main__":
    main()
