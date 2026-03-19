"""
ui.app — Streamlit page configuration, global CSS, header, and tab layout.

Entry point: `streamlit run app.py` (root app.py imports and calls `render()`).
"""

from __future__ import annotations

import streamlit as st

from ui.components.insight_tab import render_insight_tab
from ui.components.decision_tab import render_decision_tab

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Main background ─────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #080d18 0%, #0c1220 60%, #080d18 100%);
    color: #e2e8f0;
}

/* ── Hero header ─────────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #111d36 0%, #0e1929 100%);
    border: 1px solid #1e3461;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 40px rgba(79,142,247,0.10);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4f8ef7 0%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0.45rem;
}
.hero-badges { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.22rem 0.75rem;
    border-radius: 999px;
    border: 1px solid #1e3461;
    color: #93c5fd;
    background: #0e1929;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0c1525;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e3461;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #64748b;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.6rem;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(30,58,138,0.5);
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stTextInput input, .stSelectbox > div {
    background: #0c1525 !important;
    border: 1px solid #1e3461 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
}
.stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.25) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.45) !important;
}

/* ── Cards / sections ────────────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.55rem;
    margin-top: 0.2rem;
}
.report-block {
    background: #0c1525;
    border: 1px solid #1e3461;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Confidence badges ───────────────────────────────────────────────── */
.conf-high   { background:#052e16; color:#4ade80; border:1px solid #16a34a; }
.conf-medium { background:#431407; color:#fb923c; border:1px solid #ea580c; }
.conf-low    { background:#3f0f0f; color:#f87171; border:1px solid #dc2626; }
.conf-badge {
    display: inline-block;
    padding: 0.22rem 0.9rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.6px;
}

/* ── Agent step list ─────────────────────────────────────────────────── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.7rem 1rem;
    background: #0c1525;
    border: 1px solid #1e3461;
    border-radius: 10px;
    margin-bottom: 0.45rem;
    font-size: 0.86rem;
    color: #94a3b8;
}
.step-icon { font-size: 1rem; flex-shrink: 0; }

/* ── Expander ────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #0c1525 !important;
    border: 1px solid #1e3461 !important;
    border-radius: 9px !important;
    color: #64748b !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border-color: #1e3461 !important; margin: 0.8rem 0 !important; }

/* ── Select box ──────────────────────────────────────────────────────── */
.stSelectbox label { color: #64748b !important; font-size: 0.85rem !important; }
.stTextInput label { color: #94a3b8 !important; font-size: 0.88rem !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render() -> None:
    """Configure the Streamlit page and render both tabs."""

    st.set_page_config(
        page_title="AI Leadership Insight Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Hero header
    st.markdown(
        """
        <div class="hero-header">
          <div class="hero-title">🧠 AI Leadership Insight &amp; Decision Agent</div>
          <div class="hero-subtitle">
            Grounded intelligence from your company documents &nbsp;&middot;&nbsp;
            Powered by Gemini 1.5 Flash &nbsp;&middot;&nbsp;
            Hybrid RAG + LangGraph Agentic Reasoning
          </div>
          <div class="hero-badges">
            <span class="badge">⬡ Gemini 1.5 Flash</span>
            <span class="badge">⬡ ChromaDB + BM25 Hybrid Retrieval</span>
            <span class="badge">⬡ LangGraph Agent</span>
            <span class="badge">⬡ Reciprocal Rank Fusion</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📊  Insight — Q&A Report", "🤖  Decision — Strategic Agent"])

    with tab1:
        render_insight_tab()

    with tab2:
        render_decision_tab()

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#1e3461;font-size:0.76rem;'>"
        "AI Leadership Insight Agent &nbsp;·&nbsp; "
        "Gemini 1.5 Flash &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; LangGraph"
        "</p>",
        unsafe_allow_html=True,
    )
