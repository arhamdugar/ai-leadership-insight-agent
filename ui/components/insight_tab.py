"""
ui.components.insight_tab — Task 1 Insight Q&A report tab.

Renders the question input, calls the RAG pipeline, and displays
a structured report with confidence badge, section cards, and auto-chart.
"""

from __future__ import annotations

import streamlit as st

from insight_agent.retrieval.retriever import retrieve
from insight_agent.reasoning.generator import generate_answer, compute_confidence
from insight_agent.output.charts import generate_chart
from insight_agent.exceptions import NotIngestedError, InsightAgentError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXAMPLE_QUESTIONS = [
    "What is our current revenue trend?",
    "Which regions are underperforming and why?",
    "What are the key strategic risks for FY2025?",
    "How has EBITDA margin changed over the past two years?",
    "What drove the APAC underperformance in FY2024?",
]

_SECTION_CONFIG: dict[str, tuple[str, bool]] = {
    # section_key → (icon, expand_by_default)
    "SUMMARY":          ("📋", True),
    "KEY FINDINGS":     ("🔍", True),
    "SUPPORTING DATA":  ("📊", True),
    "RISKS & CAVEATS":  ("⚠️", True),
    "SOURCES":          ("📄", False),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sections(markdown: str) -> dict[str, str]:
    """Split a structured markdown report into named sections."""
    sections = {k: "" for k in _SECTION_CONFIG}
    current: str | None = None

    for line in markdown.splitlines():
        stripped = line.strip()
        matched = False
        for key in sections:
            if stripped.upper().startswith(f"## {key}") or stripped.upper() == key:
                current = key
                matched = True
                break
        if not matched and current:
            sections[current] += line + "\n"

    return {k: v.strip() for k, v in sections.items()}


def _confidence_badge(level: str) -> str:
    cls_map = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}
    return (
        f"<span class='conf-badge {cls_map.get(level, 'conf-low')}'>"
        f"⬡&nbsp;{level} CONFIDENCE</span>"
    )


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render_insight_tab() -> None:
    """Render the full Task 1 Insight Q&A tab."""
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-top:0.2rem;'>"
        "Ask a question about your company documents — get a structured leadership report "
        "grounded in your ingested documents with citations and auto-generated charts.</p>",
        unsafe_allow_html=True,
    )

    # Example questions selector
    selected = st.selectbox(
        "📌 Try an example question  (or type your own below):",
        ["— select —"] + _EXAMPLE_QUESTIONS,
        key="insight_example",
    )
    default_q = "" if selected == "— select —" else selected

    question = st.text_input(
        "Your question:",
        value=default_q,
        placeholder="e.g.  What is our revenue trend in APAC?",
        key="insight_question",
    )

    col_btn, _ = st.columns([1, 6])
    with col_btn:
        submitted = st.button("🔍  Generate Report", key="insight_submit")

    if not submitted:
        return

    if not question.strip():
        st.warning("Please enter a question before submitting.")
        return

    with st.spinner("Retrieving relevant context and generating report…"):
        try:
            chunks = retrieve(question)
            answer = generate_answer(question, chunks)
            confidence = compute_confidence(chunks)
        except NotIngestedError:
            st.error(
                "⚠️  No documents indexed yet.  "
                "Run `python scripts/ingest.py` after placing documents in `data/docs/`."
            )
            return
        except InsightAgentError as exc:
            st.error(f"Pipeline error: {exc}")
            return

    # Confidence + source stats
    unique_sources = len({c.get("source") for c in chunks})
    st.markdown(
        f"<div style='margin-bottom:1.4rem;'>"
        f"{_confidence_badge(confidence)}"
        f"&nbsp;&nbsp;<span style='color:#475569;font-size:0.8rem;'>"
        f"{len(chunks)} chunks retrieved &nbsp;·&nbsp; {unique_sources} source(s)"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # Report sections
    sections = _parse_sections(answer)
    for key, (icon, expanded) in _SECTION_CONFIG.items():
        content = sections.get(key, "").strip()
        if not content:
            continue

        if key == "SOURCES":
            with st.expander(f"{icon}  {key}", expanded=expanded):
                st.markdown(content)
        else:
            st.markdown(
                f"<div class='section-label'>{icon}&nbsp; {key}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(content)
            st.markdown("<hr>", unsafe_allow_html=True)

    # Auto chart
    fig = generate_chart(answer, question=question)
    if fig:
        st.markdown(
            "<div class='section-label'>📈&nbsp; AUTO-GENERATED CHART</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(fig)

    # Debug expander
    with st.expander("🗂  Retrieved Chunks (debug view)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**[{i}]** `{chunk.get('source', '?')}` "
                f"p.{chunk.get('page', '?')} "
                f"*(rrf_score={chunk.get('rrf_score', 'n/a')})*"
            )
            st.code(chunk["chunk_text"][:350], language=None)
