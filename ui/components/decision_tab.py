"""
ui.components.decision_tab — Task 2 LangGraph Decision Agent tab.

Displays live reasoning steps as the agent decomposes, retrieves, synthesises,
and reflects. Shows sub-questions used and the final structured recommendation.
"""

from __future__ import annotations

import streamlit as st

from insight_agent.reasoning.agent import run_agent
from insight_agent.output.charts import generate_chart
from insight_agent.exceptions import NotIngestedError, InsightAgentError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXAMPLE_QUESTIONS = [
    "Should we expand into the Southeast Asian market in FY2025?",
    "How should we respond to the APAC segment underperformance?",
    "What strategic investments should we prioritise given current performance?",
    "Is an acquisition strategy better than organic growth for APAC expansion?",
]

_AGENT_SECTIONS: dict[str, tuple[str, bool]] = {
    # key → (icon, expanded_by_default)
    "RECOMMENDATION": ("🎯", True),
    "EVIDENCE":       ("🔍", True),
    "RISKS":          ("⚠️", True),
    "CONFIDENCE":     ("📊", True),
    "SOURCES":        ("📄", False),
}

_STEP_LABELS = [
    ("🔀", "Decomposing strategic question into sub-questions…"),
    ("🔎", "Retrieving evidence from document corpus…"),
    ("✍️", "Synthesising structured recommendation…"),
    ("🔁", "Self-critiquing and refining answer…"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_agent_sections(text: str) -> dict[str, str]:
    sections = {k: "" for k in _AGENT_SECTIONS}
    current: str | None = None
    for line in text.splitlines():
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


def _render_step(icon: str, label: str, done: bool) -> str:
    tick = "✅" if done else "⟳"
    return (
        f"<div class='step-row'>"
        f"<span class='step-icon'>{tick}</span>"
        f"<span>{label}</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render_decision_tab() -> None:
    """Render the full Task 2 Decision Agent tab."""
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-top:0.2rem;'>"
        "The LangGraph agent autonomously decomposes your question, retrieves evidence, "
        "synthesises a recommendation, and self-critiques before finalising. "
        "Watch every reasoning step in real time.</p>",
        unsafe_allow_html=True,
    )

    selected = st.selectbox(
        "📌 Try a strategic question:",
        ["— select —"] + _EXAMPLE_QUESTIONS,
        key="decision_example",
    )
    default_q = "" if selected == "— select —" else selected

    question = st.text_input(
        "Strategic question:",
        value=default_q,
        placeholder="e.g.  Should we expand into Southeast Asia in FY2025?",
        key="decision_question",
    )

    col_btn, _ = st.columns([1, 6])
    with col_btn:
        submitted = st.button("🤖  Run Agent", key="decision_submit")

    if not submitted:
        return

    if not question.strip():
        st.warning("Please enter a strategic question before submitting.")
        return

    # -- Live step display -------------------------------------------------
    step_container = st.empty()

    def _draw_steps(done_count: int) -> None:
        html = "".join(
            _render_step(icon, label, done=(i < done_count))
            for i, (icon, label) in enumerate(_STEP_LABELS)
        )
        step_container.markdown(html, unsafe_allow_html=True)

    _draw_steps(0)  # all pending

    try:
        final_answer, sub_qs = run_agent(question)
    except NotIngestedError:
        st.error(
            "⚠️  No documents indexed yet. "
            "Run `python scripts/ingest.py` after placing documents in `data/docs/`."
        )
        return
    except InsightAgentError as exc:
        st.error(f"Agent error: {exc}")
        return

    _draw_steps(len(_STEP_LABELS))  # all done

    st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)

    # -- Sub-questions used ------------------------------------------------
    if sub_qs:
        st.markdown(
            "<div class='section-label'>🔀&nbsp; SUB-QUESTIONS RESEARCHED</div>",
            unsafe_allow_html=True,
        )
        for i, sq in enumerate(sub_qs, 1):
            st.markdown(
                f"<div style='color:#64748b;font-size:0.87rem;padding:0.25rem 0;'>"
                f"<span style='color:#3b82f6;font-weight:600;'>Q{i}:</span>&nbsp;{sq}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)

    # -- Agent report sections ---------------------------------------------
    sections = _parse_agent_sections(final_answer)
    for key, (icon, expanded) in _AGENT_SECTIONS.items():
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

    # -- Auto chart --------------------------------------------------------
    fig = generate_chart(final_answer, question=question)
    if fig:
        st.markdown(
            "<div class='section-label'>📈&nbsp; AUTO-GENERATED CHART</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(fig)
