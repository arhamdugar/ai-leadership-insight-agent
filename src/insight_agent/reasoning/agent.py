"""
insight_agent.reasoning.agent — LangGraph Decision Agent (Task 2).

Implements a four-node DAG:
    decompose → retrieve → synthesize → reflect
                  ↑                         |
                  └─── if not done (max 2) ─┘
                                            |
                                       if done → END

The agent self-critiques its draft and optionally loops once or twice to
research additional sub-questions before finalising.
"""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict
import operator

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from insight_agent.config import Config
from insight_agent.exceptions import AgentError
from insight_agent.retrieval.retriever import retrieve, FusedChunk

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    question: str
    sub_questions: Annotated[list[str], operator.add]
    retrieved_chunks: Annotated[list[dict], operator.add]
    draft_answer: str
    iteration: int
    final_answer: str
    done: bool


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _llm(prompt: str, cfg: Config) -> str:
    """Thin LLM wrapper with error conversion."""
    try:
        genai.configure(api_key=cfg.api_key)
        model = genai.GenerativeModel(
            cfg.llm_model,
            generation_config=genai.GenerationConfig(
                temperature=cfg.llm_temperature,
                max_output_tokens=cfg.llm_max_tokens,
            ),
        )
        return model.generate_content(prompt).text
    except Exception as exc:
        raise AgentError(f"Agent LLM call failed: {exc}") from exc


def _format_chunks(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | {c.get('source', '?')} p.{c.get('page', '?')}]\n"
            f"{c['chunk_text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def _node_decompose(state: AgentState) -> dict:
    """Break the strategic question into 2–3 focused sub-questions."""
    cfg = Config()
    prompt = (
        f"You are a strategic research assistant.\n"
        f"Break this strategic question into exactly 2-3 specific, focused sub-questions "
        f"that together would fully answer the original question.\n"
        f"Return ONLY the sub-questions as a numbered list (1., 2., 3.) — nothing else.\n\n"
        f"Question: {state['question']}"
    )
    text = _llm(prompt, cfg)

    # Parse numbered list
    sub_qs: list[str] = []
    for line in text.strip().splitlines():
        clean = line.lstrip("0123456789.-)  ").strip()
        if clean and len(clean) > 8:
            sub_qs.append(clean)

    sub_qs = sub_qs[:3]  # hard cap at 3
    logger.info("[agent] Decomposed into %d sub-questions: %s", len(sub_qs), sub_qs)
    return {"sub_questions": sub_qs, "iteration": 0}


def _node_retrieve(state: AgentState) -> dict:
    """Run hybrid retrieval for every new sub-question."""
    existing_texts = {c["chunk_text"][:80] for c in state.get("retrieved_chunks", [])}
    new_chunks: list[dict] = []

    for q in state["sub_questions"]:
        results: list[FusedChunk] = retrieve(q)
        for chunk in results:
            key = chunk["chunk_text"][:80]
            if key not in existing_texts:
                new_chunks.append(dict(chunk))
                existing_texts.add(key)

    logger.info(
        "[agent] Retrieved %d new unique chunks (total: %d)",
        len(new_chunks),
        len(state.get("retrieved_chunks", [])) + len(new_chunks),
    )
    return {"retrieved_chunks": new_chunks}


def _node_synthesise(state: AgentState) -> dict:
    """Synthesise a structured strategic recommendation from all retrieved chunks."""
    cfg = Config()
    context = _format_chunks(state.get("retrieved_chunks", []))

    prompt = (
        "You are a senior strategic advisor preparing a briefing for executive leadership.\n"
        "Write a comprehensive strategic report using EXACTLY these markdown section headers:\n\n"
        "## RECOMMENDATION\n"
        "Clear, actionable recommendation answering the question.\n\n"
        "## EVIDENCE\n"
        "Key facts and data points from the sources that support the recommendation.\n\n"
        "## RISKS\n"
        "Potential risks, challenges, and suggested mitigation strategies.\n\n"
        "## CONFIDENCE\n"
        "HIGH / MEDIUM / LOW — with a 1–2 sentence justification.\n\n"
        "## SOURCES\n"
        "All source documents and pages referenced.\n\n"
        "---\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"---\n\nSTRATEGIC QUESTION: {state['question']}"
    )

    draft = _llm(prompt, cfg)
    logger.info("[agent] Draft synthesised (%d chars)", len(draft))
    return {"draft_answer": draft}


def _node_reflect(state: AgentState) -> dict:
    """Self-critique the draft; loop if insufficient and under iteration limit."""
    cfg = Config()
    iteration = state.get("iteration", 0)

    if iteration >= _MAX_ITERATIONS:
        logger.info("[agent] Max iterations (%d) reached — finalising.", _MAX_ITERATIONS)
        return {"final_answer": state["draft_answer"], "done": True, "iteration": iteration}

    prompt = (
        f'You are a critical reviewer evaluating a strategic executive report.\n\n'
        f'Question: "{state["question"]}"\n\n'
        f"Draft answer:\n{state['draft_answer']}\n\n"
        f"Is this draft sufficient to advise a strategic decision?\n"
        f"Reply with exactly one of:\n"
        f'  YES\n'
        f'  NO: <one concise new sub-question that would fill the most important gap>\n\n'
        f"Your response:"
    )
    verdict = _llm(prompt, cfg).strip()
    logger.info("[agent] Reflect verdict (iter %d): %s", iteration, verdict[:80])

    if verdict.upper().startswith("YES"):
        return {
            "final_answer": state["draft_answer"],
            "done": True,
            "iteration": iteration + 1,
        }

    # Extract the new sub-question after "NO:"
    new_q = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
    logger.info("[agent] Adding refinement sub-question: %s", new_q)
    return {
        "sub_questions": [new_q],
        "done": False,
        "iteration": iteration + 1,
    }


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def _route_after_reflect(state: AgentState) -> str:
    return END if state.get("done") else "retrieve"


def _build_graph() -> object:
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("decompose", _node_decompose)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("synthesise", _node_synthesise)
    graph.add_node("reflect", _node_reflect)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "synthesise")
    graph.add_edge("synthesise", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"retrieve": "retrieve", END: END},
    )

    return graph.compile()


# Compile once at import time for reuse
_AGENT = None


def _get_agent() -> object:
    global _AGENT  # noqa: PLW0603
    if _AGENT is None:
        _AGENT = _build_graph()
    return _AGENT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent(question: str) -> tuple[str, list[str]]:
    """
    Run the Decision Agent on an open-ended strategic *question*.

    Args:
        question: Strategic question from leadership.

    Returns:
        Tuple of ``(final_answer, sub_questions_used)``.

    Raises:
        AgentError: If an unexpected error occurs during agent execution.
        NotIngestedError: If the document index doesn't exist.
    """
    logger.info("[agent] Starting Decision Agent for: %s...", question[:70])

    initial: AgentState = {
        "question": question,
        "sub_questions": [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "iteration": 0,
        "final_answer": "",
        "done": False,
    }

    try:
        result = _get_agent().invoke(initial)
    except Exception as exc:
        if "AgentError" in type(exc).__name__ or "InsightAgentError" in type(exc).__name__:
            raise
        raise AgentError(f"Unexpected agent failure: {exc}") from exc

    final = result.get("final_answer") or result.get("draft_answer", "")
    sub_qs = result.get("sub_questions", [])

    logger.info(
        "[agent] Complete after %d iteration(s). Answer: %d chars",
        result.get("iteration", 0),
        len(final),
    )
    return final, sub_qs
