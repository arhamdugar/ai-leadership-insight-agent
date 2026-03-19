"""
insight_agent.reasoning.generator — LLM synthesis for Task 1 (Insight Q&A).

Takes a question + retrieved chunks, calls Gemini, and returns a structured
markdown report in five sections: SUMMARY / KEY FINDINGS / SUPPORTING DATA /
RISKS & CAVEATS / SOURCES.
"""

from __future__ import annotations

import logging

import google.generativeai as genai

from insight_agent.config import Config
from insight_agent.exceptions import GenerationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an AI assistant serving company executives and leadership.
Answer ONLY based on the context provided — do not invent facts or extrapolate beyond the sources.

Structure your response using EXACTLY these markdown section headers (include the ## prefix):

## SUMMARY
2–3 sentence direct answer to the question, grounded in the context.

## KEY FINDINGS
A bullet list of the most important facts and data points from the context.

## SUPPORTING DATA
Key numbers, percentages, revenue figures, tables, and verbatim extracts from the context.

## RISKS & CAVEATS
Data gaps, conflicting information, forward-looking limitations, or low-confidence areas.

## SOURCES
List every source document (filename) and page number referenced in your answer.
Format: → <filename>, p.<page>

---

CONTEXT:
{context}

---

QUESTION: {question}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered, source-labelled context block."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Source {i} | {chunk.get('source', 'unknown')} p.{chunk.get('page', '?')}]"
        parts.append(f"{label}\n{chunk['chunk_text']}")
    return "\n\n".join(parts)


def _call_llm(prompt: str, cfg: Config) -> str:
    """
    Make a Gemini generation call for Task 1 Q&A.

    Uses cfg.fast_llm_model (gemini-2.5-flash) — structured document Q&A
    is a fast-tier task. The Pro model is reserved for the Agent's final
    strategic synthesis in Task 2.
    """
    model_name = cfg.fast_llm_model
    try:
        genai.configure(api_key=cfg.api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                temperature=cfg.llm_temperature,
                max_output_tokens=cfg.fast_llm_max_tokens,
            ),
        )
        return model.generate_content(prompt).text
    except Exception as exc:
        raise GenerationError(f"LLM call failed [{model_name}]: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    chunks: list[dict],
    cfg: Config | None = None,
) -> str:
    """
    Generate a structured markdown report answering *question* from *chunks*.

    Args:
        question: Natural language leadership question.
        chunks:   Retrieved chunks (list of dicts with ``chunk_text``, ``source``, ``page``).
        cfg:      Config instance (uses singleton if omitted).

    Returns:
        Structured markdown report string.

    Raises:
        GenerationError: If the Gemini API call fails.
    """
    cfg = cfg or Config()

    if not chunks:
        logger.warning("generate_answer called with no chunks — returning fallback.")
        return (
            "## SUMMARY\n"
            "No relevant documents were found in the knowledge base to answer this question.\n\n"
            "## RISKS & CAVEATS\n"
            "Please ensure documents have been ingested by running `python scripts/ingest.py`."
        )

    context = _format_context(chunks)
    prompt = _SYSTEM_PROMPT.format(context=context, question=question)

    logger.info("Calling LLM (%s) for question: %s...", cfg.fast_llm_model, question[:60])
    answer = _call_llm(prompt, cfg)
    logger.debug("LLM response length: %d chars", len(answer))
    return answer


def compute_confidence(chunks: list[dict]) -> str:
    """
    Compute a heuristic confidence level based on chunk coverage.

    Returns:
        ``"HIGH"`` — 4+ chunks from 2+ distinct sources.
        ``"MEDIUM"`` — 2–3 chunks, or all from one source.
        ``"LOW"`` — 0–1 chunk.
    """
    if not chunks:
        return "LOW"
    unique_sources = {c.get("source") for c in chunks}
    if len(chunks) >= 4 and len(unique_sources) >= 2:
        return "HIGH"
    if len(chunks) >= 2:
        return "MEDIUM"
    return "LOW"
