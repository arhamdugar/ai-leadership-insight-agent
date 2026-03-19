"""
validation/evaluate.py — LLM-as-Judge evaluation script.

Runs each Q&A pair from qa_pairs.json through the insight pipeline,
then scores the answer using a second Gemini call on four dimensions:
  Groundedness, Completeness, Conciseness, Actionability (each 1–5).

Usage:
    python validation/evaluate.py
    python validation/evaluate.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import google.generativeai as genai

from insight_agent.config import Config
from insight_agent.retrieval.retriever import retrieve
from insight_agent.reasoning.generator import generate_answer

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("evaluate")

_QA_PATH = Path(__file__).parent / "qa_pairs.json"

_JUDGE_PROMPT = """\
You are an expert evaluator assessing an AI-generated leadership report.

Question: {question}
Expected answer: {expected}
Generated answer:
{generated}

Score the generated answer on each dimension from 1 (poor) to 5 (excellent):
- Groundedness: Are all claims traceable to the document context?
- Completeness: Does it cover all aspects of the question?
- Conciseness: Is the answer appropriately scoped (not too verbose or too brief)?
- Actionability: Can leadership act on this insight?

Reply ONLY with a JSON object (no markdown fences), example:
{{"groundedness": 4, "completeness": 3, "conciseness": 5, "actionability": 4}}
"""


def _judge(question: str, expected: str, generated: str, cfg: Config) -> dict[str, int]:
    """Call Gemini to score a single answer. Returns score dict or zeros on failure."""
    genai.configure(api_key=cfg.api_key)
    model = genai.GenerativeModel(cfg.llm_model)
    prompt = _JUDGE_PROMPT.format(
        question=question, expected=expected, generated=generated[:1500]
    )
    try:
        text = model.generate_content(prompt).text.strip()
        # Strip markdown fences if present
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning("Judge scoring failed: %s", exc)
        return {"groundedness": 0, "completeness": 0, "conciseness": 0, "actionability": 0}


def run_evaluation(output_path: Path | None = None) -> None:
    cfg = Config()
    qa_pairs: list[dict] = json.loads(_QA_PATH.read_text(encoding="utf-8"))

    results = []
    all_scores: list[float] = []

    print(f"\nEvaluating {len(qa_pairs)} Q&A pairs ...\n")

    for pair in qa_pairs:
        qid = pair["id"]
        question = pair["question"]
        expected = pair["expected_answer"]

        logger.info("[%s] %s", qid, question[:70])

        chunks = retrieve(question, cfg=cfg)
        generated = generate_answer(question, chunks, cfg=cfg)
        scores = _judge(question, expected, generated, cfg)

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        all_scores.append(avg)

        result = {
            "id": qid,
            "category": pair.get("category", ""),
            "difficulty": pair.get("difficulty", ""),
            "question": question,
            "scores": scores,
            "avg_score": round(avg, 2),
        }
        results.append(result)

        print(
            f"  [{qid}] avg={avg:.1f}/5 | "
            + " | ".join(f"{k}={v}" for k, v in scores.items())
        )

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n━━ Overall average score: {overall:.2f}/5.00 ━━\n")

    if output_path:
        output_path.write_text(
            json.dumps({"overall_avg": round(overall, 2), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the insight pipeline with LLM-as-Judge.")
    parser.add_argument("--output", type=Path, default=None, help="Write results to this JSON file.")
    args = parser.parse_args()
    run_evaluation(output_path=args.output)


if __name__ == "__main__":
    main()
