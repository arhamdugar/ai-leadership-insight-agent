"""
scripts/ingest.py — CLI for running the full document ingestion pipeline.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --docs-dir /path/to/my/docs
    python scripts/ingest.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the src/ package is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_agent.ingestion.pipeline import run_ingestion_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Ingest documents into the AI Leadership Insight Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/ingest.py\n"
            "  python scripts/ingest.py --docs-dir ./my_documents\n"
        ),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing PDF, DOCX, and/or Markdown files "
             "(default: data/docs/ relative to project root).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      AI Leadership Insight Agent — Document Ingestion        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    t0 = time.perf_counter()

    try:
        summary = run_ingestion_pipeline(docs_dir=args.docs_dir)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    elapsed = time.perf_counter() - t0

    if summary["documents"] == 0:
        print("  ⚠️  No documents found.")
        print("  Drop PDF, DOCX, or Markdown files into data/docs/ and re-run.\n")
        return 0

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"  ✅  Ingestion complete in {elapsed:.1f}s")
    print(f"      Documents  : {summary['documents']}")
    print(f"      Pages      : {summary['pages']}")
    print(f"      Chunks     : {summary['chunks']}")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  → Launch the app with:  streamlit run app.py")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
