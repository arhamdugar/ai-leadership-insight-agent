"""
tests/conftest.py — Shared fixtures for all test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure insight_agent is importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_agent.ingestion.loader import DocumentPage
from insight_agent.ingestion.chunker import Chunk


@pytest.fixture()
def sample_pages() -> list[DocumentPage]:
    """Two synthetic DocumentPages for use in ingestion tests."""
    return [
        DocumentPage(
            text=(
                "FY2024 revenue reached 2.4 billion dollars, growing 14 percent year over year. "
                "The Americas segment was the strongest performer at plus 18 percent year over year. "
                "APAC underperformed by 50 million dollars against its 390 million target."
            ),
            source="FY2024_Annual_Report.md",
            page=1,
            doc_title="FY2024_Annual_Report",
        ),
        DocumentPage(
            text=(
                "Q3 2024 revenue was 640 million dollars, a 14 percent year over year increase. "
                "APAC Q3 revenue of 108 million fell 22 million below the 130 million quarterly target. "
                "The Americas reached 285 million, beating its target by 5 million dollars."
            ),
            source="Q3_FY2024_Quarterly_Report.md",
            page=1,
            doc_title="Q3_FY2024_Quarterly_Report",
        ),
    ]


@pytest.fixture()
def sample_chunks(sample_pages) -> list[Chunk]:
    """Chunks derived from sample_pages with small chunk_size for test speed."""
    from insight_agent.ingestion.chunker import chunk_documents

    return chunk_documents(sample_pages, chunk_size=200, chunk_overlap=30)
