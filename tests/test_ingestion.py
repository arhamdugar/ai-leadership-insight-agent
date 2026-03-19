"""
tests/test_ingestion.py — Unit tests for the ingestion sub-package.

These tests do NOT require a Gemini API key — they test pure Python logic only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_agent.ingestion.loader import (
    load_all_documents,
    SUPPORTED_EXTENSIONS,
    DocumentPage,
)
from insight_agent.ingestion.chunker import chunk_documents, Chunk
from insight_agent.exceptions import IngestionError


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoader:
    def test_supported_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS

    def test_load_markdown(self, tmp_path: Path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Report\n\nRevenue grew 14% YoY in FY2024.", encoding="utf-8")

        pages = load_all_documents(tmp_path)

        assert len(pages) == 1
        assert pages[0]["source"] == "test.md"
        assert pages[0]["doc_title"] == "test"
        assert pages[0]["page"] == 1
        assert "14%" in pages[0]["text"]

    def test_skips_gitkeep(self, tmp_path: Path):
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
        pages = load_all_documents(tmp_path)
        assert pages == []

    def test_skips_unsupported_extension(self, tmp_path: Path, caplog):
        (tmp_path / "notes.txt").write_text("some text", encoding="utf-8")
        pages = load_all_documents(tmp_path)
        assert pages == []

    def test_empty_docs_dir_raises(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(IngestionError, match="does not exist"):
            load_all_documents(missing)

    def test_multiple_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("Document A content here.", encoding="utf-8")
        (tmp_path / "b.md").write_text("Document B content here.", encoding="utf-8")

        pages = load_all_documents(tmp_path)
        assert len(pages) == 2
        sources = {p["source"] for p in pages}
        assert sources == {"a.md", "b.md"}


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_chunk_count_reasonable(self, sample_pages):
        chunks = chunk_documents(sample_pages, chunk_size=600, chunk_overlap=80)
        # Two ~100-word pages should produce at least 1 chunk each
        assert len(chunks) >= 2

    def test_metadata_propagated(self, sample_chunks: list[Chunk]):
        for chunk in sample_chunks:
            assert chunk["source"] != ""
            assert chunk["doc_title"] != ""
            assert isinstance(chunk["page"], int)
            assert chunk["chunk_text"].strip() != ""

    def test_global_index_monotonic(self, sample_chunks: list[Chunk]):
        indices = [c["chunk_index"] for c in sample_chunks]
        assert indices == list(range(len(indices)))

    def test_small_chunk_size_produces_more_chunks(self, sample_pages):
        coarse = chunk_documents(sample_pages, chunk_size=600, chunk_overlap=0)
        fine = chunk_documents(sample_pages, chunk_size=100, chunk_overlap=0)
        assert len(fine) >= len(coarse)

    def test_empty_pages_returns_empty(self):
        result = chunk_documents([], chunk_size=600, chunk_overlap=80)
        assert result == []
