"""
tests/test_retrieval.py — Unit tests for the retrieval sub-package.

These tests do NOT require a Gemini API key or ChromaDB connection.
BM25 and RRF fusion are tested in isolation.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from insight_agent.retrieval.bm25 import BM25Index, BM25Result
from insight_agent.retrieval.retriever import reciprocal_rank_fusion, FusedChunk
from insight_agent.exceptions import NotIngestedError


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_build_and_search(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("revenue APAC", top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, dict) for r in results)
        assert all("bm25_score" in r for r in results)
        assert all("chunk_text" in r for r in results)

    def test_search_returns_fewer_than_top_k_when_corpus_small(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("anything", top_k=1000)
        assert len(results) == len(sample_chunks)

    def test_save_and_load(self, sample_chunks, tmp_path: Path):
        index = BM25Index()
        index.build(sample_chunks)

        save_path = tmp_path / "bm25_test.pkl"
        index.save(save_path)
        assert save_path.exists()

        loaded = BM25Index.load(save_path)
        results = loaded.search("revenue", top_k=2)
        assert len(results) >= 1

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(NotIngestedError):
            BM25Index.load(tmp_path / "nonexistent.pkl")

    def test_highest_score_chunk_contains_query_term(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("Americas", top_k=5)
        # The top result should contain 'Americas' (case-insensitive)
        assert "americas" in results[0]["chunk_text"].lower()

    def test_empty_build_is_safe(self):
        index = BM25Index()
        index.build([])  # should not raise
        assert index._bm25 is None


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------


class TestRRFFusion:
    def _make_dense(self, texts: list[str]) -> list[dict]:
        return [
            {"chunk_text": t, "source": "doc.md", "page": "1", "doc_title": "doc", "score": 0.9}
            for t in texts
        ]

    def _make_sparse(self, texts: list[str]) -> list[dict]:
        return [
            {
                "chunk_text": t,
                "source": "doc.md",
                "page": "1",
                "doc_title": "doc",
                "bm25_score": 5.0,
            }
            for t in texts
        ]

    def test_returns_top_k(self):
        dense = self._make_dense([f"chunk {i}" for i in range(10)])
        sparse = self._make_sparse([f"chunk {i}" for i in range(10)])
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert len(fused) == 5

    def test_boosts_chunks_in_both_lists(self):
        """A chunk ranked #1 in both lists should beat one ranked #1 in only one."""
        shared_text = "shared important chunk"
        dense = self._make_dense([shared_text, "only in dense"])
        sparse = self._make_sparse([shared_text, "only in sparse"])
        fused = reciprocal_rank_fusion(dense, sparse, top_k=3)
        assert fused[0]["chunk_text"] == shared_text

    def test_deduplication(self):
        text = "duplicated chunk"
        dense = self._make_dense([text, text])
        sparse = self._make_sparse([text])
        fused = reciprocal_rank_fusion(dense, sparse, top_k=10)
        texts = [c["chunk_text"] for c in fused]
        assert texts.count(text) == 1

    def test_rrf_score_field_present(self):
        dense = self._make_dense(["a", "b"])
        sparse = self._make_sparse(["b", "c"])
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert all("rrf_score" in c for c in fused)
        assert all(c["rrf_score"] > 0 for c in fused)

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([], [], top_k=5) == []
