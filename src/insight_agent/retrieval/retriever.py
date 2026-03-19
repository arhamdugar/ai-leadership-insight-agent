"""
insight_agent.retrieval.retriever — Hybrid Reciprocal Rank Fusion retrieval.

Merges dense (ChromaDB cosine) and sparse (BM25 keyword) result lists using
Reciprocal Rank Fusion (RRF) and returns the top-k de-duplicated chunks.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from insight_agent.config import Config
from insight_agent.retrieval.embedder import EmbedderClient, RetrievedChunk
from insight_agent.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)

_RRF_K = 60  # Standard RRF constant


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class FusedChunk(TypedDict):
    chunk_text: str
    source: str
    page: str
    doc_title: str
    rrf_score: float


# ---------------------------------------------------------------------------
# RRF fusion (pure, no I/O)
# ---------------------------------------------------------------------------


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list,  # list[BM25Result]
    *,
    top_k: int,
) -> list[FusedChunk]:
    """
    Merge *dense* and *sparse* ranked lists using RRF.

    Uses the first 120 characters of chunk_text as the deduplication key.
    Returns *top_k* fused, ranked :class:`FusedChunk` dicts.
    """
    scores: dict[str, float] = {}
    chunk_store: dict[str, dict] = {}

    for rank, chunk in enumerate(dense, start=1):
        key = chunk["chunk_text"][:120]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        chunk_store[key] = dict(chunk)

    for rank, chunk in enumerate(sparse, start=1):
        key = chunk["chunk_text"][:120]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        chunk_store.setdefault(key, dict(chunk))

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    return [
        FusedChunk(
            chunk_text=chunk_store[k]["chunk_text"],
            source=chunk_store[k].get("source", ""),
            page=str(chunk_store[k].get("page", "?")),
            doc_title=chunk_store[k].get("doc_title", ""),
            rrf_score=round(scores[k], 6),
        )
        for k in ranked_keys[:top_k]
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(query: str, cfg: Config | None = None) -> list[FusedChunk]:
    """
    Execute the full hybrid retrieval pipeline for *query*.

    Pipeline:
        1. Embed *query* via Gemini.
        2. Dense retrieval from ChromaDB (top_k_dense results).
        3. Sparse retrieval from BM25 index (top_k_sparse results).
        4. Merge with RRF → top_k_final fused chunks.

    Args:
        query: Natural language query string.
        cfg:   Config instance (uses singleton if omitted).

    Returns:
        List of :class:`FusedChunk` dicts sorted by descending RRF score.

    Raises:
        NotIngestedError: If ChromaDB or BM25 index doesn't exist.
        EmbeddingError:   If the Gemini API call fails.
    """
    cfg = cfg or Config()

    embedder = EmbedderClient(cfg)
    query_vec = embedder.embed_query(query)
    dense_results = embedder.query(query_vec, top_k=cfg.top_k_dense)

    bm25 = BM25Index.load(cfg.bm25_path)
    sparse_results = bm25.search(query, top_k=cfg.top_k_sparse)

    fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=cfg.top_k_final)

    logger.info(
        "Retrieval: dense=%d, sparse=%d → fused top %d",
        len(dense_results),
        len(sparse_results),
        len(fused),
    )
    return fused
