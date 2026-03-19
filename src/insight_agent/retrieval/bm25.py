"""
insight_agent.retrieval.bm25 — BM25 sparse keyword index.

Wraps `rank_bm25.BM25Okapi` with a clean class interface:
  build(chunks) → indexes all chunks
  save(path)    → persists to disk (pickle)
  load(path)    → restores from disk
  search(query) → returns top-k scored chunks
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TypedDict

from rank_bm25 import BM25Okapi

from insight_agent.exceptions import NotIngestedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class BM25Result(TypedDict):
    chunk_text: str
    source: str
    page: str
    doc_title: str
    bm25_score: float


# ---------------------------------------------------------------------------
# BM25Index class
# ---------------------------------------------------------------------------


class BM25Index:
    """
    Stateful BM25 index over a corpus of chunks.

    Intended workflow:
        index = BM25Index()
        index.build(chunks)
        index.save(path)

        # Later, in retrieval:
        index = BM25Index.load(path)
        results = index.search("revenue APAC", top_k=10)
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list) -> None:  # list[Chunk]
        """Build the BM25 index from *chunks*."""
        if not chunks:
            logger.warning("BM25Index.build called with empty chunk list.")
            return

        tokenised = [self._tokenise(c["chunk_text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        self._chunks = list(chunks)
        logger.info("BM25 index built with %d documents.", len(chunks))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle the index and chunk list to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, fh)
        logger.info("BM25 index saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """
        Load a previously saved index from *path*.

        Raises:
            NotIngestedError: If the index file doesn't exist.
        """
        if not path.exists():
            raise NotIngestedError(
                f"BM25 index not found at {path}. Run `python scripts/ingest.py` first."
            )
        with path.open("rb") as fh:
            data = pickle.load(fh)

        instance = cls()
        instance._bm25 = data["bm25"]
        instance._chunks = data["chunks"]
        logger.debug("BM25 index loaded from %s (%d docs)", path, len(instance._chunks))
        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[BM25Result]:
        """
        Return *top_k* chunks ranked by BM25 score for *query*.

        Raises:
            RuntimeError: If the index hasn't been built or loaded yet.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index not initialised — call build() or load() first.")

        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[BM25Result] = []
        for idx, score in ranked[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                BM25Result(
                    chunk_text=chunk["chunk_text"],
                    source=chunk.get("source", ""),
                    page=str(chunk.get("page", "?")),
                    doc_title=chunk.get("doc_title", ""),
                    bm25_score=round(float(score), 6),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lower-case, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()
