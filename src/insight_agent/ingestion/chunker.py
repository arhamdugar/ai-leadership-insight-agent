"""
insight_agent.ingestion.chunker — Splits DocumentPages into overlapping text chunks.

Each output Chunk preserves the full source metadata from its parent page.
The splitter operates on word boundaries for clean, readable chunks.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from insight_agent.ingestion.loader import DocumentPage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Chunk(TypedDict):
    chunk_text: str
    source: str       # Source filename
    page: int         # Source page number
    doc_title: str    # Source document title (stem)
    chunk_index: int  # Global monotonic index across all chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _word_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split *text* into overlapping word-based windows.

    Sizes are approximated as: 1 word ≈ 6 characters.
    """
    words = text.split()
    if not words:
        return []

    words_per_chunk = max(1, chunk_size // 6)
    words_overlap = max(0, chunk_overlap // 6)
    stride = max(1, words_per_chunk - words_overlap)

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_documents(
    pages: list[DocumentPage],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """
    Split each :class:`DocumentPage` into overlapping :class:`Chunk` dicts.

    Args:
        pages: Output of :func:`~insight_agent.ingestion.loader.load_all_documents`.
        chunk_size: Target chunk length in characters (approximate).
        chunk_overlap: Overlap between consecutive chunks in characters (approximate).

    Returns:
        Flat list of :class:`Chunk` dicts with a global `chunk_index`.
    """
    all_chunks: list[Chunk] = []
    global_idx = 0

    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        splits = _word_split(text, chunk_size, chunk_overlap)
        for split_text in splits:
            if not split_text.strip():
                continue
            all_chunks.append(
                Chunk(
                    chunk_text=split_text,
                    source=page["source"],
                    page=page["page"],
                    doc_title=page["doc_title"],
                    chunk_index=global_idx,
                )
            )
            global_idx += 1

    logger.info(
        "Chunked %d page(s) → %d chunks (size≈%d, overlap≈%d)",
        len(pages),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
