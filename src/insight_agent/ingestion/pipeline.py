"""
insight_agent.ingestion.pipeline — Orchestrates the full ingestion pipeline.

Load → Chunk → Embed into ChromaDB → Build BM25 index.
This is the single entry point called by scripts/ingest.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from insight_agent.config import Config
from insight_agent.ingestion.loader import load_all_documents
from insight_agent.ingestion.chunker import Chunk, chunk_documents
from insight_agent.retrieval.embedder import EmbedderClient
from insight_agent.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(docs_dir: Path | None = None) -> dict[str, int]:
    """
    Execute the full ingestion pipeline end-to-end.

    Steps:
        1. Load all documents from *docs_dir*.
        2. Chunk each page with config-defined size/overlap.
        3. Embed chunks and store in ChromaDB.
        4. Build and persist the BM25 sparse index.

    Args:
        docs_dir: Override for the document directory (default: config.docs_dir).

    Returns:
        Summary dict: ``{"documents": N, "pages": N, "chunks": N}``.

    Raises:
        IngestionError: If loading fails.
        EmbeddingError: If the embedding API call fails.
    """
    cfg = Config()
    docs_dir = docs_dir or cfg.docs_dir
    t0 = time.perf_counter()

    # 1. Load
    logger.info("[1/4] Loading documents from: %s", docs_dir)
    pages = load_all_documents(docs_dir)
    if not pages:
        logger.warning("No documents found — place files in %s and re-run.", docs_dir)
        return {"documents": 0, "pages": 0, "chunks": 0}

    # 2. Chunk
    logger.info("[2/4] Chunking %d page(s) ...", len(pages))
    chunks: list[Chunk] = chunk_documents(
        pages,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # 3. Embed → ChromaDB
    logger.info("[3/4] Embedding %d chunks → ChromaDB ...", len(chunks))
    client = EmbedderClient(cfg)
    client.store_chunks(chunks)

    # 4. BM25 index
    logger.info("[4/4] Building BM25 sparse index ...")
    index = BM25Index()
    index.build(chunks)
    index.save(cfg.bm25_path)

    elapsed = time.perf_counter() - t0
    unique_docs = len({p["source"] for p in pages})
    logger.info(
        "Ingestion complete in %.1fs | docs=%d, pages=%d, chunks=%d",
        elapsed,
        unique_docs,
        len(pages),
        len(chunks),
    )

    return {"documents": unique_docs, "pages": len(pages), "chunks": len(chunks)}
