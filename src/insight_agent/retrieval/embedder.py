"""
insight_agent.retrieval.embedder — Gemini embedding client and ChromaDB store.

Responsibilities:
  - Batch-embed text via `models/text-embedding-004`
  - Upsert chunks into a persistent ChromaDB collection
  - Embed query strings for dense retrieval
  - Query ChromaDB and return typed RetrievedChunk objects
"""

from __future__ import annotations

import logging
import time
from typing import TypedDict

import chromadb
import google.generativeai as genai

from insight_agent.config import Config
from insight_agent.exceptions import EmbeddingError, NotIngestedError

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "documents"
_EMBED_BATCH_SIZE = 50
_STORE_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class RetrievedChunk(TypedDict):
    chunk_text: str
    source: str
    page: str        # Stored as string in ChromaDB metadata
    doc_title: str
    score: float     # Cosine similarity (0–1, higher = more relevant)


# ---------------------------------------------------------------------------
# EmbedderClient
# ---------------------------------------------------------------------------


class EmbedderClient:
    """
    Manages Gemini embeddings and ChromaDB persistence.

    One instance per application lifetime is sufficient.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or Config()
        genai.configure(api_key=self._cfg.api_key)
        self._chroma = chromadb.PersistentClient(
            path=str(self._cfg.chroma_dir)
        )

    # ------------------------------------------------------------------
    # Public: ingestion
    # ------------------------------------------------------------------

    def store_chunks(self, chunks: list) -> None:  # list[Chunk]
        """
        Embed all *chunks* and upsert them into ChromaDB.
        Drops and recreates the collection for a clean ingest.
        """
        if not chunks:
            logger.warning("store_chunks called with empty chunk list — nothing to do.")
            return

        # Fresh collection every ingest
        try:
            self._chroma.delete_collection(_COLLECTION_NAME)
            logger.debug("Dropped existing ChromaDB collection '%s'", _COLLECTION_NAME)
        except Exception:
            pass

        collection = self._chroma.create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c["chunk_text"] for c in chunks]
        ids = [str(c["chunk_index"]) for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "page": str(c["page"]),
                "doc_title": c["doc_title"],
            }
            for c in chunks
        ]

        logger.info("Embedding %d chunks with '%s' ...", len(chunks), self._cfg.embed_model)
        embeddings = self._embed_texts(texts, task_type="retrieval_document")

        for i in range(0, len(chunks), _STORE_BATCH_SIZE):
            j = i + _STORE_BATCH_SIZE
            collection.add(
                ids=ids[i:j],
                documents=texts[i:j],
                embeddings=embeddings[i:j],
                metadatas=metadatas[i:j],
            )

        logger.info("Stored %d chunks in ChromaDB ✓", len(chunks))

    # ------------------------------------------------------------------
    # Public: retrieval
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """Embed a single *query* string for retrieval."""
        try:
            result = genai.embed_content(
                model=self._cfg.embed_model,
                content=query,
                task_type="retrieval_query",
            )
            return result["embedding"]
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """
        Run a dense vector query against ChromaDB.

        Raises:
            NotIngestedError: If the collection doesn't exist yet.
        """
        try:
            collection = self._chroma.get_collection(_COLLECTION_NAME)
        except Exception as exc:
            raise NotIngestedError(
                "ChromaDB collection not found. Run `python scripts/ingest.py` first."
            ) from exc

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                RetrievedChunk(
                    chunk_text=doc,
                    source=meta.get("source", ""),
                    page=meta.get("page", "?"),
                    doc_title=meta.get("doc_title", ""),
                    score=round(1.0 - dist, 6),  # cosine similarity
                )
            )
        return output

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Batch-embed *texts*, returning a list of float vectors."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            try:
                result = genai.embed_content(
                    model=self._cfg.embed_model,
                    content=batch,
                    task_type=task_type,
                )
                all_embeddings.extend(result["embedding"])
                logger.debug(
                    "Embedded batch %d/%d (%d items)",
                    i // _EMBED_BATCH_SIZE + 1,
                    (len(texts) - 1) // _EMBED_BATCH_SIZE + 1,
                    len(batch),
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Embedding API failed on batch starting at index {i}: {exc}"
                ) from exc

            if i + _EMBED_BATCH_SIZE < len(texts):
                time.sleep(0.4)  # gentle rate-limit courtesy delay

        return all_embeddings
