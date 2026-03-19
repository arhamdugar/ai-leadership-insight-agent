"""insight_agent.ingestion — Document loading, chunking, and pipeline orchestration."""

from insight_agent.ingestion.loader import DocumentPage, load_all_documents
from insight_agent.ingestion.chunker import Chunk, chunk_documents
from insight_agent.ingestion.pipeline import run_ingestion_pipeline

__all__ = [
    "DocumentPage",
    "load_all_documents",
    "Chunk",
    "chunk_documents",
    "run_ingestion_pipeline",
]
