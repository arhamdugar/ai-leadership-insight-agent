"""
insight_agent.exceptions — Custom exception hierarchy.

Catching `InsightAgentError` will catch any library-level error.
More specific handlers can target sub-classes as needed.
"""


class InsightAgentError(Exception):
    """Base exception for all insight_agent errors."""


class ConfigError(InsightAgentError):
    """Raised when configuration is missing or invalid."""


class IngestionError(InsightAgentError):
    """Raised during document loading or chunking."""


class EmbeddingError(InsightAgentError):
    """Raised when the embedding API call fails."""


class RetrievalError(InsightAgentError):
    """Raised when the retrieval pipeline fails (ChromaDB or BM25)."""


class GenerationError(InsightAgentError):
    """Raised when the LLM synthesis call fails."""


class AgentError(InsightAgentError):
    """Raised when the LangGraph decision agent fails."""


class NotIngestedError(RetrievalError):
    """
    Raised when retrieval is attempted but no index exists.
    Hint: run `python scripts/ingest.py` first.
    """
