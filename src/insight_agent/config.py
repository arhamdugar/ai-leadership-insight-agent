"""
insight_agent.config — Singleton configuration loader.

Reads config.yaml (or the path in INSIGHT_AGENT_CONFIG env var) once and
exposes typed properties. Supports .env files via python-dotenv.

Usage:
    from insight_agent.config import Config
    cfg = Config()
    print(cfg.llm_model)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root = two levels above this file:
# src/insight_agent/config.py → parents[0]=insight_agent, [1]=src, [2]=project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from project root if it exists (no-op if absent)
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Config:
    """
    Singleton that loads config.yaml on first instantiation and caches it.
    All subsequent `Config()` calls return the same instance.
    """

    _instance: Config | None = None
    _data: dict[str, Any] = {}

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        config_env = os.environ.get("INSIGHT_AGENT_CONFIG")
        config_path = Path(config_env) if config_env else _PROJECT_ROOT / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Set INSIGHT_AGENT_CONFIG env var or create config.yaml at the project root."
            )

        with config_path.open("r", encoding="utf-8") as fh:
            self._data = yaml.safe_load(fh) or {}

        logger.debug("Loaded config from %s", config_path)

    def _get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node

    # ------------------------------------------------------------------
    # API key (env var takes precedence over config file)
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY") or self._get("api_key", default="")

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    @property
    def llm_model(self) -> str:
        return self._get("llm", "model", default="gemini-2.5-flash")

    @property
    def llm_temperature(self) -> float:
        return float(self._get("llm", "temperature", default=0.1))

    @property
    def llm_max_tokens(self) -> int:
        return int(self._get("llm", "max_tokens", default=4096))

    # Fast LLM — used for decompose, reflect, and Task 1 Q&A calls
    @property
    def fast_llm_model(self) -> str:
        return self._get("fast_llm", "model", default="gemini-2.5-flash")

    @property
    def fast_llm_max_tokens(self) -> int:
        return int(self._get("fast_llm", "max_tokens", default=2048))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def embed_model(self) -> str:
        return self._get("embedding", "model", default="models/text-embedding-004")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @property
    def top_k_dense(self) -> int:
        return int(self._get("retrieval", "top_k_dense", default=10))

    @property
    def top_k_sparse(self) -> int:
        return int(self._get("retrieval", "top_k_sparse", default=10))

    @property
    def top_k_final(self) -> int:
        return int(self._get("retrieval", "top_k_final", default=5))

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        return int(self._get("chunking", "chunk_size", default=600))

    @property
    def chunk_overlap(self) -> int:
        return int(self._get("chunking", "chunk_overlap", default=80))

    # ------------------------------------------------------------------
    # Paths (always absolute, relative to project root)
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def docs_dir(self) -> Path:
        return _PROJECT_ROOT / "data" / "docs"

    @property
    def chroma_dir(self) -> Path:
        return _PROJECT_ROOT / "data" / "chroma_db"

    @property
    def bm25_path(self) -> Path:
        return _PROJECT_ROOT / "data" / "bm25_index.pkl"
