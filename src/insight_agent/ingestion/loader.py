"""
insight_agent.ingestion.loader — Document loaders for PDF, DOCX, and Markdown.

Each loader returns a list of `DocumentPage` TypedDicts.
All pages carry: text, source (filename), page (1-indexed), doc_title.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import pdfplumber
from docx import Document

from insight_agent.exceptions import IngestionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DocumentPage(TypedDict):
    text: str
    source: str       # Filename, e.g. "FY2024_Annual_Report.pdf"
    page: int         # 1-indexed page/section number
    doc_title: str    # Filename stem, e.g. "FY2024_Annual_Report"


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_pdf(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(
                        DocumentPage(
                            text=text,
                            source=path.name,
                            page=i,
                            doc_title=path.stem,
                        )
                    )
    except Exception as exc:
        raise IngestionError(f"Failed to read PDF '{path.name}': {exc}") from exc
    return pages


def _load_docx(path: Path) -> list[DocumentPage]:
    try:
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        if not text:
            return []
        return [
            DocumentPage(
                text=text,
                source=path.name,
                page=1,
                doc_title=path.stem,
            )
        ]
    except Exception as exc:
        raise IngestionError(f"Failed to read DOCX '{path.name}': {exc}") from exc


def _load_markdown(path: Path) -> list[DocumentPage]:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [
            DocumentPage(
                text=text,
                source=path.name,
                page=1,
                doc_title=path.stem,
            )
        ]
    except Exception as exc:
        raise IngestionError(f"Failed to read Markdown '{path.name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS: dict[str, object] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".md": _load_markdown,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def load_all_documents(docs_dir: Path) -> list[DocumentPage]:
    """
    Walk *docs_dir* and load every supported document.

    Args:
        docs_dir: Directory containing PDF, DOCX, and/or Markdown files.

    Returns:
        Flat list of :class:`DocumentPage` dicts, one per page/section.

    Raises:
        IngestionError: If any individual file fails to parse.
    """
    docs_dir = docs_dir.resolve()
    if not docs_dir.exists():
        raise IngestionError(f"docs_dir does not exist: {docs_dir}")

    pages: list[DocumentPage] = []

    for path in sorted(docs_dir.iterdir()):
        if path.name.startswith("."):
            continue
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("Skipping unsupported file: %s", path.name)
            continue

        logger.info("Loading %s ...", path.name)
        try:
            result = loader(path)  # type: ignore[operator]
            pages.extend(result)
            logger.info("  → %d page(s) extracted from %s", len(result), path.name)
        except IngestionError:
            logger.warning("Skipped %s due to parse error (see DEBUG for details)", path.name)

    logger.info("Total pages loaded: %d", len(pages))
    return pages
