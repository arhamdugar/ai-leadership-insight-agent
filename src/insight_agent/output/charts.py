"""
insight_agent.output.charts — Auto-generates a matplotlib chart from LLM response text.

Scans the generated text for ``Label: $Value[BKMG%]`` patterns and builds
a dark-themed bar or line chart. Returns ``None`` when no plottable data
is detected (caller is responsible for skipping the chart section).
"""

from __future__ import annotations

import logging
import re

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Non-interactive, thread-safe backend

# ---------------------------------------------------------------------------
# Regex for numeric extraction
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9 &/\-]{1,30})[:\-\u2013]\s*"  # Label:
    r"[\$\u20ac\u00a3]?\s*"                              # Optional currency symbol
    r"([+-]?\d+(?:\.\d+)?)\s*"                           # Numeric value
    r"([BKMG%]?)",                                       # Optional suffix
    re.IGNORECASE,
)

_SUFFIX_MULTIPLIER: dict[str, float] = {
    "B": 1e9,
    "M": 1e6,
    "K": 1e3,
    "G": 1e9,
    "%": 1.0,
    "": 1.0,
}

_MAX_SERIES_ITEMS = 10

# ---------------------------------------------------------------------------
# Dark theme constants
# ---------------------------------------------------------------------------

_BG = "#0f1117"
_GRID = "#1e2a3a"
_BAR_POS = "#4f8ef7"
_BAR_NEG = "#ff6b6b"
_TEXT = "#e2e8f0"
_MUTED = "#7a8ba8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_series(text: str) -> tuple[list[str], list[float], str]:
    """
    Extract (label, value) pairs from *text*.

    Returns:
        Tuple of ``(labels, values, unit_hint)`` where
        ``unit_hint`` is ``"%"`` if any value was a percentage, else ``"Value"``.
    """
    matches = _VALUE_RE.findall(text)
    labels: list[str] = []
    values: list[float] = []
    has_percent = False

    for raw_label, num_str, suffix in matches:
        suffix = suffix.upper()
        try:
            val = float(num_str) * _SUFFIX_MULTIPLIER.get(suffix, 1.0)
        except ValueError:
            continue
        labels.append(raw_label.strip().title())
        values.append(val)
        if suffix == "%":
            has_percent = True

    # Deduplicate (first occurrence wins)
    seen: set[str] = set()
    deduped_labels: list[str] = []
    deduped_values: list[float] = []
    for lbl, val in zip(labels, values):
        if lbl not in seen:
            seen.add(lbl)
            deduped_labels.append(lbl)
            deduped_values.append(val)

    unit = "%" if has_percent else "Value"
    return deduped_labels[:_MAX_SERIES_ITEMS], deduped_values[:_MAX_SERIES_ITEMS], unit


def _apply_dark_theme(fig: matplotlib.figure.Figure, ax: plt.Axes) -> None:
    """Apply the dark premium theme to a figure/axes pair."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.yaxis.label.set_color(_MUTED)
    ax.xaxis.label.set_color(_MUTED)
    ax.grid(axis="y", color=_GRID, linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_chart(
    response_text: str,
    *,
    question: str = "",
) -> matplotlib.figure.Figure | None:
    """
    Parse *response_text* for numeric data patterns and build a chart.

    Args:
        response_text: The LLM-generated report text to scan.
        question:      Used as the chart title (truncated to 70 chars).

    Returns:
        A :class:`matplotlib.figure.Figure` if plottable data is found,
        otherwise ``None``.
    """
    labels, values, unit = _parse_series(response_text)

    if len(labels) < 2:
        logger.debug("generate_chart: not enough data points (%d) — skipping.", len(labels))
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_dark_theme(fig, ax)

    colors = [_BAR_POS if v >= 0 else _BAR_NEG for v in values]

    if len(labels) <= 7:
        # Bar chart — clearer for small discrete datasets
        bars = ax.bar(labels, values, color=colors, edgecolor=_GRID, linewidth=0.8)
        ax.bar_label(
            bars,
            fmt=lambda v: f"{v:,.0f}" if abs(v) >= 1_000 else f"{v:.1f}",
            color=_TEXT,
            fontsize=8,
            padding=4,
        )
        plt.xticks(rotation=22, ha="right", color=_TEXT, fontsize=8)
    else:
        # Line chart — better for time-series trends
        ax.plot(labels, values, color=_BAR_POS, marker="o", linewidth=2, markersize=5)
        plt.xticks(rotation=30, ha="right", color=_TEXT, fontsize=8)

    title = (question[:70] + "…") if len(question) > 70 else question
    ax.set_title(
        title or "Data Snapshot",
        color=_TEXT,
        fontsize=11,
        pad=12,
        fontweight="600",
    )
    ax.set_ylabel(unit, color=_MUTED, fontsize=9)

    plt.tight_layout()
    return fig
