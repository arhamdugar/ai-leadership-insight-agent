"""
app.py — Root Streamlit entry point.

Adds `src/` to sys.path so `insight_agent` is importable,
then delegates all rendering to `ui.app.render()`.

Run with:
    streamlit run app.py
"""

import sys
import logging
from pathlib import Path

# Make the `insight_agent` package importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure basic logging (Streamlit-friendly: goes to stderr)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from ui.app import render  # noqa: E402  (import after sys.path patch)

render()
