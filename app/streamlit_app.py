"""Streamlit entry point: MurphysTelecom churn early-warning app.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Repository root (for the ``app`` package) and ``src`` (for ``telco_churn``)
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

from app.controllers.router import run  # noqa: E402

run()
