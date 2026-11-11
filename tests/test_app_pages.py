"""Smoke test: every page renders without an error (Section 11)."""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    "overview",
    "alerts",
    "plan_risk",
    "customer_check",
    "batch_scoring",
    "churn_insights",
    "glossary",
    "alert_settings",
]


def _harness(page: str) -> None:
    import sys
    from pathlib import Path

    root = Path.cwd()
    for path in (root / "src", root):
        sys.path.insert(0, str(path))
    import streamlit as st

    from app.controllers import pages, state

    state.init()
    st.session_state.pages = {}
    getattr(pages, page)()


@pytest.mark.usefixtures("bundle")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_function(_harness, args=(page,), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.title, "page has no title"


@pytest.mark.usefixtures("bundle")
@pytest.mark.parametrize("page", PAGES)
def test_pages_avoid_technical_jargon(page):
    """Business users never see data-preparation or model-evaluation jargon."""
    at = AppTest.from_function(_harness, args=(page,), default_timeout=60)
    at.run()
    texts = [
        e.value
        for kind in ("markdown", "caption", "info", "warning", "title")
        for e in getattr(at, kind)
    ]
    shown = " ".join(str(t) for t in texts).lower()
    jargon = (
        "clean",
        "phase a",
        "leakage",
        "notebook",
        "modelling dataset",
        "roc-auc",
        "cramér",
        "chi-square",
        "p-value",
        "calibrat",
        "analyst",
    )
    for term in jargon:
        assert term not in shown, term
