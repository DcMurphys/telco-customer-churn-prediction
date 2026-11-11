"""Cached access to artifacts and derived tables (shared across visitors).

Everything returned here is read-only. Per-visitor state lives in
``controllers.state``.
"""

from __future__ import annotations

import streamlit as st

from app.models import analysis, artifacts, eda
from app.models.alerts import AlertSettings, DemoState
from app.models.artifacts import ModelBundle
from app.models.plans import add_plan_columns
from telco_churn.config import DEFAULT_ALERT_SETTINGS


@st.cache_resource(show_spinner="Loading the churn model…")
def bundle() -> ModelBundle:
    """Model, calibrator, decision policy and preprocessing artifacts."""
    return artifacts.load_bundle()


@st.cache_data(show_spinner=False)
def reference() -> dict:
    """Reference values for the top-reasons logic."""
    return artifacts.load_reference()


@st.cache_data(show_spinner=False)
def history():
    """Training-split customers with plan columns and a 0/1 churn outcome."""
    data = add_plan_columns(artifacts.load_history())
    return data.assign(left=data["Churn Label"].eq("Yes").astype(int))


@st.cache_data(show_spinner=False)
def reports() -> dict:
    """Evaluation reports written by the training pipeline."""
    return artifacts.load_reports()


@st.cache_resource(show_spinner="Preparing the demo history…")
def seed_state() -> DemoState:
    """Pre-scored demo batches (template; every visitor gets a copy)."""
    return analysis.build_demo_state(
        bundle(),
        reference(),
        AlertSettings(**DEFAULT_ALERT_SETTINGS),
        artifacts.demo_batch_files(),
    )


@st.cache_data(show_spinner=False)
def sample_files() -> dict[str, bytes]:
    """Downloadable sample batch files."""
    return {
        name: path.read_bytes() for name, path in artifacts.sample_batch_files().items()
    }


@st.cache_data(show_spinner="Loading customer data…")
def eda_data():
    """Clean modelling dataset for Churn Insights (notebook Phase A)."""
    return eda.load_clean()


@st.cache_data(show_spinner=False)
def test_rows() -> int:
    """Number of customers in the held-out test split."""
    return len(artifacts.load_test_split()[1])


@st.cache_data(show_spinner=False)
def eda_tables() -> dict:
    """Tables that do not depend on widget choices."""
    data = eda_data()
    return {
        "stats": eda.overview(data),
        "leaderboard": eda.segment_leaderboard(data),
        "tenure": eda.tenure_profile(data),
        "add_ons": eda.add_on_effect(data),
    }
