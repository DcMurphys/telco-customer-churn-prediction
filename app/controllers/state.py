"""Per-visitor session state: demo history and kept form values.

Each visitor gets an independent copy of the seeded demo state, so one
visitor's batches, alerts and settings never affect another (requirement I-2).
"""

from __future__ import annotations

import streamlit as st

from app.controllers import resources
from app.models.alerts import DemoState

KEEP_PREFIX = "keep_"  # widget keys that must survive page switches (E-6)


def init() -> None:
    """Create the visitor's state on the first run of a session."""
    if "demo" not in st.session_state:
        st.session_state.demo = resources.seed_state().copy()
        st.session_state.toasted_alerts = set()
        st.session_state.batch_flow = {"step": 0}
    # Re-assigning keeps widget values alive when their page is not rendered.
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(KEEP_PREFIX):
            st.session_state[key] = st.session_state[key]


def demo() -> DemoState:
    """The visitor's demo state."""
    return st.session_state.demo


def reset_demo() -> None:
    """Restore the seeded demo state (requirement I-3)."""
    st.session_state.demo = resources.seed_state().copy()
    st.session_state.toasted_alerts = set()
    st.session_state.batch_flow = {"step": 0}
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(KEEP_PREFIX):
            del st.session_state[key]


def batch_flow() -> dict:
    """Progress of the Batch Scoring steps."""
    return st.session_state.batch_flow
