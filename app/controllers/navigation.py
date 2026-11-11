"""Page switching by URL path."""

from __future__ import annotations

import streamlit as st


def go_to(url: str) -> None:
    """Switch to another page by its URL path."""
    page = st.session_state.get("pages", {}).get(url)
    if page is not None:
        st.switch_page(page)
