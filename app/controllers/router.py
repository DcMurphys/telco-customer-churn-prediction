"""App shell: page config, sidebar, navigation, alert banner, error boundary."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from app.controllers import navigation, pages, resources, state
from app.models.artifacts import ArtifactsMissingError
from app.views.components import ui
from telco_churn.config import BRAND_DIR, BRAND_NAME, FOOTER_TEXT

logger = logging.getLogger("murphys_app")


@dataclass(frozen=True)
class PageSpec:
    """One page in the navigation (UI/UX requirements, Section 4.1)."""

    section: str
    url: str
    title: str
    icon: str
    render: Callable[[], None]


PAGE_SPECS = [
    PageSpec("Monitor", "overview", "Overview", ":material/dashboard:", pages.overview),
    PageSpec("Monitor", "alerts", "Alerts", ":material/notifications:", pages.alerts),
    PageSpec(
        "Monitor",
        "plan-risk",
        "Plan Risk",
        ":material/stacked_bar_chart:",
        pages.plan_risk,
    ),
    PageSpec(
        "Act",
        "customer-check",
        "Customer Check",
        ":material/person_search:",
        pages.customer_check,
    ),
    PageSpec(
        "Act",
        "batch-scoring",
        "Batch Scoring",
        ":material/upload_file:",
        pages.batch_scoring,
    ),
    PageSpec(
        "Analyse",
        "churn-insights",
        "Churn Insights",
        ":material/query_stats:",
        pages.churn_insights,
    ),
    PageSpec(
        "Help", "glossary", "Glossary & Guide", ":material/menu_book:", pages.glossary
    ),
    PageSpec(
        "Settings",
        "alert-settings",
        "Alert Settings",
        ":material/tune:",
        pages.alert_settings,
    ),
]


def _guarded(spec: PageSpec) -> Callable[[], None]:
    """Wrap a page so errors show a friendly message and a reference code (G-6)."""

    def run() -> None:
        st.session_state.current_page = spec.url
        try:
            spec.render()
        except Exception:  # noqa: BLE001 - last-resort boundary for the UI
            code = uuid.uuid4().hex[:8].upper()
            logger.exception("Unhandled error on page %s (ref %s)", spec.url, code)
            st.error(
                "Something went wrong while showing this page. Your data in this "
                f"session is safe. Please try again or reset the demo. "
                f"Reference code: **{code}**",
                icon=":material/error:",
            )

    run.__name__ = f"page_{spec.url.replace('-', '_')}"
    return run


def _confirm_reset() -> None:
    @st.dialog("Reset the demo?")
    def dialog() -> None:
        st.write(
            "This restores the demo batches, alerts, alert settings and audit log "
            "to their starting state. Your changes in this session will be lost."
        )
        left, right = st.columns(2)
        if left.button("Cancel", width="stretch"):
            st.rerun()
        if right.button("Reset demo", type="primary", width="stretch"):
            state.reset_demo()
            st.session_state.pending_toast = "Demo restored to its starting state."
            st.rerun()

    dialog()


def _sidebar_bottom() -> None:
    demo = state.demo()
    with st.sidebar:
        st.divider()
        latest = demo.latest
        if latest:
            st.caption(
                f":material/schedule: Last batch scored: "
                f"**{latest.scored_at:%d %b %Y, %H:%M}** · {latest.name}"
                + (" · Demo data" if latest.is_demo else "")
            )
        else:
            st.caption(":material/schedule: No batch scored yet.")
        if st.button("Reset demo", icon=":material/restart_alt:", width="stretch"):
            _confirm_reset()


def _toasts() -> None:
    message = st.session_state.pop("pending_toast", None)
    if message:
        st.toast(message, icon=":material/info:")
    demo = state.demo()
    seen = st.session_state.toasted_alerts
    fresh = [a for a in demo.alerts if a.status == "New" and a.alert_id not in seen]
    if fresh:
        critical = sum(a.severity == "Critical" for a in fresh)
        st.toast(
            f"{len(fresh)} new alerts ({critical} critical). See Alerts for details.",
            icon=":material/notifications_active:",
            duration="long",
        )
        seen.update(a.alert_id for a in fresh)


def run() -> None:
    """Entry point called by ``app/streamlit_app.py``."""
    st.set_page_config(
        page_title=f"{BRAND_NAME} · Churn early warning",
        page_icon=str(BRAND_DIR / "murphys_symbol.svg"),
        layout="wide",
        initial_sidebar_state="auto",
    )
    ui.inject_css()
    st.logo(
        str(BRAND_DIR / "murphys_logo.svg"),
        icon_image=str(BRAND_DIR / "murphys_symbol.svg"),
        size="large",
    )

    try:
        resources.bundle()
    except ArtifactsMissingError as error:
        logger.error("%s", error)
        st.error(
            "The churn model is not available yet, so this app cannot score "
            "customers. Ask the data team to run the training pipeline.",
            icon=":material/error:",
        )
        st.caption(str(error))
        ui.footer(FOOTER_TEXT)
        st.stop()

    state.init()

    page_objects = {}
    sections: dict[str, list] = {}
    for spec in PAGE_SPECS:
        page = st.Page(
            _guarded(spec),
            title=spec.title,
            icon=spec.icon,
            url_path=spec.url,
            default=spec.url == "overview",
        )
        page_objects[spec.url] = page
        sections.setdefault(spec.section, []).append(page)
    st.session_state.pages = page_objects
    current = st.navigation(sections, position="sidebar", expanded=True)
    pending = st.session_state.pop("pending_nav", None)
    if pending:
        navigation.go_to(pending)
    _sidebar_bottom()
    _toasts()

    critical = [
        a for a in state.demo().alerts if a.severity == "Critical" and a.status == "New"
    ]
    if critical and current.url_path != "alerts":
        ui.alert_banner(len(critical), lambda: navigation.go_to("alerts"))

    current.run()
    ui.footer(FOOTER_TEXT)
