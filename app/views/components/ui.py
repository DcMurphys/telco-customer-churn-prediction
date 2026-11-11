"""Reusable building blocks: badges, KPI cards, captions, step indicator, banner."""

from __future__ import annotations

from collections.abc import Callable

import altair as alt
import pandas as pd
import streamlit as st

from telco_churn.config import PALETTE

# Status is never colour alone: label + icon + colour (Section 3.3)
STATUS_STYLE = {
    "Critical": ("red", "▲"),
    "High": ("red", "▲"),
    "Watch": ("orange", "●"),
    "Medium": ("orange", "●"),
    "OK": ("green", "✓"),
    "Low": ("green", "✓"),
    "Resolved": ("green", "✓"),
    "New": ("blue", "◆"),
    "Acknowledged": ("gray", "○"),
}

GLOBAL_CSS = f"""
<style>
  [data-testid="stMetricValue"], [data-testid="stDataFrame"], .tabular {{
    font-variant-numeric: tabular-nums;
  }}
  *:focus-visible {{
    outline: 2px solid {PALETTE["brand-orange-dark"]} !important;
    outline-offset: 2px;
  }}
  .brand-badge {{
    display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; line-height: 1.4; white-space: nowrap;
  }}
  .brand-badge.critical {{ background: {PALETTE["status-critical"]}; color: #fff; }}
  .brand-badge.watch {{
    background: {PALETTE["status-watch"]}; color: {PALETTE["ink"]};
  }}
  .brand-badge.ok {{ background: {PALETTE["status-ok"]}; color: #fff; }}
  .brand-badge.neutral {{
    background: {PALETTE["border"]}; color: {PALETTE["ink"]};
  }}
  .brand-verdict {{ font-size: 1.15rem; font-weight: 600; margin: 0.25rem 0 0.5rem; }}
  .brand-steps {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem; }}
  .brand-step {{
    padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.85rem;
    border: 1px solid {PALETTE["border"]}; color: {PALETTE["slate"]};
  }}
  .brand-step.done {{
    color: {PALETTE["status-ok"]}; border-color: {PALETTE["status-ok"]};
  }}
  .brand-step.current {{
    background: {PALETTE["brand-orange"]}; color: {PALETTE["ink"]};
    border-color: {PALETTE["brand-orange"]}; font-weight: 600;
  }}
  .brand-footer {{ color: {PALETTE["slate"]}; font-size: 0.75rem; margin-top: 2rem; }}
</style>
"""

BADGE_CLASS = {
    "Critical": "critical",
    "High": "critical",
    "Watch": "watch",
    "Medium": "watch",
    "OK": "ok",
    "Low": "ok",
    "Resolved": "ok",
}


def inject_css() -> None:
    """Apply the small global style additions (tabular figures, focus ring)."""
    st.html(GLOBAL_CSS)


def badge_html(status: str, text: str | None = None) -> str:
    """HTML badge with icon + label + colour (used in markdown and cards)."""
    _, icon = STATUS_STYLE.get(status, ("gray", "•"))
    css = BADGE_CLASS.get(status, "neutral")
    return f'<span class="brand-badge {css}">{icon} {text or status}</span>'


def badge_text(status: str) -> str:
    """Plain-text badge for tables (icon + label)."""
    _, icon = STATUS_STYLE.get(status, ("gray", "•"))
    return f"{icon} {status}"


def show_badge(status: str, text: str | None = None) -> None:
    """Render one status badge."""
    st.html(badge_html(status, text))


def kpi_card(
    label: str,
    value: str,
    caption: str,
    delta: str | None = None,
    delta_color: str = "normal",
    help_text: str | None = None,
) -> None:
    """Bordered KPI card: metric, delta and a one-line caption.

    ``height="stretch"`` makes every card in a row as tall as the tallest one,
    whether or not it has a delta or a longer caption.
    """
    with st.container(border=True, height="stretch"):
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)
        st.caption(caption)


def chart_block(
    chart: alt.Chart, what_this_shows: str, data: pd.DataFrame, key: str
) -> None:
    """Chart + "What this shows" caption + "Show data" table (G-4)."""
    _chart_body(chart, what_this_shows)
    _show_data(data, key)


def _chart_body(chart: alt.Chart, what_this_shows: str) -> None:
    st.altair_chart(chart, width="stretch")
    st.caption(f"**What this shows:** {what_this_shows}")


def _show_data(data: pd.DataFrame, key: str) -> None:
    with st.expander("Show data", icon=":material/table:"):
        st.dataframe(data, hide_index=True, key=f"{key}_data")


def chart_card(
    title: str, chart: alt.Chart, what_this_shows: str, data: pd.DataFrame, key: str
) -> None:
    """Chart block in a bordered card that matches the height of its row.

    The card holds exactly two blocks, and ``vertical_alignment="distribute"``
    pins the second one ("Show data") to the bottom, so the expanders line up
    across cards even when the captions have different lengths.
    """
    with st.container(border=True, height="stretch", vertical_alignment="distribute"):
        with st.container():
            st.markdown(f"**{title}**")
            _chart_body(chart, what_this_shows)
        _show_data(data, key)


def reliability_panel(metrics: dict) -> None:
    """Plain-language summary of model quality on unseen customers."""
    with st.container(border=True):
        st.markdown("**How reliable are these predictions?**")
        st.markdown(
            f"- Catches **{metrics['recall']:.0%}** of customers who leave "
            f"(about {round(10 * metrics['recall'])} in 10).\n"
            f"- **{metrics['precision']:.0%}** of flagged customers actually leave "
            f"(about {round(10 * metrics['precision'])} in 10).\n"
            f"- Calls **{metrics['contacted']:.0%}** of customers, vs "
            f"{metrics['rule_contacted']:.0%} if we called every month-to-month "
            "customer, while catching almost as many leavers."
        )
        st.caption(
            f"Tested on {metrics['tested']:,} customers the model never saw. "
            "Use it as a decision aid, not a certainty."
        )


def step_indicator(steps: list[str], current: int) -> None:
    """Numbered step chips; ``current`` is 0-based."""
    chips = []
    for i, step in enumerate(steps):
        state = "done" if i < current else "current" if i == current else ""
        marker = "✓" if i < current else str(i + 1)
        aria = ' aria-current="step"' if i == current else ""
        chips.append(f'<span class="brand-step {state}"{aria}>{marker} {step}</span>')
    st.html(f'<nav class="brand-steps" aria-label="Progress">{"".join(chips)}</nav>')


def alert_banner(n_critical: int, go_to_alerts: Callable[[], None]) -> None:
    """Top banner while unacknowledged critical alerts exist (requirement G-2)."""
    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        noun = "alert needs" if n_critical == 1 else "alerts need"
        left.html(
            badge_html("Critical")
            + f" &nbsp;<b>{n_critical} critical {noun} attention.</b> "
            "Open Alerts to acknowledge or resolve them."
        )
        if right.button(
            "Open alerts", icon=":material/notifications:", key="banner_go"
        ):
            go_to_alerts()


def empty_state(
    title: str, message: str, button: str | None = None, on_click=None, key: str = ""
) -> None:
    """Designed empty state with an optional call to action (requirement G-5)."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(message)
        if button and on_click:
            st.button(button, on_click=on_click, type="primary", key=f"empty_{key}")


def footer(text: str) -> None:
    """Global footer (requirement G-3)."""
    st.html(f'<div class="brand-footer">{text}</div>')


def pct(value: float | None, digits: int = 1) -> str:
    """Format a share as a percentage."""
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def pp_delta(current: float | None, previous: float | None) -> str | None:
    """Delta in percentage points for st.metric."""
    if current is None or previous is None:
        return None
    return f"{100 * (current - previous):+.1f} pp"


def init_key(key: str, value: object) -> None:
    """Set a kept widget's starting value once (instead of ``default=``)."""
    if key not in st.session_state:
        st.session_state[key] = value
