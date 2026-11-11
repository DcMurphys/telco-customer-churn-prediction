"""Churn Insights: exploratory dashboard for business users.

Tells the story of notebook Section 4.4 (who leaves, when, and what tends to
keep them) with ranked bars and plain-language takeaways. It does not explain
data preparation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views.components import charts, ui

CAUSALITY_NOTE = (
    "These are links in historical data, not proven causes. Test any action "
    "with a control group before rolling it out."
)


def _kpis(cards: list[tuple[str, str, str]]) -> None:
    for col, (label, value, caption) in zip(st.columns(len(cards)), cards, strict=True):
        with col:
            ui.kpi_card(label, value, caption)


def churn_insights(
    stats: dict,
    leaderboard: pd.DataFrame,
    tenure: pd.DataFrame,
    add_ons: pd.DataFrame,
    takeaways: list[dict],
) -> None:
    """Story-first page for John and Alex."""
    st.title("Churn Insights")
    st.caption("Who leaves, when they leave, and what tends to keep them.")

    _kpis(
        [
            ("Customers", f"{stats['customers']:,}", "In the sample, one quarter."),
            (
                "Customers who left",
                f"{stats['churned']:,}",
                f"{stats['churn_rate']:.1%} of all customers, about 1 in 4.",
            ),
            (
                "Revenue from leavers",
                f"${stats['revenue_lost'] / 1e6:.2f}M",
                f"{stats['revenue_share']:.0%} of total revenue came from "
                "customers who later left.",
            ),
            (
                "Months before leaving",
                f"{stats['tenure_left']:.0f}",
                f"Typical leaver, vs {stats['tenure_stayed']:.0f} months for "
                "customers who stayed.",
            ),
        ]
    )

    left, right = st.columns(2, gap="medium")
    with left:
        ui.chart_card(
            "Who is most likely to leave",
            charts.ranked_rate_bars(
                leaderboard, "segment", stats["churn_rate"], "Share who left"
            ),
            "The ten customer groups with the highest share leaving (groups of "
            "at least 100 customers). The dashed line is the average.",
            leaderboard.assign(rate=(100 * leaderboard["rate"]).round(1))[
                ["segment", "rate", "customers"]
            ].rename(
                columns={
                    "segment": "Group",
                    "rate": "Share who left (%)",
                    "customers": "Customers",
                }
            ),
            "eda_leaderboard",
        )
    with right:
        ui.chart_card(
            "When customers leave",
            charts.tenure_chart(tenure, stats["churn_rate"]),
            "Share leaving by months as a customer. More than half of customers "
            "in their first six months left; risk falls steadily after year one.",
            tenure.assign(rate=(100 * tenure["rate"]).round(1))[
                ["group", "rate", "customers"]
            ].rename(
                columns={
                    "group": "Months as a customer",
                    "rate": "Share who left (%)",
                    "customers": "Customers",
                }
            ),
            "eda_tenure",
        )

    internet_only = add_ons[add_ons["Scope"] == "Internet customers"]
    ui.chart_card(
        "Do add-ons go with staying? (internet customers)",
        charts.add_on_dumbbell(internet_only, facet_scope=False),
        "Share leaving with (green) and without (orange) each add-on, among "
        "internet customers only. Security and support add-ons show the biggest "
        "gap; streaming and unlimited data make little difference.",
        internet_only.round(3),
        "eda_addons",
    )

    st.subheader("What this means")
    rows = [takeaways[i : i + 2] for i in range(0, len(takeaways), 2)]
    for row in rows:
        for col, item in zip(st.columns(2), row, strict=False):
            with col, st.container(border=True, height="stretch"):
                st.markdown(f"**{item['title']}**")
                st.write(item["finding"])
                st.markdown(f":material/arrow_forward: **Next step:** {item['action']}")
    st.caption(CAUSALITY_NOTE)
