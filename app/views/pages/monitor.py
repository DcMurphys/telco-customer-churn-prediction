"""Monitor pages: Overview, Alerts and Plan Risk."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from app.views.components import charts, ui
from app.views.glossary import label, tooltip
from app.views.playbook import talking_points_for


# ===== Overview =====
def overview(
    kpis: dict,
    trend: pd.DataFrame,
    thresholds: tuple[float, float],
    top_plans: pd.DataFrame,
    sentence: str | None,
    go: Callable[[str], None],
) -> None:
    """Render the Overview page (requirements O-1 to O-5)."""
    st.title("Overview")
    st.caption(
        "Is churn under control right now, and what needs attention? "
        "Changes are shown against the previous batch."
    )

    if sentence:
        st.info(sentence, icon=":material/summarize:")

    top_row, bottom_row = st.columns(2), st.columns(2)
    cols = [*top_row, *bottom_row]
    with cols[0]:
        ui.kpi_card(
            label("predicted_rate"),
            ui.pct(kpis["rate"]),
            "Average chance of leaving in the latest batch.",
            ui.pp_delta(kpis["rate"], kpis["prev_rate"]),
            delta_color="inverse",
            help_text=tooltip("predicted_rate"),
        )
    with cols[1]:
        delta = (
            None
            if kpis["prev_flagged"] is None
            else f"{kpis['flagged'] - kpis['prev_flagged']:+,}"
        )
        ui.kpi_card(
            label("predicted_positive"),
            f"{kpis['flagged']:,}",
            f"{kpis['flagged_share']:.0%} of {kpis['customers']:,} customers are "
            "above the risk cut-off.",
            delta,
            delta_color="inverse",
            help_text=tooltip("predicted_positive"),
        )
    with cols[2]:
        delta = (
            None
            if kpis["prev_alerts"] is None
            else f"{kpis['batch_alerts'] - kpis['prev_alerts']:+} raised"
        )
        ui.kpi_card(
            "Active alerts",
            f"{kpis['active_alerts']}",
            f"Not yet resolved, {kpis['critical_alerts']} critical.",
            delta,
            delta_color="inverse",
        )
    with cols[3]:
        ui.kpi_card(
            "Riskiest plan",
            (kpis["riskiest_plan"] or "—").split(": ")[-1],
            f"{(kpis['riskiest_plan'] or '').split(': ')[0]} · "
            f"{ui.pct(kpis['riskiest_rate'])} {label('predicted_rate').lower()}, "
            "among groups large enough to compare.",
            ui.pp_delta(kpis["riskiest_rate"], kpis["prev_riskiest_rate"]),
            delta_color="inverse",
        )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Trend per scored batch")
        ui.chart_block(
            charts.trend_chart(trend, *thresholds),
            f"{label('predicted_rate')} for every scored batch. Dashed lines "
            "are the watch and critical thresholds; triangles mark critical breaches.",
            trend.assign(Rate=(100 * trend["Rate"]).round(1)).rename(
                columns={"Rate": "Rate (%)"}
            ),
            "trend",
        )
    with right:
        st.subheader("Top 3 plans to watch")
        for _, row in top_plans.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['plan']}**")
                st.caption(
                    f"{ui.pct(row['rate'])} {label('predicted_rate').lower()} · "
                    f"{row['customers']:,} customers · "
                    f"{row['gap_pp']:+.1f} percentage points vs batch average"
                )
        st.button(
            "Open Plan Risk",
            icon=":material/arrow_forward:",
            on_click=go,
            args=("plan-risk",),
        )


# ===== Alerts =====
def alerts(
    table: pd.DataFrame,
    rules: list[str],
    export_csv: bytes,
    on_change_status: Callable[[str, str, str, str], None],
    rule_descriptions: dict[str, str],
) -> None:
    """Render the Alerts page (requirements A-1 to A-5)."""
    st.title("Alerts")
    st.caption("Alerts raised by scored batches. Acknowledge or resolve them here.")
    if table.empty:
        ui.empty_state(
            "No alerts yet",
            "Alerts appear here when a scored batch crosses a threshold.",
        )
        return

    ui.init_key("keep_alert_sev", ["Critical", "Watch"])
    ui.init_key("keep_alert_rule", list(rules))
    ui.init_key("keep_alert_status", ["New", "Acknowledged"])
    f1, f2, f3 = st.columns([1, 2, 1.4])
    severities = f1.pills(
        "Severity", ["Critical", "Watch"], selection_mode="multi", key="keep_alert_sev"
    )
    chosen_rules = f2.pills(
        "Rule", rules, selection_mode="multi", key="keep_alert_rule"
    )
    statuses = f3.pills(
        "Status",
        ["New", "Acknowledged", "Resolved"],
        selection_mode="multi",
        key="keep_alert_status",
        help="Default: not resolved",
    )
    shown = table[
        table["severity"].isin(severities or [])
        & table["rule"].isin(chosen_rules or [])
        & table["status"].isin(statuses or [])
    ]
    st.caption(f"Showing {len(shown)} of {len(table)} alerts.")
    display = pd.DataFrame(
        {
            "ID": shown["alert_id"],
            "Time": shown["raised_at"],
            "Severity": shown["severity"].map(ui.badge_text),
            "Rule": shown["rule"],
            "Batch": shown["batch"],
            "Value": shown["value"],
            "Threshold": shown["threshold"],
            "Detail": shown["detail"],
            "Status": shown["status"].map(ui.badge_text),
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        column_config={
            "Time": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
            "Value": st.column_config.NumberColumn(
                format="percent", help="Predicted churn rate that triggered the rule"
            ),
            "Threshold": st.column_config.NumberColumn(format="percent"),
            "Detail": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "Export alerts (CSV)",
        export_csv,
        file_name="murphys_alerts_demo-data.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.subheader("Follow up an alert")
    open_ids = shown.loc[shown["status"] != "Resolved", "alert_id"].tolist()
    if not open_ids:
        st.caption("No open alerts in the current filter.")
    else:
        details = shown.set_index("alert_id")
        with st.form("alert_action", border=True):
            alert_id = st.selectbox(
                "Alert",
                open_ids,
                format_func=lambda a: (
                    f"{a} · {details.loc[a, 'severity']} · "
                    f"{details.loc[a, 'rule']} · {details.loc[a, 'batch']}"
                ),
            )
            action = st.radio("Action", ["Acknowledge", "Resolve"], horizontal=True)
            note = st.text_input("Note (optional)", max_chars=200)
            changed_by = st.text_input(
                "Your name",
                max_chars=60,
                key="keep_alert_by",
                help="Recorded in the alert's history (no login in this prototype).",
            )
            submitted = st.form_submit_button("Apply", type="primary")
        if submitted:
            status = "Acknowledged" if action == "Acknowledge" else "Resolved"
            _confirm_status(alert_id, status, note, changed_by, on_change_status)

    with st.expander("What each rule means", icon=":material/help:"):
        for rule, text in rule_descriptions.items():
            st.markdown(f"**{rule}:** {text}")


def _confirm_status(alert_id, status, note, changed_by, on_change_status) -> None:
    @st.dialog(f"{status[:-1] if status.endswith('d') else status} alert {alert_id}?")
    def dialog() -> None:
        st.write(f"The alert's status will change to **{status}**.")
        if note:
            st.write(f"Note: {note}")
        left, right = st.columns(2)
        if left.button("Cancel", width="stretch"):
            st.rerun()
        if right.button("Confirm", type="primary", width="stretch"):
            on_change_status(alert_id, status, note, changed_by)
            st.rerun()

    dialog()


# ===== Plan Risk =====
def plan_risk(
    attributes: list[str],
    sources: list[str],
    get_table: Callable[[str, str], dict],
    combination: Callable[[str, str, str], pd.DataFrame],
    batch_available: bool,
) -> None:
    """Render the Plan Risk page (requirements P-1 to P-5, P-7)."""
    st.title("Plan Risk")
    st.caption("Which subscription plans make customers leave?")

    ui.init_key("keep_plan_attr", attributes[0])
    ui.init_key("keep_plan_source", sources[0])
    c1, c2 = st.columns([1, 1])
    attribute = c1.segmented_control(
        "Plan attribute", attributes, required=True, key="keep_plan_attr"
    )
    source = c2.segmented_control(
        "Data source",
        sources,
        required=True,
        key="keep_plan_source",
        help="Past quarter: customers who actually left. Latest batch: expected "
        "share leaving, predicted for the newest scored batch.",
    )
    if source == sources[1] and not batch_available:
        ui.empty_state(
            "No batch scored yet", "Score a batch to see expected churn by plan."
        )
        return

    result = get_table(attribute, source)
    table: pd.DataFrame = result["table"]
    rate_name = "Share who left" if source == sources[0] else label("predicted_rate")
    st.markdown(
        f"**Source:** {source} · {result['customers']:,} customers · overall "
        f"{rate_name.lower()} {ui.pct(result['overall'])}"
    )

    chart_table = table.assign(Status=table["status"].fillna("—"))
    ui.chart_block(
        charts.plan_bar_chart(chart_table, result["overall"], False, rate_name),
        f"{rate_name} per {attribute.lower()} group, highest first. The dashed line "
        f"is the overall average; grey bars have fewer than {result['min_group']} "
        "customers and are too small to compare.",
        _plan_display_table(table),
        "plan",
    )

    st.subheader("Plans and what to do")
    for _, row in table.iterrows():
        status = row["status"]
        header = (
            f"{row['group']} · {ui.pct(row['rate'])} · {row['customers']:,} customers"
        )
        if row["too_small"]:
            header += " · Too few customers to compare"
        with st.expander(header, expanded=bool(status)):
            if status:
                ui.show_badge(status, f"{status}: plan warning in the latest batch")
            if row["too_small"]:
                st.caption("Too few customers to compare. Treat this rate with care.")
            st.markdown("**What to do**")
            for point in talking_points_for(attribute, row["group"]):
                st.markdown(f"- {point}")

    with st.expander("Compare two plan attributes", icon=":material/grid_on:"):
        rows_attr, cols_attr = st.columns(2)
        ui.init_key("keep_combo_rows", attributes[0])
        ui.init_key("keep_combo_cols", attributes[2])
        first = rows_attr.selectbox("Rows", attributes, key="keep_combo_rows")
        second = cols_attr.selectbox("Columns", attributes, key="keep_combo_cols")
        if first == second:
            st.caption("Choose two different attributes.")
        else:
            combo = combination(first, second, source)
            ui.chart_block(
                charts.heatmap(combo, first, second, rate_name),
                f"{rate_name} for each combination, with the number of customers.",
                combo,
                "combo",
            )


def _plan_display_table(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Group": table["group"],
            "Customers": table["customers"],
            "Rate (%)": (100 * table["rate"]).round(1),
            "Gap vs average (pp)": table["gap_pp"].round(1),
            "Too few to compare": table["too_small"],
            "Plan warning": table["status"].fillna(""),
        }
    )
