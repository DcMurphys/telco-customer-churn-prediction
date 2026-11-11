"""Help and Settings pages: Glossary & Guide and Alert Settings."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from app.views.components import ui
from app.views.glossary import TERMS

GUIDES = {
    "John · customer service agent": [
        "A customer calls: open **Customer Check**, pick the customer ID or enter "
        "their details, and read the risk level, top reasons and suggested offer.",
        "Use the copy icon on the talking points to paste them into the CRM.",
        "Preparing the week: open **Plan Risk** to see which plans churn most and "
        "the suggested conversation points for each.",
    ],
    "Alex · customer relations supervisor": [
        "Start of shift: the **Overview** shows the latest batch, the trend and a "
        "banner if a critical alert is open.",
        "Open **Alerts**, acknowledge what you are following up, resolve what is "
        "done, and export the list for team reports.",
        "New batch: **Batch Scoring** checks every row before scoring. Fix the "
        "flagged rows or download them, then confirm.",
        "Thresholds live in **Alert Settings**; every change is logged with your name.",
        "**Churn Insights** explains who leaves and when, with suggested next steps.",
    ],
}
LIMITATIONS = [
    "Results show links in past data, not proven causes. Test any action with a "
    "control group before rolling it out.",
    "The data is a fictional sample covering one quarter; real carriers lose far "
    "fewer customers.",
    "Customers on one- or two-year contracts who leave are often missed, so do "
    "not rely on a Low risk level alone for them.",
    "There is no login: Alert Settings and Alerts record the name you type.",
    "Demo changes (batches, alerts, settings) stay in your browser session and "
    "are lost when it ends.",
]


def glossary(
    rule_descriptions: dict[str, str], risk_rules: list[dict], reliability: dict
) -> None:
    """Glossary, guides, reliability, alert rules and limitations (Section 5.10)."""
    st.title("Glossary & Guide")
    st.caption("The words this app uses, and how to use it in two minutes.")

    st.subheader("Glossary")
    st.dataframe(
        pd.DataFrame(
            {
                "Term": [t.label for t in TERMS.values()],
                "What it means": [t.definition for t in TERMS.values()],
            }
        ),
        hide_index=True,
    )

    st.subheader("Two-minute guides")
    tabs = st.tabs(list(GUIDES))
    for tab, steps in zip(tabs, GUIDES.values(), strict=True):
        with tab:
            for i, step in enumerate(steps, start=1):
                st.markdown(f"{i}. {step}")

    st.subheader("Risk levels")
    st.dataframe(pd.DataFrame(risk_rules), hide_index=True)
    ui.reliability_panel(reliability)

    st.subheader("Alert rules")
    for rule, text in rule_descriptions.items():
        st.markdown(f"- **{rule}:** {text}")

    st.subheader("Limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")


def alert_settings(
    current: dict,
    defaults: dict,
    observed_rate: float,
    preview: Callable[[dict], pd.DataFrame | dict],
    on_save: Callable[[dict, str], int],
    on_reset: Callable[[str], int],
    audit_log: pd.DataFrame,
) -> None:
    """Alert Settings with validation, preview and audit log (S-1 to S-4)."""
    st.title("Alert Settings")
    st.caption("Thresholds used to raise alerts when a batch is scored.")
    st.info(
        f"Reference point: the IBM sample's observed quarterly churn is "
        f"{observed_rate:.1%}. This is context, not a recommended target.",
        icon=":material/info:",
    )

    with st.form("alert_settings", border=True):
        c1, c2 = st.columns(2)
        watch = c1.number_input(
            "Watch threshold (%)",
            0.0,
            100.0,
            100 * current["watch_threshold"],
            0.5,
            help="Predicted churn rate of a batch that raises a Watch alert.",
        )
        critical = c2.number_input(
            "Critical threshold (%)",
            0.0,
            100.0,
            100 * current["critical_threshold"],
            0.5,
            help="Predicted churn rate of a batch that raises a Critical alert.",
        )
        c3, c4 = st.columns(2)
        gap = c3.number_input(
            "Plan warning gap (percentage points)",
            0.5,
            100.0,
            current["plan_gap_pp"],
            0.5,
            help="How far above the batch average a plan must be to raise a warning.",
        )
        min_n = c4.number_input(
            "Minimum group size (customers)",
            1,
            10_000,
            int(current["min_group_size"]),
            1,
            help="Smaller plan groups never raise alerts and are greyed out in "
            "Plan Risk.",
        )
        changed_by = st.text_input(
            "Changed by (required)", max_chars=60, key="keep_settings_by"
        )
        b1, b2 = st.columns(2)
        preview_clicked = b1.form_submit_button(
            "Preview impact", icon=":material/visibility:"
        )
        save_clicked = b2.form_submit_button("Save settings", type="primary")

    candidate = {
        "watch_threshold": watch / 100,
        "critical_threshold": critical / 100,
        "plan_gap_pp": float(gap),
        "min_group_size": int(min_n),
    }
    result = preview(candidate)
    if isinstance(result, dict):  # validation errors
        for message in result.values():
            st.error(message, icon=":material/error:")
    elif preview_clicked or save_clicked:
        st.markdown("**Preview: past batches that would have triggered each rule**")
        st.dataframe(result, hide_index=True)

    if save_clicked and not isinstance(result, dict):
        if not changed_by.strip():
            st.error(
                "Enter your name in “Changed by” to save.", icon=":material/error:"
            )
        elif candidate == current:
            st.info("Nothing changed.", icon=":material/info:")
        else:
            _confirm_save(candidate, changed_by.strip(), on_save)

    if st.button("Reset to defaults", icon=":material/restart_alt:"):
        _confirm_reset(defaults, st.session_state.get("keep_settings_by", ""), on_reset)

    st.subheader("Audit log")
    if audit_log.empty:
        st.caption("No changes yet in this session.")
    else:
        st.dataframe(
            audit_log,
            hide_index=True,
            column_config={
                "Time": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm:ss")
            },
        )


def _confirm_save(candidate: dict, changed_by: str, on_save) -> None:
    @st.dialog("Save new alert settings?")
    def dialog() -> None:
        st.write(
            f"Watch {candidate['watch_threshold']:.1%} · Critical "
            f"{candidate['critical_threshold']:.1%} · Plan gap "
            f"{candidate['plan_gap_pp']:g} pp · Minimum group "
            f"{candidate['min_group_size']} customers"
        )
        st.caption("New settings apply to batches scored from now on.")
        left, right = st.columns(2)
        if left.button("Cancel", width="stretch"):
            st.rerun()
        if right.button("Save", type="primary", width="stretch"):
            n = on_save(candidate, changed_by)
            st.session_state.pending_toast = (
                f"Alert settings saved: {n} field(s) changed."
            )
            st.rerun()

    dialog()


def _confirm_reset(defaults: dict, changed_by: str, on_reset) -> None:
    @st.dialog("Reset alert settings to defaults?")
    def dialog() -> None:
        st.write(
            f"Watch {defaults['watch_threshold']:.0%} · Critical "
            f"{defaults['critical_threshold']:.0%} · Plan gap "
            f"{defaults['plan_gap_pp']:g} pp · Minimum group "
            f"{defaults['min_group_size']} customers"
        )
        name = st.text_input("Changed by (required)", value=changed_by, max_chars=60)
        left, right = st.columns(2)
        if left.button("Cancel", width="stretch"):
            st.rerun()
        if right.button(
            "Reset", type="primary", width="stretch", disabled=not name.strip()
        ):
            n = on_reset(name.strip())
            st.session_state.pending_toast = f"Defaults restored: {n} field(s) changed."
            st.rerun()

    dialog()
