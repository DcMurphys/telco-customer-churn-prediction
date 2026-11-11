"""Act pages: Customer Check and Batch Scoring."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from app.models.validation import (
    DERIVED_IN_FORM,
    SCHEMA,
    CategoryField,
    NumberField,
)
from app.views.components import charts, ui
from app.views.glossary import label, tooltip

DECISION_AID_NOTE = (
    "This is a decision aid. Use your judgement and the customer's own words."
)
FORM_PREFIX = "keep_cc_"


# ===== Customer Check =====
def result_card(result: dict, reliability: dict) -> None:
    """Risk level, verdict, top reasons and suggested offers (C-4 to C-7)."""
    with st.container(border=True):
        st.caption(f"Result for **{result['customer']}**")
        st.html(
            ui.badge_html(result["risk_level"], f"{result['risk_level']} risk")
            + f'<div class="brand-verdict">{result["verdict"]}</div>'
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"**{label('importance')}**")
            if result["reasons"]:
                for i, reason in enumerate(result["reasons"], start=1):
                    st.markdown(f"{i}. {reason}")
            else:
                st.caption("No strong risk signals from the playbook reasons.")
        with right:
            st.markdown("**Suggested next steps**")
            for action in result["actions"] or [result["guidance"]]:
                st.markdown(f"- {action}")
        st.markdown("**Talking points** (use the copy icon to paste them into the CRM)")
        st.code(result["talking_points"], language=None, wrap_lines=True)
        st.caption(f":material/info: {DECISION_AID_NOTE}")

    with st.expander("How reliable is this?", icon=":material/verified:"):
        ui.reliability_panel(reliability)


def _form_field(spec: NumberField | CategoryField, errors: dict[str, str]) -> None:
    key = FORM_PREFIX + spec.name
    if isinstance(spec, NumberField):
        kwargs = {
            "min_value": int(spec.min_value) if spec.integer else float(spec.min_value),
            "max_value": int(spec.max_value) if spec.integer else float(spec.max_value),
            "step": 1 if spec.integer else float(spec.step),
        }
        if not spec.integer:
            kwargs["format"] = "%.2f"
        title = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
        st.number_input(
            title,
            key=key,
            help=f"Allowed range: {spec.min_value:g} to {spec.max_value:g}",
            **kwargs,
        )
    else:
        st.selectbox(spec.label, spec.options, key=key, help=spec.help or None)
    if spec.name in errors:
        st.error(errors[spec.name], icon=":material/error:")


def customer_form(
    defaults: dict,
    errors: dict[str, str],
    on_submit: Callable[[dict], None],
) -> None:
    """Manual entry form grouped Account → Plan & services → Billing (C-2, E-5)."""
    for name, value in defaults.items():
        ui.init_key(FORM_PREFIX + name, value)
    if errors:
        st.error(
            f"Please fix {len(errors)} field(s) below before scoring.",
            icon=":material/error:",
        )
    with st.form("customer_form", border=True):
        for group, fields in SCHEMA.groups.items():
            st.markdown(f"##### {group}")
            asked = [f for f in fields if f.name not in DERIVED_IN_FORM]
            columns = st.columns(3)
            for i, spec in enumerate(asked):
                with columns[i % 3]:
                    _form_field(spec, errors)
        st.caption(
            "Under 30, senior citizen, dependents, referral flag, internet service, "
            "total long-distance charges and total revenue are derived from your "
            "answers, so they cannot contradict them."
        )
        _, right = st.columns([3, 1])
        submitted = right.form_submit_button(
            "Review and score", type="primary", width="stretch"
        )
    if submitted:
        values = {
            spec.name: st.session_state[FORM_PREFIX + spec.name]
            for spec in SCHEMA.fields.values()
            if spec.name not in DERIVED_IN_FORM
        }
        on_submit(values)


def confirm_record(record: dict, on_confirm: Callable[[dict], None]) -> None:
    """Summary dialog listing every value before scoring (C-3)."""

    @st.dialog("Check the details before scoring", width="large")
    def dialog() -> None:
        rows = []
        for group, fields in SCHEMA.groups.items():
            for spec in fields:
                derived = " (derived)" if spec.name in DERIVED_IN_FORM else ""
                rows.append(
                    {
                        "Group": group,
                        "Field": spec.label + derived,
                        "Value": str(record[spec.name]),
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=420)
        left, right = st.columns(2)
        if left.button("Edit details", width="stretch"):
            st.rerun()
        if right.button("Confirm and score", type="primary", width="stretch"):
            on_confirm(record)
            st.rerun()

    dialog()


# ===== Batch Scoring =====
STEPS = ["Upload", "Check", "Confirm", "Results"]


def upload_step(
    template_csv: bytes,
    samples: dict[str, bytes],
    max_mb: int,
    max_rows: int,
    on_upload: Callable[[object], None],
) -> None:
    """Step 1: upload, template and sample files (B-2, I-4, I-5)."""
    st.warning(
        "Use sample or synthetic data only. Do not upload real personal data. "
        "Files are processed in memory and never stored.",
        icon=":material/privacy_tip:",
    )
    uploaded = st.file_uploader(
        f"Customer batch (CSV, up to {max_mb} MB and {max_rows:,} rows)",
        type=["csv"],
        max_upload_size=max_mb,
        key="batch_upload",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Template**")
        st.caption("Exact columns and one example row.")
        st.download_button(
            "Download template",
            template_csv,
            file_name="murphys_batch_template_demo-data.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    with c2:
        st.markdown("**Sample batches**")
        st.caption(
            "A file without errors, a file with deliberate input errors, and a "
            "high-risk file that triggers alerts."
        )
        for name, data in samples.items():
            st.download_button(
                name,
                data,
                file_name=name.replace(".csv", "_demo-data.csv"),
                mime="text/csv",
                icon=":material/description:",
                key=f"dl_{name}",
            )
    if uploaded is not None and st.button(
        "Check file", type="primary", icon=":material/fact_check:"
    ):
        on_upload(uploaded)


def check_step(
    flow: dict,
    errors_csv: bytes | None,
    on_recheck: Callable[[pd.DataFrame], None],
    on_continue: Callable[[], None],
    on_restart: Callable[[], None],
    column_config: dict,
) -> None:
    """Step 2: summary + row-level errors + optional fixing (B-3 to B-5)."""
    check = flow["check"]
    if flow.get("read_error"):
        st.error(flow["read_error"], icon=":material/error:")
        st.button("Upload another file", on_click=on_restart)
        return
    if check.missing_columns:
        st.error(
            "The file is missing required columns, so rows cannot be checked: "
            + ", ".join(check.missing_columns)
            + ". Download the template to see the exact layout.",
            icon=":material/error:",
        )
        st.button("Upload another file", on_click=on_restart)
        return
    if check.n_rows == 0:
        st.error("The file has no data rows.", icon=":material/error:")
        st.button("Upload another file", on_click=on_restart)
        return

    ready = len(check.valid_rows)
    if check.n_invalid:
        st.warning(
            f"**{ready:,} rows ready · {check.n_invalid:,} rows need fixing.** "
            "Invalid rows are never dropped or changed automatically.",
            icon=":material/rule:",
        )
    else:
        st.success(
            f"**All {ready:,} rows passed every check.**",
            icon=":material/check_circle:",
        )
    if check.unknown_columns:
        st.info(
            "These columns are not used and will be ignored: "
            + ", ".join(check.unknown_columns),
            icon=":material/info:",
        )

    if check.n_invalid:
        st.markdown("**Rows that need fixing**")
        st.caption("Row = data row number in the file (the header row is not counted).")
        st.dataframe(check.errors, hide_index=True)
        st.download_button(
            "Download rows to fix (CSV)",
            errors_csv,
            file_name="murphys_rows_to_fix_demo-data.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        with st.expander("Fix the flagged rows here", icon=":material/edit:"):
            st.caption("Only flagged rows are shown. Edit, then re-check.")
            edited = st.data_editor(
                flow["invalid_rows"],
                key=f"fix_editor_{flow['upload_id']}",
                column_config=column_config,
                num_rows="fixed",
            )
            if st.button("Re-check fixed rows", icon=":material/refresh:"):
                on_recheck(edited)

    left, right = st.columns([1, 1])
    left.button(
        "Upload another file", on_click=on_restart, icon=":material/arrow_back:"
    )
    if ready:
        label_text = (
            f"Continue with {ready:,} valid rows"
            if check.n_invalid
            else f"Continue with {ready:,} rows"
        )
        right.button(
            label_text,
            type="primary",
            on_click=on_continue,
            icon=":material/arrow_forward:",
        )


def confirm_step(
    n_rows: int,
    existing_names: list[str],
    on_confirm: Callable[[str, object], None],
    on_back: Callable[[], None],
) -> None:
    """Step 3: required batch name and date, then confirmation (B-6, E-3)."""
    with st.form("batch_meta", border=True):
        name = st.text_input(
            "Batch name (required)", max_chars=40, placeholder="e.g. Nov-W3"
        )
        batch_date = st.date_input("Batch date (required)", format="DD/MM/YYYY")
        submitted = st.form_submit_button("Score batch", type="primary")
    st.button("Back to check", on_click=on_back, icon=":material/arrow_back:")
    if submitted:
        clean = name.strip()
        if not clean:
            st.error("Enter a batch name.", icon=":material/error:")
        elif clean.lower() in {n.lower() for n in existing_names}:
            st.error(
                f"A batch named “{clean}” already exists. Choose another name.",
                icon=":material/error:",
            )
        elif batch_date is None:
            st.error("Choose the batch date.", icon=":material/error:")
        else:
            _confirm_batch(clean, batch_date, n_rows, on_confirm)


def _confirm_batch(name, batch_date, n_rows, on_confirm) -> None:
    @st.dialog("Score this batch?")
    def dialog() -> None:
        st.write(
            f"**{name}** ({batch_date:%d %b %Y}) with **{n_rows:,} customers** will "
            "be scored and added to the batch history. Alerts are checked "
            "automatically."
        )
        left, right = st.columns(2)
        if left.button("Cancel", width="stretch"):
            st.rerun()
        if right.button("Score batch", type="primary", width="stretch"):
            with st.spinner("Scoring customers…"):
                on_confirm(name, batch_date)
            st.rerun()

    dialog()


def results_step(
    summary: dict,
    risk_dist: pd.DataFrame,
    plan_table: pd.DataFrame,
    raised: pd.DataFrame,
    scored_csv: bytes,
    on_restart: Callable[[], None],
    go: Callable[[str], None],
) -> None:
    """Step 4: KPIs, risk distribution, plans, alerts and download (B-7)."""
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.kpi_card(
            label("predicted_rate"),
            ui.pct(summary["rate"]),
            "Average chance of leaving in this batch.",
            ui.pp_delta(summary["rate"], summary["prev_rate"]),
            delta_color="inverse",
            help_text=tooltip("predicted_rate"),
        )
    with c2:
        ui.kpi_card(
            label("predicted_positive"),
            f"{summary['flagged']:,}",
            f"{summary['flagged'] / summary['customers']:.0%} of "
            f"{summary['customers']:,} customers.",
            help_text=tooltip("predicted_positive"),
        )
    with c3:
        ui.kpi_card(
            "Alerts raised",
            f"{len(raised)}",
            f"{int((raised['severity'] == 'Critical').sum())} critical.",
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Customers by risk level")
        ui.chart_block(
            charts.risk_distribution_chart(risk_dist),
            "How many customers fall in each risk level.",
            risk_dist,
            "risk_dist",
        )
    with right:
        st.subheader("Plans in this batch")
        st.dataframe(plan_table, hide_index=True, height=300)

    st.subheader("Alerts raised by this batch")
    if raised.empty:
        st.success("No thresholds were crossed.", icon=":material/check_circle:")
    else:
        st.dataframe(
            raised.assign(severity=raised["severity"].map(ui.badge_text))[
                ["severity", "rule", "detail"]
            ].rename(columns=str.capitalize),
            hide_index=True,
        )
        st.button(
            "Open Alerts",
            on_click=go,
            args=("alerts",),
            icon=":material/notifications:",
        )

    st.download_button(
        "Download scored customers (CSV)",
        scored_csv,
        file_name=f"murphys_scored_{summary['name'].replace(' ', '_')}_demo-data.csv",
        mime="text/csv",
        type="primary",
        icon=":material/download:",
    )
    st.button("Score another batch", on_click=on_restart, icon=":material/restart_alt:")
