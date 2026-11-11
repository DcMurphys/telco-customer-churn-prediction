"""Page controllers: gather data from models, pass it to views, handle actions."""

from __future__ import annotations

import io
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from app.controllers import resources, state
from app.models import analysis, plans, scoring
from app.models.alerts import (
    RULE_DESCRIPTIONS,
    RULES,
    AlertSettings,
    alerts_table,
    change_status,
    preview_rules,
    summarise_batch,
)
from app.models.artifacts import read_upload_csv
from app.models.validation import (
    ID_COL,
    SCHEMA,
    UPLOAD_COLS,
    CategoryField,
    check_batch,
    check_record,
    complete_form_record,
)
from app.views.components import ui
from app.views.pages import act, explore, help_settings, monitor
from telco_churn.config import (
    DEFAULT_ALERT_SETTINGS,
    DEMO_LABEL,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_ROWS,
    OBSERVED_CHURN_RATE,
)

SOURCES = ["Historical (observed churn)", "Latest batch (predicted churn)"]


def _csv(data: pd.DataFrame) -> bytes:
    """CSV export carrying the demo label as a header comment (I-8)."""
    buffer = io.StringIO()
    buffer.write(f"# {DEMO_LABEL}\n")
    data.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _policy() -> dict:
    policy = resources.bundle().policy
    return {
        "recall_threshold": policy.recall_threshold,
        "breakeven_threshold": policy.breakeven_threshold,
        "target_recall": policy.target_recall,
    }


def _reliability() -> dict:
    """Plain-language model quality from the test report (unseen customers)."""
    report = resources.reports()["test_report"]
    model = report.loc["model_at_recall_threshold"]
    rule = report.loc["month_to_month_rule"]
    return {
        "recall": float(model["recall"]),
        "precision": float(model["precision"]),
        "contacted": float(model["contacted_pct"]) / 100,
        "rule_contacted": float(rule["contacted_pct"]) / 100,
        "tested": resources.test_rows(),
    }


def _go(url: str) -> None:
    st.session_state.pending_nav = url


# ===== Overview =====
def overview() -> None:
    """Overview page."""
    demo = state.demo()
    latest, previous = demo.latest, demo.previous
    if latest is None:
        st.title("Overview")
        ui.empty_state(
            "No batch scored yet",
            "The Overview fills in once a batch of customers has been scored. "
            "Start with one of the sample batches.",
            "Go to Batch Scoring",
            lambda: _go("batch-scoring"),
            key="overview",
        )
        return

    settings = demo.settings
    batch_alerts = {b.name: 0 for b in demo.batches}
    for alert in demo.alerts:
        batch_alerts[alert.batch] = batch_alerts.get(alert.batch, 0) + 1
    open_alerts = demo.open_alerts()
    kpis = {
        "rate": latest.predicted_rate,
        "prev_rate": previous.predicted_rate if previous else None,
        "flagged": latest.flagged,
        "prev_flagged": previous.flagged if previous else None,
        "customers": latest.customers,
        "flagged_share": latest.flagged / latest.customers,
        "active_alerts": len(open_alerts),
        "critical_alerts": sum(a.severity == "Critical" for a in open_alerts),
        "batch_alerts": batch_alerts.get(latest.name, 0),
        "prev_alerts": batch_alerts.get(previous.name, 0) if previous else None,
        "riskiest_plan": latest.riskiest_plan,
        "riskiest_rate": latest.riskiest_rate,
        "prev_riskiest_rate": previous.riskiest_rate if previous else None,
    }

    def status(rate: float) -> str:
        if rate >= settings.critical_threshold:
            return "Critical"
        return "Watch" if rate >= settings.watch_threshold else "OK"

    trend = pd.DataFrame(
        {
            "Batch": [b.name for b in demo.batches],
            "Date": [pd.Timestamp(b.batch_date) for b in demo.batches],
            "Rate": [b.predicted_rate for b in demo.batches],
            "Customers": [b.customers for b in demo.batches],
            "Status": [status(b.predicted_rate) for b in demo.batches],
        }
    )
    groups = plans.all_groups(latest.scored, "probability", settings.min_group_size)
    top = groups[~groups["too_small"]].head(3)

    sentence = None
    if previous:
        direction = (
            "rose" if latest.predicted_rate > previous.predicted_rate else "fell"
        )
        sentence = (
            f"Since the last batch: predicted churn {direction} from "
            f"{previous.predicted_rate:.1%} to {latest.predicted_rate:.1%}"
        )
        if latest.predicted_rate >= settings.critical_threshold:
            sentence += (
                f", above the {settings.critical_threshold:.0%} critical threshold."
            )
        elif latest.predicted_rate >= settings.watch_threshold:
            sentence += f", above the {settings.watch_threshold:.0%} watch threshold."
        else:
            sentence += ", below the alert thresholds."
    monitor.overview(
        kpis,
        trend,
        (settings.watch_threshold, settings.critical_threshold),
        top,
        sentence,
        _go,
    )


# ===== Alerts =====
def alerts() -> None:
    """Alerts page."""
    demo = state.demo()
    table = alerts_table(demo.alerts)
    by_id = {a.alert_id: a for a in demo.alerts}

    def on_change(alert_id: str, status: str, note: str, changed_by: str) -> None:
        change_status(by_id[alert_id], status, note, changed_by)
        st.session_state.pending_toast = f"Alert {alert_id} marked as {status.lower()}."

    export = table.rename(columns=str.capitalize)
    monitor.alerts(
        table,
        list(RULES.values()),
        _csv(export),
        on_change,
        {RULES[k]: v for k, v in RULE_DESCRIPTIONS.items()},
    )


# ===== Plan Risk =====
def plan_risk() -> None:
    """Plan Risk page."""
    demo = state.demo()
    settings = demo.settings
    history = resources.history()
    latest = demo.latest
    flagged = (
        plans.plan_warnings(
            latest.scored, "probability", settings.plan_gap_pp, settings.min_group_size
        )
        if latest
        else pd.DataFrame(columns=["plan"])
    )
    severity = {
        alert.subject: alert.severity
        for alert in demo.alerts
        if latest and alert.batch == latest.name and alert.subject
    }

    def source_data(source: str) -> tuple[pd.DataFrame, str]:
        if source == SOURCES[0]:
            return history, "left"
        return latest.scored, "probability"

    def get_table(attribute: str, source: str) -> dict:
        data, outcome = source_data(source)
        column = plans.PLAN_ATTRIBUTES[attribute]
        table = plans.churn_by_group(data, column, outcome, settings.min_group_size)
        labels = attribute + ": " + table["group"]
        if source == SOURCES[1]:
            table["status"] = [
                severity.get(lbl, "Watch") if lbl in set(flagged["plan"]) else None
                for lbl in labels
            ]
        else:
            table["status"] = None
        return {
            "table": table,
            "overall": float(data[outcome].mean()),
            "customers": len(data),
            "min_group": settings.min_group_size,
        }

    def combination(rows: str, cols: str, source: str) -> pd.DataFrame:
        data, outcome = source_data(source)
        table = plans.combination_table(
            data, plans.PLAN_ATTRIBUTES[rows], plans.PLAN_ATTRIBUTES[cols], outcome
        )
        return table.rename(
            columns={
                plans.PLAN_ATTRIBUTES[rows]: rows,
                plans.PLAN_ATTRIBUTES[cols]: cols,
            }
        )

    monitor.plan_risk(
        list(plans.PLAN_ATTRIBUTES),
        SOURCES,
        get_table,
        combination,
        batch_available=latest is not None,
    )


# ===== Customer Check =====
def _result_from_row(row: pd.Series, customer: str) -> dict:
    return {
        "customer": customer,
        "risk_level": row["risk_level"],
        "verdict": scoring.VERDICTS[row["risk_level"]],
        "guidance": row["guidance"],
        "reasons": row["top_reasons"],
        "actions": row["suggested_actions"],
        "probability": row["probability"],
        "talking_points": scoring.talking_points(row, customer),
    }


def _form_defaults() -> dict:
    history = resources.history()
    defaults = {}
    for spec in SCHEMA.fields.values():
        if isinstance(spec, CategoryField):
            defaults[spec.name] = spec.options[0]
        else:
            value = float(history[spec.name].median())
            value = min(max(value, spec.min_value), spec.max_value)
            defaults[spec.name] = int(round(value)) if spec.integer else round(value, 2)
    defaults.update(
        {
            "Contract": "Month-to-Month",
            "Internet Type": "Fiber Optic",
            "Phone Service": "Yes",
            "Payment Method": "Bank Withdrawal",
            "Tenure in Months": 12,
            "Monthly Charge": 85.0,
            "Total Charges": 1020.0,
            "Avg Monthly Long Distance Charges": 25.0,
            "Total Refunds": 0.0,
            "Total Extra Data Charges": 0.0,
            "Avg Monthly GB Download": 20.0,
        }
    )
    return defaults


def customer_check() -> None:
    """Customer Check page."""
    demo = state.demo()
    st.title("Customer Check")
    st.caption("Is this customer likely to leave, why, and what should I offer?")
    st.info(act.DECISION_AID_NOTE, icon=":material/info:")
    reliability = _reliability()
    tab_find, tab_enter = st.tabs(["Find a customer", "Enter details"])

    with tab_find:
        latest = demo.latest
        if latest is None:
            ui.empty_state(
                "No scored batch",
                "Score a batch first, or enter the customer's details in the "
                "other tab.",
            )
        else:
            scored = latest.scored.set_index(ID_COL)
            st.caption(
                f"Customers in the latest batch: **{latest.name}** ({len(scored):,})."
            )
            customer_id = st.selectbox(
                "Customer ID",
                scored.index.tolist(),
                index=None,
                placeholder="Type to search, e.g. 0013-EXCHZ",
                key="keep_cc_lookup",
            )
            if customer_id:
                act.result_card(
                    _result_from_row(scored.loc[customer_id], customer_id), reliability
                )

    with tab_enter:
        errors = st.session_state.get("cc_errors", {})

        def on_submit(values: dict) -> None:
            record = complete_form_record(values)
            problems = check_record(record)
            st.session_state.cc_errors = {p["Column"]: p["Reason"] for p in problems}
            if problems:
                st.rerun()
            st.session_state.pop("cc_result", None)
            act.confirm_record(record, on_confirm)

        def on_confirm(record: dict) -> None:
            frame = pd.DataFrame([record])
            scored = scoring.score_customers(
                resources.bundle(), frame, resources.reference()
            )
            st.session_state.cc_result = _result_from_row(
                scored.iloc[0], "Manual entry " + datetime.now().strftime("%H:%M:%S")
            )
            st.session_state.pending_toast = "Customer scored."

        act.customer_form(_form_defaults(), errors, on_submit)
        if "cc_result" in st.session_state:
            act.result_card(st.session_state.cc_result, reliability)


# ===== Batch Scoring =====
def _template() -> bytes:
    example = resources.sample_files()["sample_batch_no_errors.csv"]
    first_rows = read_upload_csv(io.BytesIO(example)).head(1)
    return _csv(first_rows[UPLOAD_COLS])


def _editor_config() -> dict:
    config = {}
    for spec in SCHEMA.fields.values():
        if isinstance(spec, CategoryField):
            config[spec.name] = st.column_config.SelectboxColumn(
                spec.name, options=list(spec.options)
            )
    return config


def batch_scoring() -> None:
    """Batch Scoring page."""
    demo = state.demo()
    flow = state.batch_flow()
    st.title("Batch Scoring")
    st.caption(
        "Upload a batch of customers, check it, confirm, and review the results."
    )
    ui.step_indicator(act.STEPS, flow["step"])

    def restart() -> None:
        flow.clear()
        flow["step"] = 0

    def run_check(raw: pd.DataFrame) -> None:
        check = check_batch(raw, MAX_UPLOAD_ROWS)
        flow["raw"] = raw
        flow["check"] = check
        flow["upload_id"] = uuid.uuid4().hex[:6]
        if check.n_invalid and not check.blocking:
            bad_rows = sorted(set(check.errors["Row"]) - {"All"})
            invalid = raw[UPLOAD_COLS].reset_index(drop=True)
            invalid.index = invalid.index + 1
            flow["invalid_rows"] = invalid.loc[bad_rows].astype(str)
        flow["step"] = 1

    def on_upload(uploaded) -> None:
        try:
            raw = read_upload_csv(uploaded)
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            flow["read_error"] = (
                "The file could not be read as a CSV. Save it as UTF-8 CSV and try "
                f"again. ({type(error).__name__})"
            )
            flow["check"] = check_batch(pd.DataFrame(), MAX_UPLOAD_ROWS)
            flow["step"] = 1
            st.rerun()
        flow.pop("read_error", None)
        run_check(raw)
        st.rerun()

    def on_recheck(edited: pd.DataFrame) -> None:
        raw = flow["raw"][UPLOAD_COLS].reset_index(drop=True).astype(str)
        raw.index = raw.index + 1
        raw.loc[edited.index, UPLOAD_COLS] = edited[UPLOAD_COLS].astype(str)
        run_check(raw.reset_index(drop=True))
        st.session_state.pending_toast = "Fixed rows re-checked."
        st.rerun()

    def on_continue() -> None:
        flow["step"] = 2

    def on_back() -> None:
        flow["step"] = 1

    def on_confirm(name: str, batch_date) -> None:
        check = flow["check"]
        scored = scoring.score_customers(
            resources.bundle(), check.valid_rows, resources.reference()
        )
        summary = summarise_batch(
            name, batch_date, scored, demo.settings, is_demo=False
        )
        raised = demo.add_batch(summary)
        flow.update(step=3, batch_name=name, raised_ids=[a.alert_id for a in raised])
        st.session_state.pending_toast = (
            f"Batch {name} scored: {summary.customers:,} customers, "
            f"{len(raised)} alerts."
        )

    step = flow["step"]
    if step == 0:
        act.upload_step(
            _template(),
            resources.sample_files(),
            MAX_UPLOAD_MB,
            MAX_UPLOAD_ROWS,
            on_upload,
        )
    elif step == 1:
        check = flow["check"]
        errors_csv = None
        if check.n_invalid and "invalid_rows" in flow:
            errors_csv = _csv(
                flow["invalid_rows"]
                .reset_index(names="Row")
                .merge(
                    check.errors.groupby("Row")["Reason"]
                    .apply("; ".join)
                    .reset_index(),
                    on="Row",
                )
            )
        act.check_step(
            flow, errors_csv, on_recheck, on_continue, restart, _editor_config()
        )
    elif step == 2:
        act.confirm_step(
            len(flow["check"].valid_rows),
            [b.name for b in demo.batches],
            on_confirm,
            on_back,
        )
    else:
        batch = next(b for b in demo.batches if b.name == flow["batch_name"])
        index = demo.batches.index(batch)
        previous = demo.batches[index - 1] if index > 0 else None
        raised = alerts_table(
            [a for a in demo.alerts if a.alert_id in flow["raised_ids"]]
        )
        groups = plans.all_groups(
            batch.scored, "probability", demo.settings.min_group_size
        )
        plan_table = pd.DataFrame(
            {
                "Plan": groups["plan"],
                "Customers": groups["customers"],
                "Predicted churn (%)": (100 * groups["rate"]).round(1),
                "Too few to compare": groups["too_small"],
            }
        )
        act.results_step(
            {
                "name": batch.name,
                "rate": batch.predicted_rate,
                "prev_rate": previous.predicted_rate if previous else None,
                "flagged": batch.flagged,
                "customers": batch.customers,
            },
            analysis.risk_distribution(batch.scored),
            plan_table,
            raised,
            _csv(scoring.scored_export(batch.scored)),
            restart,
            _go,
        )


# ===== Churn Insights =====
def _takeaways(tables: dict) -> list[dict]:
    """Business takeaways computed from the data (notebook Insights 2–7)."""
    board = tables["leaderboard"].set_index("segment")["rate"]
    tenure = tables["tenure"].set_index("group")["rate"]
    add_ons = tables["add_ons"]
    security = add_ons[
        (add_ons["Scope"] == "Internet customers")
        & (add_ons["Add-on"] == "Online Security")
    ].iloc[0]
    stats = tables["stats"]
    return [
        {
            "title": "Contract length is the strongest signal",
            "finding": f"{board.get('Month-to-month contract', float('nan')):.0%} "
            "of month-to-month customers left, far above the "
            f"{stats['churn_rate']:.1%} average.",
            "action": "Offer one- or two-year plans with a loyalty discount.",
        },
        {
            "title": "The first months are the riskiest",
            "finding": f"{tenure.get('1–6', float('nan')):.0%} of customers in "
            f"their first six months left, vs {tenure.get('49–72', float('nan')):.0%} "
            "after four years.",
            "action": "Run a welcome check-in and fix onboarding issues early.",
        },
        {
            "title": "Security and support add-ons go with staying",
            "finding": f"Internet customers with online security left at "
            f"{security['With add-on']:.0%}, vs {security['Without add-on']:.0%} "
            "without it.",
            "action": "Offer a trial of online security or premium tech support.",
        },
        {
            "title": "Some offers and payment methods need a closer look",
            "finding": f"{board.get('Offer E promotion', float('nan')):.0%} of "
            "Offer E customers left; mailed-check and bank-withdrawal payers "
            "also leave more often than card payers.",
            "action": "Review who receives Offer E and encourage card payment.",
        },
    ]


def churn_insights() -> None:
    """Churn Insights page: exploratory dashboard."""
    tables = resources.eda_tables()
    explore.churn_insights(
        tables["stats"],
        tables["leaderboard"],
        tables["tenure"],
        tables["add_ons"],
        _takeaways(tables),
    )


# ===== Glossary & Guide =====
def glossary() -> None:
    """Glossary & Guide page."""
    policy = _policy()
    risk_rules = [
        {
            "Level": ui.badge_text("High"),
            "Rule": f"Chance of leaving {policy['recall_threshold']:.0%} or higher",
            "Agent guidance": scoring.GUIDANCE["High"],
        },
        {
            "Level": ui.badge_text("Medium"),
            "Rule": f"Chance of leaving between {policy['breakeven_threshold']:.0%} "
            f"and {policy['recall_threshold']:.0%}",
            "Agent guidance": scoring.GUIDANCE["Medium"],
        },
        {
            "Level": ui.badge_text("Low"),
            "Rule": f"Chance of leaving below {policy['breakeven_threshold']:.0%}",
            "Agent guidance": scoring.GUIDANCE["Low"],
        },
    ]
    help_settings.glossary(
        {RULES[k]: v for k, v in RULE_DESCRIPTIONS.items()}, risk_rules, _reliability()
    )


# ===== Alert Settings =====
def alert_settings() -> None:
    """Alert Settings page."""
    demo = state.demo()
    current = vars(demo.settings).copy()

    def preview(candidate: dict):
        settings = AlertSettings(**candidate)
        problems = settings.errors()
        return problems if problems else preview_rules(demo.batches, settings)

    def on_save(candidate: dict, changed_by: str) -> int:
        return demo.update_settings(AlertSettings(**candidate), changed_by)

    def on_reset(changed_by: str) -> int:
        return demo.update_settings(AlertSettings(**DEFAULT_ALERT_SETTINGS), changed_by)

    help_settings.alert_settings(
        current,
        DEFAULT_ALERT_SETTINGS,
        OBSERVED_CHURN_RATE,
        preview,
        on_save,
        on_reset,
        pd.DataFrame(demo.audit_log),
    )
