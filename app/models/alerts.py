"""Batch summaries, alert rules and the per-visitor demo state.

Everything here is plain Python data so the controller can keep it in
``st.session_state`` (one copy per visitor) and tests can run without Streamlit.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import pandas as pd

from app.models.plans import add_plan_columns, plan_warnings, riskiest_plan

SEVERITY_ORDER = {"Critical": 0, "Watch": 1}
STATUSES = ("New", "Acknowledged", "Resolved")
RULES = {
    "overall_critical": "Overall churn threshold",
    "overall_watch": "Overall churn watch",
    "plan_warning": "Plan warning",
    "rising_trend": "Rising trend",
}
RULE_DESCRIPTIONS = {
    "overall_critical": "Predicted churn rate of a batch is at or above the "
    "critical threshold.",
    "overall_watch": "Predicted churn rate is at or above the watch threshold "
    "but below the critical threshold.",
    "plan_warning": "A plan's predicted churn rate exceeds the batch average by "
    "at least the plan gap, and the plan has at least the minimum number of "
    "customers. Critical if it is also the riskiest plan for two consecutive "
    "batches.",
    "rising_trend": "Predicted churn rate increased in three consecutive batches.",
}


@dataclass
class AlertSettings:
    """Editable alert thresholds (Alert Settings page)."""

    watch_threshold: float
    critical_threshold: float
    plan_gap_pp: float
    min_group_size: int

    def errors(self) -> dict[str, str]:
        """Field-level validation messages (requirement S-2)."""
        problems = {}
        for name in ("watch_threshold", "critical_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                problems[name] = "Must be between 0% and 100%"
        if self.watch_threshold >= self.critical_threshold:
            problems["watch_threshold"] = "Watch must be lower than critical"
        if not 0 < self.plan_gap_pp <= 100:
            problems["plan_gap_pp"] = "Must be between 0 and 100 percentage points"
        if not 1 <= self.min_group_size <= 10_000:
            problems["min_group_size"] = "Must be between 1 and 10,000 customers"
        return problems


@dataclass
class BatchSummary:
    """What is stored per scored batch (requirement B-8)."""

    name: str
    batch_date: date
    scored_at: datetime
    customers: int
    predicted_rate: float
    flagged: int
    risk_counts: dict[str, int]
    riskiest_plan: str | None
    riskiest_rate: float | None
    is_demo: bool
    scored: pd.DataFrame = field(repr=False)


@dataclass
class Alert:
    """One raised alert with its status history (requirement A-1/A-2)."""

    alert_id: str
    raised_at: datetime
    severity: str
    rule: str
    batch: str
    value: float
    threshold: float
    detail: str
    status: str = "New"
    subject: str = ""  # plan label for plan warnings
    log: list[dict] = field(default_factory=list)


def summarise_batch(
    name: str,
    batch_date: date,
    scored: pd.DataFrame,
    settings: AlertSettings,
    is_demo: bool,
    scored_at: datetime | None = None,
) -> BatchSummary:
    """Build the summary record of a scored batch."""
    with_plans = add_plan_columns(scored)
    top = riskiest_plan(with_plans, "probability", settings.min_group_size)
    counts = scored["risk_level"].value_counts()
    return BatchSummary(
        name=name,
        batch_date=batch_date,
        scored_at=scored_at or datetime.now(),
        customers=len(scored),
        predicted_rate=float(scored["probability"].mean()),
        flagged=int(counts.get("High", 0)),
        risk_counts={
            level: int(counts.get(level, 0)) for level in ("High", "Medium", "Low")
        },
        riskiest_plan=top[0] if top else None,
        riskiest_rate=top[1] if top else None,
        is_demo=is_demo,
        scored=with_plans,
    )


def evaluate_rules(
    batch: BatchSummary, history: list[BatchSummary], settings: AlertSettings
) -> list[Alert]:
    """Alerts raised by ``batch`` given the batches scored before it.

    Args:
        batch: The newly scored batch.
        history: Earlier batches, oldest first (``batch`` not included).
        settings: Active alert settings.

    Returns:
        New alerts, most severe first.
    """
    alerts: list[Alert] = []

    def raise_alert(
        severity: str,
        rule: str,
        value: float,
        threshold: float,
        detail: str,
        subject: str = "",
    ) -> None:
        alerts.append(
            Alert(
                alert_id=uuid.uuid4().hex[:8],
                raised_at=batch.scored_at,
                severity=severity,
                rule=RULES[rule],
                batch=batch.name,
                value=value,
                threshold=threshold,
                detail=detail,
                subject=subject,
            )
        )

    rate = batch.predicted_rate
    if rate >= settings.critical_threshold:
        raise_alert(
            "Critical",
            "overall_critical",
            rate,
            settings.critical_threshold,
            f"Predicted churn {rate:.1%} ≥ critical {settings.critical_threshold:.0%}",
        )
    elif rate >= settings.watch_threshold:
        raise_alert(
            "Watch",
            "overall_watch",
            rate,
            settings.watch_threshold,
            f"Predicted churn {rate:.1%} ≥ watch {settings.watch_threshold:.0%}",
        )

    flagged = plan_warnings(
        batch.scored, "probability", settings.plan_gap_pp, settings.min_group_size
    )
    previous = history[-1] if history else None
    for _, plan in flagged.iterrows():
        repeat_riskiest = (
            plan["plan"] == batch.riskiest_plan
            and previous is not None
            and previous.riskiest_plan == batch.riskiest_plan
        )
        raise_alert(
            "Critical" if repeat_riskiest else "Watch",
            "plan_warning",
            plan["rate"],
            rate + settings.plan_gap_pp / 100,
            f"{plan['plan']}: {plan['rate']:.1%} vs batch average {rate:.1%} "
            f"(+{plan['gap_pp']:.1f} pp, {plan['customers']} customers)"
            + (" · riskiest plan two batches in a row" if repeat_riskiest else ""),
            subject=plan["plan"],
        )

    rates = [b.predicted_rate for b in history[-3:]] + [rate]
    if len(rates) == 4 and all(b > a for a, b in zip(rates, rates[1:], strict=False)):
        raise_alert(
            "Watch",
            "rising_trend",
            rate,
            rates[0],
            "Predicted churn rose in three consecutive batches: "
            + " → ".join(f"{r:.1%}" for r in rates),
        )
    return sorted(alerts, key=lambda a: SEVERITY_ORDER[a.severity])


def preview_rules(batches: list[BatchSummary], settings: AlertSettings) -> pd.DataFrame:
    """How many past batches would have triggered each rule (requirement S-2)."""
    counts = {label: 0 for label in RULES.values()}
    for i, batch in enumerate(batches):
        triggered = {a.rule for a in evaluate_rules(batch, batches[:i], settings)}
        for rule in triggered:
            counts[rule] += 1
    return pd.DataFrame(
        {
            "Rule": list(counts),
            "Past batches that would trigger it": list(counts.values()),
        }
    )


def alerts_table(alerts: list[Alert]) -> pd.DataFrame:
    """Alerts as a table (newest first)."""
    columns = [
        "alert_id",
        "raised_at",
        "severity",
        "rule",
        "batch",
        "value",
        "threshold",
        "detail",
        "status",
    ]
    if not alerts:
        return pd.DataFrame(columns=columns)
    rows = [
        {k: v for k, v in asdict(a).items() if k not in ("log", "subject")}
        for a in alerts
    ]
    return pd.DataFrame(rows)[columns].sort_values("raised_at", ascending=False)


def change_status(alert: Alert, status: str, note: str, changed_by: str) -> None:
    """Acknowledge or resolve an alert and log the change."""
    if status not in STATUSES:
        raise ValueError(f"Unknown status: {status}")
    alert.log.append(
        {
            "time": datetime.now(),
            "from": alert.status,
            "to": status,
            "note": note,
            "by": changed_by or "—",
        }
    )
    alert.status = status


@dataclass
class DemoState:
    """Per-visitor state: batch history, alerts, settings and audit log."""

    batches: list[BatchSummary]
    alerts: list[Alert]
    settings: AlertSettings
    audit_log: list[dict]

    def copy(self) -> DemoState:
        """Independent copy, so one visitor's changes never reach another."""
        return copy.deepcopy(self)

    @property
    def latest(self) -> BatchSummary | None:
        """Most recently scored batch."""
        return self.batches[-1] if self.batches else None

    @property
    def previous(self) -> BatchSummary | None:
        """Batch before the latest one."""
        return self.batches[-2] if len(self.batches) > 1 else None

    def open_alerts(self, severity: str | None = None) -> list[Alert]:
        """Alerts that are not resolved."""
        return [
            a
            for a in self.alerts
            if a.status != "Resolved" and (severity is None or a.severity == severity)
        ]

    def add_batch(self, batch: BatchSummary) -> list[Alert]:
        """Store a confirmed batch and the alerts it raises."""
        new_alerts = evaluate_rules(batch, self.batches, self.settings)
        self.batches.append(batch)
        self.alerts.extend(new_alerts)
        return new_alerts

    def update_settings(self, new: AlertSettings, changed_by: str) -> int:
        """Apply new settings and write one audit entry per changed field."""
        now = datetime.now()
        changes = 0
        for name, old_value in asdict(self.settings).items():
            new_value = getattr(new, name)
            if new_value != old_value:
                self.audit_log.append(
                    {
                        "Time": now,
                        "Field": name,
                        "Old value": old_value,
                        "New value": new_value,
                        "Changed by": changed_by,
                    }
                )
                changes += 1
        self.settings = new
        return changes
