"""Scoring, alert rules, settings and per-visitor demo state."""

import pytest

from app.models import analysis, artifacts, scoring
from app.models.alerts import AlertSettings, summarise_batch
from app.models.validation import check_batch
from telco_churn.config import DEFAULT_ALERT_SETTINGS, SAMPLE_BATCH_DIR


@pytest.fixture
def settings():
    return AlertSettings(**DEFAULT_ALERT_SETTINGS)


@pytest.fixture
def demo(bundle, reference, settings):
    return analysis.build_demo_state(
        bundle, reference, settings, artifacts.demo_batch_files()
    )


def _scored(bundle, reference, name):
    data = artifacts.read_upload_csv(SAMPLE_BATCH_DIR / name)
    valid = check_batch(data, 5_000).valid_rows
    return scoring.score_customers(bundle, valid, reference)


def test_risk_levels_follow_policy(bundle, reference):
    scored = _scored(bundle, reference, "sample_batch_no_errors.csv")
    policy = bundle.policy
    high = scored["probability"] >= policy.recall_threshold
    assert (scored.loc[high, "risk_level"] == "High").all()
    low = scored["probability"] < policy.breakeven_threshold
    assert (scored.loc[low, "risk_level"] == "Low").all()
    assert scored["top_reasons"].map(len).le(3).all()
    assert scored["suggested_actions"].map(len).le(2).all()


def test_high_risk_batch_raises_critical_alert(bundle, reference, demo, settings):
    scored = _scored(bundle, reference, "sample_batch_high_risk.csv")
    batch_date = demo.latest.batch_date
    batch = summarise_batch("High risk", batch_date, scored, settings, False)
    raised = demo.add_batch(batch)
    assert batch.predicted_rate >= settings.critical_threshold
    assert any(
        a.severity == "Critical" and a.rule == "Overall churn threshold" for a in raised
    )


def test_plan_warning_marks_a_plan(demo):
    plan_alerts = [a for a in demo.alerts if a.rule == "Plan warning"]
    assert plan_alerts and all(a.subject for a in plan_alerts)


def test_settings_validation(settings):
    assert settings.errors() == {}
    bad = AlertSettings(0.4, 0.3, 10, 30)
    assert "watch_threshold" in bad.errors()


def test_settings_changes_are_audited(demo):
    new = AlertSettings(0.25, 0.30, 20.0, 30)
    changed = demo.update_settings(new, "Alex")
    assert changed == 2
    assert {entry["Field"] for entry in demo.audit_log} == {
        "watch_threshold",
        "critical_threshold",
    }
    assert all(entry["Changed by"] == "Alex" for entry in demo.audit_log)


def test_visitor_copies_are_independent(demo):
    visitor_a, visitor_b = demo.copy(), demo.copy()
    visitor_a.update_settings(AlertSettings(0.2, 0.3, 5, 10), "A")
    visitor_a.alerts[0].status = "Resolved"
    assert visitor_b.audit_log == []
    assert visitor_b.alerts[0].status == "New"
