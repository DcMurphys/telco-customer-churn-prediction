"""Score customers: calibrated probability, risk level, top reasons, next steps.

Risk levels follow the Retention Playbook (UI/UX requirements, Section 7):

* High: probability >= recall threshold (flagged for contact)
* Medium: between the break-even and the recall threshold
* Low: below the break-even threshold

Top reasons are what-if contributions: for every playbook reason that applies
to a customer, the model is re-scored with that one characteristic changed to
a typical, lower-risk value. The drop in the model score is the reason's
contribution. This is a local sensitivity check, not a causal effect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.models.artifacts import ModelBundle
from app.models.validation import ID_COL, to_model_input
from telco_churn import policy

RISK_LEVELS = ("High", "Medium", "Low")


@dataclass(frozen=True)
class Reason:
    """A playbook reason with its condition and what-if change."""

    key: str
    label: str
    action: str
    feature: str  # technical feature name shown in Analyst view
    applies: Callable[[pd.DataFrame, dict], pd.Series]
    what_if: Callable[[pd.DataFrame, dict], pd.DataFrame]


def _scale_totals(data: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Scale billing totals consistently when tenure or charges change."""
    out = data.copy()
    for col in ("Total Charges", "Total Long Distance Charges"):
        out[col] = out[col] * factor
    out["Total Revenue"] = (
        out["Total Charges"]
        - out["Total Refunds"]
        + out["Total Extra Data Charges"]
        + out["Total Long Distance Charges"]
    )
    return out


def _contract_what_if(data: pd.DataFrame, _: dict) -> pd.DataFrame:
    return data.assign(Contract="One Year")


def _referral_what_if(data: pd.DataFrame, _: dict) -> pd.DataFrame:
    return data.assign(**{"Number of Referrals": 1, "Referred a Friend": "Yes"})


def _tenure_what_if(data: pd.DataFrame, ref: dict) -> pd.DataFrame:
    target = ref["reference_tenure_months"]
    out = _scale_totals(data, target / data["Tenure in Months"])
    return out.assign(**{"Tenure in Months": target})


def _charge_what_if(data: pd.DataFrame, ref: dict) -> pd.DataFrame:
    target = ref["reference_monthly_charge"]
    factor = target / data["Monthly Charge"]
    out = data.copy()
    out["Total Charges"] = out["Total Charges"] * factor
    out["Total Revenue"] = (
        out["Total Charges"]
        - out["Total Refunds"]
        + out["Total Extra Data Charges"]
        + out["Total Long Distance Charges"]
    )
    return out.assign(**{"Monthly Charge": target})


def _support_what_if(data: pd.DataFrame, _: dict) -> pd.DataFrame:
    return data.assign(**{"Premium Tech Support": "Yes", "Online Security": "Yes"})


REASONS = [
    Reason(
        "contract",
        "On a month-to-month contract",
        "Present a one- or two-year plan with a loyalty discount",
        "Contract_Month-to-Month",
        lambda d, _: d["Contract"].eq("Month-to-Month"),
        _contract_what_if,
    ),
    Reason(
        "referrals",
        "Has not referred anyone",
        "Mention the referral reward",
        "Number of Referrals",
        lambda d, _: d["Number of Referrals"].eq(0),
        _referral_what_if,
    ),
    Reason(
        "tenure",
        "Short time as a customer",
        "Check onboarding issues; offer a welcome check-in",
        "Tenure in Months (Pc_1, Pc_2)",
        lambda d, ref: d["Tenure in Months"].le(ref["short_tenure_months"]),
        _tenure_what_if,
    ),
    Reason(
        "charges",
        "High monthly charges for the services used",
        "Review the plan; suggest a better-fitting bundle",
        "Monthly Charge (Pc_1, Pc_2)",
        lambda d, ref: d["Monthly Charge"].gt(ref["high_monthly_charge"]),
        _charge_what_if,
    ),
    Reason(
        "support",
        "Fiber internet with no support add-ons",
        "Offer a trial of premium tech support or online security",
        "Internet Type_Fiber Optic × support add-ons",
        lambda d, _: (
            d["Internet Type"].eq("Fiber Optic")
            & d["Premium Tech Support"].eq("No")
            & d["Online Security"].eq("No")
        ),
        _support_what_if,
    ),
]
REASON_BY_KEY = {reason.key: reason for reason in REASONS}
VERDICTS = {
    "High": "Likely to leave: offer a retention plan today",
    "Medium": "Some risk: mention benefits and schedule a follow-up",
    "Low": "Unlikely to leave: no retention action needed",
}
GUIDANCE = {
    "High": "Make a retention offer in this interaction",
    "Medium": "Mention benefits; schedule a follow-up",
    "Low": "No retention action needed",
}


def model_score(bundle: ModelBundle, raw: pd.DataFrame) -> np.ndarray:
    """Uncalibrated model score (balanced-training scale)."""
    features = bundle.preprocessor.transform(to_model_input(raw), bundle.schema)
    return bundle.model.predict_proba(features)[:, 1]


def calibrate(bundle: ModelBundle, score: np.ndarray) -> np.ndarray:
    """Map model scores to calibrated churn probabilities."""
    return policy.calibrate(bundle.calibrator, score)


def risk_level(probability: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Risk band per customer from the decision policy (shared with serving)."""
    thresholds = policy.Thresholds(
        recall=bundle.policy.recall_threshold,
        breakeven=bundle.policy.breakeven_threshold,
    )
    return policy.risk_levels(probability, thresholds)


def score_customers(
    bundle: ModelBundle, raw: pd.DataFrame, reference: dict
) -> pd.DataFrame:
    """Score validated customers.

    Args:
        bundle: Loaded model bundle.
        raw: Validated rows in upload layout (explicit "None" labels allowed).
        reference: Reference values from ``reference_values.json``.

    Returns:
        ``raw`` plus ``probability``, ``risk_level``, one contribution column per
        reason (``reason_<key>``, NaN when the reason does not apply),
        ``top_reasons``, ``suggested_actions`` and ``guidance``.
    """
    raw = raw.copy()
    base_score = model_score(bundle, raw)
    probability = calibrate(bundle, base_score)
    out = raw.assign(
        probability=probability,
        model_score=base_score,
        risk_level=risk_level(probability, bundle),
    )
    for reason in REASONS:
        applies = reason.applies(raw, reference).to_numpy()
        contribution = np.full(len(raw), np.nan)
        if applies.any():
            subset = raw.loc[applies]
            changed = reason.what_if(subset, reference)
            contribution[applies] = base_score[applies] - model_score(bundle, changed)
        out[f"reason_{reason.key}"] = contribution

    reason_cols = [f"reason_{r.key}" for r in REASONS]
    contributions = out[reason_cols].to_numpy()
    top_reasons, top_actions = [], []
    for row in contributions:
        order = [i for i in np.argsort(-np.nan_to_num(row, nan=-1)) if row[i] > 0][:3]
        top_reasons.append([REASONS[i].label for i in order])
        top_actions.append([REASONS[i].action for i in order][:2])
    out["top_reasons"] = top_reasons
    out["suggested_actions"] = top_actions
    out["guidance"] = out["risk_level"].map(GUIDANCE)
    return out


def scored_export(scored: pd.DataFrame) -> pd.DataFrame:
    """Rows for the scored-batch download (requirement B-7)."""
    return pd.DataFrame(
        {
            ID_COL: scored[ID_COL],
            "Risk level": scored["risk_level"],
            "Chance of leaving (%)": (100 * scored["probability"]).round(1),
            "Top reasons": scored["top_reasons"].map("; ".join),
            "Suggested action": [
                "; ".join(actions) if actions else guidance
                for actions, guidance in zip(
                    scored["suggested_actions"], scored["guidance"], strict=True
                )
            ],
            "Contract": scored["Contract"],
            "Offer": scored["Offer"],
            "Internet Type": scored["Internet Type"],
        }
    )


def talking_points(row: pd.Series, customer_label: str) -> str:
    """Short text an agent can paste into the CRM (requirement C-5)."""
    lines = [
        f"Customer: {customer_label}",
        f"Risk level: {row['risk_level']} — {VERDICTS[row['risk_level']]}",
    ]
    if row["top_reasons"]:
        lines.append("Why: " + "; ".join(row["top_reasons"]))
    if row["suggested_actions"]:
        lines.append("Suggested: " + "; ".join(row["suggested_actions"]))
    lines.append("Decision aid only; use your judgement and the customer's own words.")
    return "\n".join(lines)
