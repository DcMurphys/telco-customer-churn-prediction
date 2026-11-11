"""Churn rates by plan attribute, with the statistics shown in Analyst view.

A "plan" combines the plan-related fields in the dataset: contract type,
promotional offer, internet type and the number of add-on services.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.models.validation import ADD_ON_COLS, NONE_LABEL

PLAN_ATTRIBUTES = {
    "Contract": "Contract",
    "Offer": "Offer",
    "Internet type": "Internet Type",
    "Add-ons": "Add-ons",
}
ADD_ON_ORDER = ["No internet", "0 add-ons", "1–2 add-ons", "3–4 add-ons", "5+ add-ons"]


def add_plan_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add display-ready plan columns (explicit "None" labels, add-on bands)."""
    out = data.copy()
    for col in ("Offer", "Internet Type"):
        out[col] = out[col].fillna(NONE_LABEL).replace("No", NONE_LABEL)
    n_add_ons = out[ADD_ON_COLS].eq("Yes").sum(axis=1)
    bands = pd.cut(n_add_ons, bins=[-1, 0, 2, 4, 99], labels=ADD_ON_ORDER[1:]).astype(
        str
    )
    out["Add-ons"] = np.where(out["Internet Service"].eq("No"), "No internet", bands)
    return out


def wilson_interval(
    successes: np.ndarray, totals: np.ndarray, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a proportion (robust for small groups)."""
    z = norm.ppf(0.5 + confidence / 2)
    totals = np.asarray(totals, dtype=float)
    p = np.divide(successes, totals, out=np.zeros_like(totals), where=totals > 0)
    denominator = 1 + z**2 / totals
    centre = (p + z**2 / (2 * totals)) / denominator
    half = z * np.sqrt(p * (1 - p) / totals + z**2 / (4 * totals**2)) / denominator
    return centre - half, centre + half


def churn_by_group(
    data: pd.DataFrame, column: str, outcome: str, min_group_size: int
) -> pd.DataFrame:
    """Churn rate per value of ``column``.

    Args:
        data: Customers with plan columns (see :func:`add_plan_columns`).
        column: Plan column to group by.
        outcome: 0/1 observed churn or a churn probability per customer.
        min_group_size: Groups smaller than this are marked as too small.

    Returns:
        One row per group, sorted by rate, with the gap to the overall rate in
        percentage points and a 95% Wilson interval (observed outcomes only).
    """
    overall = data[outcome].mean()
    grouped = data.groupby(column, observed=True)[outcome].agg(["size", "mean", "sum"])
    grouped.columns = ["customers", "rate", "events"]
    low, high = wilson_interval(grouped["events"].to_numpy(), grouped["customers"])
    grouped["ci_low"], grouped["ci_high"] = low, high
    grouped["gap_pp"] = 100 * (grouped["rate"] - overall)
    grouped["too_small"] = grouped["customers"] < min_group_size
    grouped.index.name = "group"
    return grouped.sort_values("rate", ascending=False).reset_index()


def combination_table(
    data: pd.DataFrame, rows: str, cols: str, outcome: str
) -> pd.DataFrame:
    """Rate and count for each combination of two plan attributes (long format)."""
    grouped = data.groupby([rows, cols], observed=True)[outcome].agg(["mean", "size"])
    return grouped.rename(columns={"mean": "rate", "size": "customers"}).reset_index()


def plan_warnings(
    data: pd.DataFrame, outcome: str, gap_pp: float, min_group_size: int
) -> pd.DataFrame:
    """Plans whose rate exceeds the batch average by at least ``gap_pp`` points.

    Returns:
        One row per flagged plan with ``plan`` ("Contract: Month-to-Month"),
        ``rate``, ``customers`` and ``gap_pp``, sorted by rate.
    """
    rows = []
    for label, column in PLAN_ATTRIBUTES.items():
        table = churn_by_group(data, column, outcome, min_group_size)
        hits = table[~table["too_small"] & (table["gap_pp"] >= gap_pp)]
        for _, hit in hits.iterrows():
            rows.append(
                {
                    "attribute": label,
                    "plan": f"{label}: {hit['group']}",
                    "rate": hit["rate"],
                    "customers": int(hit["customers"]),
                    "gap_pp": hit["gap_pp"],
                }
            )
    columns = ["attribute", "plan", "rate", "customers", "gap_pp"]
    return pd.DataFrame(rows, columns=columns).sort_values("rate", ascending=False)


def riskiest_plan(
    data: pd.DataFrame, outcome: str, min_group_size: int
) -> tuple[str, float, int] | None:
    """The plan group with the highest rate among groups large enough to compare."""
    best: tuple[str, float, int] | None = None
    for label, column in PLAN_ATTRIBUTES.items():
        table = churn_by_group(data, column, outcome, min_group_size)
        table = table[~table["too_small"]]
        if table.empty:
            continue
        top = table.iloc[0]
        if best is None or top["rate"] > best[1]:
            best = (
                f"{label}: {top['group']}",
                float(top["rate"]),
                int(top["customers"]),
            )
    return best


def all_groups(data: pd.DataFrame, outcome: str, min_group_size: int) -> pd.DataFrame:
    """Every plan group across all attributes, highest rate first."""
    tables = []
    for label, column in PLAN_ATTRIBUTES.items():
        table = churn_by_group(data, column, outcome, min_group_size)
        tables.append(table.assign(attribute=label, plan=label + ": " + table["group"]))
    return pd.concat(tables, ignore_index=True).sort_values("rate", ascending=False)
