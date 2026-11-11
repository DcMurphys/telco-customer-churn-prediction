"""Exploratory analysis of the clean modelling dataset (Churn Insights page).

Mirrors notebook Section 4.4 on the dataset after Phase A (Section 5.1): all
7,043 customers, no identifiers, geography or leakage columns, structural
blanks spelled out as "None". Plain pandas/SciPy only (no Streamlit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.models.plans import wilson_interval
from app.models.validation import ADD_ON_COLS
from telco_churn.config import APP_EDA_PATH

TARGET = "Churn Label"
TENURE_BANDS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["1–6", "7–12", "13–24", "25–48", "49–72"]

# Categorical variables grouped as in the notebook (Insights 3, 5 and 6)
CATEGORY_GROUPS = {
    "Demographics": [
        "Gender",
        "Senior Citizen",
        "Under 30",
        "Married",
        "Dependents",
        "Referred a Friend",
    ],
    "Products and services": [
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Internet Type",
        *ADD_ON_COLS,
    ],
    "Contract, offer and billing": [
        "Contract",
        "Offer",
        "Paperless Billing",
        "Payment Method",
    ],
}
# Business wording for segments shown in Business view
SEGMENT_LABELS = {
    ("Contract", "Month-to-Month"): "Month-to-month contract",
    ("Contract", "One Year"): "One-year contract",
    ("Contract", "Two Year"): "Two-year contract",
    ("Internet Type", "Fiber Optic"): "Fiber internet",
    ("Internet Type", "Cable"): "Cable internet",
    ("Internet Type", "DSL"): "DSL internet",
    ("Internet Type", "None"): "No internet",
    ("Senior Citizen", "Yes"): "Senior (65+)",
    ("Married", "No"): "Not married",
    ("Dependents", "No"): "No dependents",
    ("Referred a Friend", "No"): "Never referred a friend",
    ("Referred a Friend", "Yes"): "Referred a friend",
    ("Payment Method", "Mailed Check"): "Pays by mailed check",
    ("Payment Method", "Bank Withdrawal"): "Pays by bank withdrawal",
    ("Payment Method", "Credit Card"): "Pays by credit card",
    ("Paperless Billing", "Yes"): "Paperless billing",
    ("Online Security", "No"): "No online security",
    ("Premium Tech Support", "No"): "No premium tech support",
    ("Unlimited Data", "Yes"): "Unlimited data",
}
PROTECTIVE_ADD_ONS = [
    "Online Security",
    "Premium Tech Support",
    "Online Backup",
    "Device Protection Plan",
]


def load_clean(path: Path = APP_EDA_PATH) -> pd.DataFrame:
    """Clean dataset with a 0/1 ``left`` column ("None" kept as a label)."""
    data = pd.read_csv(path, keep_default_na=False)
    return data.assign(left=data[TARGET].eq("Yes").astype(int))


def overview(data: pd.DataFrame) -> dict[str, float]:
    """Headline figures for the KPI row."""
    left = data["left"].astype(bool)
    revenue = data["Total Revenue"]
    return {
        "customers": len(data),
        "churned": int(left.sum()),
        "churn_rate": float(left.mean()),
        "revenue_lost": float(revenue[left].sum()),
        "revenue_share": float(revenue[left].sum() / revenue.sum()),
        "tenure_left": float(data.loc[left, "Tenure in Months"].median()),
        "tenure_stayed": float(data.loc[~left, "Tenure in Months"].median()),
        "class_ratio": float((~left).sum() / left.sum()),
    }


def rate_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate per value of one column, with a 95% Wilson interval."""
    grouped = data.groupby(column)["left"].agg(customers="size", churned="sum")
    grouped["rate"] = grouped["churned"] / grouped["customers"]
    low, high = wilson_interval(grouped["churned"].to_numpy(), grouped["customers"])
    grouped["ci_low"], grouped["ci_high"] = low, high
    grouped["gap_pp"] = 100 * (grouped["rate"] - data["left"].mean())
    return grouped.reset_index(names="group").assign(variable=column)


def segment_rates(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rate tables for several columns stacked in long format."""
    return pd.concat([rate_table(data, col) for col in columns], ignore_index=True)


def segment_label(variable: str, group: str) -> str:
    """Plain-language name of one segment."""
    if variable == "Offer":
        return "No promotion" if group == "None" else f"{group} promotion"
    return SEGMENT_LABELS.get((variable, group), f"{variable}: {group}")


def segment_leaderboard(
    data: pd.DataFrame, min_customers: int = 100, top: int = 10
) -> pd.DataFrame:
    """Highest-churn segments with business labels (Business view)."""
    columns = [c for group in CATEGORY_GROUPS.values() for c in group]
    table = segment_rates(data, columns)
    table = table[table["customers"] >= min_customers].copy()
    table["segment"] = [
        segment_label(v, g)
        for v, g in zip(table["variable"], table["group"], strict=True)
    ]
    table = table.sort_values("rate", ascending=False).drop_duplicates("segment")
    return table.head(top).reset_index(drop=True)


def tenure_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Churn rate by tenure band (months)."""
    bands = pd.cut(
        data["Tenure in Months"], bins=TENURE_BANDS, labels=TENURE_LABELS
    ).astype(str)
    table = rate_table(data.assign(band=bands), "band")
    order = {label: i for i, label in enumerate(TENURE_LABELS)}
    return table.sort_values("group", key=lambda s: s.map(order)).reset_index(drop=True)


def add_on_effect(data: pd.DataFrame) -> pd.DataFrame:
    """Churn with vs without each add-on, for all customers and internet users only.

    Answers the notebook's follow-up check (Insight 5): customers without
    internet answer "No" to every add-on and churn far less, which distorts the
    all-customer comparison.
    """
    internet = data[data["Internet Service"].eq("Yes")]
    rows = []
    for col in ADD_ON_COLS:
        for scope, frame in (("All customers", data), ("Internet customers", internet)):
            rates = frame.groupby(col)["left"].mean()
            rows.append(
                {
                    "Add-on": col,
                    "Scope": scope,
                    "With add-on": rates.get("Yes", np.nan),
                    "Without add-on": rates.get("No", np.nan),
                }
            )
    table = pd.DataFrame(rows)
    table["Difference (pp)"] = 100 * (table["With add-on"] - table["Without add-on"])
    return table
