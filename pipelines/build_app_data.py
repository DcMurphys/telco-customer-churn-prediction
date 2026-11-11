"""Build the demo data files used by the Streamlit app.

Reproduces the notebook's stratified train/test split of the raw IBM sample
(Section 5.2) and writes small, app-ready CSV files to ``app/data/``:

* ``historical_customers.csv``: training split (observed churn) for Plan Risk and
  Churn Drivers. No customer identifiers.
* ``test_customers.csv``: test split with Customer ID and without the churn label,
  the pool for demo batches and sample uploads.
* ``eda_customers.csv``: the clean modelling dataset for the exploratory
  dashboard: all 7,043 customers after notebook Phase A (Section 5.1), i.e.
  without identifiers, geography, target-leakage columns or Satisfaction Score,
  with structural blanks spelled out as "None". Every row passes the app's
  validation rules.
* ``reference_values.json``: reference values for the "top reasons" logic.
* ``demo_batches/*.csv``: four pre-built weekly batches that seed the demo history.
* ``sample_batches/*.csv``: downloadable uploads (clean, with deliberate errors,
  high risk) so visitors can try validation and alerts.

Only the IBM sample is used; MurphysTelecom is a fictional company.

Usage (from the repository root):
    make data          # or: PYTHONPATH=src:. python -m pipelines.build_app_data
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.models.validation import check_batch
from telco_churn.config import (
    APP_DATA_DIR,
    APP_EDA_PATH,
    APP_HISTORY_PATH,
    APP_REFERENCE_PATH,
    APP_TEST_CUSTOMERS_PATH,
    DEMO_BATCH_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SAMPLE_BATCH_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from telco_churn.preprocessing import (
    IDENTIFIER_GEO_COLS,
    LEAKAGE_SUSPECT_COLS,
    STRUCTURAL_MISSING_FILL,
    TARGET_LEAKAGE_COLS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Columns of an upload file (raw IBM layout minus identifiers, geography,
# leakage columns and Gender, which the model does not use)
UPLOAD_COLS = [
    "Customer ID",
    "Age",
    "Under 30",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Number of Dependents",
    "Referred a Friend",
    "Number of Referrals",
    "Tenure in Months",
    "Offer",
    "Phone Service",
    "Avg Monthly Long Distance Charges",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Avg Monthly GB Download",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "CLTV",
]
# Upload files spell structural missing values out instead of leaving blanks
UPLOAD_MISSING_LABEL = {"Offer": "None", "Internet Type": "None"}
DEMO_BATCH_SIZE = 300
DEMO_BATCH_NAMES = ["Demo Oct-W3", "Demo Oct-W4", "Demo Nov-W1", "Demo Nov-W2"]
# Share of month-to-month customers per demo batch: a gentle rise, so the demo
# history shows a trend and at least one alert (demo data, not a real pattern)
DEMO_MTM_SHARES = [0.42, 0.47, 0.52, 0.60]


def write_eda_file() -> int:
    """Clean modelling dataset (notebook Phase A) for the exploratory dashboard.

    Raises:
        ValueError: If any row breaks the app's validation rules, or if
            duplicates or non-structural missing values are found.
    """
    raw = pd.read_csv(RAW_DATA_PATH)
    if raw.duplicated().any() or raw["Customer ID"].duplicated().any():
        raise ValueError("Duplicate rows or Customer IDs in the raw data")
    clean = raw.drop(
        columns=IDENTIFIER_GEO_COLS + TARGET_LEAKAGE_COLS + LEAKAGE_SUSPECT_COLS
    ).fillna(value=UPLOAD_MISSING_LABEL)
    if clean.isna().any().any():
        raise ValueError("Unexpected missing values after the structural fill")
    check = check_batch(
        to_upload_layout(raw.fillna(value=STRUCTURAL_MISSING_FILL)).astype(str),
        max_rows=len(raw),
    )
    if check.n_invalid:
        raise ValueError(f"{check.n_invalid} rows fail the validation rules")
    clean.to_csv(APP_EDA_PATH, index=False)
    return len(clean)


def split_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the notebook split on the raw data (same seed and stratification)."""
    raw = pd.read_csv(RAW_DATA_PATH)
    keep = raw.drop(
        columns=[
            col
            for col in IDENTIFIER_GEO_COLS + TARGET_LEAKAGE_COLS + LEAKAGE_SUSPECT_COLS
            if col != "Customer ID"
        ]
    ).fillna(value=STRUCTURAL_MISSING_FILL)
    train, test = train_test_split(
        keep,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=keep[TARGET_COL],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def to_upload_layout(data: pd.DataFrame) -> pd.DataFrame:
    """Return rows in upload layout with explicit "None" labels."""
    out = data[UPLOAD_COLS].copy()
    money = out.select_dtypes("float").columns
    out[money] = out[money].round(2)
    for col, label in UPLOAD_MISSING_LABEL.items():
        out[col] = out[col].replace("No", label)
    return out


def sample_with_mtm_share(
    pool: pd.DataFrame, size: int, share: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw customers without replacement with a given month-to-month share."""
    is_mtm = pool["Contract"].eq("Month-to-Month")
    n_mtm = round(size * share)
    mtm = pool[is_mtm].sample(n=n_mtm, random_state=rng.integers(1_000_000_000))
    other = pool[~is_mtm].sample(
        n=size - n_mtm, random_state=rng.integers(1_000_000_000)
    )
    return pd.concat([mtm, other]).sample(
        frac=1, random_state=rng.integers(1_000_000_000)
    )


def build_error_sample(clean: pd.DataFrame) -> pd.DataFrame:
    """Inject a documented set of input errors into a copy of ``clean``."""
    bad = clean.copy().astype(object)
    bad.loc[1, "Monthly Charge"] = 999  # out of range
    bad.loc[2, "Contract"] = "Monthly"  # unknown category label
    bad.loc[3, "Age"] = "forty"  # wrong type
    bad.loc[4, "Customer ID"] = bad.loc[5, "Customer ID"]  # duplicate ID
    # first customer without internet outside the rows edited here
    no_internet = bad.index[bad["Internet Service"].eq("No") & (bad.index > 10)]
    if len(no_internet):
        bad.loc[no_internet[0], "Streaming TV"] = "Yes"  # add-on without internet
    bad.loc[7, "Tenure in Months"] = ""  # missing value
    bad.loc[8, "Under 30"] = "Yes" if bad.loc[8, "Age"] >= 30 else "No"  # contradiction
    bad.loc[9, "Total Charges"] = 1.0  # lower than one monthly charge
    return bad


def main() -> None:
    """Write every app data file."""
    train, test = split_raw()
    rng = np.random.default_rng(RANDOM_STATE)
    for directory in (APP_DATA_DIR, DEMO_BATCH_DIR, SAMPLE_BATCH_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    n_eda = write_eda_file()
    history_cols = [col for col in UPLOAD_COLS if col != "Customer ID"] + [TARGET_COL]
    train[history_cols].to_csv(APP_HISTORY_PATH, index=False)
    test_upload = to_upload_layout(test)
    test_upload.to_csv(APP_TEST_CUSTOMERS_PATH, index=False)

    stayers = train[train[TARGET_COL].eq("No")]
    reference = {
        "short_tenure_months": 12,
        "reference_tenure_months": float(stayers["Tenure in Months"].median()),
        "high_monthly_charge": float(train["Monthly Charge"].quantile(0.75)),
        "reference_monthly_charge": float(train["Monthly Charge"].median()),
        "source": "training split of the IBM Telco Customer Churn sample",
    }
    APP_REFERENCE_PATH.write_text(json.dumps(reference, indent=2))

    # Seed batches: disjoint draws from the test split
    pool = test_upload.copy()
    for name, share in zip(DEMO_BATCH_NAMES, DEMO_MTM_SHARES, strict=True):
        batch = sample_with_mtm_share(pool, DEMO_BATCH_SIZE, share, rng)
        pool = pool.drop(index=batch.index)
        slug = name.lower().replace(" ", "_").replace("-", "_")
        batch.to_csv(DEMO_BATCH_DIR / f"{slug}.csv", index=False)

    # Sample uploads (from the customers not used in the seed history)
    clean = pool.sample(n=200, random_state=RANDOM_STATE).reset_index(drop=True)
    clean.to_csv(SAMPLE_BATCH_DIR / "sample_batch_no_errors.csv", index=False)
    build_error_sample(clean).to_csv(
        SAMPLE_BATCH_DIR / "sample_batch_with_errors.csv", index=False
    )
    risky = pool[
        pool["Contract"].eq("Month-to-Month") & pool["Number of Referrals"].eq(0)
    ]
    risky.sample(n=min(150, len(risky)), random_state=RANDOM_STATE).to_csv(
        SAMPLE_BATCH_DIR / "sample_batch_high_risk.csv", index=False
    )
    logger.info(
        "Wrote app data: %d EDA rows, %d historical rows, %d test rows, "
        "%d demo batches",
        n_eda,
        len(train),
        len(test_upload),
        len(DEMO_BATCH_NAMES),
    )


if __name__ == "__main__":
    main()
