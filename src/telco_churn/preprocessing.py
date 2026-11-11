"""Transform raw customer records into the model's feature matrix.

Replays the notebook's data preparation (Section 5) with the artifacts it saved,
in the same order, so new customers are processed exactly like the test split:

1. Fill structural missing values (no offer / no internet).
2. Cap continuous columns at the training IQR limits (``outlier_bounds.joblib``).
3. ``log1p`` the skewed columns (``log_transform_cols.joblib``).
4. One-hot encode and align to the training schema (``predictor_encoder.joblib``).
5. Standardize each numerical column (``<column>_scaler.joblib``).
6. Replace the collinear billing columns with ``Pc_1``/``Pc_2`` (``pca.joblib``).

This module has no Streamlit dependency, so the app, the tests and the serving
scripts can share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from telco_churn.config import PREPROCESSING_DIR, TARGET_COL

# Columns removed before modelling (notebook Section 3.2 / 5.1)
IDENTIFIER_GEO_COLS = [
    "Customer ID",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "Quarter",
]
TARGET_LEAKAGE_COLS = [
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
]
LEAKAGE_SUSPECT_COLS = ["Satisfaction Score"]
# Dropped by the chi-square screening on the training split (notebook Insight 9)
INDEPENDENT_COLS = ["Gender", "Phone Service"]
STRUCTURAL_MISSING_FILL = {"Offer": "No", "Internet Type": "No"}
PCA_COLS = [
    "Total Revenue",
    "Total Charges",
    "Tenure in Months",
    "Total Long Distance Charges",
    "Monthly Charge",
]
PC_NAMES = ["Pc_1", "Pc_2"]

CATEGORICAL_COLS = [
    "Under 30",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Referred a Friend",
    "Offer",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
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
]
NUMERICAL_COLS = [
    "Age",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "CLTV",
]
MODEL_INPUT_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


@dataclass(frozen=True)
class Preprocessor:
    """Fitted preprocessing parameters loaded from ``artifacts/preprocessing``."""

    outlier_bounds: dict[str, tuple[float, float]]
    log_transform_cols: list[str]
    encoded_columns: list[str]
    scalers: dict[str, object]
    pca: object

    @classmethod
    def load(cls, directory: Path = PREPROCESSING_DIR) -> Preprocessor:
        """Load every preprocessing artifact saved by the notebook.

        Args:
            directory: Folder that holds the ``.joblib`` files.

        Returns:
            A ready-to-use preprocessor.

        Raises:
            FileNotFoundError: If an artifact is missing.
        """
        encoded = [
            col
            for col in joblib.load(directory / "predictor_encoder.joblib")
            if col != TARGET_COL
        ]
        scalers = {
            col: joblib.load(directory / f"{col}_scaler.joblib")
            for col in NUMERICAL_COLS
        }
        return cls(
            outlier_bounds=joblib.load(directory / "outlier_bounds.joblib"),
            log_transform_cols=list(
                joblib.load(directory / "log_transform_cols.joblib")
            ),
            encoded_columns=encoded,
            scalers=scalers,
            pca=joblib.load(directory / "pca.joblib"),
        )

    def transform(self, raw: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
        """Turn raw customer rows into model features.

        Args:
            raw: One row per customer with at least ``MODEL_INPUT_COLS``.
                Values must already be validated (see ``app/models/validation``).
            schema: Final feature order (``features_schema.txt``).

        Returns:
            Feature matrix with the columns of ``schema`` and the index of ``raw``.

        Raises:
            KeyError: If a required input column is missing.
        """
        missing = [col for col in MODEL_INPUT_COLS if col not in raw.columns]
        if missing:
            raise KeyError(f"Missing input columns: {missing}")

        data = raw[MODEL_INPUT_COLS].copy()
        data = data.fillna(value=STRUCTURAL_MISSING_FILL)
        data[NUMERICAL_COLS] = data[NUMERICAL_COLS].astype(float)

        # 2. Winsorize, 3. log1p (all logged columns are non-negative)
        for col, (lower, upper) in self.outlier_bounds.items():
            data[col] = data[col].clip(lower=lower, upper=upper)
        data[self.log_transform_cols] = np.log1p(data[self.log_transform_cols])

        # 4. One-hot encode and align to the training columns
        encoded = pd.get_dummies(data, columns=CATEGORICAL_COLS, dtype=int)
        encoded = encoded.reindex(columns=self.encoded_columns, fill_value=0)

        # 5. Standardize (scalers were fitted on NumPy arrays)
        for col, scaler in self.scalers.items():
            encoded[col] = scaler.transform(encoded[[col]].to_numpy()).ravel()

        # 6. PCA on the collinear billing features
        components = pd.DataFrame(
            self.pca.transform(encoded[PCA_COLS]),
            columns=PC_NAMES,
            index=encoded.index,
        )
        features = pd.concat([encoded.drop(columns=PCA_COLS), components], axis=1)
        return features.reindex(columns=schema)
