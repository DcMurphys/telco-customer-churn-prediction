"""MLflow pyfunc model: classifier + calibrator + decision policy.

Logged by ``pipelines/train.py`` as ``churn_policy_model`` and served with
``mlflow models serve``. It returns what the Streamlit app shows (calibrated
probability, contact flag, risk level) instead of the classifier's default
0.5 cut-off on undersampled scores.

Input: model features in ``features_schema.txt`` order (as in
``data/processed/test_pca.csv``). Output: one row per customer with
``churn_probability``, ``flagged`` and ``risk_level``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow.pyfunc
import pandas as pd

from telco_churn.policy import Thresholds, apply_policy

# Artifact keys expected in ``context.artifacts``
ARTIFACT_KEYS = ("model", "calibrator", "decision_policy", "features_schema")


class ChurnPolicyModel(mlflow.pyfunc.PythonModel):
    """Serve calibrated probabilities and decisions in one call."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the classifier, calibrator, thresholds and feature order."""
        artifacts = context.artifacts
        self.model = joblib.load(artifacts["model"])
        self.calibrator = joblib.load(artifacts["calibrator"])
        self.thresholds = Thresholds.from_file(artifacts["decision_policy"])
        self.schema = Path(artifacts["features_schema"]).read_text().splitlines()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame,
        params: dict | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Score customers given their model features.

        Raises:
            ValueError: If a feature of the schema is missing or empty.
        """
        features = model_input.reindex(columns=self.schema)
        empty = features.columns[features.isna().any()].tolist()
        if empty:
            raise ValueError(f"Missing or empty feature values: {empty}")
        scores = self.model.predict_proba(features)[:, 1]
        return apply_policy(scores, self.calibrator, self.thresholds)
