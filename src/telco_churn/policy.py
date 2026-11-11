"""Decision policy shared by the Streamlit app, MLflow serving and the tests.

One place turns a model score into what the business sees:

1. calibrated churn probability (isotonic calibrator from ``pipelines/train.py``),
2. ``flagged`` = probability >= recall threshold (contact the customer),
3. risk level: High (flagged), Medium (>= break-even threshold), Low.

``telco_churn.policy_model.ChurnPolicyModel`` wraps these rules as an MLflow
pyfunc model, so ``mlflow models serve`` returns exactly what the app shows.
This module has no MLflow dependency, so the app can import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RISK_LEVELS = ("High", "Medium", "Low")


@dataclass(frozen=True)
class Thresholds:
    """Cut-offs on the calibrated churn probability."""

    recall: float
    breakeven: float | None = None

    @classmethod
    def from_policy(cls, policy: dict) -> Thresholds:
        """Read the thresholds from a ``decision_policy.json`` dictionary."""
        breakeven = policy.get("breakeven_threshold")
        return cls(
            recall=float(policy["recall_threshold"]),
            breakeven=None if breakeven is None else float(breakeven),
        )

    @classmethod
    def from_file(cls, path: Path) -> Thresholds:
        """Read the thresholds from ``decision_policy.json``."""
        return cls.from_policy(json.loads(Path(path).read_text()))


def calibrate(calibrator: object, scores: np.ndarray) -> np.ndarray:
    """Map raw model scores to calibrated churn probabilities in [0, 1]."""
    return np.clip(calibrator.predict(np.asarray(scores, dtype=float)), 0.0, 1.0)


def risk_levels(probability: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """High / Medium / Low per customer (Retention Playbook, UI/UX Section 7)."""
    probability = np.asarray(probability, dtype=float)
    breakeven = (
        thresholds.recall if thresholds.breakeven is None else thresholds.breakeven
    )
    return np.select(
        [probability >= thresholds.recall, probability >= breakeven],
        ["High", "Medium"],
        default="Low",
    )


def apply_policy(
    scores: np.ndarray, calibrator: object, thresholds: Thresholds
) -> pd.DataFrame:
    """Calibrated probability, contact flag and risk level for raw scores."""
    probability = calibrate(calibrator, scores)
    return pd.DataFrame(
        {
            "churn_probability": probability,
            "flagged": (probability >= thresholds.recall).astype(int),
            "risk_level": risk_levels(probability, thresholds),
        }
    )
