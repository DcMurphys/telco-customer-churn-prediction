"""Request payloads and response parsing for the serving stack.

Pure Python + pandas (no FastAPI), so the logic is unit-tested without the
serving dependencies. Payloads use MLflow's ``dataframe_split`` format with the
model features of ``data/processed/test_pca.csv`` in ``features_schema.txt``
order; the model is ``churn_policy_model`` (see ``telco_churn.policy_model``).

Usage (write one sample request to ``data/payloads/payload_input.json``):
    python -m serving.payloads
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from telco_churn.config import MODEL_DIR, PAYLOAD_DIR, TARGET_COL, TEST_PATH

PAYLOAD_PATH = PAYLOAD_DIR / "payload_input.json"
SCHEMA_PATH = MODEL_DIR / "features_schema.txt"
OUTCOMES = {
    (1, 1): "true_positive",
    (1, 0): "false_positive",
    (0, 1): "false_negative",
    (0, 0): "true_negative",
}


@dataclass(frozen=True)
class Prediction:
    """One customer's result from ``churn_policy_model``."""

    churn_probability: float
    flagged: int
    risk_level: str


def load_schema(path: Path = SCHEMA_PATH) -> list[str]:
    """Feature order expected by the model."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_test_split(path: Path = TEST_PATH) -> pd.DataFrame:
    """Processed test split with the 0/1 target (real customers, not dummies)."""
    return pd.read_csv(path)


def build_request(
    test_split: pd.DataFrame,
    schema: list[str],
    rng: np.random.Generator,
    n_rows: int = 1,
) -> tuple[dict, list[int]]:
    """Sample customers and build an MLflow ``dataframe_split`` request.

    Returns:
        The request body and the true labels (1 = churned) of the sampled rows.
    """
    rows = test_split.sample(n=n_rows, random_state=int(rng.integers(1_000_000_000)))
    body = {
        "dataframe_split": {
            "columns": schema,
            "data": rows[schema].astype(float).to_numpy().tolist(),
        }
    }
    return body, rows[TARGET_COL].astype(int).tolist()


def feature_values(body: dict) -> dict[str, list[float]]:
    """Numeric feature values per column of a ``dataframe_split`` request."""
    split = body.get("dataframe_split") if isinstance(body, dict) else None
    if not isinstance(split, dict):
        return {}
    columns, data = split.get("columns", []), split.get("data", [])
    values: dict[str, list[float]] = {name: [] for name in columns}
    for row in data:
        for name, value in zip(columns, row, strict=False):
            if isinstance(value, int | float) and not isinstance(value, bool):
                values[name].append(float(value))
    return values


def parse_predictions(response: dict) -> list[Prediction]:
    """Read ``churn_policy_model`` output from an MLflow response.

    Raises:
        ValueError: If the response does not contain the expected fields.
    """
    records = response.get("predictions") if isinstance(response, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Response has no 'predictions' list")
    try:
        return [
            Prediction(
                churn_probability=float(r["churn_probability"]),
                flagged=int(r["flagged"]),
                risk_level=str(r["risk_level"]),
            )
            for r in records
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unexpected prediction format: {records[0]!r}") from error


def outcome(flagged: int, actual: int) -> str:
    """Confusion-matrix cell of one prediction once the true label is known.

    Raises:
        ValueError: If either value is not 0 or 1.
    """
    key = (int(flagged), int(actual))
    if key not in OUTCOMES:
        raise ValueError(f"prediction and actual must be 0 or 1, got {key}")
    return OUTCOMES[key]


def write_payload(path: Path = PAYLOAD_PATH, seed: int | None = None) -> Path:
    """Write one sample request (and its true label) to a JSON file."""
    body, actual = build_request(
        load_test_split(), load_schema(), np.random.default_rng(seed)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, "actual": actual}, indent=2))
    return path


if __name__ == "__main__":
    print(f"Wrote {write_payload()}")
