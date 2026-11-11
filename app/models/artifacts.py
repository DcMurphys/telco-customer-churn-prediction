"""Load the model, calibrator, decision policy, reports and demo data files."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from telco_churn.config import (
    APP_HISTORY_PATH,
    APP_REFERENCE_PATH,
    APP_TEST_CUSTOMERS_PATH,
    DEMO_BATCH_DIR,
    MODEL_DIR,
    PREPROCESSING_DIR,
    SAMPLE_BATCH_DIR,
    TARGET_COL,
    TEST_PATH,
)
from telco_churn.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class ArtifactsMissingError(RuntimeError):
    """Raised when the model artifacts have not been generated yet."""


@dataclass(frozen=True)
class DecisionPolicy:
    """Thresholds from ``decision_policy.json`` (never hard-coded in the app)."""

    model_name: str
    recall_threshold: float
    breakeven_threshold: float | None
    target_recall: float
    churn_prior: float
    pipeline_version: str
    calibration: str


@dataclass(frozen=True)
class ModelBundle:
    """Everything needed to score raw customers."""

    preprocessor: Preprocessor
    model: object
    calibrator: object
    policy: DecisionPolicy
    schema: list[str]
    trained_at: datetime


def _load_preprocessor(directory: Path) -> Preprocessor:
    # The notebook saved these scalers/PCA with scikit-learn 1.6.1. Their output
    # under 1.7.0 was verified to match test_pca.csv exactly (tests/), so the
    # version warning is logged once instead of shown to users.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        preprocessor = Preprocessor.load(directory)
    if caught:
        logger.warning("Preprocessing artifacts: %s", caught[0].message)
    return preprocessor


def load_bundle(
    model_dir: Path = MODEL_DIR, preprocessing_dir: Path = PREPROCESSING_DIR
) -> ModelBundle:
    """Load the scoring bundle exported by ``pipelines/train.py``.

    Raises:
        ArtifactsMissingError: If a required file is missing.
    """
    required = [
        model_dir / name
        for name in (
            "model.joblib",
            "calibrator.joblib",
            "decision_policy.json",
            "features_schema.txt",
        )
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise ArtifactsMissingError(
            f"Missing model artifacts: {', '.join(missing)}. "
            "Run `python pipelines/train.py --offline` first."
        )
    raw_policy = json.loads((model_dir / "decision_policy.json").read_text())
    policy = DecisionPolicy(
        model_name=str(raw_policy.get("model", "unknown")),
        recall_threshold=float(raw_policy["recall_threshold"]),
        breakeven_threshold=(
            None
            if raw_policy.get("breakeven_threshold") is None
            else float(raw_policy["breakeven_threshold"])
        ),
        target_recall=float(raw_policy.get("target_recall", 0.85)),
        churn_prior=float(raw_policy.get("churn_prior", float("nan"))),
        pipeline_version=str(raw_policy.get("pipeline_version", "unknown")),
        calibration=str(raw_policy.get("calibration", "none")),
    )
    model_path = model_dir / "model.joblib"
    return ModelBundle(
        preprocessor=_load_preprocessor(preprocessing_dir),
        model=joblib.load(model_path),
        calibrator=joblib.load(model_dir / "calibrator.joblib"),
        policy=policy,
        schema=(model_dir / "features_schema.txt").read_text().splitlines(),
        trained_at=datetime.fromtimestamp(model_path.stat().st_mtime),
    )


def load_reports(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    """Read the evaluation reports written by ``pipelines/train.py``."""
    return {
        "test_report": pd.read_csv(model_dir / "test_report.csv", index_col=0),
        "cv_comparison": pd.read_csv(model_dir / "cv_comparison.csv", index_col=0),
        "permutation_importance": pd.read_csv(
            model_dir / "permutation_importance.csv", index_col=0
        ),
        "criteria": json.loads((model_dir / "criteria_checks.json").read_text()),
        "best_params": json.loads((model_dir / "optuna_best_params.json").read_text()),
    }


def load_test_split(path: Path = TEST_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Processed test split (features and 0/1 target) for evaluation curves."""
    data = pd.read_csv(path)
    return data.drop(columns=TARGET_COL), data[TARGET_COL]


def read_upload_csv(path_or_buffer: object) -> pd.DataFrame:
    """Read a batch file as text so validation sees the values as typed."""
    return pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False, comment="#")


def load_history(path: Path = APP_HISTORY_PATH) -> pd.DataFrame:
    """Training-split customers with observed churn (no identifiers)."""
    return pd.read_csv(path)


def load_test_customers(path: Path = APP_TEST_CUSTOMERS_PATH) -> pd.DataFrame:
    """Test-split customers in upload layout."""
    return read_upload_csv(path)


def load_reference(path: Path = APP_REFERENCE_PATH) -> dict[str, float]:
    """Reference values for the top-reasons logic."""
    return json.loads(path.read_text())


def demo_batch_files(directory: Path = DEMO_BATCH_DIR) -> list[Path]:
    """Seed batch files in chronological order (file names sort by week)."""
    order = {"oct": 10, "nov": 11}

    def key(path: Path) -> tuple[int, str]:
        month = path.stem.split("_")[1]
        return order.get(month, 99), path.stem

    return sorted(directory.glob("*.csv"), key=key)


def sample_batch_files(directory: Path = SAMPLE_BATCH_DIR) -> dict[str, Path]:
    """Downloadable sample uploads keyed by file name."""
    return {path.name: path for path in sorted(directory.glob("*.csv"))}
