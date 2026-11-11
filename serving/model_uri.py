"""Print the MLflow URI of the latest ``churn_policy_model`` (for ``make serve-model``).

Usage:
    python -m serving.model_uri            # latest run in the local ./mlruns
"""

from __future__ import annotations

import sys

import mlflow

from telco_churn.config import MLRUNS_DIR

EXPERIMENT_NAME = "gradient_boost_tuned"
RUN_NAME = "gb_optuna_f2_calibrated"
MODEL_NAME = "churn_policy_model"


def latest_model_uri() -> str:
    """URI of the policy model logged by the most recent tracked run.

    Raises:
        LookupError: If no run has been logged yet.
    """
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise LookupError("No training run found. Run `make train` first.")
    return f"runs:/{runs.iloc[0]['run_id']}/{MODEL_NAME}"


if __name__ == "__main__":
    try:
        print(latest_model_uri())
    except LookupError as error:
        sys.exit(str(error))
