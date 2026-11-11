"""Tracked training of the churn model with Optuna, MLflow and DagsHub.

This script repeats the modelling prototype of
``notebooks/01_churn_analysis_and_modelling.ipynb``
(Sections 6-7) as a reproducible, tracked run:

1. Score two baselines (logistic regression, default Gradient Boosting) with
   5-fold stratified cross-validation on the balanced training split.
2. Tune Gradient Boosting with Optuna. The study is stored in a journal file
   (one JSON record per line) and resumes after an interruption.
3. Select the candidate with the best cross-validated F2 score.
4. Calibrate its probabilities to the real churn rate with isotonic regression
   on out-of-fold predictions, re-weighting non-churners to undo undersampling.
5. Choose two decision thresholds on the out-of-fold predictions: the recall
   threshold (criterion D3) and the break-even threshold (campaign economics).
6. Evaluate once on the test split against the month-to-month rule and log
   everything to MLflow on DagsHub.

Inputs (exported by the notebook, Sections 5.4 and 6.6):
    data/processed/train_pca.csv, data/processed/test_pca.csv,
    artifacts/prototype/decision_policy.json

Outputs:
    MLflow run on DagsHub (or ./mlruns with --offline) with two models:
    ``model`` (the classifier) and ``churn_policy_model`` (classifier +
    calibrator + decision policy, served by ``mlflow models serve``), plus a
    local copy of the model, calibrator, decision policy, feature schema and
    reports in artifacts/model/.

Usage (from the repository root):
    make train                              # offline: log to ./mlruns
    python -m pipelines.train               # log to DagsHub (needs a token)
    python -m pipelines.train --offline --n-trials 20
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.utils import estimator_html_repr

import telco_churn
from telco_churn.config import (
    MLRUNS_DIR,
    MODEL_DIR,
    OPTUNA_DIR,
    PROTOTYPE_POLICY_PATH,
    RANDOM_STATE,
    RULE_COL,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
)
from telco_churn.policy import Thresholds, apply_policy, calibrate
from telco_churn.policy_model import ChurnPolicyModel

# ===== Configuration =====
POLICY_PATH = PROTOTYPE_POLICY_PATH

# Tracking repository on DagsHub (unchanged from the original script)
DAGSHUB_REPO_OWNER = "dharma.dwicahaya"
DAGSHUB_REPO_NAME = "telco-customer-churn-prediction"  # same name as on GitHub
MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
)
EXPERIMENT_NAME = "gradient_boost_tuned"
RUN_NAME = "gb_optuna_f2_calibrated"
PIPELINE_VERSION = "v2-optuna-f2-calibrated"

CV_FOLDS = 5
N_TRIALS = 50
F_BETA = 2  # recall weighted twice as much as precision (criterion D3)
TARGET_RECALL = 0.85  # criterion D3
OPTUNA_STUDY_PREFIX = "gb_churn_tracked"
POLICY_MODEL_NAME = "churn_policy_model"  # pyfunc: model + calibrator + policy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===== Data =====
@dataclass(frozen=True)
class Economics:
    """Illustrative campaign economics exported by the notebook (Section 2.4).

    Attributes:
        clv: Customer value saved when a churner is retained.
        offer_cost: Cost of one retention offer.
        contact_cost: Cost of one contact.
        acceptance_rate: Share of contacted churners who accept and stay.
    """

    clv: float
    offer_cost: float
    contact_cost: float
    acceptance_rate: float

    @property
    def breakeven_threshold(self) -> float:
        """Churn probability above which one contact has positive expected value."""
        return (self.offer_cost + self.contact_cost) / (
            self.acceptance_rate * (self.clv - self.offer_cost) + self.offer_cost
        )

    def net_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Expected net value of contacting the flagged customers.

        Args:
            y_true: True labels (1 = churn).
            y_pred: Predicted labels (1 = contacted).

        Returns:
            Expected net value in the currency units of CLTV.
        """
        churned = np.asarray(y_true)[np.asarray(y_pred).astype(bool)]
        gains = self.acceptance_rate * (self.clv - self.offer_cost) * churned
        costs = self.offer_cost * (1 - churned) + self.contact_cost
        return float((gains - costs).sum())


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one processed split and separate features from the target.

    Args:
        path: CSV file exported by the notebook (Section 5.4).

    Returns:
        Features and target (1 = churn).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Export it from the notebook first.")
    data = pd.read_csv(path)
    return data.drop(columns=TARGET_COL), data[TARGET_COL]


def load_policy(path: Path) -> tuple[float, Economics | None]:
    """Read the real churn rate and, if present, the campaign economics.

    Args:
        path: ``decision_policy.json`` exported by the notebook (Section 6.6).

    Returns:
        The real churn rate of the training split before undersampling, and the
        economics, or ``None`` if the file does not contain them.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Export it from the notebook first.")
    policy = json.loads(path.read_text())
    economics_keys = (
        "clv_reference",
        "offer_cost",
        "contact_cost",
        "offer_acceptance_rate",
    )
    if all(key in policy for key in economics_keys):
        economics = Economics(*(float(policy[key]) for key in economics_keys))
    else:
        logger.warning(
            "No campaign economics in %s; net value is not logged.", path.name
        )
        economics = None
    return float(policy["churn_prior"]), economics


def fingerprint(data: pd.DataFrame) -> str:
    """Return a short, stable hash of a DataFrame (used to name the Optuna study)."""
    hashed = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()[:8]


# ===== Modelling =====
def make_cv() -> StratifiedKFold:
    """Return the shared stratified cross-validation splitter."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


F_BETA_SCORER = make_scorer(fbeta_score, beta=F_BETA)
CV_SCORING = {f"f{F_BETA}": F_BETA_SCORER, "recall": "recall", "roc_auc": "roc_auc"}


def cross_validate_model(
    model: ClassifierMixin, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Score a model with the shared cross-validation.

    Precision is not reported: on balanced folds it overstates the precision
    at the real churn rate.

    Args:
        model: Unfitted classifier.
        features: Balanced training features.
        target: Balanced training target.

    Returns:
        Mean and standard deviation of every metric in ``CV_SCORING``.
    """
    scores = cross_validate(
        model, features, target, cv=make_cv(), scoring=CV_SCORING, n_jobs=-1
    )
    results = {}
    for name in CV_SCORING:
        results[f"cv_{name}_mean"] = float(scores[f"test_{name}"].mean())
        results[f"cv_{name}_std"] = float(scores[f"test_{name}"].std())
    return results


def gb_search_space(trial: optuna.Trial) -> dict[str, object]:
    """Sample one Gradient Boosting configuration (same ranges as the notebook)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_features": trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2"]
        ),
    }


def tune_gradient_boosting(
    features: pd.DataFrame, target: pd.Series, n_trials: int, study_name: str
) -> optuna.Study:
    """Run (or resume) the Optuna study that maximizes cross-validated F2.

    Args:
        features: Balanced training features.
        target: Balanced training target.
        n_trials: Total number of completed trials the study should reach.
        study_name: Study name; it includes a fingerprint of the training data.

    Returns:
        The study with all trials.
    """

    def objective(trial: optuna.Trial) -> float:
        model = GradientBoostingClassifier(
            random_state=RANDOM_STATE, **gb_search_space(trial)
        )
        scores = cross_validate(
            model, features, target, cv=make_cv(), scoring=F_BETA_SCORER, n_jobs=-1
        )
        return float(scores["test_score"].mean())

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(
            JournalFileBackend(str(OPTUNA_DIR / f"{study_name}.log"))
        ),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, n_trials - completed))
    logger.info(
        "Optuna: %d trials, best CV F%d = %.4f",
        len(study.trials),
        F_BETA,
        study.best_value,
    )
    return study


def fit_calibrator(
    model: ClassifierMixin,
    features: pd.DataFrame,
    target: pd.Series,
    churn_prior: float,
) -> tuple[IsotonicRegression, np.ndarray]:
    """Calibrate out-of-fold scores to the real churn rate.

    The model is trained on balanced data, so its raw scores overstate churn
    risk. Isotonic regression is fitted on out-of-fold scores, with every
    non-churner weighted by ``1 / beta`` (``beta`` = share of non-churners kept
    by undersampling). This restores the real class mix, so the calibrated
    output is an estimate of the real churn probability (Dal Pozzolo et al., 2015).

    Args:
        model: Unfitted classifier.
        features: Balanced training features.
        target: Balanced training target.
        churn_prior: Real churn rate of the training split before undersampling.

    Returns:
        The fitted calibrator and the calibrated out-of-fold probabilities.
    """
    oof_scores = cross_val_predict(
        model, features, target, cv=make_cv(), method="predict_proba", n_jobs=-1
    )[:, 1]
    beta = churn_prior / (1 - churn_prior)
    weights = np.where(target.to_numpy() == 1, 1.0, 1.0 / beta)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(oof_scores, target, sample_weight=weights)
    return calibrator, calibrator.predict(oof_scores)


def recall_threshold(
    target: pd.Series, proba: np.ndarray, churn_prior: float, min_recall: float
) -> float:
    """Return the most precise threshold that keeps recall >= ``min_recall``.

    Precision at the real churn rate is recomputed from the true and false
    positive rates, because balanced data overstates it.

    Args:
        target: Balanced training target.
        proba: Calibrated out-of-fold churn probabilities.
        churn_prior: Real churn rate.
        min_recall: Minimum recall to keep (criterion D3).

    Returns:
        The chosen probability threshold.
    """
    positives = target.to_numpy() == 1
    best_threshold, best_precision = float(proba.min()), -1.0
    for threshold in np.unique(proba):
        flagged = proba >= threshold
        tpr, fpr = flagged[positives].mean(), flagged[~positives].mean()
        if tpr < min_recall:
            continue
        precision = churn_prior * tpr / (churn_prior * tpr + (1 - churn_prior) * fpr)
        if precision > best_precision:
            best_threshold, best_precision = float(threshold), precision
    return best_threshold


def test_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    economics: Economics | None,
    proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute churn-class metrics on the test split.

    Args:
        y_true: True labels (1 = churn).
        y_pred: Predicted labels (1 = contacted).
        economics: Campaign economics, or ``None`` to skip net value.
        proba: Optional calibrated probabilities for threshold-free metrics.

    Returns:
        Recall, precision, F1, F-beta, share contacted, lift and, if available,
        net value, PR-AUC, ROC-AUC and Brier score.
    """
    metrics = {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        f"f{F_BETA}": fbeta_score(y_true, y_pred, beta=F_BETA, zero_division=0),
        "contacted_pct": 100 * float(np.mean(y_pred)),
    }
    metrics["lift"] = metrics["precision"] / float(y_true.mean())
    if economics is not None:
        metrics["net_value"] = economics.net_value(y_true, y_pred)
    if proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, proba)
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
        metrics["brier"] = brier_score_loss(y_true, proba)
    return {key: float(value) for key, value in metrics.items()}


# ===== MLflow logging helpers =====
def setup_tracking(offline: bool) -> None:
    """Choose the MLflow tracking server.

    * ``--offline``: a local ./mlruns folder.
    * ``MLFLOW_TRACKING_URI`` set (CI): that server; MLflow reads the
      credentials from ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``.
    * otherwise: DagsHub via ``dagshub.init`` (token or interactive login).
    """
    if offline:
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        import dagshub

        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True
        )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_json(
    data: dict, artifact_dir: Path, name: str, subdir: str = "reports"
) -> None:
    """Write a dictionary to JSON and log it as an MLflow artifact."""
    path = artifact_dir / name
    path.write_text(json.dumps(data, indent=2, default=str))
    mlflow.log_artifact(str(path), artifact_path=subdir)


def log_frame(data: pd.DataFrame, artifact_dir: Path, name: str) -> None:
    """Write a DataFrame to CSV and log it as an MLflow artifact."""
    path = artifact_dir / name
    data.to_csv(path, index=True)
    mlflow.log_artifact(str(path), artifact_path="reports")


def log_model(model: ClassifierMixin, features: pd.DataFrame) -> None:
    """Log the fitted classifier with its signature and an input example."""
    example = features.head(5).astype(float)  # one-hot ints accept floats too
    signature = infer_signature(example, model.predict(example))
    name_argument = (
        "name"
        if "name" in inspect.signature(mlflow.sklearn.log_model).parameters
        else "artifact_path"
    )
    mlflow.sklearn.log_model(
        sk_model=model,
        input_example=example,
        signature=signature,
        **{name_argument: "model"},
    )


def log_evaluation_figures(
    y_true: pd.Series, proba: np.ndarray, y_pred: np.ndarray, model_name: str
) -> None:
    """Log the confusion matrix, PR curve, ROC curve and calibration curve."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No", "Yes"], colorbar=False, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Confusion matrix at the recall threshold (test)")
    PrecisionRecallDisplay.from_predictions(
        y_true, proba, name=model_name, plot_chance_level=True, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Precision-recall curve (test)")
    RocCurveDisplay.from_predictions(y_true, proba, name=model_name, ax=axes[1, 0])
    axes[1, 0].set_title("ROC curve (test)")
    CalibrationDisplay.from_predictions(
        y_true, proba, n_bins=10, strategy="quantile", name=model_name, ax=axes[1, 1]
    )
    axes[1, 1].set_title("Calibration curve (test)")
    fig.tight_layout()
    mlflow.log_figure(fig, "figures/test_evaluation.png")
    plt.close(fig)


def log_policy_model(features: pd.DataFrame, artifact_dir: Path) -> None:
    """Log classifier + calibrator + decision policy as one pyfunc model.

    ``mlflow models serve`` on this model returns the calibrated probability,
    contact flag and risk level used by the app (see ``telco_churn.policy``).
    """
    artifacts = {
        "model": str(artifact_dir / "model.joblib"),
        "calibrator": str(artifact_dir / "calibrator.joblib"),
        "decision_policy": str(artifact_dir / "decision_policy.json"),
        "features_schema": str(artifact_dir / "features_schema.txt"),
    }
    example = features.head(5).astype(float)  # one-hot ints accept floats too
    policy_model = ChurnPolicyModel()
    output = apply_policy(
        joblib.load(artifacts["model"]).predict_proba(example)[:, 1],
        joblib.load(artifacts["calibrator"]),
        Thresholds.from_file(Path(artifacts["decision_policy"])),
    )
    name_argument = (
        "name"
        if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters
        else "artifact_path"
    )
    mlflow.pyfunc.log_model(
        python_model=policy_model,
        artifacts=artifacts,
        code_paths=[str(Path(telco_churn.__file__).parent)],
        # Explicit pins: the local telco_churn package ships via code_paths
        pip_requirements=[
            f"{name}=={version(name)}"
            for name in ("mlflow", "scikit-learn", "pandas", "numpy", "joblib")
        ],
        input_example=example,
        signature=infer_signature(example, output),
        **{name_argument: POLICY_MODEL_NAME},
    )


def write_run_info(artifact_dir: Path) -> None:
    """Record the run ID so CI and serving can find the logged models."""
    run = mlflow.active_run()
    info = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "experiment_name": EXPERIMENT_NAME,
        "policy_model_uri": f"runs:/{run.info.run_id}/{POLICY_MODEL_NAME}",
        "tracking": "local"
        if mlflow.get_tracking_uri().startswith("file:")
        else "remote",
    }
    (artifact_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    logger.info("Run ID: %s", run.info.run_id)


# ===== Workflow steps =====
@dataclass
class TrainingData:
    """Balanced training split and untouched test split."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def compare_candidates(
    data: TrainingData, n_trials: int, study_name: str
) -> tuple[pd.DataFrame, dict[str, ClassifierMixin], optuna.Study]:
    """Score the baselines and the tuned model, each in a nested MLflow run."""
    candidates: dict[str, ClassifierMixin] = {
        "logistic_regression": LogisticRegression(max_iter=1_000),
        "gb_default": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    study = tune_gradient_boosting(data.x_train, data.y_train, n_trials, study_name)
    candidates["gb_tuned"] = GradientBoostingClassifier(
        random_state=RANDOM_STATE, **study.best_params
    )
    cv_table = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name, nested=True):
            scores = cross_validate_model(model, data.x_train, data.y_train)
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            mlflow.log_metrics(scores)
            cv_table[name] = scores
    df_cv = pd.DataFrame(cv_table).T.sort_values(f"cv_f{F_BETA}_mean", ascending=False)
    return df_cv, candidates, study


def evaluate_on_test(
    data: TrainingData,
    test_proba: np.ndarray,
    thresholds: dict[str, float],
    economics: Economics | None,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Compare the model at each threshold with the naive benchmarks."""
    predictions = {
        "majority_class": np.zeros(len(data.y_test), dtype=int),
        "month_to_month_rule": data.x_test[RULE_COL].to_numpy().astype(int),
        **{
            f"model_at_{k}_threshold": (test_proba >= v).astype(int)
            for k, v in thresholds.items()
        },
    }
    report = {
        name: test_metrics(
            data.y_test,
            y_pred,
            economics,
            test_proba if name.startswith("model") else None,
        )
        for name, y_pred in predictions.items()
    }
    return report, predictions


def success_criteria(
    report: dict[str, dict[str, float]], economics: Economics | None
) -> dict[str, bool]:
    """Criteria D3, D4 and B3 (part 1) from the notebook (Section 2.4)."""
    model_row = report["model_at_recall_threshold"]
    rule_row = report["month_to_month_rule"]
    criteria = {
        "D3_recall_at_least_target": model_row["recall"] >= TARGET_RECALL,
        "D4_precision_above_rule": model_row["precision"] > rule_row["precision"],
        "D4_f1_above_rule": model_row["f1"] > rule_row["f1"],
    }
    if economics is not None:
        criteria["B3_net_value_positive"] = model_row["net_value"] > 0
    return {key: bool(value) for key, value in criteria.items()}


def importance_table(model: ClassifierMixin, data: TrainingData) -> pd.DataFrame:
    """Permutation importance on the test split (drop in PR-AUC)."""
    importance = permutation_importance(
        model,
        data.x_test,
        data.y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "mean_drop_pr_auc": importance.importances_mean,
            "std": importance.importances_std,
        },
        index=data.x_test.columns,
    ).sort_values("mean_drop_pr_auc", ascending=False)


# ===== Main workflow =====
def main(n_trials: int, offline: bool) -> None:
    """Run the tracked training and evaluation."""
    x_train, y_train = load_split(TRAIN_PATH)
    x_test, y_test = load_split(TEST_PATH)
    data = TrainingData(x_train, y_train, x_test, y_test)
    churn_prior, economics = load_policy(POLICY_PATH)
    study_name = (
        f"{OPTUNA_STUDY_PREFIX}_{fingerprint(pd.concat([x_train, y_train], axis=1))}"
    )

    setup_tracking(offline)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags(
            {"pipeline_version": PIPELINE_VERSION, "optuna_study": study_name}
        )
        mlflow.log_params(
            {
                "cv_folds": CV_FOLDS,
                "n_trials": n_trials,
                "f_beta": F_BETA,
                "target_recall": TARGET_RECALL,
                "random_state": RANDOM_STATE,
                "churn_prior": churn_prior,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "n_features": x_train.shape[1],
            }
        )

        # 1-3. Baselines, tuning and selection on cross-validated F2
        df_cv, candidates, study = compare_candidates(data, n_trials, study_name)
        selected_name = str(df_cv.index[0])
        selected_model = clone(candidates[selected_name]).fit(x_train, y_train)
        mlflow.set_tag("selected_model", selected_name)
        mlflow.log_params({f"best_{k}": str(v) for k, v in study.best_params.items()})
        mlflow.log_metric("optuna_best_cv_f2", study.best_value)
        logger.info("Selected model: %s", selected_name)

        # 4. Calibration and 5. thresholds (out-of-fold, training split only)
        calibrator, oof_proba = fit_calibrator(
            candidates[selected_name], x_train, y_train, churn_prior
        )
        thresholds = {
            "recall": recall_threshold(y_train, oof_proba, churn_prior, TARGET_RECALL)
        }
        if economics is not None:
            thresholds["breakeven"] = economics.breakeven_threshold
        mlflow.log_metrics({f"threshold_{k}": v for k, v in thresholds.items()})

        # 6. Single evaluation on the test split
        test_proba = calibrate(calibrator, selected_model.predict_proba(x_test)[:, 1])
        report, predictions = evaluate_on_test(data, test_proba, thresholds, economics)
        for name, metrics in report.items():
            mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})
        criteria = success_criteria(report, economics)
        mlflow.set_tags({f"criterion_{k}": str(v) for k, v in criteria.items()})

        decision_policy = {
            "model": selected_name,
            "pipeline_version": PIPELINE_VERSION,
            "churn_prior": churn_prior,
            "calibration": "isotonic regression on out-of-fold scores",
            "calibrator_artifact": "model_extras/calibrator.joblib",
            "recall_threshold": thresholds["recall"],
            "breakeven_threshold": thresholds.get("breakeven"),
            "target_recall": TARGET_RECALL,
        }

        # 7. Export: local copy in artifacts/model/ (app, serving) + MLflow
        artifact_dir = MODEL_DIR
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_model(selected_model, x_train)
        joblib.dump(selected_model, artifact_dir / "model.joblib")
        joblib.dump(calibrator, artifact_dir / "calibrator.joblib")
        mlflow.log_artifact(
            str(artifact_dir / "calibrator.joblib"), artifact_path="model_extras"
        )
        log_json(decision_policy, artifact_dir, "decision_policy.json", "model_extras")
        schema_path = artifact_dir / "features_schema.txt"
        schema_path.write_text("\n".join(x_train.columns))
        mlflow.log_artifact(str(schema_path), artifact_path="model_extras")
        log_policy_model(x_train, artifact_dir)

        log_frame(df_cv, artifact_dir, "cv_comparison.csv")
        log_frame(pd.DataFrame(report).T, artifact_dir, "test_report.csv")
        log_frame(
            importance_table(selected_model, data),
            artifact_dir,
            "permutation_importance.csv",
        )
        log_json(criteria, artifact_dir, "criteria_checks.json")
        log_frame(study.trials_dataframe(), artifact_dir, "optuna_trials.csv")
        log_json(
            {"best_cv_f2": study.best_value, "params": study.best_params},
            artifact_dir,
            "optuna_best_params.json",
        )
        estimator_path = artifact_dir / "estimator.html"
        estimator_path.write_text(estimator_html_repr(selected_model), encoding="utf-8")
        mlflow.log_artifact(str(estimator_path), artifact_path="reports")

        log_evaluation_figures(
            y_test, test_proba, predictions["model_at_recall_threshold"], selected_name
        )
        logger.info("Criteria: %s", criteria)
        write_run_info(artifact_dir)
        logger.info("Artifacts written to %s", artifact_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Optuna trials")
    parser.add_argument(
        "--offline", action="store_true", help="log to ./mlruns instead of DagsHub"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(n_trials=args.n_trials, offline=args.offline)
