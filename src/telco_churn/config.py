"""Project-wide paths and constants.

Shared by the training pipeline, the serving scripts and the Streamlit app, so
every component reads and writes the same locations. Paths are resolved
relative to the repository root, which requires an editable install
(``pip install -e .``) or running from a checkout.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ===== Data =====
DATA_DIR = PROJECT_ROOT / "data"
# Raw CSV, downloaded by the notebook and not committed
RAW_DATA_PATH = DATA_DIR / "raw" / "telco.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train_pca.csv"
TEST_PATH = PROCESSED_DIR / "test_pca.csv"
PAYLOAD_DIR = DATA_DIR / "payloads"  # sample requests for the inference client

# ===== Artifacts =====
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"  # encoders, scalers, PCA, bounds
PROTOTYPE_DIR = ARTIFACTS_DIR / "prototype"  # notebook prototype (Sections 6-7)
PROTOTYPE_POLICY_PATH = PROTOTYPE_DIR / "decision_policy.json"
MODEL_DIR = ARTIFACTS_DIR / "model"  # final model exported by pipelines/train.py
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"  # journal of the tracked Optuna study
MLRUNS_DIR = PROJECT_ROOT / "mlruns"  # local MLflow store for offline runs

# ===== Modelling constants =====
TARGET_COL = "Churn Label"
RULE_COL = "Contract_Month-to-Month"  # month-to-month rule benchmark
RANDOM_STATE = 126
TEST_SIZE = 0.2  # notebook Section 3.2

# ===== Streamlit app =====
APP_DIR = PROJECT_ROOT / "app"
APP_DATA_DIR = APP_DIR / "data"  # demo data built by pipelines/build_app_data.py
APP_TEST_CUSTOMERS_PATH = APP_DATA_DIR / "test_customers.csv"
APP_HISTORY_PATH = APP_DATA_DIR / "historical_customers.csv"
APP_REFERENCE_PATH = APP_DATA_DIR / "reference_values.json"
# Full IBM sample for the exploratory dashboard (Churn Insights page)
APP_EDA_PATH = APP_DATA_DIR / "eda_customers.csv"
SAMPLE_BATCH_DIR = APP_DATA_DIR / "sample_batches"  # downloadable demo uploads
DEMO_BATCH_DIR = APP_DATA_DIR / "demo_batches"  # pre-scored seed batches
BRAND_DIR = APP_DIR / "assets" / "brand"

BRAND_NAME = "MurphysTelecom"
DEMO_LABEL = f"Demo data · {BRAND_NAME} is a fictional company"
FOOTER_TEXT = (
    f"Portfolio prototype · {BRAND_NAME} is a fictional company · "
    "Data: IBM Telco Customer Churn sample."
)

# Palette tokens (docs/UI_UX_REQUIREMENTS.md, Section 3.3)
PALETTE = {
    "brand-orange": "#FF6A13",
    "brand-orange-dark": "#C2410C",
    "ink": "#1B1B1F",
    "slate": "#475467",
    "surface": "#FFFFFF",
    "surface-tint": "#FFF4EC",
    "border": "#E7E7EA",
    "status-critical": "#B42318",
    "status-watch": "#FDB022",
    "status-ok": "#067647",
    "info": "#2E5AAC",
    "context-grey": "#98A2B3",
}

# Default alert settings (agreed with the project owner on 2026-11-11)
DEFAULT_ALERT_SETTINGS = {
    "watch_threshold": 0.27,  # predicted churn rate of a batch
    "critical_threshold": 0.32,
    "plan_gap_pp": 20.0,  # plan rate above batch average, percentage points
    "min_group_size": 30,  # smallest plan group that may raise an alert
}
OBSERVED_CHURN_RATE = 0.265  # IBM sample, shown as a reference point only
MAX_UPLOAD_MB = 5
MAX_UPLOAD_ROWS = 5_000
