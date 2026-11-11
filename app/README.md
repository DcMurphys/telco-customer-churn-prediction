# Streamlit app: MurphysTelecom churn early warning

A portfolio prototype for the retention team of **MurphysTelecom**, a fictional U.S. carrier. It offers one plain-language interface for business users with overview, alerts, plan risk, customer check, batch scoring, churn insights (exploratory dashboard), glossary and alert settings. Statistical detail and model evaluation live in the notebook and `artifacts/model/` reports, not in the app.

## Run locally

```bash
make setup   # once: .venv with all dependencies
make data    # once: app/data/ from data/raw/telco.csv
make app     # http://localhost:8501
```

Without `make`: `pip install -r app/requirements.txt`, then `streamlit run app/streamlit_app.py` from the repository root. `.streamlit/config.toml` (theme, self-hosted font, minimal toolbar, hidden error details) is read from the repository root. Deployment to Streamlit Community Cloud is described in the root README.

## Layout (Model-View-Controller)

| Folder | Responsibility | May import |
|---|---|---|
| `models/` | Artifacts, validation rules, scoring and top reasons, plan statistics, alert rules, EDA tables, demo state | `telco_churn` only (never `streamlit`); risk levels come from `telco_churn.policy`, shared with serving |
| `controllers/` | Navigation (`router.py`), per-visitor state (`state.py`), cached resources (`resources.py`), page orchestration (`pages.py`) | `models`, `views` |
| `views/` | Pages (`views/pages/`), components (`views/components/`), shared vocabulary (`glossary.py`), playbook wording (`playbook.py`) | constants from `models.validation`; nothing that reads files or calls models |
| `streamlit_app.py` | Entry point: adds `src/` and the repository root to `sys.path`, then runs the router | `controllers` only |

## Inputs

* `artifacts/preprocessing/`: outlier bounds, log-transform columns, encoders, scalers, PCA (notebook)
* `artifacts/model/`: `model.joblib`, `calibrator.joblib`, `decision_policy.json`, `features_schema.txt`, test reports (`pipelines/train.py`)
* `data/processed/test_pca.csv`: size of the test split shown in the reliability summary
* `app/data/`: demo batches, sample uploads, historical customers and `eda_customers.csv` (`pipelines/build_app_data.py`)

Risk levels read the thresholds from `decision_policy.json`; nothing is hard-coded.

## Notes

* **Demo state is per visitor** (`st.session_state`): batch history, alerts, settings and audit log. "Reset demo" restores the seeded state.
* **Preprocessing artifacts were pickled with scikit-learn 1.6.1** (notebook run). Their output under the pinned 1.7.0 matches `test_pca.csv` exactly (`tests/test_preprocessing.py`); the version warning is logged once at start-up.
* **Top reasons** are what-if contributions for the Retention Playbook reasons (model re-scored with one characteristic changed), not causal effects.
* **Churn Insights uses clean data only**: the notebook's Phase A dataset (7,043 customers, 35 features + target), without identifiers, location, target-leakage columns or Satisfaction Score, with structural blanks as "None". The build script fails if any row breaks the validation rules. The page shows ranked segments, tenure and add-on effects (internet customers only, to avoid confounding) with takeaways, and never mentions data preparation to the user.
* **No jargon in the interface**: `tests/test_app_pages.py` fails if any page shows terms such as ROC-AUC, p-value, calibration, leakage or "clean data". Model quality appears only as a plain-language summary (share of leavers caught, share of flags that are right, share of customers called).
* **Uploads** are processed in memory and never stored. Use sample or synthetic data only.
* **Default alert settings** (agreed 2026-11-11): watch 27%, critical 32%, plan gap 20 pp, minimum group 30 customers (`src/telco_churn/config.py`).
