# Common tasks. Run from the repository root; override PYTHON to use another env,
# e.g. `make test PYTHON=python3`.
PYTHON ?= .venv/bin/python
# Optional local settings (MLflow tracking, serving endpoints); see .env.example
-include .env
export
export PYTHONPATH := src:.
export MPLBACKEND := Agg

.PHONY: help setup data train train-online app test lint format serve-model exporter client

help:  ## List the available tasks
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create .venv and install everything for development
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## Build app/data/ from data/raw/telco.csv (clean EDA data, demo batches)
	$(PYTHON) -m pipelines.build_app_data

train:  ## Tracked training, logged to the local ./mlruns
	$(PYTHON) -m pipelines.train --offline

train-online:  ## Tracked training, logged to DagsHub (needs DAGSHUB_USER_TOKEN)
	$(PYTHON) -m pipelines.train

app:  ## Run the Streamlit app on http://localhost:8501
	$(PYTHON) -m streamlit run app/streamlit_app.py

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Lint and check formatting (same as CI)
	$(PYTHON) -m ruff check src pipelines serving app tests notebooks
	$(PYTHON) -m ruff format --check src pipelines serving app tests

format:  ## Fix lint issues and format the code
	$(PYTHON) -m ruff check --fix src pipelines serving app tests
	$(PYTHON) -m ruff format src pipelines serving app tests

serve-model:  ## Serve the latest churn_policy_model on port 5005
	PATH="$$(dirname $(PYTHON)):$$PATH" $(PYTHON) -m mlflow models serve \
		-m "$$($(PYTHON) -m serving.model_uri)" -p 5005 --env-manager local

exporter:  ## Start the Prometheus exporter on port 8080 (needs serve-model)
	$(PYTHON) -m serving.exporter

client:  ## Send 20 test-split customers through the exporter
	$(PYTHON) -m serving.inference_client --count 20
