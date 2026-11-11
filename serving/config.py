"""Endpoints and timing for the serving stack (override with environment variables)."""

from __future__ import annotations

import os

# MLflow model server (``mlflow models serve ... -p 5005``)
MLFLOW_INVOCATIONS_URL = os.getenv(
    "MLFLOW_INVOCATIONS_URL", "http://127.0.0.1:5005/invocations"
)
# Prometheus exporter in front of the model server
EXPORTER_HOST = os.getenv("EXPORTER_HOST", "127.0.0.1")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8080"))
EXPORTER_URL = os.getenv("EXPORTER_URL", f"http://{EXPORTER_HOST}:{EXPORTER_PORT}")

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
CLIENT_INTERVAL_SECONDS = float(os.getenv("CLIENT_INTERVAL_SECONDS", "1"))
