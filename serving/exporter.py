"""Prometheus exporter: FastAPI proxy in front of the MLflow model server.

``POST /predict`` forwards the request to ``mlflow models serve`` (model
``churn_policy_model``) and records API, model and system metrics.
``POST /feedback`` receives the true label of a prediction and updates the
confusion-matrix counters. ``GET /metrics`` is scraped by Prometheus.

Metric names are unchanged from the original exporter, so existing Prometheus
and Grafana dashboards keep working; three model metrics were added
(``model_churn_probability``, ``model_risk_level_total``, and the
``status_code`` label now records the real response code).

Usage (from the repository root, needs ``requirements-serving.txt``):
    python -m serving.exporter
"""

from __future__ import annotations

import logging
import time

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel, Field

from serving.config import (
    EXPORTER_HOST,
    EXPORTER_PORT,
    MLFLOW_INVOCATIONS_URL,
    REQUEST_TIMEOUT_SECONDS,
)
from serving.payloads import feature_values, outcome, parse_predictions

logger = logging.getLogger(__name__)
app = FastAPI(title="Churn model exporter")
REGISTRY = CollectorRegistry()

# ===== API metrics =====
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total predict requests by response status",
    labelnames=["method", "status_code"],
    registry=REGISTRY,
)
ACTIVE_REQUESTS = Gauge(
    "fastapi_active_connections", "Current active connections", registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", registry=REGISTRY
)
THROUGHPUT = Counter("api_throughput", "Throughput metric", registry=REGISTRY)
REQUEST_SIZE = Histogram(
    "api_request_size_bytes", "Request payload size in bytes", registry=REGISTRY
)
UPSTREAM_REQUEST_ERRORS = Counter(
    "upstream_mlflow_errors_total",
    "HTTP and network failures when calling MLflow serving",
    registry=REGISTRY,
)
PAYLOAD_PARSING_ERRORS = Counter(
    "api_payload_parsing_errors_total",
    "Payload JSON decoding and schema parsing errors",
    registry=REGISTRY,
)

# ===== Model metrics =====
PREDICTION_CLASS_TOTAL = Counter(
    "model_prediction_class",
    "Predicted class frequency (1 = flagged for contact)",
    labelnames=["pred_class"],
    registry=REGISTRY,
)
RISK_LEVEL_TOTAL = Counter(
    "model_risk_level_total",
    "Predicted risk levels (High / Medium / Low)",
    labelnames=["risk_level"],
    registry=REGISTRY,
)
CHURN_PROBABILITY = Histogram(
    "model_churn_probability",
    "Calibrated churn probability",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)
FEATURE_VALUES = Histogram(
    "model_input_feature_value",
    "Distribution of input feature values",
    labelnames=["feature_name"],
    registry=REGISTRY,
)
CONFUSION = {
    "true_positive": Counter(
        "model_true_positives_total", "Total true positives", registry=REGISTRY
    ),
    "false_positive": Counter(
        "model_false_positives_total", "Total false positives", registry=REGISTRY
    ),
    "false_negative": Counter(
        "model_false_negatives_total", "Total false negatives", registry=REGISTRY
    ),
    "true_negative": Counter(
        "model_true_negatives_total", "Total true negatives", registry=REGISTRY
    ),
}

# ===== System metrics =====
CPU_USAGE = Gauge(
    "system_cpu_usage_percent", "Host CPU usage in percent", registry=REGISTRY
)
RAM_USAGE = Gauge(
    "system_ram_usage_percent", "Host RAM usage in percent", registry=REGISTRY
)
ProcessCollector(registry=REGISTRY)  # process CPU seconds and resident memory
psutil.cpu_percent(interval=None)  # first call sets the CPU baseline


class Feedback(BaseModel):
    """True label of an earlier prediction."""

    prediction: int = Field(ge=0, le=1, description="1 = flagged for contact")
    actual: int = Field(ge=0, le=1, description="1 = customer churned")


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _record_model_metrics(body: dict, result: dict) -> None:
    for name, values in feature_values(body).items():
        for value in values:
            FEATURE_VALUES.labels(feature_name=name).observe(value)
    for prediction in parse_predictions(result):
        PREDICTION_CLASS_TOTAL.labels(pred_class=str(prediction.flagged)).inc()
        RISK_LEVEL_TOTAL.labels(risk_level=prediction.risk_level).inc()
        CHURN_PROBABILITY.observe(prediction.churn_probability)


@app.post("/predict")
async def predict(request: Request) -> dict:
    """Forward a request to MLflow serving and record metrics."""
    start = time.perf_counter()
    status_code = 200
    THROUGHPUT.inc()
    ACTIVE_REQUESTS.inc()
    try:
        REQUEST_SIZE.observe(len(await request.body()))
        try:
            body = await request.json()
        except ValueError as error:
            PAYLOAD_PARSING_ERRORS.inc()
            raise HTTPException(
                status_code=400, detail="Invalid JSON payload"
            ) from error

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(MLFLOW_INVOCATIONS_URL, json=body)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPStatusError as error:
                UPSTREAM_REQUEST_ERRORS.inc()
                # MLflow answers 400 for schema errors: pass that on to the client
                code = 400 if error.response.status_code == 400 else 502
                if code == 400:
                    PAYLOAD_PARSING_ERRORS.inc()
                raise HTTPException(
                    status_code=code, detail=f"MLflow serving error: {error}"
                ) from error
            except (httpx.RequestError, ValueError) as error:
                UPSTREAM_REQUEST_ERRORS.inc()
                raise HTTPException(
                    status_code=502, detail=f"Cannot reach MLflow serving: {error}"
                ) from error

        try:
            _record_model_metrics(body, result)
        except ValueError:
            PAYLOAD_PARSING_ERRORS.inc()
            logger.warning("Unexpected MLflow response: %s", result)
        return result
    except HTTPException as error:
        status_code = error.status_code
        raise
    finally:
        REQUEST_COUNT.labels(method=request.method, status_code=status_code).inc()
        ACTIVE_REQUESTS.dec()
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.post("/feedback")
async def receive_feedback(feedback: Feedback) -> dict:
    """Update the confusion-matrix counters with a true label."""
    CONFUSION[outcome(feedback.prediction, feedback.actual)].inc()
    return {"status": "success", "message": "Feedback recorded"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.exporter:app", host=EXPORTER_HOST, port=EXPORTER_PORT)
