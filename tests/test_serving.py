"""Serving: payloads, response parsing, client, pyfunc model and exporter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serving import payloads
from telco_churn.config import MODEL_DIR, TEST_PATH
from telco_churn.policy import Thresholds, apply_policy, risk_levels

MODEL_FILES = ["model.joblib", "calibrator.joblib", "decision_policy.json"]
needs_model = pytest.mark.skipif(
    not all((MODEL_DIR / name).exists() for name in MODEL_FILES),
    reason="model artifacts not trained",
)


# ===== Shared decision policy =====
def test_risk_levels_follow_thresholds():
    levels = risk_levels(np.array([0.5, 0.31, 0.2, 0.1]), Thresholds(0.3, 0.15))
    assert levels.tolist() == ["High", "High", "Medium", "Low"]


def test_policy_without_breakeven_has_no_medium_band():
    levels = risk_levels(np.array([0.5, 0.2]), Thresholds(0.3, None))
    assert levels.tolist() == ["High", "Low"]


# ===== Payloads and parsing =====
def test_build_request_uses_schema_order_and_true_labels():
    schema = payloads.load_schema()
    body, actual = payloads.build_request(
        payloads.load_test_split(), schema, np.random.default_rng(0), n_rows=3
    )
    split = body["dataframe_split"]
    assert split["columns"] == schema
    assert len(split["data"]) == 3 and len(split["data"][0]) == len(schema)
    assert set(actual) <= {0, 1}


def test_feature_values_ignores_malformed_bodies():
    body = {"dataframe_split": {"columns": ["a", "b"], "data": [[1, "x"], [2.5, 3]]}}
    assert payloads.feature_values(body) == {"a": [1.0, 2.5], "b": [3.0]}
    assert payloads.feature_values("not a dict") == {}


def test_parse_predictions_reads_policy_output():
    response = {
        "predictions": [
            {"churn_probability": 0.42, "flagged": 1, "risk_level": "High"},
        ]
    }
    (prediction,) = payloads.parse_predictions(response)
    assert prediction == payloads.Prediction(0.42, 1, "High")
    with pytest.raises(ValueError):
        payloads.parse_predictions({"predictions": [0, 1]})
    with pytest.raises(ValueError):
        payloads.parse_predictions({})


def test_outcome_maps_confusion_cells():
    assert payloads.outcome(1, 1) == "true_positive"
    assert payloads.outcome(1, 0) == "false_positive"
    assert payloads.outcome(0, 1) == "false_negative"
    assert payloads.outcome(0, 0) == "true_negative"
    with pytest.raises(ValueError):
        payloads.outcome(2, 0)


def test_write_payload(tmp_path):
    path = payloads.write_payload(tmp_path / "payload.json", seed=1)
    assert '"dataframe_split"' in path.read_text()


# ===== Inference client (HTTP replaced by a fake session) =====
class FakeResponse:
    def __init__(self, data: dict, status: int = 200):
        self.data, self.status_code, self.text = data, status, str(data)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        return self.data


class FakeSession:
    def __init__(self, predict_response: FakeResponse):
        self.predict_response, self.calls = predict_response, []

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/feedback"):
            return FakeResponse({"status": "success"})
        return self.predict_response


def test_client_sends_feedback_with_the_flag():
    pytest.importorskip("requests")
    from serving import inference_client

    session = FakeSession(
        FakeResponse(
            {
                "predictions": [
                    {"churn_probability": 0.8, "flagged": 1, "risk_level": "High"}
                ]
            }
        )
    )
    result = inference_client.run_inference(session, {"dataframe_split": {}}, [0])
    assert result[0].flagged == 1
    feedback = [kw["json"] for url, kw in session.calls if url.endswith("/feedback")]
    assert feedback == [{"prediction": 1, "actual": 0}]


def test_client_returns_none_on_http_error():
    pytest.importorskip("requests")
    from serving import inference_client

    session = FakeSession(FakeResponse({"detail": "bad"}, status=502))
    assert inference_client.run_inference(session, {"x": 1}, [1]) is None
    assert not any(url.endswith("/feedback") for url, _ in session.calls)


# ===== MLflow pyfunc model =====
@needs_model
def test_policy_model_matches_app_scoring():
    pytest.importorskip("mlflow")
    import joblib

    from telco_churn.policy_model import ChurnPolicyModel

    artifacts = {
        "model": str(MODEL_DIR / "model.joblib"),
        "calibrator": str(MODEL_DIR / "calibrator.joblib"),
        "decision_policy": str(MODEL_DIR / "decision_policy.json"),
        "features_schema": str(MODEL_DIR / "features_schema.txt"),
    }
    model = ChurnPolicyModel()
    model.load_context(SimpleNamespace(artifacts=artifacts))
    features = pd.read_csv(TEST_PATH).drop(columns="Churn Label").head(200)
    output = model.predict(None, features)

    expected = apply_policy(
        joblib.load(artifacts["model"]).predict_proba(features[model.schema])[:, 1],
        joblib.load(artifacts["calibrator"]),
        Thresholds.from_file(MODEL_DIR / "decision_policy.json"),
    )
    pd.testing.assert_frame_equal(output, expected)
    with pytest.raises(ValueError, match="Missing"):
        model.predict(None, features.drop(columns=model.schema[0]))


# ===== Exporter (only when the serving dependencies are installed) =====
def test_exporter_proxies_and_counts(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient

    from serving import exporter

    class FakeUpstream:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            import httpx

            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "predictions": [
                        {"churn_probability": 0.6, "flagged": 1, "risk_level": "High"}
                    ]
                },
                request=request,
            )

    monkeypatch.setattr(exporter.httpx, "AsyncClient", FakeUpstream)
    client = TestClient(exporter.app)
    body = {"dataframe_split": {"columns": ["a"], "data": [[1.0]]}}
    assert client.post("/predict", json=body).status_code == 200
    assert client.post("/predict", content="{bad").status_code == 400
    assert client.post("/feedback", json={"prediction": 1, "actual": 1}).ok
    assert (
        client.post("/feedback", json={"prediction": 3, "actual": 1}).status_code == 422
    )
    # prometheus_client strips a trailing "_total" and adds it back on export
    metrics = client.get("/metrics").text
    assert 'api_request_count_total{method="POST",status_code="400"}' in metrics
    assert 'model_risk_level_total{risk_level="High"}' in metrics
    assert "model_true_positives_total 1.0" in metrics
