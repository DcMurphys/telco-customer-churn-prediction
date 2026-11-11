"""Inference client: send test-split customers to the exporter and report back.

Each request samples real customers from ``data/processed/test_pca.csv`` (not
dummy data), calls ``POST /predict`` on the exporter, logs the result and
latency, then sends the true label to ``POST /feedback`` so the exporter can
track true and false positives.

Usage (from the repository root, with the model server and exporter running):
    python -m serving.inference_client               # run until Ctrl+C
    python -m serving.inference_client --count 20    # 20 requests, then stop
    python -m serving.inference_client --invalid     # one malformed request
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid

import numpy as np
import requests

from serving.config import (
    CLIENT_INTERVAL_SECONDS,
    EXPORTER_URL,
    REQUEST_TIMEOUT_SECONDS,
)
from serving.payloads import (
    Prediction,
    build_request,
    load_schema,
    load_test_split,
    parse_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def send_feedback(
    session: requests.Session, prediction: Prediction, actual: int
) -> bool:
    """Send the true label of one prediction; return True on success."""
    payload = {"prediction": prediction.flagged, "actual": int(actual)}
    try:
        response = session.post(
            f"{EXPORTER_URL}/feedback", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as error:
        logger.error("Feedback failed: %s", error)
        return False
    return True


def run_inference(
    session: requests.Session, body: dict | str, actual: list[int] | None = None
) -> list[Prediction] | None:
    """Call ``/predict`` once and send feedback for each customer.

    Returns:
        The predictions, or ``None`` if the request failed.
    """
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = session.post(
            f"{EXPORTER_URL}/predict",
            json=body if isinstance(body, dict) else None,
            data=body if isinstance(body, str) else None,
            headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        predictions = parse_predictions(response.json())
    except requests.HTTPError as error:
        logger.error(
            "[%s] HTTP %s: %s",
            request_id,
            error.response.status_code,
            error.response.text[:200],
        )
        return None
    except (requests.RequestException, ValueError) as error:
        logger.error("[%s] Request failed: %s", request_id, error)
        return None

    latency = time.perf_counter() - start
    for i, prediction in enumerate(predictions):
        truth = "" if actual is None else f" | actually churned: {bool(actual[i])}"
        logger.info(
            "[%s] %.3fs | chance of leaving %.1f%% | %s risk | flagged=%d%s",
            request_id,
            latency,
            100 * prediction.churn_probability,
            prediction.risk_level,
            prediction.flagged,
            truth,
        )
        if actual is not None:
            send_feedback(session, prediction, actual[i])
    return predictions


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--count", type=int, default=0, help="number of requests (0 = until Ctrl+C)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=CLIENT_INTERVAL_SECONDS,
        help="seconds between requests",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--invalid",
        action="store_true",
        help="send one malformed payload (tests api_payload_parsing_errors_total)",
    )
    return parser.parse_args()


def main() -> None:
    """Send requests in a loop until the count is reached or Ctrl+C."""
    args = parse_args()
    session = requests.Session()
    if args.invalid:
        run_inference(session, "{ invalid_json_payload: true ")
        return

    test_split, schema = load_test_split(), load_schema()
    rng = np.random.default_rng(args.seed)
    sent = 0
    logger.info("Sending requests to %s (Ctrl+C to stop)", EXPORTER_URL)
    try:
        while args.count == 0 or sent < args.count:
            body, actual = build_request(test_split, schema, rng)
            run_inference(session, body, actual)
            sent += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Stopped after %d requests", sent)


if __name__ == "__main__":
    main()
