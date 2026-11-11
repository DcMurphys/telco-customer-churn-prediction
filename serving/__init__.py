"""Model serving and monitoring: Prometheus exporter and inference client.

Run from the repository root (see the Makefile):
    mlflow models serve -m runs:/<run_id>/churn_policy_model -p 5005 --env-manager local
    python -m serving.exporter          # FastAPI proxy with Prometheus metrics
    python -m serving.inference_client  # sends test-split customers + feedback
"""
