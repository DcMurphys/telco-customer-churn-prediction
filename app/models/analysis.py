"""Model performance curves, churn-driver statistics and demo-state seeding."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from app.models.alerts import AlertSettings, DemoState, summarise_batch
from app.models.artifacts import ModelBundle, read_upload_csv
from app.models.scoring import score_customers
from app.models.validation import check_batch

DEMO_DATES = [
    date(2026, 10, 19),
    date(2026, 10, 26),
    date(2026, 11, 2),
    date(2026, 11, 9),
]


def build_demo_state(
    bundle: ModelBundle,
    reference: dict,
    settings: AlertSettings,
    batch_files: list[Path],
) -> DemoState:
    """Score the seed batches and replay the alert rules over them."""
    state = DemoState(batches=[], alerts=[], settings=settings, audit_log=[])
    for path, batch_date in zip(batch_files, DEMO_DATES, strict=False):
        check = check_batch(read_upload_csv(path), max_rows=10_000)
        if check.blocking or check.n_invalid:
            raise ValueError(f"Demo batch {path.name} failed validation")
        scored = score_customers(bundle, check.valid_rows, reference)
        name = path.stem.replace("demo_", "Demo ").replace("_", "-").title()
        name = name.replace("Demo-", "Demo ")
        summary = summarise_batch(
            name,
            batch_date,
            scored,
            settings,
            is_demo=True,
            scored_at=datetime.combine(batch_date, time(8, 0)),
        )
        state.add_batch(summary)
    return state


def risk_distribution(scored: pd.DataFrame) -> pd.DataFrame:
    """Customers per risk level, in High → Low order."""
    counts = scored["risk_level"].value_counts()
    levels = ["High", "Medium", "Low"]
    return pd.DataFrame(
        {"Risk level": levels, "Customers": [int(counts.get(lv, 0)) for lv in levels]}
    )
