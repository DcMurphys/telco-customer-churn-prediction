"""Churn Insights uses clean data and reproduces the notebook's EDA figures."""

import pytest

from app.models import eda
from telco_churn.config import APP_EDA_PATH

pytestmark = pytest.mark.skipif(not APP_EDA_PATH.exists(), reason="app data not built")


@pytest.fixture(scope="module")
def data():
    return eda.load_clean()


def test_dataset_is_clean(data):
    assert data.shape == (7043, 37)  # 36 columns + derived 0/1 target
    assert not data.isna().any().any()
    assert data.eq("").sum().sum() == 0
    removed = {"Customer ID", "City", "Churn Reason", "Satisfaction Score"}
    assert removed.isdisjoint(data.columns)
    assert not data.duplicated().any()
    revenue = (
        data["Total Charges"]
        - data["Total Refunds"]
        + data["Total Extra Data Charges"]
        + data["Total Long Distance Charges"]
    )
    assert (revenue - data["Total Revenue"]).abs().max() < 0.01


def test_headline_figures_match_notebook(data):
    stats = eda.overview(data)
    assert stats["churned"] == 1869
    assert stats["churn_rate"] == pytest.approx(0.2654, abs=1e-4)
    assert stats["tenure_left"] == 10 and stats["tenure_stayed"] == 38
    contract = eda.rate_table(data, "Contract").set_index("group")["rate"]
    assert contract["Month-to-Month"] == pytest.approx(0.4584, abs=1e-4)
    assert contract["Two Year"] == pytest.approx(0.0255, abs=1e-4)


def test_confounding_flips_for_streaming(data):
    table = eda.add_on_effect(data).set_index(["Add-on", "Scope"])
    diff = table["Difference (pp)"]
    assert diff[("Unlimited Data", "All customers")] > 0
    assert diff[("Unlimited Data", "Internet customers")] < 0


def test_leaderboard_and_tenure_match_notebook(data):
    board = eda.segment_leaderboard(data).set_index("segment")["rate"]
    assert board["Offer E promotion"] == pytest.approx(0.5292, abs=1e-4)
    tenure = eda.tenure_profile(data).set_index("group")["rate"]
    assert tenure["1–6"] > 0.5 > tenure["49–72"]
