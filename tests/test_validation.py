"""Input validation: wrong or contradictory values are never accepted (E-1 to E-7)."""

import pandas as pd
import pytest

from app.models import artifacts
from app.models.validation import (
    check_batch,
    check_record,
    complete_form_record,
)
from telco_churn.config import SAMPLE_BATCH_DIR


@pytest.fixture
def clean_batch():
    return artifacts.read_upload_csv(SAMPLE_BATCH_DIR / "sample_batch_no_errors.csv")


@pytest.fixture
def valid_record(clean_batch):
    row = check_batch(clean_batch.head(1), 10).valid_rows.iloc[0].to_dict()
    row.pop("Customer ID")
    return row


def test_clean_sample_passes(clean_batch):
    check = check_batch(clean_batch, 5_000)
    assert check.n_invalid == 0
    assert len(check.valid_rows) == len(clean_batch)


def test_error_sample_flags_every_injected_error():
    data = artifacts.read_upload_csv(SAMPLE_BATCH_DIR / "sample_batch_with_errors.csv")
    check = check_batch(data, 5_000)
    reasons = set(check.errors["Reason"])
    assert "Must be between 0 and 200" in reasons
    assert "Must be a number" in reasons
    assert "Duplicate Customer ID" in reasons
    assert "Value is missing" in reasons
    assert any(r.startswith("Must be one of") for r in reasons)
    assert any("internet service is No" in r for r in reasons)
    assert any("Does not match Age" in r for r in reasons)
    # invalid rows are excluded, never silently fixed
    assert len(check.valid_rows) + check.n_invalid == len(data)


def test_missing_columns_block_the_file(clean_batch):
    check = check_batch(clean_batch.drop(columns=["Contract"]), 5_000)
    assert check.blocking
    assert check.missing_columns == ["Contract"]


def test_unknown_columns_are_reported_not_used(clean_batch):
    check = check_batch(clean_batch.assign(Extra="x"), 5_000)
    assert check.unknown_columns == ["Extra"]
    assert check.n_invalid == 0


def test_too_many_rows_is_rejected(clean_batch):
    check = check_batch(clean_batch, max_rows=10)
    assert check.n_invalid == 1 and check.valid_rows.empty


@pytest.mark.parametrize(
    ("field", "value", "reason_part"),
    [
        ("Internet Type", "None", "add-ons"),  # add-ons still Yes
        ("Phone Service", "No", "phone service"),
        ("Total Charges", 1.0, "lower than one monthly charge"),
        ("Under 30", "Yes", "Does not match Age"),
    ],
)
def test_contradictions_are_rejected(valid_record, field, value, reason_part):
    record = dict(valid_record)
    record["Internet Service"] = record["Internet Service"]
    if field == "Internet Type":
        record["Internet Service"] = "No"
        record["Online Security"] = "Yes"
    if field == "Phone Service":
        record["Multiple Lines"] = "Yes"
    if field == "Under 30":
        record["Age"] = 45
    record[field] = value
    errors = check_record(record)
    assert any(reason_part.lower() in e["Reason"].lower() for e in errors)


def test_form_derives_dependent_fields(valid_record):
    primary = {
        k: v
        for k, v in valid_record.items()
        if k
        not in {
            "Under 30",
            "Senior Citizen",
            "Dependents",
            "Referred a Friend",
            "Internet Service",
            "Total Long Distance Charges",
            "Total Revenue",
        }
    }
    record = complete_form_record(primary)
    assert check_record(record) == []
    assert record["Under 30"] == ("Yes" if record["Age"] < 30 else "No")


def test_out_of_range_record_is_rejected(valid_record):
    record = dict(valid_record, **{"Monthly Charge": 250.0})
    errors = check_record(record)
    assert errors and errors[0]["Column"] == "Monthly Charge"
    assert isinstance(pd.DataFrame(errors), pd.DataFrame)
