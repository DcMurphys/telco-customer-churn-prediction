"""Input schema and validation rules for customer records.

One schema serves the manual form (Customer Check) and uploaded batches
(Batch Scoring), so both reject the same mistakes. Nothing is coerced silently:
every problem is returned as a row-level error with the column, the value and a
plain-language reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

YES_NO = ("No", "Yes")
NONE_LABEL = "None"  # explicit label for "no offer" / "no internet" in uploads


@dataclass(frozen=True)
class NumberField:
    """A bounded numerical input."""

    name: str
    label: str
    min_value: float
    max_value: float
    step: float = 1.0
    integer: bool = False
    unit: str = ""
    help: str = ""


@dataclass(frozen=True)
class CategoryField:
    """A categorical input restricted to a fixed list of labels."""

    name: str
    label: str
    options: tuple[str, ...]
    help: str = ""


@dataclass(frozen=True)
class Schema:
    """Every input column with its constraints, grouped in form order."""

    groups: dict[str, list[NumberField | CategoryField]] = field(default_factory=dict)

    @property
    def fields(self) -> dict[str, NumberField | CategoryField]:
        """All fields keyed by column name."""
        return {f.name: f for group in self.groups.values() for f in group}


ADD_ON_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]

SCHEMA = Schema(
    groups={
        "Account": [
            NumberField("Age", "Age", 18, 100, integer=True, unit="years"),
            CategoryField("Under 30", "Under 30", YES_NO, "Derived from age"),
            CategoryField("Senior Citizen", "Senior citizen (65+)", YES_NO),
            CategoryField("Married", "Married", YES_NO),
            NumberField(
                "Number of Dependents", "Number of dependents", 0, 10, integer=True
            ),
            CategoryField("Dependents", "Has dependents", YES_NO),
            NumberField(
                "Number of Referrals", "Number of referrals", 0, 20, integer=True
            ),
            CategoryField("Referred a Friend", "Has referred a friend", YES_NO),
            NumberField(
                "Tenure in Months", "Tenure", 1, 120, integer=True, unit="months"
            ),
            NumberField(
                "CLTV", "Customer lifetime value score", 2000, 7000, integer=True
            ),
        ],
        "Plan & services": [
            CategoryField(
                "Contract", "Contract", ("Month-to-Month", "One Year", "Two Year")
            ),
            CategoryField(
                "Offer",
                "Promotional offer",
                (NONE_LABEL, "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"),
            ),
            CategoryField("Phone Service", "Phone service", YES_NO),
            CategoryField("Multiple Lines", "Multiple lines", YES_NO),
            CategoryField("Internet Service", "Internet service", YES_NO),
            CategoryField(
                "Internet Type",
                "Internet type",
                (NONE_LABEL, "Cable", "DSL", "Fiber Optic"),
            ),
            NumberField(
                "Avg Monthly GB Download", "Avg monthly download", 0, 200, unit="GB"
            ),
            *[CategoryField(col, col, YES_NO) for col in ADD_ON_COLS],
        ],
        "Billing": [
            CategoryField("Paperless Billing", "Paperless billing", YES_NO),
            CategoryField(
                "Payment Method",
                "Payment method",
                ("Bank Withdrawal", "Credit Card", "Mailed Check"),
            ),
            NumberField(
                "Monthly Charge", "Monthly charge", 0, 200, step=0.05, unit="USD"
            ),
            NumberField(
                "Total Charges", "Total charges", 0, 20_000, step=0.01, unit="USD"
            ),
            NumberField(
                "Avg Monthly Long Distance Charges",
                "Avg monthly long-distance charges",
                0,
                100,
                step=0.01,
                unit="USD",
            ),
            NumberField(
                "Total Long Distance Charges",
                "Total long-distance charges",
                0,
                10_000,
                step=0.01,
                unit="USD",
            ),
            NumberField(
                "Total Refunds", "Total refunds", 0, 1_000, step=0.01, unit="USD"
            ),
            NumberField(
                "Total Extra Data Charges",
                "Total extra data charges",
                0,
                1_000,
                step=1,
                unit="USD",
            ),
            NumberField(
                "Total Revenue", "Total revenue", 0, 25_000, step=0.01, unit="USD"
            ),
        ],
    }
)
ID_COL = "Customer ID"
UPLOAD_COLS = [ID_COL, *SCHEMA.fields]
# Fields the form derives from other answers, so they cannot contradict them
DERIVED_IN_FORM = [
    "Under 30",
    "Senior Citizen",
    "Dependents",
    "Referred a Friend",
    "Internet Service",
    "Total Long Distance Charges",
    "Total Revenue",
]
MONEY_TOLERANCE = 1.0  # USD, rounding tolerance for derived totals


def _error(row: object, column: str, value: object, reason: str) -> dict:
    return {"Row": row, "Column": column, "Value": value, "Reason": reason}


def _check_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing = [col for col in UPLOAD_COLS if col not in data.columns]
    unknown = [col for col in data.columns if col not in UPLOAD_COLS]
    return missing, unknown


def _check_values(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Check types, ranges and labels; return typed data and the errors."""
    errors: list[dict] = []
    typed = data.copy()
    for name, spec in SCHEMA.fields.items():
        raw = data[name]
        if isinstance(spec, NumberField):
            numbers = pd.to_numeric(raw, errors="coerce")
            blank = raw.isna() | raw.astype(str).str.strip().eq("")
            for row in raw.index[blank]:
                errors.append(_error(row, name, "", "Value is missing"))
            for row in raw.index[numbers.isna() & ~blank]:
                errors.append(_error(row, name, raw[row], "Must be a number"))
            valid = numbers.notna()
            out_of_range = valid & ~numbers.between(spec.min_value, spec.max_value)
            for row in raw.index[out_of_range]:
                errors.append(
                    _error(
                        row,
                        name,
                        raw[row],
                        f"Must be between {spec.min_value:g} and {spec.max_value:g}",
                    )
                )
            if spec.integer:
                fractional = valid & ~out_of_range & (numbers % 1 != 0)
                for row in raw.index[fractional]:
                    errors.append(_error(row, name, raw[row], "Must be a whole number"))
            typed[name] = numbers
        else:
            labels = raw.astype("string").str.strip()
            unknown = ~labels.isin(spec.options) | labels.isna()
            for row in raw.index[unknown]:
                value = "" if pd.isna(raw[row]) else raw[row]
                allowed = ", ".join(spec.options)
                errors.append(_error(row, name, value, f"Must be one of: {allowed}"))
            typed[name] = labels
    return typed, errors


def cross_field_errors(data: pd.DataFrame) -> list[dict]:
    """Domain rules across fields (requirement E-7).

    Args:
        data: Rows whose individual values are already valid.

    Returns:
        One error per broken rule and row.
    """
    errors: list[dict] = []

    def flag(mask: pd.Series, column: str, reason: str) -> None:
        for row in data.index[mask.fillna(False)]:
            errors.append(_error(row, column, data.loc[row, column], reason))

    yes = data.eq("Yes")
    flag(
        yes["Under 30"] != (data["Age"] < 30),
        "Under 30",
        "Does not match Age (Yes only if age is below 30)",
    )
    flag(
        yes["Senior Citizen"] != (data["Age"] >= 65),
        "Senior Citizen",
        "Does not match Age (Yes only if age is 65 or older)",
    )
    flag(
        yes["Dependents"] != (data["Number of Dependents"] > 0),
        "Dependents",
        "Does not match Number of Dependents",
    )
    flag(
        yes["Referred a Friend"] != (data["Number of Referrals"] > 0),
        "Referred a Friend",
        "Does not match Number of Referrals",
    )
    no_internet = data["Internet Service"].eq("No")
    for col in ADD_ON_COLS:
        flag(
            no_internet & yes[col],
            col,
            "Internet add-ons must be No when internet service is No",
        )
    flag(
        no_internet & data["Internet Type"].ne(NONE_LABEL),
        "Internet Type",
        "Must be None when internet service is No",
    )
    flag(
        ~no_internet & data["Internet Type"].eq(NONE_LABEL),
        "Internet Type",
        "Choose the internet type (Cable, DSL or Fiber Optic)",
    )
    flag(
        no_internet & (data["Avg Monthly GB Download"] > 0),
        "Avg Monthly GB Download",
        "Must be 0 when internet service is No",
    )
    no_phone = data["Phone Service"].eq("No")
    flag(
        no_phone & yes["Multiple Lines"],
        "Multiple Lines",
        "Must be No when phone service is No",
    )
    flag(
        no_phone & (data["Avg Monthly Long Distance Charges"] > 0),
        "Avg Monthly Long Distance Charges",
        "Must be 0 when phone service is No",
    )
    flag(
        (data["Tenure in Months"] >= 1)
        & (data["Total Charges"] < data["Monthly Charge"] - 0.01),
        "Total Charges",
        "Cannot be lower than one monthly charge",
    )
    flag(
        data["Total Refunds"] > data["Total Charges"],
        "Total Refunds",
        "Cannot be higher than total charges",
    )
    expected_ld = data["Avg Monthly Long Distance Charges"] * data["Tenure in Months"]
    tolerance = np.maximum(MONEY_TOLERANCE, 0.01 * expected_ld)
    flag(
        (data["Total Long Distance Charges"] - expected_ld).abs() > tolerance,
        "Total Long Distance Charges",
        "Should equal avg monthly long-distance charges × tenure",
    )
    expected_revenue = (
        data["Total Charges"]
        - data["Total Refunds"]
        + data["Total Extra Data Charges"]
        + data["Total Long Distance Charges"]
    )
    flag(
        (data["Total Revenue"] - expected_revenue).abs() > MONEY_TOLERANCE,
        "Total Revenue",
        "Should equal total charges − refunds + extra data + long-distance charges",
    )
    return errors


@dataclass
class BatchCheck:
    """Result of checking an uploaded batch (requirement B-3/B-4)."""

    missing_columns: list[str]
    unknown_columns: list[str]
    errors: pd.DataFrame
    valid_rows: pd.DataFrame
    n_rows: int

    @property
    def blocking(self) -> bool:
        """True when the file cannot be checked row by row."""
        return bool(self.missing_columns) or self.n_rows == 0

    @property
    def n_invalid(self) -> int:
        """Number of rows with at least one error."""
        return 0 if self.errors.empty else self.errors["Row"].nunique()


def check_batch(data: pd.DataFrame, max_rows: int) -> BatchCheck:
    """Validate an uploaded batch without changing any value.

    Args:
        data: Uploaded rows, read as text (``dtype=str``).
        max_rows: Largest accepted number of rows.

    Returns:
        Column problems, a row-level error table (rows numbered from 1, as in a
        spreadsheet without header) and the rows that passed every check.
    """
    missing, unknown = _check_columns(data)
    empty_errors = pd.DataFrame(columns=["Row", "Column", "Value", "Reason"])
    if missing or data.empty:
        return BatchCheck(missing, unknown, empty_errors, data.iloc[0:0], len(data))
    if len(data) > max_rows:
        error = _error("All", "File", len(data), f"More than {max_rows:,} rows")
        return BatchCheck([], unknown, pd.DataFrame([error]), data.iloc[0:0], len(data))

    data = data[UPLOAD_COLS].reset_index(drop=True)
    data.index = data.index + 1
    typed, errors = _check_values(data)

    ids = data[ID_COL].astype("string").str.strip()
    for row in ids.index[ids.isna() | ids.eq("")]:
        errors.append(_error(row, ID_COL, "", "Value is missing"))
    for row in ids.index[ids.duplicated(keep=False) & ids.notna() & ids.ne("")]:
        errors.append(_error(row, ID_COL, ids[row], "Duplicate Customer ID"))

    bad_rows = {e["Row"] for e in errors}
    checkable = typed.loc[~typed.index.isin(bad_rows)]
    errors.extend(cross_field_errors(checkable))

    table = pd.DataFrame(errors, columns=["Row", "Column", "Value", "Reason"])
    table["Value"] = table["Value"].astype(str)
    invalid = set(table["Row"])
    valid = typed.loc[~typed.index.isin(invalid)].copy()
    valid[ID_COL] = ids[valid.index]
    return BatchCheck(
        [], unknown, table.sort_values(["Row", "Column"]), valid, len(data)
    )


def complete_form_record(values: dict) -> dict:
    """Derive the dependent fields of a manual form entry.

    The form asks only for primary answers; flags and totals that follow from
    them are computed here so they can never contradict each other.

    Args:
        values: Primary answers from the Customer Check form.

    Returns:
        A full record in upload layout (without Customer ID).
    """
    record = dict(values)
    yes_no = {True: "Yes", False: "No"}
    record["Under 30"] = yes_no[record["Age"] < 30]
    record["Senior Citizen"] = yes_no[record["Age"] >= 65]
    record["Dependents"] = yes_no[record["Number of Dependents"] > 0]
    record["Referred a Friend"] = yes_no[record["Number of Referrals"] > 0]
    record["Internet Service"] = yes_no[record["Internet Type"] != NONE_LABEL]
    record["Total Long Distance Charges"] = round(
        record["Avg Monthly Long Distance Charges"] * record["Tenure in Months"], 2
    )
    record["Total Revenue"] = round(
        record["Total Charges"]
        - record["Total Refunds"]
        + record["Total Extra Data Charges"]
        + record["Total Long Distance Charges"],
        2,
    )
    return record


def check_record(record: dict) -> list[dict]:
    """Validate one complete record; return field errors (empty when valid)."""
    frame = pd.DataFrame([record], index=[1])
    typed, errors = _check_values(frame[list(SCHEMA.fields)])
    if not errors:
        errors = cross_field_errors(typed)
    return errors


def to_model_input(data: pd.DataFrame) -> pd.DataFrame:
    """Map explicit "None" labels back to the training spelling ("No")."""
    out = data.copy()
    for col in ("Offer", "Internet Type"):
        out[col] = out[col].replace(NONE_LABEL, "No")
    return out
