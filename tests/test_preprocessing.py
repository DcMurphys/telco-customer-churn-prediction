"""The raw-data transform must reproduce the notebook's processed test split."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from telco_churn import preprocessing as pre
from telco_churn.config import RANDOM_STATE, RAW_DATA_PATH, TARGET_COL, TEST_PATH


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw data not downloaded")
def test_transform_matches_processed_test_split():
    raw = pd.read_csv(RAW_DATA_PATH)
    model_df = raw.drop(
        columns=pre.IDENTIFIER_GEO_COLS
        + pre.TARGET_LEAKAGE_COLS
        + pre.LEAKAGE_SUSPECT_COLS
    ).fillna(pre.STRUCTURAL_MISSING_FILL)
    _, test = train_test_split(
        model_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=model_df[TARGET_COL],
    )
    expected = pd.read_csv(TEST_PATH)
    schema = [c for c in expected.columns if c != TARGET_COL]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # artifacts pickled with sklearn 1.6.1
        got = pre.Preprocessor.load().transform(test.reset_index(drop=True), schema)
    np.testing.assert_allclose(got.to_numpy(), expected[schema].to_numpy(), atol=1e-9)
