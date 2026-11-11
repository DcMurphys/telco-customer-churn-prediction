"""Shared fixtures (import paths come from ``pythonpath`` in pyproject.toml)."""

import pytest

from app.models import artifacts


@pytest.fixture(scope="session")
def bundle():
    """Scoring bundle; skips when the model has not been trained yet."""
    try:
        return artifacts.load_bundle()
    except artifacts.ArtifactsMissingError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def reference():
    """Reference values for the top-reasons logic."""
    return artifacts.load_reference()
