"""
tests/test_full_data.py

The same cleanliness checks as test_data.py, but on the FULL model-ready tables.
They run on your computer (where DVC has the data) and are skipped automatically
on GitHub Actions (where the big files do not exist).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_models.m3_model import INPUT_COLS, TARGET

READY = Path(__file__).resolve().parents[1] / "data" / "model_ready"


@pytest.fixture(scope="module")
def full_m3():
    path = READY / "m3_india_temp.csv"
    if not path.exists():
        pytest.skip("full data not available here (expected on GitHub Actions)")
    return pd.read_csv(path, usecols=INPUT_COLS + [TARGET])


@pytest.fixture(scope="module")
def full_m2():
    path = READY / "m2_co2_growth.csv"
    if not path.exists():
        pytest.skip("full data not available here (expected on GitHub Actions)")
    return pd.read_csv(path)


def test_full_m3_no_missing_values(full_m3):
    assert full_m3.isna().sum().sum() == 0


def test_full_m3_no_duplicate_square_months(full_m3):
    assert not full_m3.duplicated(["cell_id", "year", "month"]).any()


def test_full_m3_every_square_has_enough_months(full_m3):
    """Squares with more than 30% missing months were dropped, so each kept square has at least 70%."""
    months_per_square = full_m3.groupby("cell_id").size()
    assert (months_per_square >= 0.7 * 888).all(), months_per_square.min()


def test_full_m3_years_cover_1951_to_2024(full_m3):
    assert full_m3["year"].min() == 1951 and full_m3["year"].max() == 2024


def test_full_m3_anomalies_plausible(full_m3):
    assert full_m3[TARGET].abs().max() < 15


def test_full_m2_clean_and_consecutive(full_m2):
    assert full_m2.isna().sum().sum() == 0
    assert np.all(np.diff(full_m2["year"].to_numpy()) == 1)
