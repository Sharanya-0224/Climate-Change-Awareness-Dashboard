"""
tests/conftest.py

Shared setup for all tests (pytest loads this file automatically).
- Puts src/ on the import path so tests can import climate_models and quality.
- Loads the sample data once per test session.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = ROOT / "tests" / "sample_data"


def _load(name):
    path = SAMPLE / name
    if not path.exists():
        pytest.skip(f"{path} missing: run python3 tests/make_sample_data.py once and commit the files")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def m1():
    return _load("m1_emissions.csv").sort_values(["country", "year"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def m2():
    return _load("m2_co2_growth.csv").sort_values("year").reset_index(drop=True)


@pytest.fixture(scope="session")
def m3():
    df = _load("m3_india_temp.csv")
    return df.sort_values(["cell_id", "year", "month"]).reset_index(drop=True)
