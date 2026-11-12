"""
tests/test_rain_data.py

Data checks on real val rows (tests/sample_data/rain_sample.csv, 12 points x 2003-2013).
The full-data versions of these checks are the gates in precipitation/src/quality/rain_gates.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "rain_package"
SAMPLE = ROOT / "tests" / "sample_data" / "rain_sample.csv"


def _features():
    return json.loads((PKG / "feature_list.json").read_text())["features"]


@pytest.fixture(scope="module")
def sample():
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE} missing: run python3 precipitation/src/quality/make_rain_sample.py")
    df = pd.read_csv(SAMPLE, dtype={c: "float32" for c in _features()})
    return df.sort_values(["key", "year", "month"]).reset_index(drop=True)


def _prev(df, k):
    """Row positions i whose row i-k is the same point exactly k months earlier."""
    t = df["year"].to_numpy() * 12 + df["month"].to_numpy()
    key = df["key"].to_numpy()
    i = np.arange(k, len(df))
    return i[(key[i] == key[i - k]) & (t[i] - t[i - k] == k)]


def test_required_columns(sample):
    need = ["key", "year", "month", "region", "split", "target_z", "target_mm",
            "target_mean", "target_sd", "pred_z"] + _features()
    assert not [c for c in need if c not in sample.columns]


def test_no_duplicates_and_no_missing_inputs(sample):
    assert not sample.duplicated(["key", "year", "month"]).any()
    assert not sample[_features()].isna().any().any()


def test_sample_is_real_val_rows(sample):
    assert set(sample["split"]) == {"val"}
    assert sample["year"].between(2003, 2013).all()


def test_rain_lag_looks_backwards(sample):
    i = _prev(sample, 1)
    assert len(i) > 1000
    assert np.allclose(sample["rain_z_lag1"].values[i], sample["rain_z_t"].values[i - 1], atol=1e-5)


def test_rain_roll3_uses_past_three_months(sample):
    i = _prev(sample, 2)
    z = sample["rain_z_t"].values.astype(float)
    expected = (z[i] + z[i - 1] + z[i - 2]) / 3
    assert np.allclose(sample["rain_z_roll3"].values[i], expected, atol=1e-4)


@pytest.mark.parametrize("prefix", ["nino34", "dmi"])
def test_index_lags_look_backwards(sample, prefix):
    t = sample[f"{prefix}_t"].values.astype(float)
    for k in (1, 2, 3):
        i = _prev(sample, k)
        assert np.allclose(sample[f"{prefix}_lag{k}"].values[i], t[i - k], atol=1e-5), f"{prefix}_lag{k}"
    i = _prev(sample, 2)
    assert np.allclose(sample[f"{prefix}_mean3"].values[i], (t[i] + t[i - 1] + t[i - 2]) / 3, atol=1e-4)
    if prefix == "nino34":
        i = _prev(sample, 3)
        assert np.allclose(sample["nino34_chg3"].values[i], t[i] - t[i - 3], atol=1e-4)


def test_target_uses_package_normals(sample):
    nrm = pd.read_csv(PKG / "normals.csv")
    m = sample.merge(nrm, left_on=["key", "month"], right_on=["key", "origin_month"],
                     how="left", validate="many_to_one")
    assert m["target_mean_mm"].notna().all()
    assert np.allclose(m["target_mean"], m["target_mean_mm"], atol=1e-2)
    assert np.allclose(m["target_sd"], m["target_sd_mm"], atol=1e-2)
    assert (m["target_sd_mm"] >= 5).all()          # sd floor 5 mm
    z = (m["target_mm"] - m["target_mean"]) / m["target_sd"]
    assert np.allclose(z, m["target_z"], atol=1e-3)


def test_normals_declared_from_training_years():
    meta = json.loads((PKG / "metadata.json").read_text())
    assert meta["training_years"] == "1901-2002"
    assert "1901-2002" in meta["normals"]