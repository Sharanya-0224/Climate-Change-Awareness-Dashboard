"""
tests/test_rain_model.py

Model checks: the package loads, predictions are reproducible, finite and deterministic,
and served rainfall totals are never negative.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from api import rain_router as rr

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "rain_package"
SAMPLE = ROOT / "tests" / "sample_data" / "rain_sample.csv"
PACKAGE_FILES = ["model.txt", "feature_list.json", "normals.csv", "points.csv",
                 "lookup_latest.csv", "skill_table.csv", "metadata.json"]


@pytest.fixture(scope="module")
def features():
    return json.loads((PKG / "feature_list.json").read_text())["features"]


@pytest.fixture(scope="module")
def booster():
    return lgb.Booster(model_file=str(PKG / "model.txt"))


@pytest.fixture(scope="module")
def sample(features):
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE} missing")
    return pd.read_csv(SAMPLE, dtype={c: "float32" for c in features})


def test_package_files_present_and_small():
    sizes = {f: (PKG / f).stat().st_size / 1e6 for f in PACKAGE_FILES}
    assert sum(sizes.values()) < 20
    assert max(sizes.values()) < 50                 # GitHub rejects files near 100 MB


def test_lightgbm_version_matches_package():
    meta = json.loads((PKG / "metadata.json").read_text())
    assert lgb.__version__ == meta["lightgbm_version"]


def test_feature_order(booster, features):
    assert len(features) == 23
    assert booster.feature_name() == features


def test_predictions_match_stored(booster, features, sample):
    p = booster.predict(sample[features])
    assert np.abs(p - sample["pred_z"].values).max() < 1e-6


def test_predictions_finite_and_deterministic(booster, features, sample):
    a = booster.predict(sample[features])
    b = booster.predict(sample[features])
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_served_total_never_negative(sample):
    assert rr.load_rain_model(str(PKG))
    point = rr.snap(float(sample["lat"].iloc[0]), float(sample["lon"].iloc[0]))
    out = rr._result(point, 1, None, -50.0)          # forced very dry forecast
    assert out["pred_total_mm"] == 0.0
    assert out["clipped_at_zero"] is True
    for _, r in sample.head(50).iterrows():
        res = rr.forecast_like = None
        body = {f: float(r[f]) for f in rr.RainPredictRequest.model_fields if f not in ("lat", "lon", "month")}
        res = rr.predict(float(r["lat"]), float(r["lon"]), int(r["month"]), body)
        assert res["pred_total_mm"] >= 0