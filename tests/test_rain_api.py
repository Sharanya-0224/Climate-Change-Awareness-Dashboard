"""
tests/test_rain_api.py

API checks with FastAPI's TestClient: good requests give 200, bad input gives 422,
a missing package gives 503. Uses a small app with only the rain router (prefix /rain).
"""
import json
import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import rain_router as rr

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "rain_package"
SAMPLE = ROOT / "tests" / "sample_data" / "rain_sample.csv"
TIME_FEATURES = [f for f in rr.RainPredictRequest.model_fields if f not in ("lat", "lon", "month")]
RESPONSE_KEYS = {"key", "point_lat", "point_lon", "distance_deg", "cell_id", "state", "region",
                 "origin_year", "origin_month", "season", "months_covered", "pred_z", "pred_anom_mm",
                 "pred_total_mm", "clipped_at_zero", "has_skill", "beyond_test_period",
                 "note", "served_model", "served_version"}


@pytest.fixture
def client():
    assert rr.load_rain_model(str(PKG))
    app = FastAPI()
    app.include_router(rr.router, prefix="/rain")
    return TestClient(app)


@pytest.fixture(scope="module")
def row():
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE} missing")
    feats = json.loads((PKG / "feature_list.json").read_text())["features"]
    return pd.read_csv(SAMPLE, dtype={c: "float32" for c in feats}).iloc[100]


def _finite(body):
    return all(not (isinstance(v, float) and not math.isfinite(v)) for v in body.values())


def test_forecast_ok(client):
    r = client.get("/rain/forecast", params={"lat": 26.9, "lon": 75.8})
    assert r.status_code == 200
    body = r.json()
    assert RESPONSE_KEYS <= set(body)
    assert _finite(body)
    assert body["cell_id"] == "26.5_75.5"            # friend's 1-degree rule on the requested place
    assert body["distance_deg"] <= 0.25
    assert body["beyond_test_period"] is True
    assert body["served_version"] == json.loads((PKG / "metadata.json").read_text())["registered_version"]


def test_forecast_snaps_to_nearest_point(client, row):
    r = client.get("/rain/forecast", params={"lat": float(row["lat"]) + 0.05, "lon": float(row["lon"]) - 0.05})
    assert r.status_code == 200
    assert r.json()["key"] == int(row["key"])


def test_forecast_chosen_origin(client):
    r = client.get("/rain/forecast", params={"lat": 26.9, "lon": 75.8, "year": 2025, "month": 10})
    assert r.status_code == 200
    assert r.json()["origin_month"] == 10


@pytest.mark.parametrize("params", [
    {"lat": 7.0, "lon": 70.0},                          # sea, inside the box but outside the grid
    {"lat": 500, "lon": 75.8},
    {"lat": 26.9, "lon": 500},
    {"lat": 26.9, "lon": 75.8, "year": 2025, "month": 13},
    {"lat": 26.9, "lon": 75.8, "year": 2025},           # year without month
    {"lat": 26.9, "lon": 75.8, "year": 2020, "month": 1},  # origin not in the package
])
def test_forecast_bad_input_422(client, params):
    assert client.get("/rain/forecast", params=params).status_code == 422


def test_predict_matches_model(client, row):
    body = {"lat": float(row["lat"]), "lon": float(row["lon"]), "month": int(row["month"]),
            **{f: float(row[f]) for f in TIME_FEATURES}}
    r = client.post("/rain/predict", json=body)
    assert r.status_code == 200
    assert _finite(r.json())
    assert abs(r.json()["pred_z"] - float(row["pred_z"])) < 1e-3   # response is rounded to 4 decimals


@pytest.mark.parametrize("change", [
    {"drop": "dmi_t"},                  # missing feature
    {"add": ("foo", 1.0)},              # unknown feature
    {"set": ("nino34_t", 10.0)},        # out of range
    {"set": ("month", 0)},
    {"set": ("lat", 500.0)},
])
def test_predict_bad_input_422(client, row, change):
    body = {"lat": float(row["lat"]), "lon": float(row["lon"]), "month": int(row["month"]),
            **{f: float(row[f]) for f in TIME_FEATURES}}
    if "drop" in change:
        body.pop(change["drop"])
    if "add" in change:
        body[change["add"][0]] = change["add"][1]
    if "set" in change:
        body[change["set"][0]] = change["set"][1]
    assert client.post("/rain/predict", json=body).status_code == 422


def test_model_info(client):
    r = client.get("/rain/model-info")
    assert r.status_code == 200
    assert r.json()["registered_name"] == "rain_v2_3month_lgbm"
    assert len(r.json()["skill_table"]) == 9


def test_status_and_smoke(client):
    assert rr.rain_status()["loaded"] is True
    assert math.isfinite(rr.rain_smoke())


def test_missing_package_gives_503(tmp_path):
    assert rr.load_rain_model(str(tmp_path / "no_package")) is False
    assert rr.rain_status()["loaded"] is False
    app = FastAPI()
    app.include_router(rr.router, prefix="/rain")
    c = TestClient(app)
    assert c.get("/rain/forecast", params={"lat": 26.9, "lon": 75.8}).status_code == 503
    assert c.get("/rain/model-info").status_code == 503
    rr.load_rain_model(str(PKG))                      # leave the module loaded for other tests