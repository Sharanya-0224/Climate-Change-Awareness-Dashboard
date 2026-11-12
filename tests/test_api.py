"""
tests/test_api.py

Tests for the web layer (src/api/main.py), using FastAPI's built-in test client.
No real server and no DagsHub: models fitted on the sample data are placed
directly into the model store.
"""
import os

import pytest
from sklearn.linear_model import LinearRegression

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
os.environ["API_SKIP_MODEL_LOAD"] = "1"       # never contact the registry during tests

from fastapi.testclient import TestClient

from api import model_store, service
from api.main import app
from climate_models.m1_model import M1GrowthModel
from climate_models.m3_model import M3Model, MODELS, INPUT_COLS, TARGET

GOOD_M3 = {"lat": 26.9, "lon": 75.8, "year": 2030, "month": 5, "co2_ppm": 440.0}
GOOD_M2 = {"total_emissions_gtc": 11.0, "nino34": 0.5, "nino34_lag1": 0.0}
GOOD_M1 = {"co2_last_mt": 2800.0, "population_change_pct": 8.0, "income_change_pct": 60.0, "years_ahead": 10}


@pytest.fixture(scope="module")
def client(m1, m2, m3):
    model_store.clear()
    model_store.set_model("m1", M1GrowthModel(step=5).fit(m1))
    model_store.set_model("m2", LinearRegression().fit(m2[service.M2_FEATURES], m2["growth_ppm"]))
    model_store.set_model("m3", M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET]))
    from api import rain_router
    assert rain_router.load_rain_model(), "rain_package/ should load from the repo root"
    yield TestClient(app)
    model_store.clear()


def test_health_reports_all_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"m1", "m2", "m3", "rain"}
    assert body["models"]["rain"]["source"] == "rain package"


def test_rain_routes_are_mounted_in_the_main_app(client):
    r = client.get("/rain/forecast", params={"lat": 26.9, "lon": 75.8})
    assert r.status_code == 200
    assert r.json()["cell_id"] == "26.5_75.5"      # same 1-degree square rule as the temperature model
    assert client.get("/rain/model-info").status_code == 200


def test_india_temperature_returns_a_number_and_the_model_version(client):
    r = client.post("/predict/india_temperature", json=GOOD_M3)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["anomaly_degc"], float)
    assert body["grid_square"] == "26.5_75.5"
    assert body["served_model"] == "m3_india_temperature" and body["served_version"] == "test"
    assert body["beyond_training_range"] is True          # 440 ppm is above the training maximum


def test_co2_growth_and_country_emissions_work(client):
    r2 = client.post("/predict/co2_growth", json=GOOD_M2)
    assert r2.status_code == 200 and isinstance(r2.json()["co2_growth_ppm"], float)
    r1 = client.post("/predict/country_emissions", json=GOOD_M1)
    assert r1.status_code == 200 and r1.json()["co2_mt"] > 0


@pytest.mark.parametrize("field,value", [("lat", 500), ("lon", -10), ("month", 13),
                                         ("co2_ppm", 5), ("year", 1800), ("nino_lag1", 99)])
def test_impossible_inputs_are_rejected_with_422(client, field, value):
    r = client.post("/predict/india_temperature", json={**GOOD_M3, field: value})
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in GOOD_M3.items() if k != "co2_ppm"}
    assert client.post("/predict/india_temperature", json=bad).status_code == 422


def test_negative_emissions_rejected(client):
    assert client.post("/predict/co2_growth", json={**GOOD_M2, "total_emissions_gtc": -1}).status_code == 422
    assert client.post("/predict/country_emissions", json={**GOOD_M1, "co2_last_mt": 0}).status_code == 422


def test_unloaded_model_gives_503(client):
    model, info = model_store.get("m2")
    model_store._models.pop("m2")
    try:
        assert client.post("/predict/co2_growth", json=GOOD_M2).status_code == 503
        assert client.get("/health").json()["status"] == "degraded"
    finally:
        model_store.set_model("m2", model, info["version"])
