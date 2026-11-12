"""
tests/test_service.py

Tests for the logic between a request and a model (src/api/service.py).
No web server is needed: models are fitted on the sample data and called directly.
"""
import math

import pytest
from sklearn.linear_model import LinearRegression

from api import limits as L
from api import service
from climate_models.m1_model import M1GrowthModel
from climate_models.m3_model import M3Model, MODELS, INPUT_COLS, TARGET, co2_forcing_from_ppm


@pytest.fixture(scope="module")
def model_m1(m1):
    return M1GrowthModel(step=5).fit(m1)


@pytest.fixture(scope="module")
def model_m2(m2):
    return LinearRegression().fit(m2[service.M2_FEATURES], m2["growth_ppm"])


@pytest.fixture(scope="module")
def model_m3(m3):
    return M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET])


# ---------------------------------------------------------------- grid squares
def test_grid_square_snaps_to_centre():
    assert service.grid_square(26.9, 75.8) == (26.5, 75.5, "26.5_75.5")
    assert service.grid_square(8.0, 77.0) == (8.5, 77.5, "8.5_77.5")


# ---------------------------------------------------------------- Model 3
def test_m3_inputs_have_the_model_columns_in_order():
    X = service.india_temperature_inputs(26.9, 75.8, 2030, 5, 440.0)
    assert list(X.columns) == INPUT_COLS and len(X) == 1
    assert X["co2_forcing"].iloc[0] == pytest.approx(float(co2_forcing_from_ppm(440.0)))
    assert X["month_sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 5 / 12))


def test_m3_prediction_is_a_plausible_number(model_m3):
    out = service.predict_india_temperature(model_m3, 26.9, 75.8, 2020, 5, 414.0)
    assert -10 < out["anomaly_degc"] < 10
    assert out["grid_square"] == "26.5_75.5"
    assert out["beyond_training_range"] is False


def test_m3_more_co2_is_warmer_and_flagged(model_m3):
    now = service.predict_india_temperature(model_m3, 26.9, 75.8, 2040, 5, 420.0)
    later = service.predict_india_temperature(model_m3, 26.9, 75.8, 2040, 5, 500.0)
    assert later["anomaly_degc"] > now["anomaly_degc"]
    assert later["beyond_training_range"] is True and "what-if" in later["note"]


def test_m3_emissions_model_requires_cumulative_emissions(m3):
    model = M3Model(**MODELS["linear_cum_emissions"]).fit(m3[INPUT_COLS], m3[TARGET])
    with pytest.raises(ValueError):
        service.predict_india_temperature(model, 26.9, 75.8, 2020, 5, 414.0)
    out = service.predict_india_temperature(model, 26.9, 75.8, 2020, 5, 414.0, cum_emissions_gtco2=1700.0)
    assert -10 < out["anomaly_degc"] < 10


# ---------------------------------------------------------------- Model 2
def test_m2_more_emissions_means_bigger_rise(model_m2, m2):
    mid = float(m2["total_emissions"].median())
    low = service.predict_co2_growth(model_m2, mid)["co2_growth_ppm"]
    high = service.predict_co2_growth(model_m2, mid * 1.2)["co2_growth_ppm"]
    assert high > low


# ---------------------------------------------------------------- Model 1
def test_m1_no_growth_short_span_stays_near_last_level(model_m1):
    out = service.predict_country_emissions(model_m1, 1000.0, 0.0, 0.0, 1)
    assert out["co2_mt"] == pytest.approx(1000.0, rel=0.05)     # only the small yearly drift applies


def test_m1_growth_raises_emissions_and_long_horizon_is_flagged(model_m1):
    base = service.predict_country_emissions(model_m1, 1000.0, 5.0, 10.0, 10)
    more = service.predict_country_emissions(model_m1, 1000.0, 5.0, 80.0, 10)
    assert more["co2_mt"] > base["co2_mt"]
    assert base["beyond_tested_horizon"] is False
    far = service.predict_country_emissions(model_m1, 1000.0, 5.0, 10.0, L.TESTED_YEARS_AHEAD + 5)
    assert far["beyond_tested_horizon"] is True
