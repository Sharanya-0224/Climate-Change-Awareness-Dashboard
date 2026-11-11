"""
tests/test_models.py

Model tests: does the model code behave as it should on the sample data?
These do not judge accuracy (that is what the exams and gates are for); they
check shape, sanity, repeatability and that the model learns the right direction.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from climate_models.m3_model import M3Model, MODELS, INPUT_COLS, TARGET
from quality import gates as G

M2_FEATURES = ["total_emissions", "nino34", "nino34_lag1"]   # the Model 2 champion's inputs
FAST_M3 = [m for m in MODELS if m != "hybrid_regional_gbm"]  # trees are slow; tested separately


# ====================================================================== Model 2
def test_m2_champion_fits_and_learns_positive_emissions_effect(m2):
    model = LinearRegression().fit(m2[M2_FEATURES], m2["growth_ppm"])
    pred = model.predict(m2[M2_FEATURES])
    assert len(pred) == len(m2) and np.isfinite(pred).all()
    assert model.coef_[0] > 0, "more emissions must mean a bigger CO2 rise"


def test_m2_beats_average_baseline_in_sample(m2):
    model = LinearRegression().fit(m2[M2_FEATURES], m2["growth_ppm"])
    rmse_model = np.sqrt(np.mean((m2["growth_ppm"] - model.predict(m2[M2_FEATURES])) ** 2))
    rmse_avg = m2["growth_ppm"].std(ddof=0)
    assert rmse_model < rmse_avg


# ====================================================================== Model 3
@pytest.mark.parametrize("name", FAST_M3)
def test_m3_every_variant_fits_and_predicts(m3, name):
    model = M3Model(**MODELS[name]).fit(m3[INPUT_COLS], m3[TARGET])
    pred = model.predict(m3[INPUT_COLS])
    assert pred.shape == (len(m3),)
    assert np.isfinite(pred).all()


def test_m3_hybrid_fits_on_small_sample(m3):
    small = m3[m3["year"] >= 2000]
    model = M3Model(**MODELS["hybrid_regional_gbm"]).fit(small[INPUT_COLS], small[TARGET])
    assert np.isfinite(model.predict(small[INPUT_COLS])).all()


def test_m3_champion_is_deterministic(m3):
    model = M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET])
    a, b = model.predict(m3[INPUT_COLS]), model.predict(m3[INPUT_COLS])
    assert np.allclose(a, b)


def test_m3_champion_learns_warming_from_co2(m3):
    model = M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET])
    w = dict(zip(model.feature_names_, model.linear_.coef_))
    assert w["co2_forcing"] > 0, "more CO2 forcing must mean a warmer anomaly"


def test_m3_champion_beats_zero_baseline_in_sample(m3):
    model = M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET])
    rmse_model = np.sqrt(np.mean((m3[TARGET] - model.predict(m3[INPUT_COLS])) ** 2))
    rmse_zero = np.sqrt(np.mean(m3[TARGET] ** 2))
    assert rmse_model < rmse_zero


def test_m3_higher_co2_gives_warmer_prediction(m3):
    """Scenario sanity: same place and month, more CO2 -> warmer."""
    model = M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET])
    row = m3[INPUT_COLS].iloc[[0]].copy()
    low, high = row.copy(), row.copy()
    low["co2_forcing"], high["co2_forcing"] = 1.0, 2.0
    assert model.predict(high)[0] > model.predict(low)[0]


def test_m3_champion_works_for_unseen_square(m3):
    """A square the model never saw (e.g. a user at the edge of the data) still gets a prediction."""
    train = m3[m3["cell_id"] != m3["cell_id"].iloc[0]]
    unseen = m3[m3["cell_id"] == m3["cell_id"].iloc[0]]
    model = M3Model(**MODELS["linear_co2"]).fit(train[INPUT_COLS], train[TARGET])
    assert np.isfinite(model.predict(unseen[INPUT_COLS])).all()


def test_m3_baseline_zero_predicts_zero(m3):
    model = M3Model(**MODELS["baseline_zero"]).fit(m3[INPUT_COLS], m3[TARGET])
    assert (model.predict(m3[INPUT_COLS]) == 0).all()


# ====================================================================== quality gates
def test_gate_no_missing_outputs_catches_nan():
    assert G.no_missing_outputs([1.0, 2.0])["passed"]
    assert not G.no_missing_outputs([1.0, np.nan])["passed"]
    assert not G.no_missing_outputs([1.0, np.inf])["passed"]


def test_gate_physical_range_uses_observed_range_plus_margin():
    observed = np.array([-2.0, 3.0])
    assert G.physical_range([-6.9, 7.9], observed, margin=5.0, unit="degC")["passed"]
    assert not G.physical_range([8.1], observed, margin=5.0, unit="degC")["passed"]


def test_gate_hit_rate_share():
    g = G.hit_rate(actual=[0, 0, 0, 0], pred=[0.1, 0.1, 0.5, 0.5], tolerance=0.2, unit="x")
    assert g["passed"] and "50%" in g["detail"]
    g = G.hit_rate(actual=[0, 0, 0, 0], pred=[0.5, 0.5, 0.5, 0.1], tolerance=0.2, unit="x")
    assert not g["passed"]


def test_gate_beats_baseline_threshold():
    assert G.beats_baseline(0.1, "zero")["passed"]
    assert not G.beats_baseline(-0.1, "zero")["passed"]


def test_gate_report_all_passed():
    ok, table = G.report([G.beats_baseline(1, "zero"), G.no_missing_outputs([1.0])])
    assert ok and len(table) == 2
    ok, _ = G.report([G.beats_baseline(1, "zero"), G.no_missing_outputs([np.nan])])
    assert not ok
