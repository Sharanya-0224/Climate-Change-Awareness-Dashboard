"""
tests/test_data.py

Data tests: do the model-ready tables obey the rules the models rely on?
If a feature script changes and breaks a rule (for example a lag that
accidentally looks into the future), these tests fail before any model is trained.
"""
import numpy as np
import pandas as pd

from climate_models.m3_model import INPUT_COLS, TARGET, co2_forcing_from_ppm
from climate_models.m1_model import PANEL_COLS

M2_COLS = ["year", "growth_ppm", "total_emissions", "nino34", "nino34_lag1", "volcano_flag"]


# ====================================================================== Model 1
def test_m1_has_required_columns(m1):
    assert set(PANEL_COLS) <= set(m1.columns)


def test_m1_no_missing_values(m1):
    assert m1[PANEL_COLS].isna().sum().sum() == 0


def test_m1_one_row_per_country_year(m1):
    assert not m1.duplicated(["country", "year"]).any()


def test_m1_logs_are_finite(m1):
    assert np.isfinite(m1[PANEL_COLS[2:]].to_numpy()).all(), "log of zero or negative value"


# ====================================================================== Model 2
def test_m2_has_required_columns(m2):
    assert set(M2_COLS) <= set(m2.columns)


def test_m2_no_missing_values(m2):
    assert m2[M2_COLS].isna().sum().sum() == 0


def test_m2_years_consecutive(m2):
    years = m2["year"].to_numpy()
    assert np.all(np.diff(years) == 1), "years must be consecutive with no gaps"


def test_m2_nino_lag_is_last_years_value(m2):
    """nino34_lag1 in year t must equal nino34 in year t-1: the lag only looks backwards."""
    expected = m2["nino34"].shift(1)
    got = m2["nino34_lag1"]
    mask = expected.notna()
    assert np.allclose(got[mask], expected[mask], atol=1e-6)


def test_m2_values_plausible(m2):
    assert m2["total_emissions"].min() > 0
    assert m2["growth_ppm"].between(-1, 6).all(), "yearly CO2 rise outside a plausible range"
    assert set(m2["volcano_flag"].unique()) <= {0, 1}


# ====================================================================== Model 3
def test_m3_has_required_columns(m3):
    assert set(INPUT_COLS + [TARGET, "co2_ppm"]) <= set(m3.columns)


def test_m3_no_missing_inputs(m3):
    assert m3[INPUT_COLS + [TARGET]].isna().sum().sum() == 0, "never fill: rows with gaps must be dropped"


def test_m3_no_duplicate_square_months(m3):
    assert not m3.duplicated(["cell_id", "year", "month"]).any()


def test_m3_location_and_month_ranges(m3):
    assert m3["lat"].between(6, 38).all() and m3["lon"].between(66, 100).all(), "outside India"
    assert m3["month"].between(1, 12).all()
    assert m3["year"].between(1951, 2030).all()


def test_m3_cell_id_matches_lat_lon(m3):
    ids = m3["lat"].astype(str) + "_" + m3["lon"].astype(str)
    assert (ids == m3["cell_id"]).all()


def test_m3_season_features_match_month(m3):
    assert np.allclose(m3["month_sin"], np.sin(2 * np.pi * m3["month"] / 12), atol=1e-4)
    assert np.allclose(m3["month_cos"], np.cos(2 * np.pi * m3["month"] / 12), atol=1e-4)


def test_m3_co2_forcing_formula(m3):
    """co2_forcing must be 5.35 * ln(ppm / 280): the API will rebuild it from ppm."""
    assert np.allclose(m3["co2_forcing"], co2_forcing_from_ppm(m3["co2_ppm"]), atol=1e-4)


def test_m3_co2_is_one_value_per_year(m3):
    assert (m3.groupby("year")["co2_ppm"].nunique() == 1).all()


def test_m3_co2_rises_over_time(m3):
    yearly = m3.groupby("year")["co2_ppm"].first()
    assert yearly.loc[yearly.index.max()] > yearly.loc[yearly.index.min()]


def test_m3_nino_lags_only_look_backwards(m3):
    """For one square, nino_lag3 at month t must equal nino_lag1 at month t-2
    (both are the El Nino value 3 months before t). If a lag were built from the
    future this would break."""
    for cell, g in m3.groupby("cell_id"):
        g = g.sort_values(["year", "month"])
        t = g["year"] * 12 + g["month"]
        consecutive = (t.diff() == 1).to_numpy()
        lag1_two_back = g["nino_lag1"].shift(2).to_numpy()
        both = consecutive & np.roll(consecutive, 1)      # t-1 and t-2 present
        both[:2] = False
        assert np.allclose(g["nino_lag3"].to_numpy()[both], lag1_two_back[both], atol=1e-6), cell


def test_m3_anomaly_plausible(m3):
    assert m3[TARGET].abs().max() < 15, "a monthly anomaly beyond 15 degC means a unit or sentinel error"


def test_m3_anomaly_centred_on_1961_1990(m3):
    """The normal is each square's 1961-1990 average, so anomalies in that period
    should average close to zero for every square (small tolerance: a few months
    may be missing)."""
    base = m3[m3["year"].between(1961, 1990)]
    means = base.groupby("cell_id")[TARGET].mean()
    assert (means.abs() < 0.15).all(), means.round(3).to_dict()
