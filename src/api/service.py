"""
api/service.py

The logic between a web request and a model: build the model's input table from
the user's simple inputs, call the model, and package the answer.

Kept free of FastAPI so it can be tested directly. main.py only does the web part.
"""
import math

import numpy as np
import pandas as pd

from api import limits as L
from climate_models.m1_model import PREDICT_COLS
from climate_models.m3_model import INPUT_COLS, co2_forcing_from_ppm

M2_FEATURES = ["total_emissions", "nino34", "nino34_lag1"]


def grid_square(lat, lon):
    """The 1-degree square a point falls in. Squares are centred on .5 values,
    so 26.9 N, 75.8 E belongs to the square '26.5_75.5'."""
    sq_lat, sq_lon = math.floor(lat) + 0.5, math.floor(lon) + 0.5
    return sq_lat, sq_lon, f"{sq_lat}_{sq_lon}"


# ---------------------------------------------------------------- Model 3
def india_temperature_inputs(lat, lon, year, month, co2_ppm, nino_lag1=0.0, nino_lag3=0.0,
                             nino_lag6=0.0, nino_prev12_mean=0.0, cum_emissions_gtco2=None):
    sq_lat, sq_lon, cell_id = grid_square(lat, lon)
    row = {"cell_id": cell_id, "year": int(year), "month": int(month),
           "lat": sq_lat, "lon": sq_lon,
           "month_sin": math.sin(2 * math.pi * month / 12),
           "month_cos": math.cos(2 * math.pi * month / 12),
           "nino_lag1": nino_lag1, "nino_lag3": nino_lag3, "nino_lag6": nino_lag6,
           "nino_prev12_mean": nino_prev12_mean,
           "co2_forcing": float(co2_forcing_from_ppm(co2_ppm)),
           "cum_emissions_gtco2": np.nan if cum_emissions_gtco2 is None else cum_emissions_gtco2}
    return pd.DataFrame([row])[INPUT_COLS]


def predict_india_temperature(model, lat, lon, year, month, co2_ppm, **optional):
    if getattr(model, "co2_col", None) == "cum_emissions_gtco2" and optional.get("cum_emissions_gtco2") is None:
        raise ValueError("this model version needs cum_emissions_gtco2")
    X = india_temperature_inputs(lat, lon, year, month, co2_ppm, **optional)
    anomaly = float(model.predict(X)[0])
    beyond = co2_ppm > L.TRAINED_CO2_PPM_MAX
    return {"anomaly_degc": round(anomaly, 3),
            "grid_square": X["cell_id"].iloc[0],
            "co2_forcing_wm2": round(float(X["co2_forcing"].iloc[0]), 4),
            "beyond_training_range": bool(beyond),
            "note": ("CO2 is above anything in the training data (max "
                     f"{L.TRAINED_CO2_PPM_MAX} ppm): treat this as a what-if, not a forecast."
                     if beyond else
                     "Anomaly = degrees warmer (+) or cooler (-) than this square's 1961-1990 normal "
                     "for that month. Single months also swing about +/-1 degC with weather.")}


# ---------------------------------------------------------------- Model 2
def predict_co2_growth(model, total_emissions_gtc, nino34=0.0, nino34_lag1=0.0):
    X = pd.DataFrame([{"total_emissions": total_emissions_gtc, "nino34": nino34,
                       "nino34_lag1": nino34_lag1}])[M2_FEATURES]
    growth = float(model.predict(X)[0])
    return {"co2_growth_ppm": round(growth, 3),
            "note": "Predicted rise of CO2 in the air this year, in ppm. Depends on total human "
                    "emissions and on El Nino (this year and last year)."}


# ---------------------------------------------------------------- Model 1
def predict_country_emissions(model, co2_last_mt, population_change_pct, income_change_pct, years_ahead):
    """Percent changes are totals over the whole period (e.g. +12 means 12% more people)."""
    X = pd.DataFrame([{"log_co2_last": math.log(co2_last_mt),
                       "d_pop": math.log(1 + population_change_pct / 100),
                       "d_income": math.log(1 + income_change_pct / 100),
                       "span": float(years_ahead)}])[PREDICT_COLS]
    co2 = float(np.exp(model.predict(X)[0]))
    beyond = years_ahead > L.TESTED_YEARS_AHEAD
    return {"co2_mt": round(co2, 3),
            "change_pct": round((co2 / co2_last_mt - 1) * 100, 2),
            "beyond_tested_horizon": bool(beyond),
            "note": (f"More than {L.TESTED_YEARS_AHEAD} years ahead was never tested: treat as a rough what-if."
                     if beyond else
                     "Emissions predicted from the last known level, grown by the learned rule for "
                     "population and income change.")}
