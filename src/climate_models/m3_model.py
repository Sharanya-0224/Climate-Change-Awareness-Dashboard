"""
climate_models/m3_model.py

The Model 3 class (India temperature anomaly) in ONE shared place.

Why a shared module?  The training script, the registry script and the API all
need the same model definition. If each had its own copy, they could drift apart
and a saved model might not load. Everything imports from here instead.

A saved model remembers that its class lives at climate_models.m3_model.M3Model,
so this folder is shipped with the model (MLflow code_paths) and must stay importable.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

TARGET = "tmean_anom"
NINO = ["nino_lag1", "nino_lag3", "nino_lag6", "nino_prev12_mean"]
PLACE_SEASON = ["lat", "lon", "month_sin", "month_cos"]
GBM_COLS = NINO + PLACE_SEASON
INPUT_COLS = ["cell_id", "year", "month"] + PLACE_SEASON + NINO + ["co2_forcing", "cum_emissions_gtco2"]
SEED = 42

# Every Model 3 variant, by name. The training script, registry and notebook all read this.
MODELS = {
    "baseline_zero": dict(kind="zero"),
    "baseline_own_trend": dict(kind="own_trend"),
    "natural_only": dict(kind="linear", co2_col=None),
    "linear_co2": dict(kind="linear", co2_col="co2_forcing"),
    "linear_cum_emissions": dict(kind="linear", co2_col="cum_emissions_gtco2"),
    "linear_co2_regional": dict(kind="linear", co2_col="co2_forcing", regional=True),
    "linear_co2_per_square": dict(kind="linear", co2_col="co2_forcing", per_square=True),
    "hybrid_regional_gbm": dict(kind="linear", co2_col="co2_forcing", regional=True,
                                residual_gbm=True),
}


def co2_forcing_from_ppm(ppm):
    """CO2's warming push in W/m2, the same formula the feature script uses."""
    return 5.35 * np.log(np.asarray(ppm, float) / 280.0)


def per_square_line(cells, x, y):
    """Straight line y = intercept + slope * x fitted separately for each square."""
    d = pd.DataFrame({"cell": np.asarray(cells), "x": np.asarray(x, float),
                      "y": np.asarray(y, float)})
    g = d.groupby("cell")
    mx, my = g["x"].mean(), g["y"].mean()
    dx = d["x"] - d["cell"].map(mx)
    dy = d["y"] - d["cell"].map(my)
    sxy = (dx * dy).groupby(d["cell"]).sum()
    sxx = (dx * dx).groupby(d["cell"]).sum()
    slope = (sxy / sxx.where(sxx > 0)).fillna(0.0)
    return pd.DataFrame({"slope": slope, "intercept": my - slope * mx})


def apply_square_line(table, cells, x):
    cells = pd.Series(np.asarray(cells))
    a = cells.map(table["intercept"]).fillna(0.0).to_numpy()
    b = cells.map(table["slope"]).fillna(0.0).to_numpy()
    return a + b * np.asarray(x, float)


def decimal_year(X):
    return X["year"].to_numpy() + (X["month"].to_numpy() - 0.5) / 12


class M3Model(BaseEstimator, RegressorMixin):
    """One class for every Model 3 variant, so each saved model takes the same input
    table (INPUT_COLS) and works the same way in the API."""

    def __init__(self, kind="linear", co2_col="co2_forcing", regional=False,
                 per_square=False, residual_gbm=False, seed=SEED):
        self.kind = kind
        self.co2_col = co2_col
        self.regional = regional
        self.per_square = per_square
        self.residual_gbm = residual_gbm
        self.seed = seed

    def _design(self, X):
        cols = {c: X[c].to_numpy(float) for c in NINO + PLACE_SEASON}
        if self.co2_col:
            c = X[self.co2_col].to_numpy(float)
            cols[self.co2_col] = c
            if self.regional:
                la = (X["lat"].to_numpy(float) - 23.0) / 10.0
                lo = (X["lon"].to_numpy(float) - 80.0) / 10.0
                cols.update({"co2_x_lat": c * la, "co2_x_lon": c * lo, "co2_x_lat_lon": c * la * lo,
                             "co2_x_lat2": c * la ** 2, "co2_x_lon2": c * lo ** 2,
                             "co2_x_month_sin": c * X["month_sin"].to_numpy(float),
                             "co2_x_month_cos": c * X["month_cos"].to_numpy(float)})
        return pd.DataFrame(cols, index=X.index)

    def fit(self, X, y):
        y = np.asarray(y, float)
        if self.kind == "zero":
            return self
        if self.kind == "own_trend":
            self.square_table_ = per_square_line(X["cell_id"], decimal_year(X), y)
            return self
        D = self._design(X)
        self.feature_names_ = list(D.columns)
        self.linear_ = LinearRegression().fit(D, y)
        resid = y - self.linear_.predict(D)
        if self.per_square:
            self.square_table_ = per_square_line(X["cell_id"], X[self.co2_col], resid)
            resid = resid - apply_square_line(self.square_table_, X["cell_id"], X[self.co2_col])
        if self.residual_gbm:
            self.gbm_ = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                l2_regularization=1.0, random_state=self.seed).fit(X[GBM_COLS], resid)
        return self

    def predict(self, X):
        if self.kind == "zero":
            return np.zeros(len(X))
        if self.kind == "own_trend":
            return apply_square_line(self.square_table_, X["cell_id"], decimal_year(X))
        p = self.linear_.predict(self._design(X))
        if self.per_square:
            p = p + apply_square_line(self.square_table_, X["cell_id"], X[self.co2_col])
        if self.residual_gbm:
            p = p + self.gbm_.predict(X[GBM_COLS])
        return p
