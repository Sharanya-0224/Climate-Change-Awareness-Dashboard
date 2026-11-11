"""
climate_models/m1_model.py

Model 1 (what drives a country's CO2 emissions) as one self-contained class.

The champion is a GROWTH RULE, not a direct predictor:

    change in log(emissions) = a * change in log(population)
                             + b * change in log(income per person)
                             + drift * number of years

To predict, it starts from a country's LAST KNOWN emissions and applies the rule.
The model saved during the exams was only the small regression inside the rule,
which cannot predict on its own. This class wraps the whole rule so the registry
and the API get something usable: give it the last known emissions, the changes
in population and income, and how many years ahead; it returns emissions.

Everything is in natural logs, as in training. A change of 0.10 in log terms is
about +10.5%.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

POP, INCOME, TARGET = "log_population", "log_gdp_per_capita", "log_co2"
PANEL_COLS = ["country", "year", POP, INCOME, TARGET]
PREDICT_COLS = ["log_co2_last", "d_pop", "d_income", "span"]
HORIZON = 12      # years predicted in each exam, as in the training script


def change_pairs(df, step):
    """For each country, every pair of years `step` apart: the change in population,
    income and emissions over that span."""
    base = df[PANEL_COLS]
    end = base.assign(start_year=base["year"] - step)
    m = end.merge(base, left_on=["country", "start_year"], right_on=["country", "year"],
                  suffixes=("", "_start"))
    return pd.DataFrame({"d_pop": m[POP] - m[f"{POP}_start"],
                         "d_income": m[INCOME] - m[f"{INCOME}_start"],
                         "d_co2": m[TARGET] - m[f"{TARGET}_start"]})


def exam_table(df, cutoff, horizon=HORIZON):
    """Rows for the years after `cutoff`, each with the country's last known values
    (at or before the cutoff) and the changes since then. Same logic as the exams."""
    train = df[df["year"] <= cutoff]
    test = df[(df["year"] > cutoff) & (df["year"] <= cutoff + horizon)]
    last = train.sort_values("year").groupby("country").last()[[POP, INCOME, TARGET, "year"]]
    t = test.join(last, on="country", rsuffix="_last").dropna(subset=[f"{TARGET}_last"])
    return pd.DataFrame({"country": t["country"].values, "year": t["year"].values,
                         "log_co2_last": t[f"{TARGET}_last"].values,
                         "d_pop": (t[POP] - t[f"{POP}_last"]).values,
                         "d_income": (t[INCOME] - t[f"{INCOME}_last"]).values,
                         "span": (t["year"] - t["year_last"]).values.astype(float),
                         "actual": t[TARGET].values,
                         "stays_same": t[f"{TARGET}_last"].values})


class M1GrowthModel(BaseEstimator, RegressorMixin):
    """Linear growth rule learned from `step`-year changes (5 for the champion)."""

    def __init__(self, step=5):
        self.step = step

    def fit(self, X, y=None):
        """X is the country panel (PANEL_COLS). The rule is learned from all pairs of
        years `step` apart inside X."""
        pairs = change_pairs(X, self.step)
        reg = LinearRegression().fit(pairs[["d_pop", "d_income"]], pairs["d_co2"])
        self.a_pop_ = float(reg.coef_[0])
        self.b_income_ = float(reg.coef_[1])
        self.drift_per_year_ = float(reg.intercept_ / self.step)
        self.n_pairs_ = int(len(pairs))
        return self

    def predict_change(self, X):
        return (self.a_pop_ * np.asarray(X["d_pop"], float)
                + self.b_income_ * np.asarray(X["d_income"], float)
                + self.drift_per_year_ * np.asarray(X["span"], float))

    def predict(self, X):
        """X has PREDICT_COLS. Returns predicted log(emissions)."""
        return np.asarray(X["log_co2_last"], float) + self.predict_change(X)
