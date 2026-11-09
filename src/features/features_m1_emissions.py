"""
features_m1_emissions.py  (put it at src/features/features_m1_emissions.py)

Model 1 model-ready table: what drives a country's CO2 emissions.

Run from the repo root:
    python3 src/features/features_m1_emissions.py

Input:  data/processed_drivers/country_panel.csv
Output: data/model_ready/m1_emissions.csv

Columns:
- keys:     country, iso_code, year
- target:   log_co2 (natural log of co2, million tonnes); co2 kept for reporting
- inputs:   log_population, log_gdp_per_capita            (main model, option A)
- extra:    log_energy_per_gdp                            (option B comparison only)
- split:    split_time   = "train" if year <= 2010 else "test"
            country_fold = 0-4, fixed per country, for "unseen country" tests

Logs are used because the EDA showed raw values are very skewed (skew 9.4 -> -0.16).
No values are filled. Rows with co2 == 0 would break the log; the build stage
already has none, and the script checks it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/processed_drivers/country_panel.csv")
OUT = Path("data/model_ready/m1_emissions.csv")
TRAIN_END = 2010
N_FOLDS = 5
SEED = 42


def main():
    df = pd.read_csv(SRC)
    assert (df["co2"] > 0).all(), "co2 must be > 0 for the log"

    out = df[["country", "iso_code", "year", "co2"]].copy()
    out["log_co2"] = np.log(df["co2"])
    out["log_population"] = np.log(df["population"])
    out["log_gdp_per_capita"] = np.log(df["gdp_per_capita"])
    out["log_energy_per_gdp"] = np.log(df["energy_per_gdp"])
    out["split_time"] = np.where(out["year"] <= TRAIN_END, "train", "test")

    # Each country gets one fixed fold (shuffled once with a seed, then round-robin).
    countries = sorted(out["country"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(countries)
    folds = {c: i % N_FOLDS for i, c in enumerate(countries)}
    out["country_fold"] = out["country"].map(folds)

    assert not out.isna().any().any(), "unexpected missing values"
    assert not out.duplicated(["country", "year"]).any()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Saved {OUT}: {len(out):,} rows, {out['country'].nunique()} countries")
    print(out["split_time"].value_counts().to_string())
    print("Countries per fold:", out.groupby("country_fold")["country"].nunique().to_dict())
    print(f"India fold: {folds.get('India')}")
    print("Main inputs: log_population, log_gdp_per_capita | target: log_co2")


if __name__ == "__main__":
    main()
