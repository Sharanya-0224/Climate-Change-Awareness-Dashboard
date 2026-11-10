"""
features_m2_co2.py  (put it at src/features/features_m2_co2.py)

Model 2 model-ready table: how much CO2 in the air rises each year.

Run from the repo root:
    python3 src/features/features_m2_co2.py

Input:  data/processed_drivers/global_carbon_budget.csv
Output: data/model_ready/m2_co2_growth.csv

Columns:
- key:     year
- target:  growth_ppm (atmospheric CO2 rise that year, ppm, from the carbon budget)
- inputs:  total_emissions (GtC/yr), nino34, nino34_lag1        (main model)
- extra:   fossil_emissions, land_use_emissions, volcano_flag   (experiments only)
- split:   split_time = "train" if year <= 2009 else "test"

volcano_flag marks years after the three large eruptions since 1959
(Agung 1963-64, El Chichon 1982-83, Pinatubo 1991-92). The dates are written
from general knowledge; verify them before reporting. Future eruptions cannot
be known, so this flag is for explaining the past only, never for scenarios.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/processed_drivers/global_carbon_budget.csv")
OUT = Path("data/model_ready/m2_co2_growth.csv")
TRAIN_END = 2009
VOLCANO_YEARS = [1963, 1964, 1982, 1983, 1991, 1992]


def main():
    g = pd.read_csv(SRC).sort_values("year").reset_index(drop=True)

    out = g[["year"]].copy()
    out["growth_ppm"] = g["atmospheric_growth_ppm"]
    out["total_emissions"] = g["total_emissions"]
    out["fossil_emissions"] = g["fossil_emissions"]
    out["land_use_emissions"] = g["land_use_emissions"]
    out["nino34"] = g["nino34"]
    out["nino34_lag1"] = g["nino34"].shift(1)
    out["volcano_flag"] = out["year"].isin(VOLCANO_YEARS).astype(int)
    out["split_time"] = np.where(out["year"] <= TRAIN_END, "train", "test")

    before = len(out)
    out = out.dropna(subset=["nino34_lag1"])          # first year has no lag
    assert not out.isna().any().any(), "unexpected missing values"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Saved {OUT}: {len(out)} rows ({before - len(out)} dropped for the lag), "
          f"years {out['year'].min()}-{out['year'].max()}")
    print(out["split_time"].value_counts().to_string())
    print("Main inputs: total_emissions, nino34, nino34_lag1 | target: growth_ppm")


if __name__ == "__main__":
    main()
