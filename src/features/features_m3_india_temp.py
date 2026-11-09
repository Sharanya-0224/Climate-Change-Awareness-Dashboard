"""
features_m3_india_temp.py  (put it at src/features/features_m3_india_temp.py)

Model 3 model-ready table: India temperature anomaly per 1-degree square per month.

Run from the repo root:
    python3 src/features/features_m3_india_temp.py

Inputs:
    data/processed_india/india_monthly_temperature.csv   (complete grid, gaps marked)
    data/raw/smoothedco2.txt                             (ice-core CO2, up to 1958)
    data/raw/co2_annmean_mlo.txt                         (Mauna Loa CO2, from 1959)
    data/raw/detrend.nino34.ascii.txt                    (monthly Nino 3.4 anomaly)
    data/raw/emissions/owid-co2-data.csv                 (World cumulative CO2)
Output:
    data/model_ready/m3_india_temp.csv

Columns:
- keys:     date, year, month, cell_id, lat, lon
- target:   tmean_anom (degC vs the square's 1961-1990 normal); tmax_anom, tmin_anom kept
- inputs, version A (CO2 in the air):     co2_forcing
- inputs, version B (human emissions):    cum_emissions_gtco2
- inputs, both versions:  nino_lag1, nino_lag3, nino_lag6, nino_prev12_mean,
                          lat, lon, month_sin, month_cos
- reporting only:         co2_ppm, co2_source
- split:    split_time = "train" if year <= 2010 else "test"

Rules:
- Rows with a missing tmean_anom are dropped (nothing is filled).
- Squares with more than MAX_MISSING_PCT of their months missing are dropped.
- CO2 is one value per year (ice core up to 1958, Mauna Loa from 1959), used for
  all 12 months of that year; CO2 changes slowly, and scenarios are yearly.
- Nino features use only months BEFORE the current month.
- Years end at 2024 because OWID World cumulative CO2 ends in 2024; all versions
  use the same rows so the A/B/C comparison is fair. Cumulative CO2 is fossil
  plus industry only (land use is not included in OWID's cumulative_co2).
"""

from pathlib import Path

import numpy as np
import pandas as pd

INDIA = Path("data/processed_india/india_monthly_temperature.csv")
ICE = Path("data/raw/smoothedco2.txt")
MLO = Path("data/raw/co2_annmean_mlo.txt")
NINO = Path("data/raw/detrend.nino34.ascii.txt")
OWID = Path("data/raw/emissions/owid-co2-data.csv")
OUT = Path("data/model_ready/m3_india_temp.csv")

START, END = 1951, 2024
SPLICE_YEAR = 1958          # ice core up to and including this year
TRAIN_END = 2010
MAX_MISSING_PCT = 30.0


def spliced_co2():
    ice = pd.read_csv(ICE, sep="\t")[["Year", "ALL_50_full"]]
    ice.columns = ["year", "co2_ppm"]
    ice["year"] = ice["year"].round().astype(int)
    ice = ice.groupby("year", as_index=False)["co2_ppm"].mean()
    ice = ice[ice["year"] <= SPLICE_YEAR].assign(co2_source="ice_core")

    mlo = pd.read_csv(MLO, comment="#", sep=r"\s+", header=None,
                      names=["year", "co2_ppm", "unc"])[["year", "co2_ppm"]]
    mlo = mlo[mlo["year"] > SPLICE_YEAR].assign(co2_source="mauna_loa")

    co2 = pd.concat([ice, mlo]).sort_values("year")
    co2 = co2[co2["year"].between(START, END)].reset_index(drop=True)
    assert list(co2["year"]) == list(range(START, END + 1)), "CO2 years incomplete"
    co2["co2_forcing"] = 5.35 * np.log(co2["co2_ppm"] / 280.0)
    step = co2.set_index("year").loc[[SPLICE_YEAR, SPLICE_YEAR + 1], "co2_ppm"]
    print(f"CO2 splice: {SPLICE_YEAR} {step.iloc[0]:.2f} ppm (ice core) -> "
          f"{SPLICE_YEAR + 1} {step.iloc[1]:.2f} ppm (Mauna Loa), step {step.iloc[1] - step.iloc[0]:+.2f}")
    return co2


def cumulative_emissions():
    w = pd.read_csv(OWID, usecols=["country", "year", "cumulative_co2"])
    w = w[(w["country"] == "World") & w["year"].between(START, END)]
    w = w.dropna(subset=["cumulative_co2"])
    assert list(w["year"]) == list(range(START, END + 1)), "World cumulative CO2 years incomplete"
    return pd.DataFrame({"year": w["year"].values,
                         "cum_emissions_gtco2": w["cumulative_co2"].values / 1000.0})


def nino_features():
    n = pd.read_csv(NINO, sep=r"\s+")
    n.columns = [str(c).strip().upper() for c in n.columns]
    n = n[n["ANOM"] > -90]
    n["date"] = pd.to_datetime(dict(year=n["YR"], month=n["MON"], day=1))
    s = n.set_index("date")["ANOM"].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    assert len(full) == len(s), "Nino series has gaps"
    f = pd.DataFrame(index=s.index)
    f["nino_lag1"] = s.shift(1)
    f["nino_lag3"] = s.shift(3)
    f["nino_lag6"] = s.shift(6)
    f["nino_prev12_mean"] = s.shift(1).rolling(12, min_periods=12).mean()
    return f.rename_axis("date").reset_index()


def main():
    india = pd.read_csv(INDIA, parse_dates=["date"])
    india = india[india["year"].between(START, END)]

    # Squares with too many missing months
    pct = india.groupby("cell_id")["tmean_anom"].apply(lambda s: 100 * s.isna().mean())
    sparse = sorted(pct[pct > MAX_MISSING_PCT].index)
    print(f"Dropping {len(sparse)} squares with > {MAX_MISSING_PCT:.0f}% missing: {sparse}")
    india = india[~india["cell_id"].isin(sparse)]

    before = len(india)
    india = india.dropna(subset=["tmean_anom"])
    print(f"Dropped {before - len(india):,} rows with missing tmean_anom")

    df = india[["date", "year", "month", "cell_id", "lat", "lon",
                "tmean_anom", "tmax_anom", "tmin_anom"]].copy()
    df = df.merge(spliced_co2(), on="year", how="left")
    df = df.merge(cumulative_emissions(), on="year", how="left")
    df = df.merge(nino_features(), on="date", how="left")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["split_time"] = np.where(df["year"] <= TRAIN_END, "train", "test")

    inputs = ["co2_forcing", "cum_emissions_gtco2", "nino_lag1", "nino_lag3",
              "nino_lag6", "nino_prev12_mean", "lat", "lon", "month_sin", "month_cos"]
    missing = df[inputs].isna().sum()
    assert missing.sum() == 0, f"missing input values:\n{missing[missing > 0]}"
    assert not df.duplicated(["cell_id", "date"]).any()

    df = df.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.5f", date_format="%Y-%m-%d")

    print(f"\nSaved {OUT}: {len(df):,} rows, {df['cell_id'].nunique()} squares, "
          f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    print(df["split_time"].value_counts().to_string())
    print("\nInput ranges:")
    print(df[inputs + ["co2_ppm"]].describe().loc[["min", "max"]].round(3).T.to_string())
    print("\nVersion A input: co2_forcing | Version B: cum_emissions_gtco2 | Version C: both")
    print("Values interpolated: 0")


if __name__ == "__main__":
    main()
