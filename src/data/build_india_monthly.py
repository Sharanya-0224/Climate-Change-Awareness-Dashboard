"""
build_india_monthly.py  (put it at src/data/build_india_monthly.py)

Turns IMD daily gridded temperature (tmax, tmin, 1951-2025) into one table:
one row per grid cell per month, with each cell's anomaly against its own
normal for that calendar month.

Run from the repo root:
    python3 src/data/build_india_monthly.py

Output:
    data/processed_india/india_monthly_temperature.csv

Rules followed:
- IMD's missing code 99.9 becomes NaN (compared with a tolerance, because the
  stored value is not exactly 99.9).
- A month needs at least MIN_DAYS valid days, otherwise it stays NaN.
- Nothing is filled or interpolated.
- The normal (baseline) is 1961-1990, so it lies entirely before any test
  period from 2011 onward and does not leak test data into training.
"""

from pathlib import Path

import pandas as pd
import imdlib as imd

RAW_DIR = "data/raw/imd"
OUT = Path("data/processed_india/india_monthly_temperature.csv")
START, END = 1951, 2025
MISSING = 99.9
MIN_DAYS = 20                    # minimum valid days for a monthly mean
BASE_START, BASE_END = 1961, 1990
MIN_BASE_YEARS = 20              # minimum years behind each cell-month normal
VARS = ["tmax", "tmin"]


def monthly_table(var):
    """Daily grid -> monthly mean per cell, plus the number of valid days."""
    da = imd.open_data(var, START, END, "yearwise", RAW_DIR).get_xarray()[var]
    da = da.where(abs(da - MISSING) > 0.01)
    days = da.notnull().resample(time="MS").sum()
    mean = da.resample(time="MS").mean().where(days >= MIN_DAYS)
    df = mean.to_dataframe(name=f"{var}_mean")
    df[f"{var}_days"] = days.to_dataframe(name="d")["d"].astype(int)
    return df


def add_anomalies(df):
    """Anomaly = this month's value minus the cell's 1961-1990 normal for that month."""
    base = df[df["year"].between(BASE_START, BASE_END)]
    cols = [f"{v}_mean" for v in VARS + ["tmean"]]
    grouped = base.groupby(["lat", "lon", "month"])[cols]
    normals = grouped.mean()
    counts = grouped.count()
    normals = normals.where(counts >= MIN_BASE_YEARS)
    normals.columns = [c.replace("_mean", "_normal") for c in normals.columns]
    df = df.join(normals, on=["lat", "lon", "month"])
    for v in VARS + ["tmean"]:
        df[f"{v}_anom"] = df[f"{v}_mean"] - df[f"{v}_normal"]
    return df


def check(df):
    assert not df.duplicated(["cell_id", "date"]).any(), "duplicate cell-month rows"
    for v in VARS:
        vals = df[f"{v}_mean"].dropna()
        assert vals.between(-30, 55).all(), f"{v} outside -30..55 C"
    both = df[["tmax_mean", "tmin_mean"]].dropna()
    assert (both["tmax_mean"] >= both["tmin_mean"]).all(), "monthly tmax below tmin"


def main():
    df = monthly_table("tmax").join(monthly_table("tmin")).reset_index()
    df = df.rename(columns={"time": "date"})

    # Keep only land cells: cells that have any valid tmax at all.
    has_data = df.groupby(["lat", "lon"])["tmax_mean"].transform(lambda s: s.notna().any())
    df = df[has_data].copy()

    df["tmean_mean"] = (df["tmax_mean"] + df["tmin_mean"]) / 2
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["cell_id"] = df["lat"].map("{:.1f}".format) + "_" + df["lon"].map("{:.1f}".format)
    df = add_anomalies(df)

    # Drop cells with no 1961-1990 normal at all (4 edge cells with patchy records:
    # far-east Arunachal, the far north, Nicobar). They can never get an anomaly.
    no_normal = df.groupby("cell_id")["tmean_normal"].transform(lambda s: s.isna().all())
    dropped = sorted(df.loc[no_normal, "cell_id"].unique())
    df = df[~no_normal].copy()
    print(f"Dropped {len(dropped)} cells with no baseline normal: {dropped}")

    order = ["date", "year", "month", "cell_id", "lat", "lon",
             "tmax_mean", "tmin_mean", "tmean_mean", "tmax_days", "tmin_days",
             "tmax_normal", "tmin_normal", "tmean_normal",
             "tmax_anom", "tmin_anom", "tmean_anom"]
    df = df[order].sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    check(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.3f", date_format="%Y-%m-%d")

    # ---- log ----
    print(f"Saved {OUT}")
    print(f"Rows: {len(df):,} | cells: {df['cell_id'].nunique()} | "
          f"months: {df['date'].nunique()} | {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    last = df[df["year"] == END]
    print(f"\nMonths in {END} with any data: "
          f"{last.dropna(subset=['tmax_mean'])['month'].nunique()} of 12")
    print("\nAll-India mean tmean anomaly by decade (°C, vs 1961-1990):")
    dec = df.assign(decade=(df["year"] // 10) * 10).groupby("decade")["tmean_anom"].mean()
    print(dec.round(2).to_string())
    print("\nValues interpolated: 0")


if __name__ == "__main__":
    main()