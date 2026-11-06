"""
build_country_panel.py  (put it at src/data/build_country_panel.py)

Model 1 table: one row per country per year, 1965-2022, from the OWID CO2 file.

Run from the repo root:
    python3 src/data/build_country_panel.py

Output:
    data/processed_drivers/country_panel.csv

Rules:
- Countries only (3-letter ISO code). Regions and groups such as "World" or
  "Asia" are removed, because they would double-count countries.
- 1965-2022 only: energy data starts in 1965 and GDP ends in 2022.
- A row is kept only if population, gdp, co2 and primary_energy_consumption
  are all present. Nothing is filled or interpolated.

Column roles (important):
- TARGET: co2 (and co2_per_capita).
- MODEL INPUTS: population, gdp_per_capita, energy_per_capita, energy_per_gdp.
- DISPLAY ONLY, never model inputs: carbon_intensity and the fuel shares.
  They are calculated from co2 itself. Because
  co2 = population x gdp_per_capita x energy_per_gdp x carbon_intensity
  exactly (the Kaya identity), using them as inputs would give a fake
  perfect score.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/raw/emissions/owid-co2-data.csv")
OUT = Path("data/processed_drivers/country_panel.csv")
START, END = 1965, 2022
CORE = ["population", "gdp", "co2", "primary_energy_consumption"]
FUELS = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"]


def safe_div(a, b):
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def main():
    raw = pd.read_csv(SRC)
    print(f"Read {SRC}: {len(raw):,} rows")

    iso = raw["iso_code"].astype(str)
    is_country = raw["iso_code"].notna() & (iso.str.len() == 3) & ~iso.str.startswith("OWID")
    df = raw[is_country & raw["year"].between(START, END)].copy()
    print(f"Countries only, {START}-{END}: {len(df):,} rows, {df['country'].nunique()} countries")

    keep = ["country", "iso_code", "year"] + CORE + [c for c in FUELS if c in df.columns]
    df = df[keep]
    before = len(df)
    df = df.dropna(subset=CORE)
    print(f"Dropped {before - len(df):,} rows missing any of {CORE}")

    # Zero or negative population, GDP or energy are placeholders, not real values.
    bad = (df[["population", "gdp", "primary_energy_consumption"]] <= 0).any(axis=1)
    if bad.any():
        print(f"Dropped {int(bad.sum())} rows with zero or negative population/GDP/energy:")
        print(df.loc[bad].groupby("country")["year"].agg(["count", "min", "max"]).to_string())
    df = df[~bad].copy()

    # Model inputs
    df["gdp_per_capita"] = safe_div(df["gdp"], df["population"])
    df["energy_per_capita"] = safe_div(df["primary_energy_consumption"], df["population"])
    df["energy_per_gdp"] = safe_div(df["primary_energy_consumption"], df["gdp"])
    # Target per person
    df["co2_per_capita"] = safe_div(df["co2"] * 1e6, df["population"])   # tonnes per person
    # Display only (derived from the target)
    df["carbon_intensity"] = safe_div(df["co2"], df["primary_energy_consumption"])
    for f in [c for c in FUELS if c in df.columns]:
        df[f"{f.replace('_co2', '')}_share_of_co2"] = safe_div(df[f], df["co2"])

    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Checks
    assert not df.duplicated(["country", "year"]).any(), "duplicate country-year rows"
    for c in ["population", "gdp", "primary_energy_consumption"]:
        assert (df[c] > 0).all(), f"{c} has zero or negative values"
    assert (df["co2"] >= 0).all(), "negative co2"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Log
    print(f"\nSaved {OUT}")
    print(f"Rows: {len(df):,} | countries: {df['country'].nunique()} | "
          f"years {df['year'].min()}-{df['year'].max()}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nZero-co2 rows (check these):", int((df["co2"] == 0).sum()))
    last = df[df["year"] == df["year"].max()].nlargest(5, "co2")
    print(f"\nTop 5 emitters in {df['year'].max()} (co2, million tonnes; sanity check):")
    print(last[["country", "co2", "co2_per_capita"]].to_string(index=False))
    india = df[df["country"] == "India"]
    print(f"\nIndia: {len(india)} rows, {india['year'].min()}-{india['year'].max()}")
    print("\nModel inputs: population, gdp_per_capita, energy_per_capita, energy_per_gdp")
    print("Display only: carbon_intensity, *_share_of_co2")
    print("Values interpolated: 0")


if __name__ == "__main__":
    main()