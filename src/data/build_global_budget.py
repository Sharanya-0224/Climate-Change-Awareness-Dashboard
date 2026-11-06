"""
build_global_budget.py  (put it at src/data/build_global_budget.py)

Model 2 table: one row per year (1959 onward) with global emissions, where the
carbon went (air, ocean, land), and the yearly El Nino average.
Also prints the "emissions bookkeeping" check: did humans emit more carbon
than the air gained?

Run from the repo root:
    python3 src/data/build_global_budget.py

Output:
    data/processed_drivers/global_carbon_budget.csv

Units: GtC per year (billion tonnes of carbon, not CO2).
GTC_PER_PPM below is the standard conversion from memory; the script prints
the sheet's own notes so it can be checked against the source.
"""

from pathlib import Path

import pandas as pd

SRC = Path("data/raw/emissions/Global_Carbon_Budget_2025_v1.0.xlsx")
SHEET = "Global Carbon Budget"
HEADER_ROW = 21                      # row with the column names (0-based)
MLO = Path("data/raw/co2_annmean_mlo.txt")           # Mauna Loa annual mean CO2
NINO = Path("data/raw/detrend.nino34.ascii.txt")     # monthly Nino 3.4 anomaly
OUT = Path("data/processed_drivers/global_carbon_budget.csv")
GTC_PER_PPM = 2.124                  # CHECK against the sheet notes printed below

NAMES = ["year", "fossil_emissions", "land_use_emissions", "atmospheric_growth",
         "ocean_sink", "land_sink", "cement_carbonation_sink", "budget_imbalance"]


def load_mauna_loa():
    """Mauna Loa annual mean CO2 (ppm). '#' comment lines, then: year mean unc."""
    df = pd.read_csv(MLO, comment="#", sep=r"\s+", header=None,
                     names=["year", "co2_ppm_mauna_loa", "unc"])
    return df[["year", "co2_ppm_mauna_loa"]]


def load_nino_annual():
    """Calendar-year mean of the monthly Nino 3.4 anomaly, only for complete years."""
    n = pd.read_csv(NINO, sep=r"\s+")
    n.columns = [str(c).strip().upper() for c in n.columns]
    n = n[n["ANOM"] > -90]                        # guard against missing-value codes
    g = n.groupby("YR")["ANOM"].agg(["mean", "count"])
    g = g[g["count"] == 12]
    return g["mean"].rename("nino34").rename_axis("year").reset_index()


def main():
    notes = pd.read_excel(SRC, sheet_name=SHEET, header=None, nrows=HEADER_ROW)
    print("Sheet notes (check units and the ppm conversion):")
    for v in notes[1].dropna():
        print("  -", v)

    raw = pd.read_excel(SRC, sheet_name=SHEET, header=HEADER_ROW)
    print(f"\nOriginal column names: {list(raw.columns[:len(NAMES)])}")
    df = raw.iloc[:, :len(NAMES)].copy()
    df.columns = NAMES
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])                       # drops footnote rows
    df["year"] = df["year"].astype(int)
    for c in NAMES[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Derived
    df["total_emissions"] = df["fossil_emissions"] + df["land_use_emissions"]
    df["airborne_fraction"] = df["atmospheric_growth"] / df["total_emissions"]
    df["atmospheric_growth_ppm"] = df["atmospheric_growth"] / GTC_PER_PPM

    # Add Mauna Loa CO2 and yearly El Nino straight from the raw files
    df = df.merge(load_mauna_loa(), on="year", how="left")
    df = df.merge(load_nino_annual(), on="year", how="left")
    df["co2_growth_ppm_mauna_loa"] = df["co2_ppm_mauna_loa"].diff()

    # Checks
    assert df["year"].is_unique and df["year"].is_monotonic_increasing
    assert (df["year"].diff().dropna() == 1).all(), "gap in years"
    assert (df["total_emissions"] > 0).all()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Log
    print(f"\nSaved {OUT}")
    print(f"Rows: {len(df)} | years {df['year'].min()}-{df['year'].max()}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    both = df.dropna(subset=["co2_growth_ppm_mauna_loa"])
    diff = (both["atmospheric_growth_ppm"] - both["co2_growth_ppm_mauna_loa"]).abs()
    print(f"\nCross-check: GCB growth vs Mauna Loa growth, mean absolute difference "
          f"{diff.mean():.2f} ppm/yr (small = conversion and data agree)")

    print("\n=== Emissions bookkeeping (human-cause evidence) ===")
    tot_emit = df["total_emissions"].sum()
    tot_air = df["atmospheric_growth"].sum()
    print(f"{df['year'].min()}-{df['year'].max()}: humans emitted {tot_emit:.0f} GtC; "
          f"the air gained {tot_air:.0f} GtC ({100 * tot_air / tot_emit:.0f}% of emissions).")
    print(f"Years where the air gained MORE than humans emitted: "
          f"{int((df['atmospheric_growth'] > df['total_emissions']).sum())} of {len(df)}")
    print(f"Airborne fraction: mean {df['airborne_fraction'].mean():.2f}, "
          f"range {df['airborne_fraction'].min():.2f}-{df['airborne_fraction'].max():.2f}")
    print("Values interpolated: 0")


if __name__ == "__main__":
    main()