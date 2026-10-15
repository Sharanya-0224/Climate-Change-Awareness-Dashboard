"""
build_dataset.py  (put it at src/data/build_dataset.py)

Merges the raw climate files into clean, model-ready tables:
  data/processed/climate_annual.csv    1880 -> last complete year
  data/processed/climate_monthly.csv   1959-01 -> latest month
  data/processed/co2_annual_full.csv   1850 -> last year (ice core + Mauna Loa),
                                       kept so rolling-mean features can look back before 1880

Run from the repo root:   python src/data/build_dataset.py

Raw files are never modified. Optional files (ENSO, sunspots) are skipped with a
warning if they are missing, so you can add them one at a time.
Change the file names below if yours differ.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

F_GISTEMP = RAW / "GLB.Ts+dSST.csv"
F_MLO_ANN = RAW / "co2_annmean_mlo.txt"
F_MLO_MON = RAW / "co2_mm_mlo.txt"
F_ICE = RAW / "smoothedco2.txt"
F_ENSO = RAW / "detrend.nino34.ascii.txt"   # optional (monthly dataset)
F_SUN_Y = RAW / "SN_y_tot_V2.0.csv"         # optional (annual dataset)
F_SUN_M = RAW / "SN_m_tot_V2.0.csv"         # optional (monthly dataset)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ICE_LAST_YEAR = 1958       # ice core used up to here, Mauna Loa from 1959
ANNUAL_START = 1880
MONTHLY_START = "1959-01-01"


# ----------------------------------------------------------------- loaders
def load_gistemp():
    """GISTEMP global means: one row per year, columns Jan..Dec, J-D (annual mean).
    First line is a title row. Missing values are written as ***."""
    df = pd.read_csv(F_GISTEMP, skiprows=1, na_values=["***"])
    if df.shape[1] < 10:                       # file turned out to be tab-separated
        df = pd.read_csv(F_GISTEMP, skiprows=1, na_values=["***"], sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    for c in MONTHS + ["J-D"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Annual: J-D is NaN for an incomplete year, so the current year drops out automatically
    annual = (df[["Year", "J-D"]]
              .rename(columns={"Year": "year", "J-D": "temp_anomaly"})
              .dropna())

    # Monthly: reshape Jan..Dec columns into one row per month
    monthly = df.melt(id_vars="Year", value_vars=MONTHS,
                      var_name="mon", value_name="temp_anomaly")
    monthly["month"] = monthly["mon"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    monthly["date"] = pd.to_datetime(dict(year=monthly["Year"],
                                          month=monthly["month"], day=1))
    monthly = (monthly.dropna(subset=["temp_anomaly"])[["date", "temp_anomaly"]]
               .sort_values("date").reset_index(drop=True))
    return annual.reset_index(drop=True), monthly


def load_ice_core():
    """Smoothed ice core CO2 (tab-separated, one row per year, 1000-2004).
    ALL_50_full is the only column with no gaps."""
    ice = pd.read_csv(F_ICE, sep="\t", usecols=["Year", "ALL_50_full"])
    return ice.rename(columns={"Year": "year", "ALL_50_full": "co2_ice"})


def load_mlo_annual():
    mlo = pd.read_csv(F_MLO_ANN, comment="#", sep=r"\s+", header=None,
                      names=["year", "co2_mlo", "unc"])
    return mlo[["year", "co2_mlo"]]


def load_mlo_monthly():
    """Columns: year month decimal_date monthly_average de-seasonalized ndays stdev unc.
    NOTE: negative ndays/stdev/unc (e.g. -1, -9.99) only flag interpolated months or
    Scripps-era data with no day counts. The CO2 values themselves are valid, so we
    do NOT drop those rows."""
    cols = ["year", "month", "decimal_date", "co2_monthly", "co2_trend",
            "ndays", "stdev", "unc"]
    mm = pd.read_csv(F_MLO_MON, comment="#", sep=r"\s+", header=None, names=cols)
    mm["date"] = pd.to_datetime(dict(year=mm["year"], month=mm["month"], day=1))
    return mm[["date", "co2_monthly", "co2_trend"]]


def load_enso():
    """Nino 3.4 SST anomaly, monthly from 1950. Header: YR MON TOTAL ClimAdjust ANOM."""
    e = pd.read_csv(F_ENSO, sep=r"\s+")
    e["date"] = pd.to_datetime(dict(year=e["YR"], month=e["MON"], day=1))
    return e[["date", "ANOM"]].rename(columns={"ANOM": "enso_anom"})


def load_sunspots_yearly():
    """SILSO yearly file: semicolon-separated, no header. Year may be 1700.5 (mid-year)."""
    s = pd.read_csv(F_SUN_Y, sep=";", header=None, skipinitialspace=True,
                    names=["year_mid", "sunspots", "sun_std", "sun_nobs", "sun_definitive"])
    s["year"] = np.floor(s["year_mid"]).astype(int)
    s.loc[s["sunspots"] < 0, "sunspots"] = np.nan           # -1 means missing
    return s[["year", "sunspots"]]


def load_sunspots_monthly():
    s = pd.read_csv(F_SUN_M, sep=";", header=None, skipinitialspace=True,
                    names=["year", "month", "decimal_year", "sunspots",
                           "sun_std", "sun_nobs", "sun_marker"])
    s["date"] = pd.to_datetime(dict(year=s["year"], month=s["month"], day=1))
    s.loc[s["sunspots"] < 0, "sunspots"] = np.nan
    return s[["date", "sunspots"]]


# ------------------------------------------------------------- CO2 splice
def build_co2_annual(ice, mlo):
    """Ice core up to 1958, Mauna Loa annual means from 1959."""
    old = ice[ice["year"] <= ICE_LAST_YEAR].rename(columns={"co2_ice": "co2"})
    new = mlo.rename(columns={"co2_mlo": "co2"})
    co2 = pd.concat([old, new], ignore_index=True).sort_values("year")
    co2["co2_source"] = np.where(co2["year"] <= ICE_LAST_YEAR, "ice_core", "mauna_loa")
    return co2.reset_index(drop=True)


# -------------------------------------------------------------- checks
def validate(df, key, co2_col, name):
    """These checks become PyTest tests later."""
    assert df[key].is_monotonic_increasing, f"{name}: {key} not sorted"
    assert not df[key].duplicated().any(), f"{name}: duplicate {key}"
    assert df["temp_anomaly"].notna().all(), f"{name}: missing target values"
    assert df[co2_col].notna().all(), f"{name}: missing CO2 values"
    assert df[co2_col].between(250, 500).all(), f"{name}: CO2 outside 250-500 ppm"
    assert df["temp_anomaly"].between(-1.5, 3.0).all(), f"{name}: anomaly outside -1.5..3"
    print(f"[ok] {name}: {len(df)} rows, {df[key].iloc[0]} -> {df[key].iloc[-1]}")
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print(f"     NaNs left (expected for optional columns): {nan_counts.to_dict()}")


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    temp_a, temp_m = load_gistemp()
    ice = load_ice_core()
    co2_a = build_co2_annual(ice, load_mlo_annual())
    co2_a[co2_a["year"] >= 1850].to_csv(OUT / "co2_annual_full.csv", index=False)

    # ---- annual table
    annual = temp_a.merge(co2_a[["year", "co2", "co2_source"]], on="year", how="inner")
    annual = annual[annual["year"] >= ANNUAL_START]
    if F_SUN_Y.exists():
        annual = annual.merge(load_sunspots_yearly(), on="year", how="left")
    else:
        print(f"[skip] {F_SUN_Y.name} not found: no sunspots in the annual table")
    annual = annual.sort_values("year").reset_index(drop=True)
    validate(annual, "year", "co2", "annual")
    annual.to_csv(OUT / "climate_annual.csv", index=False)

    # ---- monthly table
    mm = load_mlo_monthly()
    monthly = temp_m.merge(mm, on="date", how="inner")
    monthly = monthly[monthly["date"] >= MONTHLY_START]
    if F_ENSO.exists():
        monthly = monthly.merge(load_enso(), on="date", how="left")
    else:
        print(f"[skip] {F_ENSO.name} not found: no El Nino column")
    if F_SUN_M.exists():
        monthly = monthly.merge(load_sunspots_monthly(), on="date", how="left")
    else:
        print(f"[skip] {F_SUN_M.name} not found: no sunspot column")
    monthly = monthly.sort_values("date").reset_index(drop=True)
    validate(monthly, "date", "co2_trend", "monthly")
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")
    monthly.to_csv(OUT / "climate_monthly.csv", index=False)

    print(f"Saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
