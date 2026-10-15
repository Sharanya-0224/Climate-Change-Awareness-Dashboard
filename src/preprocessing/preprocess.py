"""
Data Preprocessing Pipeline
Project: Climate Change Awareness Dashboard
Target: NASA GISTEMP Global Surface Temperature Anomaly
Drivers: CO2, Sunspots, Nino 3.4 SST Anomaly, CH4, N2O

Strict Time-Series Rules:
- No forward-fill or back-fill
- No interpolation or extrapolation (values interpolated: 0)
- No scaling or normalisation (deferred to modelling pipeline)
- Standardised snake_case column names, ISO dates, integer years
"""

import os
import sys
import numpy as np
import pandas as pd


# Directory configuration (project-relative paths)
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def log_intentionally_unused():
    """Logs raw files intentionally excluded from processed outputs."""
    print("=" * 70)
    print("INTENTIONALLY UNUSED RAW FILES (from EDA decisions):")
    print("  - ghg-concentrations_fig-2.csv (EPA CH4): Hidden 0 ppb bad values,")
    print("    ~150 ppb source offset between Law Dome and modern stations.")
    print("  - ghg-concentrations_fig-3.csv (EPA N2O): 41-year temporal gap")
    print("    (1938-1978) and completely unpopulated column.")
    print("  - global-atmospheric-co2-concentration.csv (OWID CO2): Redundant with")
    print("    Mauna Loa CO2, starts 21 years later (1979) with calibration offset.")
    print("=" * 70)


def load_gistemp(raw_dir):
    """
    Loads NASA GISTEMP temperature anomalies.
    Input layout: 1 title line, header (Year, Jan..Dec, J-D, D-N, DJF, MAM, JJA, SON)
    Sentinel: '***' denotes missing values.
    """
    path = os.path.join(raw_dir, "GLB.Ts+dSST.csv")
    df_raw = pd.read_csv(path, skiprows=1, na_values=["***"])
    
    # Log sentinels
    raw_text = pd.read_csv(path, skiprows=1)
    star_counts = (raw_text == "***").sum()
    stars_found = star_counts[star_counts > 0].to_dict()
    print("GLB.Ts+dSST.csv: read %d rows. Converted '***' to NaN: %s" % (len(df_raw), stars_found))
    
    # Annual: keep Year and J-D
    df_ann = df_raw[["Year", "J-D"]].dropna().rename(columns={"Year": "year", "J-D": "temp_anomaly"})
    df_ann["year"] = df_ann["year"].astype(int)
    
    # Monthly: melt Jan..Dec into long format
    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    records = []
    for _, row in df_raw.iterrows():
        yr = int(row["Year"])
        for m_idx, m_name in enumerate(month_cols, start=1):
            val = row[m_name]
            if pd.notna(val):
                records.append({
                    "year": yr,
                    "month": m_idx,
                    "temp_anomaly": float(val)
                })
    df_monthly = pd.DataFrame(records)
    df_monthly = df_monthly.sort_values(["year", "month"]).reset_index(drop=True)
    
    return df_ann, df_monthly


def load_co2(raw_dir):
    """
    Loads paleoclimate ice-core CO2 and modern Mauna Loa CO2 (annual and monthly).
    Ice core: smoothedco2.txt (Year, ALL_50_full)
    MLO Annual: co2_annmean_mlo.txt (year, mean, unc)
    MLO Monthly: co2_mm_mlo.txt (year, month, decimal_date, monthly_average, deseasonalized, ndays, sdev_days, unc_mon_mean)
    """
    # 1. Ice core
    ice_path = os.path.join(raw_dir, "smoothedco2.txt")
    df_ice = pd.read_csv(ice_path, sep="\t")
    print("smoothedco2.txt: read %d rows. Missing values in ALL_50_full: %d" % (len(df_ice), df_ice["ALL_50_full"].isnull().sum()))
    df_ice_clean = df_ice[["Year", "ALL_50_full"]].rename(columns={"Year": "year", "ALL_50_full": "co2"})
    df_ice_clean["year"] = df_ice_clean["year"].astype(int)
    df_ice_clean["co2_source"] = "ice_core"
    
    # 2. Mauna Loa Annual
    mlo_ann_path = os.path.join(raw_dir, "co2_annmean_mlo.txt")
    df_mlo_ann = pd.read_csv(mlo_ann_path, comment="#", sep=r"\s+", names=["year", "co2", "unc"])
    print("co2_annmean_mlo.txt: read %d rows. Missing values: %d" % (len(df_mlo_ann), df_mlo_ann["co2"].isnull().sum()))
    df_mlo_ann["year"] = df_mlo_ann["year"].astype(int)
    df_mlo_ann["co2_source"] = "mauna_loa"
    df_mlo_ann_clean = df_mlo_ann[["year", "co2", "co2_source"]]
    
    # Spliced annual CO2 series: Ice core <= 1958, Mauna Loa >= 1959
    df_co2_spliced = pd.concat([
        df_ice_clean[df_ice_clean["year"] <= 1958],
        df_mlo_ann_clean[df_mlo_ann_clean["year"] >= 1959]
    ]).sort_values("year").reset_index(drop=True)
    
    # 3. Mauna Loa Monthly
    mlo_m_path = os.path.join(raw_dir, "co2_mm_mlo.txt")
    mlo_m_cols = ["year", "month", "decimal_date", "co2_monthly", "co2_trend", "ndays", "sdev_days", "unc_mon_mean"]
    df_mlo_m = pd.read_csv(mlo_m_path, comment="#", sep=r"\s+", names=mlo_m_cols)
    
    # Log sentinels in metadata before dropping them
    n_days_sent = (df_mlo_m["ndays"] == -1).sum()
    sdev_sent = (df_mlo_m["sdev_days"] == -9.99).sum()
    unc_sent = (df_mlo_m["unc_mon_mean"] == -0.99).sum()
    print("co2_mm_mlo.txt: read %d rows. Metadata sentinels: ndays==-1: %d, sdev==-9.99: %d, unc==-0.99: %d" % (
        len(df_mlo_m), n_days_sent, sdev_sent, unc_sent
    ))
    df_mlo_m["year"] = df_mlo_m["year"].astype(int)
    df_mlo_m["month"] = df_mlo_m["month"].astype(int)
    df_mlo_m_clean = df_mlo_m[["year", "month", "co2_monthly", "co2_trend"]].copy()
    
    return df_co2_spliced, df_mlo_m_clean


def load_sunspots(raw_dir):
    """
    Loads WDC-SILSO sunspot numbers (annual and monthly).
    Annual: SN_y_tot_V2.0.csv (semicolon, no header: year_frac, sn_mean, sn_std, n_obs, is_definitive)
    Monthly: SN_m_tot_V2.0.csv (semicolon, no header: year, month, frac_year, sn_mean, sn_std, n_obs, is_definitive)
    Sentinel: -1 in sn_std and n_obs denotes missing observer stats prior to 1818.
    """
    # Annual
    sn_y_path = os.path.join(raw_dir, "SN_y_tot_V2.0.csv")
    sn_y = pd.read_csv(sn_y_path, sep=";", header=None, names=["year_frac", "sunspots", "sn_std", "n_obs", "is_definitive"])
    std_sent_y = (sn_y["sn_std"] == -1.0).sum()
    obs_sent_y = (sn_y["n_obs"] == -1).sum()
    print("SN_y_tot_V2.0.csv: read %d rows. Sentinels (-1) in sn_std: %d, n_obs: %d" % (len(sn_y), std_sent_y, obs_sent_y))
    sn_y["year"] = np.floor(sn_y["year_frac"]).astype(int)
    sn_y_clean = sn_y[["year", "sunspots"]].copy()
    
    # Monthly
    sn_m_path = os.path.join(raw_dir, "SN_m_tot_V2.0.csv")
    sn_m = pd.read_csv(sn_m_path, sep=";", header=None, names=["year", "month", "frac_year", "sunspots", "sn_std", "n_obs", "sunspots_definitive"])
    std_sent_m = (sn_m["sn_std"] == -1.0).sum()
    obs_sent_m = (sn_m["n_obs"] == -1).sum()
    print("SN_m_tot_V2.0.csv: read %d rows. Sentinels (-1) in sn_std: %d, n_obs: %d" % (len(sn_m), std_sent_m, obs_sent_m))
    sn_m["year"] = sn_m["year"].astype(int)
    sn_m["month"] = sn_m["month"].astype(int)
    sn_m_clean = sn_m[["year", "month", "sunspots", "sunspots_definitive"]].copy()
    
    return sn_y_clean, sn_m_clean


def load_nino34(raw_dir):
    """
    Loads NOAA CPC detrended Nino 3.4 SST anomalies.
    detrend.nino34.ascii.txt (whitespace: YR, MON, TOTAL, ClimAdjust, ANOM)
    """
    path = os.path.join(raw_dir, "detrend.nino34.ascii.txt")
    df = pd.read_csv(path, sep=r"\s+")
    print("detrend.nino34.ascii.txt: read %d rows. Missing values: %d" % (len(df), df["ANOM"].isnull().sum()))
    df = df.rename(columns={"YR": "year", "MON": "month", "ANOM": "nino34"})
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    
    # Monthly table
    df_m = df[["year", "month", "nino34"]].copy()
    
    # Annual mean: strictly calendar-year mean for years with all 12 complete months
    year_counts = df.groupby("year")["nino34"].agg(["count", "mean"]).reset_index()
    complete_years = year_counts[year_counts["count"] == 12]
    df_ann = complete_years[["year", "mean"]].rename(columns={"mean": "nino34"})
    
    return df_ann, df_m


def load_ghg_noaa(raw_dir):
    """
    Loads NOAA Global Marine Boundary Layer CH4 and N2O (annual and monthly).
    Sentinels: -9.99 in uncertainty columns for provisional recent months.
    """
    # 1. CH4 Annual (1984-2025)
    ch4_ann_path = os.path.join(raw_dir, "ch4_annmean_gl.txt")
    df_ch4_ann = pd.read_csv(ch4_ann_path, comment="#", sep=r"\s+", names=["year", "ch4", "unc"])
    print("ch4_annmean_gl.txt: read %d rows. Missing values: %d" % (len(df_ch4_ann), df_ch4_ann["ch4"].isnull().sum()))
    df_ch4_ann["year"] = df_ch4_ann["year"].astype(int)
    df_ch4_ann = df_ch4_ann[["year", "ch4"]]
    
    # 2. CH4 Monthly (1983-07 to 2026-06)
    ch4_m_path = os.path.join(raw_dir, "ch4_mm_gl.txt")
    ch4_cols = ["year", "month", "decimal", "ch4", "average_unc", "ch4_trend", "trend_unc"]
    df_ch4_m = pd.read_csv(ch4_m_path, comment="#", sep=r"\s+", names=ch4_cols)
    unc_sent_ch4 = (df_ch4_m["average_unc"] == -9.99).sum()
    print("ch4_mm_gl.txt: read %d rows. Uncertainty sentinels (-9.99): %d" % (len(df_ch4_m), unc_sent_ch4))
    df_ch4_m["year"] = df_ch4_m["year"].astype(int)
    df_ch4_m["month"] = df_ch4_m["month"].astype(int)
    df_ch4_m = df_ch4_m[["year", "month", "ch4", "ch4_trend"]]
    
    # 3. N2O Annual (2001-2025)
    n2o_ann_path = os.path.join(raw_dir, "n2o_annmean_gl.txt")
    df_n2o_ann = pd.read_csv(n2o_ann_path, comment="#", sep=r"\s+", names=["year", "n2o", "unc"])
    print("n2o_annmean_gl.txt: read %d rows. Missing values: %d" % (len(df_n2o_ann), df_n2o_ann["n2o"].isnull().sum()))
    df_n2o_ann["year"] = df_n2o_ann["year"].astype(int)
    df_n2o_ann = df_n2o_ann[["year", "n2o"]]
    
    # 4. N2O Monthly (2001-01 to 2026-06)
    n2o_m_path = os.path.join(raw_dir, "n2o_mm_gl.txt")
    n2o_cols = ["year", "month", "decimal", "n2o", "average_unc", "n2o_trend", "trend_unc"]
    df_n2o_m = pd.read_csv(n2o_m_path, comment="#", sep=r"\s+", names=n2o_cols)
    unc_sent_n2o = (df_n2o_m["average_unc"] == -9.99).sum()
    print("n2o_mm_gl.txt: read %d rows. Uncertainty sentinels (-9.99): %d" % (len(df_n2o_m), unc_sent_n2o))
    df_n2o_m["year"] = df_n2o_m["year"].astype(int)
    df_n2o_m["month"] = df_n2o_m["month"].astype(int)
    df_n2o_m = df_n2o_m[["year", "month", "n2o", "n2o_trend"]]
    
    return df_ch4_ann, df_ch4_m, df_n2o_ann, df_n2o_m


def build_climate_annual(df_glb_ann, df_co2_ann, df_sn_ann, df_nino_ann, df_ch4_ann, df_n2o_ann):
    """
    Builds climate_annual.csv:
    Complete years 1880-2025 (146 rows). Year 2026 dropped due to missing J-D.
    Columns: year, temp_anomaly, co2, co2_source, sunspots, nino34, ch4, n2o
    """
    df = pd.DataFrame({"year": range(1880, 2026)})
    df = df.merge(df_glb_ann, on="year", how="left")
    df = df.merge(df_co2_ann, on="year", how="left")
    df = df.merge(df_sn_ann, on="year", how="left")
    df = df.merge(df_nino_ann, on="year", how="left")
    df = df.merge(df_ch4_ann, on="year", how="left")
    df = df.merge(df_n2o_ann, on="year", how="left")
    
    cols = ["year", "temp_anomaly", "co2", "co2_source", "sunspots", "nino34", "ch4", "n2o"]
    df = df[cols].sort_values("year").reset_index(drop=True)
    return df


def build_drivers_annual_full(df_co2_ann, df_sn_ann, df_nino_ann, df_ch4_ann, df_n2o_ann):
    """
    Builds drivers_annual_full.csv:
    Historical years 1850-2025 (176 rows), without temp_anomaly.
    Columns: year, co2, co2_source, sunspots, nino34, ch4, n2o
    """
    df = pd.DataFrame({"year": range(1850, 2026)})
    df = df.merge(df_co2_ann, on="year", how="left")
    df = df.merge(df_sn_ann, on="year", how="left")
    df = df.merge(df_nino_ann, on="year", how="left")
    df = df.merge(df_ch4_ann, on="year", how="left")
    df = df.merge(df_n2o_ann, on="year", how="left")
    
    cols = ["year", "co2", "co2_source", "sunspots", "nino34", "ch4", "n2o"]
    df = df[cols].sort_values("year").reset_index(drop=True)
    return df


def build_climate_monthly(df_glb_m, df_co2_m, df_sn_m, df_nino_m, df_ch4_m, df_n2o_m):
    """
    Builds climate_monthly.csv:
    Months 1958-03 to 2026-08 where BOTH temperature and Mauna Loa CO2 exist (822 rows).
    Columns: date, temp_anomaly, co2_monthly, co2_trend, sunspots, sunspots_definitive,
             nino34, ch4, ch4_trend, n2o, n2o_trend
    """
    # Base grid where both temperature and CO2 exist
    df = pd.merge(df_glb_m, df_co2_m, on=["year", "month"], how="inner")
    
    # Merge other drivers
    df = df.merge(df_sn_m, on=["year", "month"], how="left")
    df = df.merge(df_nino_m, on=["year", "month"], how="left")
    df = df.merge(df_ch4_m, on=["year", "month"], how="left")
    df = df.merge(df_n2o_m, on=["year", "month"], how="left")
    
    # ISO date string YYYY-MM-01
    df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    
    cols = [
        "date", "temp_anomaly", "co2_monthly", "co2_trend",
        "sunspots", "sunspots_definitive", "nino34",
        "ch4", "ch4_trend", "n2o", "n2o_trend"
    ]
    df = df[cols].sort_values("date").reset_index(drop=True)
    return df


def validate_table(name, df, is_monthly=False):
    """Runs strict domain and structural assertions on output tables."""
    print("\n--- Validating %s ---" % name)
    print("Shape: %s" % str(df.shape))
    
    # 1. Sorted keys and no duplicates
    if is_monthly:
        assert df["date"].is_monotonic_increasing, "%s: dates are not monotonically increasing!" % name
        assert df["date"].duplicated().sum() == 0, "%s: duplicate dates found!" % name
        print("Date range: %s to %s" % (df["date"].min(), df["date"].max()))
    else:
        assert df["year"].is_monotonic_increasing, "%s: years are not monotonically increasing!" % name
        assert df["year"].duplicated().sum() == 0, "%s: duplicate years found!" % name
        print("Year range: %d to %d" % (df["year"].min(), df["year"].max()))
        
    # 2. Target check
    if "temp_anomaly" in df.columns:
        assert df["temp_anomaly"].isnull().sum() == 0, "%s: missing target temp_anomaly!" % name
        assert df["temp_anomaly"].between(-1.5, 3.0).all(), "%s: temp_anomaly out of domain range (-1.5 to 3.0)!" % name
        
    # 3. CO2 checks
    co2_col = "co2_monthly" if is_monthly else "co2"
    assert df[co2_col].isnull().sum() == 0, "%s: missing CO2 values!" % name
    assert df[co2_col].between(250.0, 500.0).all(), "%s: CO2 out of domain range (250 to 500)!" % name
    
    # 4. Driver domain checks
    if "sunspots" in df.columns:
        assert (df["sunspots"] >= 0).all(), "%s: negative sunspots found!" % name
    if "nino34" in df.columns:
        valid_nino = df["nino34"].dropna()
        assert valid_nino.between(-3.0, 4.0).all(), "%s: nino34 out of domain range (-3.0 to 4.0)!" % name
    if "ch4" in df.columns:
        valid_ch4 = df["ch4"].dropna()
        assert valid_ch4.between(600.0, 2100.0).all(), "%s: ch4 out of domain range (600 to 2100)!" % name
    if "n2o" in df.columns:
        valid_n2o = df["n2o"].dropna()
        assert valid_n2o.between(250.0, 360.0).all(), "%s: n2o out of domain range (250 to 360)!" % name
        
    print("NaN counts per column:\n%s" % df.isnull().sum().to_string())
    print("All assertions passed for %s." % name)


def check_co2_seam(df_ann):
    """
    Prints inspection of CO2 splice seam (1955-1962).
    Evaluates year-over-year step across 1958 (ice core) -> 1959 (Mauna Loa).
    """
    print("\n" + "=" * 70)
    print("CO2 SPLICE SEAM CHECK (1955-1962):")
    sub = df_ann[df_ann["year"].between(1955, 1962)][["year", "co2", "co2_source"]].copy()
    for _, r in sub.iterrows():
        print("  Year %d: CO2 = %.2f ppm (Source: %s)" % (int(r["year"]), r["co2"], r["co2_source"]))
        
    co2_1958 = df_ann.loc[df_ann["year"] == 1958, "co2"].values[0]
    co2_1959 = df_ann.loc[df_ann["year"] == 1959, "co2"].values[0]
    yoy_step = co2_1959 - co2_1958
    print("1958 (ice core: %.2f) -> 1959 (mauna loa: %.2f) YoY Step: %+.2f ppm" % (co2_1958, co2_1959, yoy_step))
    if abs(yoy_step) > 3.0:
        print("WARNING: Splice seam YoY step exceeds 3.0 ppm!")
    else:
        print("Splice seam step is clean and within physical annual growth bounds (<= 3.0 ppm).")
    print("=" * 70)


def main():
    print("Starting climate data preprocessing pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Log intentionally unused raw files
    log_intentionally_unused()
    
    # 1. Load raw files
    print("\n--- Loading Raw Data Files ---")
    df_glb_ann, df_glb_m = load_gistemp(RAW_DIR)
    df_co2_ann, df_co2_m = load_co2(RAW_DIR)
    df_sn_ann, df_sn_m = load_sunspots(RAW_DIR)
    df_nino_ann, df_nino_m = load_nino34(RAW_DIR)
    df_ch4_ann, df_ch4_m, df_n2o_ann, df_n2o_m = load_ghg_noaa(RAW_DIR)
    
    # 2. Build target tables
    print("\n--- Building Processed Tables ---")
    climate_annual = build_climate_annual(df_glb_ann, df_co2_ann, df_sn_ann, df_nino_ann, df_ch4_ann, df_n2o_ann)
    drivers_annual_full = build_drivers_annual_full(df_co2_ann, df_sn_ann, df_nino_ann, df_ch4_ann, df_n2o_ann)
    climate_monthly = build_climate_monthly(df_glb_m, df_co2_m, df_sn_m, df_nino_m, df_ch4_m, df_n2o_m)
    
    # 3. Validate tables
    validate_table("climate_annual.csv", climate_annual, is_monthly=False)
    validate_table("drivers_annual_full.csv", drivers_annual_full, is_monthly=False)
    validate_table("climate_monthly.csv", climate_monthly, is_monthly=True)
    
    # 4. Check CO2 seam
    check_co2_seam(climate_annual)
    
    # 5. Sanity check key benchmark years
    print("\n--- Benchmark Sanity Checks ---")
    r2024 = climate_annual[climate_annual["year"] == 2024].iloc[0]
    r2025 = climate_annual[climate_annual["year"] == 2025].iloc[0]
    print("2024 Benchmark: temp_anomaly = %.2f (expected ~1.29), co2 = %.2f (expected ~424.61)" % (
        r2024["temp_anomaly"], r2024["co2"]
    ))
    print("2025 Benchmark: temp_anomaly = %.2f (expected ~1.19), co2 = %.2f (expected ~427.35)" % (
        r2025["temp_anomaly"], r2025["co2"]
    ))
    
    # Complete rows check
    complete_ann = climate_annual.dropna().shape[0]
    print("Complete rows in climate_annual (all columns present): %d (expected: 25 rows, 2001-2025)" % complete_ann)
    
    # Values interpolated statement
    print("values interpolated: 0")
    
    # 6. Save outputs
    print("\n--- Saving Output Files to %s ---" % PROCESSED_DIR)
    path_ann = os.path.join(PROCESSED_DIR, "climate_annual.csv")
    path_drivers = os.path.join(PROCESSED_DIR, "drivers_annual_full.csv")
    path_m = os.path.join(PROCESSED_DIR, "climate_monthly.csv")
    
    climate_annual.to_csv(path_ann, index=False)
    drivers_annual_full.to_csv(path_drivers, index=False)
    climate_monthly.to_csv(path_m, index=False)
    
    print("Saved: %s" % path_ann)
    print("Saved: %s" % path_drivers)
    print("Saved: %s" % path_m)
    print("Preprocessing pipeline finished successfully!")


if __name__ == "__main__":
    main()
