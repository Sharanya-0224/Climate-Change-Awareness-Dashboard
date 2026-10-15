"""Create auditable annual CH4/N2O reconstructed datasets.

Uses only pandas and numpy. It DOES NOT overwrite the original processed CSVs.

Important: reconstructed values are estimates, not observations. The method uses
linear interpolation between selected historical-source anchors and NOAA global
annual observations. The long source gaps are assumption-heavy and must not be
presented as validated measurements.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # if copied to repo/src/preprocessing/
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
START_YEAR = 1880
END_YEAR = 2025


def read_epa_numeric_csv(filename: str) -> pd.DataFrame:
    """Read EPA's multi-section CSV; parse comma-formatted numeric cells safely."""
    path = RAW / filename
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    for col in raw.columns:
        raw[col] = raw[col].astype(str).str.replace(",", "", regex=False).str.strip()
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def read_noaa_annual(filename: str, value_name: str) -> pd.DataFrame:
    path = RAW / filename
    df = pd.read_csv(
        path, comment="#", sep=r"\s+", header=None,
        names=["year", value_name, "uncertainty"], engine="python"
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["year", value_name]).copy()
    df["year"] = df["year"].astype(int)
    return df[["year", value_name]].sort_values("year")


def historical_anchors(filename: str, year_col: int, value_col: int,
                       min_year: int, max_year: int, label: str) -> pd.DataFrame:
    raw = read_epa_numeric_csv(filename)
    anchors = pd.DataFrame({
        "year_raw": raw.iloc[:, year_col],
        "value": raw.iloc[:, value_col],
    })
    anchors = anchors.dropna(subset=["year_raw", "value"]).copy()
    anchors = anchors[anchors["year_raw"].between(min_year, max_year)]
    # EPA data can contain decimal years. Keep their timing, then map to nearest
    # calendar year for this annual model; document that this is an approximation.
    anchors["year"] = np.rint(anchors["year_raw"]).astype(int)
    anchors = anchors.groupby("year", as_index=False)["value"].mean()
    anchors["anchor_source"] = label
    if anchors.empty:
        raise ValueError(f"No historical anchors found in {filename}, column {value_col}.")
    return anchors[["year", "value", "anchor_source"]]


def make_reconstruction(gas: str, historical: pd.DataFrame,
                        modern: pd.DataFrame, min_year: int = START_YEAR,
                        max_year: int = END_YEAR) -> pd.DataFrame:
    """Build annual estimates from historical anchors and modern NOAA means."""
    modern = modern.rename(columns={gas: "value"}).copy()
    modern["anchor_source"] = "NOAA_global_annual_observation"
    historical = historical.copy()

    # NOAA values take precedence for their observed years.
    anchors = pd.concat([historical, modern], ignore_index=True)
    anchors = anchors.dropna(subset=["year", "value"])
    anchors["year"] = anchors["year"].astype(int)
    anchors = anchors.sort_values("year")
    anchors = anchors.drop_duplicates("year", keep="last")
    anchors = anchors.groupby("year", as_index=False).agg(
        value=("value", "mean"), anchor_source=("anchor_source", "last")
    ).sort_values("year")

    if anchors["year"].nunique() < 2:
        raise ValueError(f"Not enough anchors to reconstruct {gas}.")

    years = np.arange(min_year, max_year + 1, dtype=int)
    x = anchors["year"].to_numpy(dtype=float)
    y = anchors["value"].to_numpy(dtype=float)

    # Piecewise-linear interpolation between anchors. No endpoint extrapolation
    # is used for the requested 1880-2025 window because anchors bracket it.
    estimates = np.interp(years.astype(float), x, y)
    result = pd.DataFrame({"year": years, f"{gas}_reconstructed": estimates})

    source_map = anchors.set_index("year")["anchor_source"].to_dict()
    result[f"{gas}_data_status"] = [
        source_map.get(int(year), "synthetic_linear_interpolation") for year in years
    ]
    result[f"{gas}_is_synthetic"] = ~result["year"].isin(anchors["year"])
    result[f"{gas}_method_note"] = "piecewise_linear_between_anchors"

    # Make long unobserved bridges visible in the output.
    if gas == "ch4":
        result.loc[result["year"].between(1959, 1983), f"{gas}_method_note"] = \
            "synthetic_bridge_1959_1983_no_NOAA_global_annual_data"
        result.loc[result["year"].between(START_YEAR, 1958), f"{gas}_method_note"] = \
            "historical_interpolation_from_sparse_EPA_Law_Dome_anchors"
    elif gas == "n2o":
        result.loc[result["year"].between(START_YEAR, 1937), f"{gas}_method_note"] = \
            "historical_interpolation_from_very_sparse_EPA_EPICA_anchors"
        result.loc[result["year"].between(1938, 2000), f"{gas}_method_note"] = \
            "synthetic_bridge_1938_2000_no_NOAA_global_annual_data"

    # Retain anchor values as source evidence in a separate table-like column.
    anchor_value_map = anchors.set_index("year")["value"].to_dict()
    result[f"{gas}_anchor_value"] = result["year"].map(anchor_value_map)
    return result


def interpolation_cv(historical: pd.DataFrame) -> dict:
    """LOO-style interior-anchor interpolation check; not a long-gap validation."""
    h = historical.sort_values("year").drop_duplicates("year")
    if len(h) < 4:
        return {"n_tested": 0, "mae": np.nan, "rmse": np.nan,
                "note": "Too few historical anchors for a meaningful interpolation CV."}
    x = h["year"].to_numpy(dtype=float)
    y = h["value"].to_numpy(dtype=float)
    preds, actual = [], []
    # Only interior anchors: each is estimated from its two neighbours.
    for i in range(1, len(h) - 1):
        pred = np.interp(x[i], [x[i - 1], x[i + 1]], [y[i - 1], y[i + 1]])
        preds.append(pred)
        actual.append(y[i])
    err = np.asarray(preds) - np.asarray(actual)
    return {"n_tested": len(err), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "note": "Historical-anchor interpolation only; does not validate the long bridge to NOAA."}


def add_to_processed(source_path: Path, gas_tables: dict, output_path: Path,
                     start_year: int = START_YEAR) -> None:
    df = pd.read_csv(source_path)
    if "year" not in df.columns:
        raise ValueError(f"Missing year column in {source_path}")

    for gas, recon in gas_tables.items():
        if gas not in df.columns:
            raise ValueError(f"Missing {gas} column in {source_path}")
        # Preserve the original measured values before replacing model input columns.
        observed_col = f"{gas}_observed_noaa"
        if observed_col not in df.columns:
            df[observed_col] = df[gas]
        df = df.merge(recon, on="year", how="left", validate="many_to_one")
        estimate_col = f"{gas}_reconstructed"
        fill_mask = df["year"].ge(start_year) & df[gas].isna()
        df.loc[fill_mask, gas] = df.loc[fill_mask, estimate_col]
        # Keep original NOAA observations unchanged and mark which values were filled.
        df[f"{gas}_was_filled"] = fill_mask
        df[f"{gas}_final_status"] = np.where(
            df[observed_col].notna(), "original_processed_NOAA_value",
            np.where(df[gas].notna(), df[f"{gas}_data_status"], "missing_not_reconstructed")
        )
        df = df.drop(columns=[estimate_col])

    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print("Rows:", len(df), "Year range:", int(df.year.min()), "-", int(df.year.max()))
    for gas in gas_tables:
        print(f"{gas}: missing={df[gas].isna().sum()}, filled={df[f'{gas}_was_filled'].sum()}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # EPA methane: columns are Year, EPICA Dome C, Law Dome, Cape Grim,
    # Mauna Loa, Shetland Islands. Use Law Dome only as historical anchors;
    # do not mix the problematic station columns into the global series.
    ch4_hist = historical_anchors(
        "ghg-concentrations_fig-2.csv", year_col=0, value_col=2,
        min_year=1800, max_year=1958, label="EPA_Law_Dome_historical_anchor"
    )
    ch4_noaa = read_noaa_annual("ch4_annmean_gl.txt", "ch4")
    ch4_recon = make_reconstruction("ch4", ch4_hist, ch4_noaa)

    # EPA N2O: columns are Year, EPICA Dome C, Antarctica (Battle et al.),
    # Cape Grim, Mauna Loa, South Pole, Barrow. Use EPICA historical points only;
    # do not treat station records as global annual means.
    n2o_hist = historical_anchors(
        "ghg-concentrations_fig-3.csv", year_col=0, value_col=1,
        min_year=1800, max_year=1937, label="EPA_EPICA_historical_anchor"
    )
    n2o_noaa = read_noaa_annual("n2o_annmean_gl.txt", "n2o")
    n2o_recon = make_reconstruction("n2o", n2o_hist, n2o_noaa)

    # Save the reconstruction ledger independently for auditing.
    ledger = ch4_recon.merge(n2o_recon, on="year", how="outer", validate="one_to_one")
    ledger_path = PROCESSED / "ghg_annual_reconstruction_ledger.csv"
    ledger.to_csv(ledger_path, index=False)
    print(f"Saved reconstruction ledger: {ledger_path}")

    # Historical-anchor interpolation diagnostics. They do NOT validate the
    # long 1959-1983 CH4 or 1938-2000 N2O bridges.
    print("\nCH4 historical-anchor interpolation CV:", interpolation_cv(ch4_hist))
    print("N2O historical-anchor interpolation CV:", interpolation_cv(n2o_hist))
    print("\nWARNING: Long source gaps are synthetic and not independently validated.")
    print("Do not interpret reconstructed values as direct observations.")

    ch4_table = ch4_recon[["year", "ch4_reconstructed", "ch4_data_status",
                           "ch4_is_synthetic", "ch4_method_note", "ch4_anchor_value"]]
    n2o_table = n2o_recon[["year", "n2o_reconstructed", "n2o_data_status",
                           "n2o_is_synthetic", "n2o_method_note", "n2o_anchor_value"]]

    # Do not overwrite original processed datasets.
    add_to_processed(
        PROCESSED / "climate_annual.csv",
        {"ch4": ch4_table, "n2o": n2o_table},
        PROCESSED / "climate_annual_reconstructed.csv"
    )
    add_to_processed(
        PROCESSED / "drivers_annual_full.csv",
        {"ch4": ch4_table, "n2o": n2o_table},
        PROCESSED / "drivers_annual_full_reconstructed.csv"
    )

    print("\nMonthly data were intentionally not changed.")
    print("Annual reconstruction must not be copied into monthly rows: that would fabricate monthly seasonality.")


if __name__ == "__main__":
    main()
