
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "ghg_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)

ledger = pd.read_csv(PROCESSED / "ghg_annual_reconstruction_ledger.csv")
reconstructed = pd.read_csv(
    PROCESSED / "climate_annual_reconstructed.csv"
)
original = pd.read_csv(PROCESSED / "climate_annual.csv")

GASES = ["ch4", "n2o"]


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    error = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)

    return {
        "n": len(actual),
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error**2)),
        "R2": (
            1 - np.sum(error**2) / denominator
            if denominator > 0
            else np.nan
        ),
    }


def validate_gas(gas):
    print("\n" + "=" * 65)
    print(f"SCIENTIFIC VALIDATION: {gas.upper()}")
    print("=" * 65)

    value_col = f"{gas}_reconstructed"
    anchor_col = f"{gas}_anchor_value"
    status_col = f"{gas}_data_status"
    synthetic_col = f"{gas}_is_synthetic"

    required = [
        "year", value_col, anchor_col, status_col, synthetic_col
    ]
    missing_columns = [c for c in required if c not in ledger.columns]

    if missing_columns:
        raise ValueError(
            f"Ledger is missing columns for {gas}: {missing_columns}"
        )

    data = ledger[
        ["year", value_col, anchor_col, status_col, synthetic_col]
    ].copy()

    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data[anchor_col] = pd.to_numeric(data[anchor_col], errors="coerce")
    data = data.dropna(subset=["year"]).sort_values("year")

    # A genuine anchor must have an explicitly recorded anchor value.
    anchors = (
        data.dropna(subset=[anchor_col])
        .drop_duplicates("year")
        .sort_values("year")
        .reset_index(drop=True)
    )

    print("\n1. DATA COMPLETENESS")
    print("Years in ledger:", len(data))
    print("Genuine recorded anchors:", len(anchors))
    print("Missing reconstructed values:", data[value_col].isna().sum())
    print("Synthetic rows:", data[synthetic_col].astype(str).str.lower()
          .isin(["true", "1"]).sum())

    if len(anchors) < 3:
        print(
            "WARNING: Fewer than three anchors. "
            "Interpolation validation is extremely limited."
        )

    # TEST A: Does the reconstructed series preserve recorded anchors?
    print("\n2. ANCHOR PRESERVATION")

    anchor_check = anchors.merge(
        data[["year", value_col]],
        on="year",
        suffixes=("_anchor_record", "_ledger"),
    )

    # Compare the reconstruction against the anchor measurement.
    anchor_check["anchor_error"] = (
        anchor_check[value_col + "_ledger"]
        - anchor_check[anchor_col]
    )

    anchor_errors = anchor_check["anchor_error"].dropna().to_numpy()

    print("Anchors compared:", len(anchor_errors))
    print(
        "Maximum absolute anchor error:",
        np.max(np.abs(anchor_errors)) if len(anchor_errors) else np.nan,
    )
    print(
        "Anchors matching within 1e-6:",
        int(np.sum(np.isclose(anchor_errors, 0, atol=1e-6)))
        if len(anchor_errors) else 0,
        "/",
        len(anchor_errors),
    )

    anchor_check.to_csv(
        OUT / f"{gas}_anchor_preservation.csv",
        index=False,
    )

    # TEST B: Leave-one-anchor-out interpolation validation.
    # Hide each interior anchor and interpolate between its two neighbors.
    print("\n3. LEAVE-ONE-ANCHOR-OUT VALIDATION")

    years = anchors["year"].to_numpy(dtype=float)
    values = anchors[anchor_col].to_numpy(dtype=float)
    statuses = anchors[status_col].astype(str).to_numpy()

    predictions = []

    for i in range(1, len(anchors) - 1):
        left_year, right_year = years[i - 1], years[i + 1]
        left_value, right_value = values[i - 1], values[i + 1]

        if right_year <= left_year:
            continue

        fraction = (years[i] - left_year) / (right_year - left_year)
        prediction = left_value + fraction * (right_value - left_value)

        predictions.append({
            "year": years[i],
            "actual_anchor": values[i],
            "predicted_anchor": prediction,
            "error": values[i] - prediction,
            "absolute_error": abs(values[i] - prediction),
            "neighbor_span_years": right_year - left_year,
            "anchor_status": statuses[i],
        })

    cv = pd.DataFrame(predictions)

    if cv.empty:
        print("Not enough interior anchors for this test.")
    else:
        print("All eligible held-out anchors:")
        print(metrics(cv["actual_anchor"], cv["predicted_anchor"]))

        # Stratify performance by how far apart the two surrounding
        # training anchors were. This exposes weak long-gap performance.
        bins = [-np.inf, 5, 15, 30, np.inf]
        labels = ["<=5 years", "6-15 years", "16-30 years", ">30 years"]
        cv["gap_group"] = pd.cut(
            cv["neighbor_span_years"],
            bins=bins,
            labels=labels,
        )

        print("\nError by span between surrounding anchors:")
        for group, subset in cv.groupby("gap_group", observed=False):
            if len(subset):
                print(str(group), metrics(
                    subset["actual_anchor"],
                    subset["predicted_anchor"],
                ))

        cv.to_csv(OUT / f"{gas}_leave_one_anchor_out.csv", index=False)

    # TEST C: How much time separates genuine measurements?
    print("\n4. OBSERVATION GAP AUDIT")

    if len(anchors) >= 2:
        anchors["next_anchor_year"] = anchors["year"].shift(-1)
        anchors["gap_years"] = (
            anchors["next_anchor_year"] - anchors["year"]
        )

        gaps = anchors.dropna(subset=["gap_years"])[
            ["year", "next_anchor_year", "gap_years"]
        ].copy()

        print("Largest interval between anchors:")
        print(gaps.nlargest(10, "gap_years").to_string(index=False))

        gaps.to_csv(OUT / f"{gas}_anchor_gaps.csv", index=False)

    # TEST D: Compare against original processed observations.
    # This is a consistency check, not independent validation.
    print("\n5. AGREEMENT WITH ORIGINAL PROCESSED FILE")

    overlap = original[["year", gas]].merge(
        reconstructed[["year", gas]],
        on="year",
        suffixes=("_original", "_reconstructed"),
    ).dropna()

    if len(overlap):
        agreement = metrics(
            overlap[f"{gas}_original"],
            overlap[f"{gas}_reconstructed"],
        )
        print(agreement)
        print(
            "Maximum absolute difference:",
            np.max(np.abs(
                overlap[f"{gas}_original"]
                - overlap[f"{gas}_reconstructed"]
            )),
        )
    else:
        print("No overlapping observed values to compare.")

    # TEST E: Trend and shape diagnostics. These are warning indicators,
    # not proof that the reconstruction is correct.
    print("\n6. TREND DIAGNOSTICS")

    series = data.dropna(subset=[value_col]).copy()
    series["annual_change"] = series[value_col].diff()
    series["year_gap"] = series["year"].diff()

    annual_rate = (
        series["annual_change"] / series["year_gap"]
    ).replace([np.inf, -np.inf], np.nan).dropna()

    print("Negative annual changes:", int((annual_rate < 0).sum()))
    print("Zero annual changes:", int(np.isclose(annual_rate, 0).sum()))
    print("Largest absolute annual rate:", annual_rate.abs().max())

    series.to_csv(OUT / f"{gas}_trend_diagnostics.csv", index=False)

    print("\nInterpretation:")
    print(
        "- Low held-out errors support interpolation only where "
        "the tested anchor spacing is comparable."
    )
    print(
        "- Large errors, long anchor gaps, or source changes need "
        "investigation."
    )
    print(
        "- These tests cannot verify unobserved historical values "
        "without independent measurements."
    )


for gas in GASES:
    validate_gas(gas)

print("\nReports saved to:", OUT)
print("Scientific validation run completed.")

