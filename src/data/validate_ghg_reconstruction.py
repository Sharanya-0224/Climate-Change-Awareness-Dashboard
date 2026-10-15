
from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "ghg_reconstruction_comprehensive"
OUT.mkdir(parents=True, exist_ok=True)

GASES = ["ch4", "n2o"]
GAP_LENGTHS = [1, 3, 5, 10, 15, 20]
RANDOM_SEED = 42


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan,
                "Bias": np.nan, "R2": np.nan}

    error = predicted - actual
    variance = np.sum((actual - actual.mean()) ** 2)

    return {
        "n": len(actual),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
        "R2": (
            float(1 - np.sum(error ** 2) / variance)
            if variance > 0 else np.nan
        ),
    }


def read_csv_checked(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def parse_bool(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )


def linear_predict(train_years, train_values, test_years):
    order = np.argsort(train_years)
    x = np.asarray(train_years, dtype=float)[order]
    y = np.asarray(train_values, dtype=float)[order]
    q = np.asarray(test_years, dtype=float)

    if len(x) < 2 or len(np.unique(x)) < 2:
        return np.full(len(q), np.nan)

    # Do not extrapolate outside the training range.
    result = np.interp(q, x, y)
    result[(q < x.min()) | (q > x.max())] = np.nan
    return result


def nearest_anchor_predict(train_years, train_values, test_years):
    x = np.asarray(train_years, dtype=float)
    y = np.asarray(train_values, dtype=float)
    q = np.asarray(test_years, dtype=float)

    result = []
    for year in q:
        idx = np.argmin(np.abs(x - year))
        result.append(y[idx])
    return np.asarray(result)


def make_local_trend_predict(train_years, train_values, test_years):
    """
    Predict with a local linear trend using up to three anchors
    on each side of the held-out block. No extrapolation permitted.
    """
    x = np.asarray(train_years, dtype=float)
    y = np.asarray(train_values, dtype=float)
    q = np.asarray(test_years, dtype=float)
    result = []

    for year in q:
        left = np.where(x < year)[0][-3:]
        right = np.where(x > year)[0][:3]
        idx = np.concatenate([left, right])

        if len(idx) < 2:
            result.append(np.nan)
            continue

        # Fit only nearby anchors to reduce global-trend distortion.
        idx = idx[np.argsort(np.abs(x[idx] - year))]
        idx = idx[:min(6, len(idx))]
        slope, intercept = np.polyfit(x[idx], y[idx], 1)
        result.append(slope * year + intercept)

    return np.asarray(result)


# ------------------------------------------------------------
# Load and verify input files
# ------------------------------------------------------------

ledger = read_csv_checked(
    PROCESSED / "ghg_annual_reconstruction_ledger.csv"
)
original = read_csv_checked(PROCESSED / "climate_annual.csv")
reconstructed = read_csv_checked(
    PROCESSED / "climate_annual_reconstructed.csv"
)
drivers_path = PROCESSED / "drivers_annual_full_reconstructed.csv"
drivers = read_csv_checked(drivers_path)

required_ledger = ["year"]
for gas in GASES:
    required_ledger += [
        f"{gas}_reconstructed",
        f"{gas}_data_status",
        f"{gas}_is_synthetic",
        f"{gas}_anchor_value",
    ]

missing = [c for c in required_ledger if c not in ledger.columns]
if missing:
    raise ValueError(f"Ledger columns missing: {missing}")

input_files = [
    PROCESSED / "climate_annual.csv",
    PROCESSED / "climate_annual_reconstructed.csv",
    PROCESSED / "drivers_annual_full_reconstructed.csv",
    PROCESSED / "ghg_annual_reconstruction_ledger.csv",
]

file_manifest = pd.DataFrame([
    {
        "path": str(p),
        "exists": p.exists(),
        "bytes": p.stat().st_size if p.exists() else np.nan,
        "sha256_current": sha256(p) if p.exists() else None,
    }
    for p in input_files
])
file_manifest.to_csv(OUT / "input_file_manifest.csv", index=False)

print("Inputs loaded. Output:", OUT)
print("\nCurrent input-file hashes saved.")
print("Note: these hashes cannot prove whether a file changed before this run.")


# ------------------------------------------------------------
# Raw-source inventory
# ------------------------------------------------------------

print("\n=== RAW SOURCE INVENTORY ===")
raw_inventory = []

if RAW.exists():
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            raw_inventory.append({
                "file": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            })
else:
    print("WARNING: data/raw directory not found.")

raw_df = pd.DataFrame(raw_inventory)
raw_df.to_csv(OUT / "raw_source_inventory.csv", index=False)

if not raw_df.empty:
    print(raw_df.to_string(index=False))

print(
    "\nRaw-source files are inventoried, but inventory alone does not "
    "confirm their measurements match the ledger. Source-specific parsing "
    "and unit/calibration checks must be based on each file's actual schema."
)


# ------------------------------------------------------------
# Anchor checks and time-series diagnostics
# ------------------------------------------------------------

summary_rows = []
all_cv_rows = []
all_transition_rows = []
all_gap_rows = []
all_sensitivity_rows = []

for gas in GASES:
    print("\n" + "=" * 72)
    print("GAS:", gas.upper())
    print("=" * 72)

    value_col = f"{gas}_reconstructed"
    anchor_col = f"{gas}_anchor_value"
    status_col = f"{gas}_data_status"
    synthetic_col = f"{gas}_is_synthetic"

    d = ledger[
        ["year", value_col, anchor_col, status_col, synthetic_col]
    ].copy()

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d[anchor_col] = pd.to_numeric(d[anchor_col], errors="coerce")
    d = d.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)

    if d["year"].duplicated().any():
        raise ValueError(f"{gas}: duplicate years in ledger.")

    anchor = d.dropna(subset=[anchor_col]).copy()
    anchor = anchor.sort_values("year").reset_index(drop=True)
    x = anchor["year"].to_numpy(dtype=float)
    y = anchor[anchor_col].to_numpy(dtype=float)

    synthetic_mask = parse_bool(d[synthetic_col])
    synthetic_years = d.loc[synthetic_mask, "year"].to_numpy(dtype=int)

    print("Total annual rows:", len(d))
    print("Anchor rows:", len(anchor))
    print("Synthetic rows:", len(synthetic_years))
    print("Missing reconstructed values:", d[value_col].isna().sum())

    # A. Anchor preservation
    anchor_compare = anchor[["year", anchor_col]].merge(
        d[["year", value_col]],
        on="year",
        how="left",
    )
    anchor_compare["error"] = (
        anchor_compare[value_col] - anchor_compare[anchor_col]
    )
    anchor_compare["abs_error"] = anchor_compare["error"].abs()

    anchor_metrics = metrics(
        anchor_compare[anchor_col],
        anchor_compare[value_col],
    )

    print("\nA. Anchor preservation:", anchor_metrics)
    anchor_compare.to_csv(OUT / f"{gas}_anchor_check.csv", index=False)

    # B. Original processed observations versus reconstruction
    if gas in original.columns and gas in reconstructed.columns:
        overlap = original[["year", gas]].merge(
            reconstructed[["year", gas]],
            on="year",
            suffixes=("_original", "_reconstructed"),
        ).dropna()

        overlap_metrics = metrics(
            overlap[f"{gas}_original"],
            overlap[f"{gas}_reconstructed"],
        )
        print("B. Original processed agreement:", overlap_metrics)
    else:
        overlap_metrics = metrics([], [])
        print("B. Original processed agreement unavailable.")

    # C. Leave-one-anchor-out: hide one interior anchor
    loo_rows = []
    for i in range(1, len(x) - 1):
        train_x = np.delete(x, i)
        train_y = np.delete(y, i)
        pred = linear_predict(train_x, train_y, [x[i]])[0]

        loo_rows.append({
            "gas": gas,
            "year": x[i],
            "actual": y[i],
            "predicted": pred,
            "error": pred - y[i],
            "left_right_span_years": x[i + 1] - x[i - 1],
        })

    loo = pd.DataFrame(loo_rows)
    if not loo.empty:
        loo.to_csv(OUT / f"{gas}_leave_one_anchor_out.csv", index=False)
        loo_metrics = metrics(loo["actual"], loo["predicted"])
    else:
        loo_metrics = metrics([], [])

    print("C. Leave-one-anchor-out:", loo_metrics)

    # D. Artificial contiguous calendar-year gap experiments.
    # Mask all anchors inside each interval, retaining anchors on both sides.
    gap_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for gap_length in GAP_LENGTHS:
        eligible_starts = []

        for start_year in range(int(x.min()) + 1, int(x.max()) - gap_length):
            stop_year = start_year + gap_length - 1
            test_idx = np.where((x >= start_year) & (x <= stop_year))[0]

            if len(test_idx) == 0:
                continue

            train_mask = np.ones(len(x), dtype=bool)
            train_mask[test_idx] = False
            train_x, train_y = x[train_mask], y[train_mask]

            # Only evaluate gaps bracketed by remaining observations.
            if (
                len(train_x) >= 2
                and train_x.min() < x[test_idx].min()
                and train_x.max() > x[test_idx].max()
            ):
                eligible_starts.append((test_idx, train_mask, start_year))

        # Test all eligible intervals when feasible; otherwise deterministic sample.
        if len(eligible_starts) > 250:
            chosen = rng.choice(len(eligible_starts), 250, replace=False)
            eligible_starts = [eligible_starts[i] for i in sorted(chosen)]

        for test_idx, train_mask, start_year in eligible_starts:
            test_x, test_y = x[test_idx], y[test_idx]
            train_x, train_y = x[train_mask], y[train_mask]

            predictions = {
                "linear": linear_predict(train_x, train_y, test_x),
                "nearest": nearest_anchor_predict(train_x, train_y, test_x),
                "local_trend": make_local_trend_predict(
                    train_x, train_y, test_x
                ),
            }

            for method, pred in predictions.items():
                m = metrics(test_y, pred)
                gap_rows.append({
                    "gas": gas,
                    "gap_length_calendar_years": gap_length,
                    "gap_start_year": start_year,
                    "method": method,
                    **m,
                })

    gap_df = pd.DataFrame(gap_rows)
    if not gap_df.empty:
        gap_df.to_csv(OUT / f"{gas}_artificial_gap_validation.csv", index=False)
        gap_summary = (
            gap_df.groupby(["gap_length_calendar_years", "method"], dropna=False)
            .apply(
                lambda g: pd.Series({
                    "n_blocks": len(g),
                    "total_test_points": g["n"].sum(),
                    "mean_block_MAE": g["MAE"].mean(),
                    "mean_block_RMSE": g["RMSE"].mean(),
                }),
                include_groups=False,
            )
            .reset_index()
        )
        gap_summary.to_csv(
            OUT / f"{gas}_artificial_gap_summary.csv", index=False
        )
        print("\nD. Artificial gap summary:")
        print(gap_summary.to_string(index=False))
    else:
        gap_summary = pd.DataFrame()
        print("D. No eligible artificial gap experiments.")

    # E. Longest intervals between actual anchors
    if len(anchor) >= 2:
        gap_audit = pd.DataFrame({
            "gas": gas,
            "left_anchor_year": x[:-1],
            "right_anchor_year": x[1:],
            "gap_years": np.diff(x),
        }).sort_values("gap_years", ascending=False)

        gap_audit.to_csv(OUT / f"{gas}_anchor_gap_audit.csv", index=False)
        all_gap_rows.extend(gap_audit.to_dict("records"))

        print("\nE. Largest real anchor gaps:")
        print(gap_audit.head(10).to_string(index=False))

    # F. Transition diagnostics: source/method/status changes
    d["previous_year"] = d["year"].shift()
    d["year_step"] = d["year"] - d["previous_year"]
    d["annual_change"] = d[value_col].diff()
    d["annual_rate"] = d["annual_change"] / d["year_step"]
    d["previous_status"] = d[status_col].shift()
    d["status_changed"] = d[status_col] != d["previous_status"]

    transitions = d.loc[
        d["status_changed"] | d["year_step"].ne(1),
        [
            "year", value_col, status_col, "previous_status",
            "year_step", "annual_change", "annual_rate",
        ],
    ].copy()
    transitions.insert(0, "gas", gas)

    transitions.to_csv(OUT / f"{gas}_transition_audit.csv", index=False)
    all_transition_rows.extend(transitions.to_dict("records"))

    print("\nF. Status transitions and non-consecutive years:")
    print(transitions.to_string(index=False))

    # G. Sensitivity: remove one interior anchor and compare interpolated
    # values over the interval around it with the full-anchor interpolation.
    sens_rows = []
    if len(x) >= 3:
        for i in range(1, len(x) - 1):
            left_year, right_year = x[i - 1], x[i + 1]
            interval_years = np.arange(
                int(left_year), int(right_year) + 1, dtype=float
            )

            full_pred = linear_predict(x, y, interval_years)
            reduced_pred = linear_predict(np.delete(x, i), np.delete(y, i),
                                          interval_years)

            delta = np.abs(reduced_pred - full_pred)
            sens_rows.append({
                "gas": gas,
                "removed_anchor_year": x[i],
                "left_anchor_year": left_year,
                "right_anchor_year": right_year,
                "span_years": right_year - left_year,
                "max_abs_change": np.nanmax(delta),
                "mean_abs_change": np.nanmean(delta),
            })

    sens = pd.DataFrame(sens_rows)
    if not sens.empty:
        sens.to_csv(OUT / f"{gas}_anchor_sensitivity.csv", index=False)
        print("\nG. Highest sensitivity to removing one anchor:")
        print(sens.nlargest(10, "max_abs_change").to_string(index=False))

    # H. Combined plot: anchors and full reconstructed curve
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        d["year"], d[value_col],
        linewidth=2, label="Reconstructed series",
    )
    ax.scatter(
        anchor["year"], anchor[anchor_col],
        s=28, edgecolors="black", linewidths=0.5,
        label="Recorded anchors", zorder=3,
    )
    ax.scatter(
        synthetic_years,
        d.set_index("year").reindex(synthetic_years)[value_col],
        s=12, marker="x", label="Synthetic years", zorder=2,
    )

    # Shade intervals between adjacent anchors based on their duration.
    if len(x) >= 2:
        for i in range(len(x) - 1):
            span = x[i + 1] - x[i]
            if span >= 10:
                ax.axvspan(x[i], x[i + 1], alpha=0.12)

    ax.set_title(f"{gas.upper()}: anchors and reconstruction")
    ax.set_xlabel("Year")
    ax.set_ylabel("Concentration (ppb)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{gas}_comprehensive_plot.png", dpi=180)
    plt.close(fig)

    summary_rows.append({
        "gas": gas,
        "annual_rows": len(d),
        "anchor_count": len(anchor),
        "synthetic_count": int(synthetic_mask.sum()),
        "anchor_max_abs_error": (
            anchor_compare["abs_error"].max()
            if len(anchor_compare) else np.nan
        ),
        "LOO_MAE": loo_metrics["MAE"],
        "LOO_RMSE": loo_metrics["RMSE"],
        "LOO_R2": loo_metrics["R2"],
        "original_overlap_MAE": overlap_metrics["MAE"],
        "largest_anchor_gap_years": (
            float(np.max(np.diff(x))) if len(x) >= 2 else np.nan
        ),
    })


# ------------------------------------------------------------
# Full-driver consistency checks
# ------------------------------------------------------------

print("\n=== FULL DRIVER DATASET CONSISTENCY ===")
driver_rows = []

for gas in GASES:
    col = gas
    if col not in drivers.columns:
        print(f"{gas}: no {col!r} column in full drivers.")
        continue

    n_missing = int(drivers[col].isna().sum())
    driver_rows.append({
        "gas": gas,
        "rows": len(drivers),
        "year_min": drivers["year"].min(),
        "year_max": drivers["year"].max(),
        "missing_values": n_missing,
    })
    print(gas, driver_rows[-1])

pd.DataFrame(driver_rows).to_csv(
    OUT / "full_driver_consistency.csv", index=False
)

# ------------------------------------------------------------
# Save combined reports
# ------------------------------------------------------------

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "validation_summary.csv", index=False)

if all_gap_rows:
    pd.DataFrame(all_gap_rows).to_csv(
        OUT / "all_anchor_gaps.csv", index=False
    )

if all_transition_rows:
    pd.DataFrame(all_transition_rows).to_csv(
        OUT / "all_transition_audits.csv", index=False
    )

print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)
print(summary.to_string(index=False))

print("\nReports written to:", OUT)
print(
    "\nSCIENTIFIC LIMITATION: artificial-gap performance only validates "
    "gap lengths and data regimes represented by the observed anchors. "
    "It cannot establish the true values inside an unobserved 27-year "
    "or 64-year interval. Raw-source values, calibration scales, units, "
    "and uncertainty must be independently checked."
)
