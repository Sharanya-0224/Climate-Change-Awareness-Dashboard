
"""Strict validation of annual CH4 and N2O reconstructions.

Read-only for raw and processed inputs.
Writes reports only to reports/ghg_reconstruction_strict/.
This evaluates interpolation performance; it cannot prove values in
unobserved historical periods are correct.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "ghg_reconstruction_strict"
REPORT.mkdir(parents=True, exist_ok=True)

# Reuse the exact anchor-loading functions from the reconstruction script.
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
from reconstruct_ghg import historical_anchors, read_noaa_annual

CONFIG = {
    "ch4": {
        "epa_file": "ghg-concentrations_fig-2.csv",
        "epa_value_col": 2,  # Law Dome
        "epa_max_year": 1958,
        "noaa_file": "ch4_annmean_gl.txt",
    },
    "n2o": {
        "epa_file": "ghg-concentrations_fig-3.csv",
        "epa_value_col": 1,  # EPICA Dome C
        "epa_max_year": 1937,
        "noaa_file": "n2o_annmean_gl.txt",
    },
}


def metrics(actual, predicted):
    """Calculate errors between withheld observations and estimates."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual

    return {
        "n_tested": len(error),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


def load_anchors(gas, cfg):
    """Load the same historical and NOAA anchors used by reconstruction."""
    historical = historical_anchors(
        cfg["epa_file"],
        year_col=0,
        value_col=cfg["epa_value_col"],
        min_year=1800,
        max_year=cfg["epa_max_year"],
        label="EPA_historical",
    )

    modern = read_noaa_annual(cfg["noaa_file"], gas)
    modern = modern.rename(columns={gas: "value"})
    modern["anchor_source"] = "NOAA_annual"

    # NOAA takes precedence when the same year appears in both sources.
    anchors = pd.concat([historical, modern], ignore_index=True)
    anchors = anchors.sort_values(
        ["year", "anchor_source"], kind="stable"
    )
    anchors = anchors.drop_duplicates("year", keep="last")
    anchors = anchors.sort_values("year").reset_index(drop=True)

    return anchors[["year", "value", "anchor_source"]]


def leave_one_out(anchors):
    """Hide each interior anchor and estimate it from its neighbours."""
    rows = []

    for i in range(1, len(anchors) - 1):
        left = anchors.iloc[i - 1]
        test = anchors.iloc[i]
        right = anchors.iloc[i + 1]

        # Linear interpolation using the surrounding observations only.
        fraction = (
            (test["year"] - left["year"])
            / (right["year"] - left["year"])
        )
        pred = left["value"] + fraction * (
            right["value"] - left["value"]
        )

        rows.append({
            "year": int(test["year"]),
            "actual": test["value"],
            "predicted": pred,
            "left_anchor_year": int(left["year"]),
            "right_anchor_year": int(right["year"]),
            "bracketing_gap_years": int(right["year"] - left["year"]),
            "source": test["anchor_source"],
            "error": pred - test["value"],
        })

    return pd.DataFrame(rows)


def blocked_gap_test(anchors, gap_lengths=(1, 3, 5, 10, 15, 20)):
    """Hide observed anchors in a bracketed block, then interpolate.

    Only tests blocks that contain actual anchors and have observed
    anchors on both sides. It cannot test an interval with no interior
    observations at all.
    """
    rows = []
    years = anchors["year"].to_numpy(dtype=int)
    values = anchors["value"].to_numpy(dtype=float)

    for gap in gap_lengths:
        for start in range(len(anchors)):
            end = start + gap

            # Need at least one anchor on either side of the hidden block.
            if end >= len(anchors) - 1:
                continue

            left_i = start
            right_i = end + 1
            hidden = range(start + 1, end + 1)

            left_year, right_year = years[left_i], years[right_i]

            for i in hidden:
                year = years[i]
                prediction = np.interp(
                    year,
                    [left_year, right_year],
                    [values[left_i], values[right_i]],
                )

                rows.append({
                    "requested_hidden_anchor_count": gap,
                    "year": int(year),
                    "actual": values[i],
                    "predicted": float(prediction),
                    "error": float(prediction - values[i]),
                    "left_anchor_year": int(left_year),
                    "right_anchor_year": int(right_year),
                    "bracketing_year_span": int(right_year - left_year),
                })

    return pd.DataFrame(rows)


def gap_audit(anchors, gas):
    """Report actual gaps between successive anchors."""
    a = anchors.sort_values("year").reset_index(drop=True)
    rows = []

    for i in range(1, len(a)):
        left = a.iloc[i - 1]
        right = a.iloc[i]
        span = int(right["year"] - left["year"])

        rows.append({
            "gas": gas.upper(),
            "left_year": int(left["year"]),
            "left_source": left["anchor_source"],
            "right_year": int(right["year"]),
            "right_source": right["anchor_source"],
            "year_span": span,
            "missing_intermediate_calendar_years": max(0, span - 1),
            "source_changes": (
                left["anchor_source"] != right["anchor_source"]
            ),
            "requires_long_gap_review": span > 5,
        })

    return pd.DataFrame(rows)


def validate_gas(gas, cfg):
    anchors = load_anchors(gas, cfg)

    if anchors["year"].duplicated().any():
        raise ValueError(f"{gas}: duplicate anchor years remain.")
    if not np.isfinite(anchors["value"]).all():
        raise ValueError(f"{gas}: non-finite anchor values found.")

    loo = leave_one_out(anchors)
    blocked = blocked_gap_test(anchors)
    gaps = gap_audit(anchors, gas)

    loo.to_csv(REPORT / f"{gas}_leave_one_out.csv", index=False)
    blocked.to_csv(REPORT / f"{gas}_blocked_gap_test.csv", index=False)
    gaps.to_csv(REPORT / f"{gas}_anchor_gaps.csv", index=False)
    anchors.to_csv(REPORT / f"{gas}_anchors_used.csv", index=False)

    loo_summary = metrics(loo["actual"], loo["predicted"]) if not loo.empty else {
        "n_tested": 0, "MAE": np.nan, "RMSE": np.nan, "bias": np.nan
    }

    blocked_summary = []
    if not blocked.empty:
        for requested_gap, group in blocked.groupby(
            "requested_hidden_anchor_count"
        ):
            row = metrics(group["actual"], group["predicted"])
            row["requested_hidden_anchor_count"] = int(requested_gap)
            row["actual_year_span_max"] = int(
                group["bracketing_year_span"].max()
            )
            blocked_summary.append(row)

    # The real maximum gap is directly measurable from anchor dates.
    max_gap_row = gaps.loc[gaps["year_span"].idxmax()]

    print(f"\n{'=' * 65}")
    print(gas.upper())
    print(f"Anchors used: {len(anchors)}")
    print(
        f"Anchor coverage: {anchors['year'].min()}–"
        f"{anchors['year'].max()}"
    )
    print(
        "Largest anchor gap: "
        f"{int(max_gap_row['left_year'])}–"
        f"{int(max_gap_row['right_year'])} "
        f"({int(max_gap_row['year_span'])} years)"
    )
    print("Leave-one-out metrics:", loo_summary)

    if blocked_summary:
        print("\nBlocked-gap metrics:")
        print(pd.DataFrame(blocked_summary).to_string(index=False))

    print(
        "\nIMPORTANT: These tests measure performance only where "
        "observations exist. They do not validate the true values "
        "inside unobserved historical gaps."
    )

    return {
        "gas": gas.upper(),
        "anchor_count": len(anchors),
        "first_anchor_year": int(anchors["year"].min()),
        "last_anchor_year": int(anchors["year"].max()),
        "largest_anchor_gap_years": int(max_gap_row["year_span"]),
        "largest_gap_start": int(max_gap_row["left_year"]),
        "largest_gap_end": int(max_gap_row["right_year"]),
        "loo_n": loo_summary["n_tested"],
        "loo_MAE": loo_summary["MAE"],
        "loo_RMSE": loo_summary["RMSE"],
        "loo_bias": loo_summary["bias"],
        "interpretation": "Partial validation; long unobserved gaps unverified",
    }


def main():
    summaries = []

    for gas, cfg in CONFIG.items():
        summaries.append(validate_gas(gas, cfg))

    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORT / "validation_summary.csv", index=False)

    print(f"\nReports written to: {REPORT}")
    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print(
        "\nNo raw data or original processed datasets were modified."
    )


if __name__ == "__main__":
    main()

