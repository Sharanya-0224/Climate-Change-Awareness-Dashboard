"""Final practical validation for CH4/N2O interpolation.
Reads existing source files; writes reports only to a new report folder.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "ghg_validation_final"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
from reconstruct_ghg import historical_anchors, read_noaa_annual

CONFIG = {
    "ch4": {
        "epa": "ghg-concentrations_fig-2.csv",
        "column": 2,  # Law Dome
        "historical_max": 1958,
        "noaa": "ch4_annmean_gl.txt",
    },
    "n2o": {
        "epa": "ghg-concentrations_fig-3.csv",
        "column": 1,  # EPICA Dome C
        "historical_max": 1937,
        "noaa": "n2o_annmean_gl.txt",
    },
}


def error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    return {
        "n": len(err),
        "MAE": float(np.mean(np.abs(err))) if len(err) else np.nan,
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
        "bias": float(np.mean(err)) if len(err) else np.nan,
    }


def load_sources(gas, cfg):
    hist = historical_anchors(
        cfg["epa"], year_col=0, value_col=cfg["column"],
        min_year=1880, max_year=cfg["historical_max"],
        label="EPA_historical",
    )
    noaa = read_noaa_annual(cfg["noaa"], gas)
    noaa = noaa.rename(columns={gas: "value"})
    noaa["source"] = "NOAA_annual"
    hist["source"] = "EPA_historical"

    # Do not combine these sources for the validation metrics.
    hist = hist[["year", "value", "source"]].sort_values("year")
    noaa = noaa[["year", "value", "source"]].sort_values("year")
    return hist, noaa


def historical_anchor_test(hist):
    """Estimate each historical anchor using its immediate neighbours."""
    rows = []
    h = hist.sort_values("year").reset_index(drop=True)

    for i in range(1, len(h) - 1):
        left, test, right = h.iloc[i - 1], h.iloc[i], h.iloc[i + 1]
        fraction = (test.year - left.year) / (right.year - left.year)
        pred = left.value + fraction * (right.value - left.value)
        rows.append({
            "held_out_year": int(test.year),
            "actual": test.value,
            "predicted": pred,
            "left_year": int(left.year),
            "right_year": int(right.year),
            "bracket_span_years": int(right.year - left.year),
            "error": pred - test.value,
        })
    return pd.DataFrame(rows)


def contiguous_noaa_test(noaa, lengths=(1, 3, 5, 10)):
    """Hide exact consecutive calendar years from NOAA; interpolate endpoints."""
    n = noaa.sort_values("year").reset_index(drop=True)
    rows = []
    years = n.year.to_numpy(dtype=int)
    vals = n.value.to_numpy(dtype=float)

    for length in lengths:
        for left_i in range(len(n)):
            right_i = left_i + length + 1
            if right_i >= len(n):
                break

            # Require every calendar year in the test interval to exist.
            if years[right_i] - years[left_i] != length + 1:
                continue

            for i in range(left_i + 1, right_i):
                pred = np.interp(
                    years[i],
                    [years[left_i], years[right_i]],
                    [vals[left_i], vals[right_i]],
                )
                rows.append({
                    "hidden_consecutive_years": length,
                    "held_out_year": int(years[i]),
                    "actual": vals[i],
                    "predicted": float(pred),
                    "error": float(pred - vals[i]),
                    "left_year": int(years[left_i]),
                    "right_year": int(years[right_i]),
                })

    return pd.DataFrame(rows)


def anchor_gaps(hist, noaa, gas):
    anchors = pd.concat([hist, noaa], ignore_index=True)
    anchors = anchors.sort_values("year").drop_duplicates("year")
    anchors = anchors.sort_values("year").reset_index(drop=True)
    rows = []

    for i in range(1, len(anchors)):
        a, b = anchors.iloc[i - 1], anchors.iloc[i]
        rows.append({
            "gas": gas.upper(),
            "left_year": int(a.year),
            "left_source": a.source,
            "right_year": int(b.year),
            "right_source": b.source,
            "span_years": int(b.year - a.year),
            "missing_intermediate_years": int(max(0, b.year - a.year - 1)),
        })
    return pd.DataFrame(rows)


def run_gas(gas, cfg):
    hist, noaa = load_sources(gas, cfg)
    hist_test = historical_anchor_test(hist)
    noaa_test = contiguous_noaa_test(noaa)
    gaps = anchor_gaps(hist, noaa, gas)

    hist.to_csv(OUT / f"{gas}_historical_anchors.csv", index=False)
    noaa.to_csv(OUT / f"{gas}_noaa_anchors.csv", index=False)
    hist_test.to_csv(OUT / f"{gas}_historical_holdout.csv", index=False)
    noaa_test.to_csv(OUT / f"{gas}_noaa_contiguous_holdout.csv", index=False)
    gaps.to_csv(OUT / f"{gas}_anchor_gaps.csv", index=False)

    hist_metrics = (
        error_metrics(hist_test.actual, hist_test.predicted)
        if not hist_test.empty else error_metrics([], [])
    )

    summaries = []
    if not noaa_test.empty:
        for length, group in noaa_test.groupby("hidden_consecutive_years"):
            m = error_metrics(group.actual, group.predicted)
            summaries.append({
                "gas": gas.upper(),
                "test": f"NOAA consecutive {length}-year gaps",
                **m,
            })

    summaries.append({
        "gas": gas.upper(),
        "test": "EPA historical anchor holdout (irregular spacing)",
        **hist_metrics,
    })

    combined = pd.concat([hist, noaa], ignore_index=True).sort_values("year")
    combined = combined.drop_duplicates("year").sort_values("year")
    gap_row = gaps.loc[gaps.span_years.idxmax()]

    print(f"\n{'=' * 60}\n{gas.upper()}")
    print("Historical EPA anchors:", len(hist))
    print("NOAA annual observations:", len(noaa))
    print("EPA historical holdout:", hist_metrics)
    print("Largest combined anchor gap:",
          int(gap_row.left_year), "to", int(gap_row.right_year),
          f"({int(gap_row.span_years)} years)")
    print("NOAA contiguous-gap test:")
    if noaa_test.empty:
        print("Not enough consecutive observations for this test.")
    else:
        print(noaa_test.groupby("hidden_consecutive_years").apply(
            lambda g: pd.Series(error_metrics(g.actual, g.predicted)),
            include_groups=False
        ).to_string())
    print("NOTE: NOAA tests validate modern-series interpolation only.")
    print("They do not validate the EPA-to-NOAA historical bridge.")

    return summaries


def main():
    all_summaries = []
    for gas, cfg in CONFIG.items():
        all_summaries.extend(run_gas(gas, cfg))

    pd.DataFrame(all_summaries).to_csv(
        OUT / "validation_metrics.csv", index=False
    )
    print(f"\nReports saved to: {OUT}")
    print("Original raw and processed datasets were not modified.")


if __name__ == "__main__":
    main()
