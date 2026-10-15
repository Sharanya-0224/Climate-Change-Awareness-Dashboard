
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "reports" / "ghg_reconstruction"
PLOTS.mkdir(parents=True, exist_ok=True)

original_path = PROCESSED / "climate_annual.csv"
reconstructed_path = PROCESSED / "climate_annual_reconstructed.csv"
drivers_path = PROCESSED / "drivers_annual_full_reconstructed.csv"
ledger_path = PROCESSED / "ghg_annual_reconstruction_ledger.csv"

for path in [
    original_path,
    reconstructed_path,
    drivers_path,
    ledger_path,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

original = pd.read_csv(original_path)
reconstructed = pd.read_csv(reconstructed_path)
drivers = pd.read_csv(drivers_path)
ledger = pd.read_csv(ledger_path)

print("\n=== FILE SHAPES ===")
print("Original:", original.shape)
print("Reconstructed:", reconstructed.shape)
print("Full drivers:", drivers.shape)
print("Ledger:", ledger.shape)

print("\n=== MISSING VALUES ===")
for column in ["ch4", "n2o"]:
    if column not in original or column not in reconstructed:
        print(f"ERROR: Column {column!r} not found.")
        print("Original columns:", original.columns.tolist())
        print("Reconstructed columns:", reconstructed.columns.tolist())
        raise KeyError(column)

    print(
        f"{column.upper()}: "
        f"original missing={original[column].isna().sum()}, "
        f"reconstructed missing={reconstructed[column].isna().sum()}"
    )

print("\n=== YEAR COVERAGE ===")
print(
    "Original:",
    original["year"].min(),
    "to",
    original["year"].max(),
)
print(
    "Reconstructed:",
    reconstructed["year"].min(),
    "to",
    reconstructed["year"].max(),
)

print("\n=== TRANSITION YEARS ===")
years_to_check = [
    1957, 1958, 1959, 1960,
    1978, 1979, 1980, 1983, 1984, 1985,
    1999, 2000, 2001, 2002,
]

for year in years_to_check:
    rows = reconstructed.loc[
        reconstructed["year"] == year, ["year", "ch4", "n2o"]
    ]
    if not rows.empty:
        print(rows.to_string(index=False))

print("\n=== LEDGER STRUCTURE ===")
print("Columns:", ledger.columns.tolist())
print(ledger.head(10).to_string(index=False))

for column in ["ch4", "n2o"]:
    values = reconstructed[column]
    print(f"\n=== {column.upper()} BASIC CHECKS ===")
    print("Minimum:", values.min())
    print("Maximum:", values.max())
    print("Non-finite values:", (~np.isfinite(values.dropna())).sum())

    # Plot the original available annual observations and reconstructed series.
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        reconstructed["year"],
        reconstructed[column],
        label="Reconstructed annual series",
        linewidth=1.8,
    )

    observed = original[["year", column]].dropna()
    ax.scatter(
        observed["year"],
        observed[column],
        label="Values present in original processed data",
        s=18,
    )

    ax.set_title(f"{column.upper()}: Original Available Values vs Reconstruction")
    ax.set_xlabel("Year")
    ax.set_ylabel("Concentration (ppb)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = PLOTS / f"{column}_observed_vs_reconstructed.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print("Plot saved:", output_path)

print("\n=== VALIDATION LIMITATIONS ===")
print(
    "These plots show original processed observations, not every historical "
    "raw-source measurement."
)
print(
    "A complete series and plausible-looking plots do not establish that "
    "synthetic values are scientifically accurate."
)
print(
    "The original CSV was read only; this script does not modify it."
)

print("\nValidation script completed.")

