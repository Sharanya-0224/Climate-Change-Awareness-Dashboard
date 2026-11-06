"""
inspect_emissions.py  (put it at src/data/inspect_emissions.py)

Read-only look at the OWID CO2 file and the Global Carbon Budget xlsx.
It changes nothing. Run from the repo root:
    python3 src/data/inspect_emissions.py
"""

from pathlib import Path

import pandas as pd

EMIS = Path("data/raw/emissions")
MODEL1_COLS = ["population", "gdp", "co2", "primary_energy_consumption",
               "energy_per_gdp", "coal_co2", "oil_co2", "gas_co2", "cement_co2"]

print("Files in", EMIS)
for p in sorted(EMIS.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

# ---------------- OWID ----------------
owid = pd.read_csv(EMIS / "owid-co2-data.csv")
print(f"\nOWID: {owid.shape[0]:,} rows, {owid.shape[1]} columns, "
      f"years {owid.year.min()} to {owid.year.max()}")

# Real countries have a 3-letter ISO code; regions and groups have none or start with OWID_.
is_country = owid["iso_code"].notna() & ~owid["iso_code"].astype(str).str.startswith("OWID_")
countries = owid[is_country]
print(f"Countries: {countries['country'].nunique()} "
      f"(dropped {owid.loc[~is_country, 'country'].nunique()} regions/groups, "
      f"e.g. {sorted(owid.loc[~is_country, 'country'].unique())[:5]})")

print("\nCoverage of Model 1 columns (countries only):")
rows = []
for c in MODEL1_COLS:
    if c not in countries.columns:
        rows.append({"column": c, "exists": False})
        continue
    have = countries[countries[c].notna()]
    rows.append({"column": c, "exists": True,
                 "rows_with_value": len(have),
                 "pct_missing": round(100 * countries[c].isna().mean(), 1),
                 "first_year": int(have.year.min()) if len(have) else None,
                 "last_year": int(have.year.max()) if len(have) else None})
print(pd.DataFrame(rows).to_string(index=False))

core = [c for c in ["population", "gdp", "co2", "primary_energy_consumption"]
        if c in countries.columns]
complete = countries.dropna(subset=core)
print(f"\nRows with ALL of {core}: {len(complete):,} | "
      f"countries: {complete['country'].nunique()} | "
      f"years {complete.year.min()} to {complete.year.max()}")
print("Complete rows per decade:")
print(complete.groupby((complete.year // 10) * 10).size().to_string())

india = countries[countries["country"] == "India"]
print(f"\nIndia: {len(india)} rows; complete rows: "
      f"{len(complete[complete['country'] == 'India'])}")
print(india[["year"] + core].dropna().tail(3).to_string(index=False))

# ---------------- Global Carbon Budget ----------------
xlsx = sorted(EMIS.glob("*.xlsx"))
print(f"\nGCB xlsx files: {[p.name for p in xlsx]}")
for p in xlsx:
    sheets = pd.ExcelFile(p).sheet_names
    print(f"\n{p.name} sheets: {sheets}")
    target = next((s for s in sheets if "global" in s.lower() and "budget" in s.lower()), sheets[0])
    raw = pd.read_excel(p, sheet_name=target, header=None, nrows=30)
    print(f"First 30 raw rows of sheet '{target}' (to find the header row):")
    print(raw.to_string(max_colwidth=25))