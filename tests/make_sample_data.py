"""
tests/make_sample_data.py

Creates small sample copies of the model-ready tables for the tests.
Run ONCE from the repo root, then commit tests/sample_data/ to git:

    python3 tests/make_sample_data.py

Why: GitHub Actions runs the tests on every push but cannot reach the DVC data
without credentials, and the full Model 3 table is 50 MB. The tests check the
CODE (column rules, lag rules, model behaviour) on a small, honest subset of the
real data. The full data was already checked by training.

Model 1: ten countries, every year (about 580 rows).
Model 2: the whole table (65 rows, tiny).
Model 3: six squares spread across India, every month 1951-2024 (about 5,000 rows).
"""
from pathlib import Path

import pandas as pd

OUT = Path("tests/sample_data")
OUT.mkdir(parents=True, exist_ok=True)

m1 = pd.read_csv("data/model_ready/m1_emissions.csv")
countries = ["India", "China", "United States", "Germany", "Brazil", "Nigeria", "Japan",
             "Indonesia", "South Africa", "Bangladesh"]
m1_sample = m1[m1.country.isin(countries)]
m1_sample.to_csv(OUT / "m1_emissions.csv", index=False)
print(f"m1: {len(m1_sample)} rows, {m1_sample.country.nunique()} countries -> {OUT / 'm1_emissions.csv'}")

m2 = pd.read_csv("data/model_ready/m2_co2_growth.csv")
m2.to_csv(OUT / "m2_co2_growth.csv", index=False)
print(f"m2: {len(m2)} rows -> {OUT / 'm2_co2_growth.csv'}")

m3 = pd.read_csv("data/model_ready/m3_india_temp.csv")
# six squares: west Rajasthan, Delhi region, Bihar (warming hole), Gujarat coast, Kerala, Assam
wanted = ["27.5_71.5", "28.5_77.5", "25.5_85.5", "21.5_70.5", "10.5_76.5", "26.5_92.5"]
have = [c for c in wanted if c in set(m3.cell_id)]
missing = sorted(set(wanted) - set(have))
if missing:   # fall back to any six squares if a wanted one was dropped from the table
    extra = [c for c in m3.cell_id.unique() if c not in have][: len(missing)]
    print(f"not in table: {missing}; using {extra} instead")
    have += extra
sample = m3[m3.cell_id.isin(have)].sort_values(["year", "month", "cell_id"])
sample.to_csv(OUT / "m3_india_temp.csv", index=False)
print(f"m3: {len(sample)} rows, squares {have} -> {OUT / 'm3_india_temp.csv'}")
