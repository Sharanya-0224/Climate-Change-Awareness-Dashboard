"""Make tests/sample_data/rain_sample.csv: real val rows for pytest (CI has no DVC data).

12 points (3 per region, fixed seed) x all val origins 2003-2013, so lag checks have
consecutive months. pred_z comes from rain_package/model.txt.
Run from the repo root: python3 precipitation/src/quality/make_rain_sample.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # precipitation/src

import numpy as np
import pandas as pd

import config
from training import train_lgbm as tl

OUT = config.REPO_ROOT / "tests" / "sample_data" / "rain_sample.csv"
POINTS_PER_REGION = 3


def main():
    assert not OUT.exists(), f"{OUT} exists, stop"
    features = json.load(open(config.SERVE_PACKAGE_DIR / "feature_list.json"))["features"]
    ids = ["key", "year", "month", "region", "split", "target_z", "target_mm", "target_mean", "target_sd"]
    val = pd.read_parquet(config.DATA_PATH, columns=ids + features,
                          filters=[("year", ">=", config.VAL_START), ("year", "<=", config.VAL_END)])

    rng = np.random.default_rng(config.RANDOM_STATE)
    keys = []
    for r in config.REGIONS:
        pool = np.sort(val.loc[val["region"] == r, "key"].unique())
        keys += [int(k) for k in rng.choice(pool, POINTS_PER_REGION, replace=False)]

    s = val[val["key"].isin(keys)].sort_values(["key", "year", "month"]).reset_index(drop=True)
    s["region"] = s["region"].astype(str)
    s["split"] = s["split"].astype(str)
    booster = tl.load_booster(config.SERVE_PACKAGE_DIR / "model.txt")
    s["pred_z"] = tl.predict_z(booster, s, features)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT, index=False, float_format="%.9g")

    back = pd.read_csv(OUT, dtype={f: "float32" for f in features})
    diff = float(np.abs(tl.predict_z(booster, back, features) - back["pred_z"].values).max())
    kb = OUT.stat().st_size / 1e3
    print(f"rows {len(back)} | points {back['key'].nunique()} | years {back['year'].min()}-{back['year'].max()}"
          f" | size {kb:.0f} KB | prediction diff after CSV {diff:.2e}")
    assert kb < 1000 and diff < 1e-6


if __name__ == "__main__":
    main()