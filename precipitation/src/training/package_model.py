"""Build the serving package rain_package/ (repo root).

model.txt is already there (the tested 1901-2002 model). This script adds:
feature_list.json, normals.csv, points.csv, lookup_latest.csv, skill_table.csv, metadata.json.
Run from the repo root: python3 precipitation/src/training/package_model.py
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # precipitation/src

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from evaluation.metrics import get_season, mse_skill
from features import build_features as bf
from training import train_lgbm as tl

PKG = config.SERVE_PACKAGE_DIR
LOOKUP_ORIGINS = [(2025, 10), (2025, 11), (2025, 12)]
HAS_SKILL_RULE = ("val skill > 0 and test 95% range lower bound > 0 "
                  "(set after the test; a reporting flag, not used for tuning)")
SEASON_COVERS = {"DJF": "origin Dec-Feb, covers Jan-May", "MAM": "origin Mar-May, covers Apr-Aug",
                 "JJA": "origin Jun-Aug, covers Jul-Nov", "SON": "origin Sep-Nov, covers Oct-Feb"}


def cell_id(lat, lon):
    """Friend's 1-degree square: centre = floor(value) + 0.5."""
    return f"{np.floor(lat) + 0.5:.1f}_{np.floor(lon) + 0.5:.1f}"


def new_file(name):
    path = PKG / name
    assert not path.exists(), f"{path} exists, stop (no overwrite)"
    return path


def main():
    # 0. model
    model_txt = PKG / "model.txt"
    assert config.file_md5(model_txt) == config.file_md5(config.EVAL_MODEL_PATH), "model.txt is not the tested model"
    booster = tl.load_booster(model_txt)
    inputs = json.load(open(config.FEATURE_PATH))["inputs"]
    tl.check_feature_order(booster, inputs)

    # 1. feature list
    with open(new_file("feature_list.json"), "w") as f:
        json.dump({"features": inputs, "n_features": len(inputs), "target": "target_z"}, f, indent=2)

    # 2. normals of the 3-month target (train years only)
    nrm = pd.read_parquet(config.NORMALS_PATH)[["key", "origin_month", "target_mean_mm", "target_sd_mm"]]
    nrm = nrm.sort_values(["key", "origin_month"]).reset_index(drop=True)
    assert (nrm["target_sd_mm"] >= config.SD_FLOOR).all() and nrm.notna().all().all()
    nrm.to_csv(new_file("normals.csv"), index=False, float_format="%.4f")

    # 3. points (kept points only)
    keys_kept = np.sort(nrm["key"].unique())
    pts = bf.get_points(config.RAIN_2025_PATH, config.POINT_REGION_PATH)
    pts = pts[pts["key"].isin(keys_kept)].copy()
    assert len(pts) == len(keys_kept)
    pts["cell_id"] = [cell_id(a, b) for a, b in zip(pts["latitude"], pts["longitude"])]
    pts = pts[["key", "latitude", "longitude", "state", "region", "cell_id"]]
    pts.to_csv(new_file("points.csv"), index=False)

    # 4. lookup: same feature code, latest origins
    keys, X = bf.monthly_matrix(pd.read_parquet(config.MONTHLY_CLEAN_PATH), config.END_YEAR)
    assert (keys == keys_kept).all(), "monthly file and normals have different points"
    mu_m, sd_m = bf.monthly_normals(X, config.TRAIN_END, config.SD_FLOOR)
    ind = bf.load_indices(config.RAW)
    arrays = bf.feature_arrays(X, mu_m, sd_m, bf.idx_series(ind, "nino34", config.END_YEAR),
                               bf.idx_series(ind, "dmi", config.END_YEAR))
    pts_idx = pts.set_index("key")

    sep = bf.rows_at(arrays, keys, pts_idx, [bf.position(2025, 9)], inputs)
    old = pd.read_parquet(config.DATA_PATH, columns=["key"] + inputs,
                          filters=[("year", "==", 2025), ("month", "==", 9)])
    m = old.merge(sep, on="key", suffixes=("_o", "_n"), how="left", validate="one_to_one")
    sep_diff = max(float(np.abs(m[f + "_o"].astype(float) - m[f + "_n"].astype(float)).max()) for f in inputs)
    print("check rebuilt Sep 2025 vs dataset_02, max diff:", sep_diff)
    assert sep_diff == 0.0

    lookup = bf.rows_at(arrays, keys, pts_idx, [bf.position(y, mo) for y, mo in LOOKUP_ORIGINS], inputs)
    lookup = lookup[["key", "year", "month"] + inputs]
    assert not lookup[inputs].isna().any().any(), "NaN in lookup"
    lookup_path = new_file("lookup_latest.csv")
    lookup.to_csv(lookup_path, index=False, float_format="%.9g")
    back = pd.read_csv(lookup_path)
    csv_diff = float(np.abs(tl.predict_z(booster, back, inputs) - tl.predict_z(booster, lookup, inputs)).max())
    print("lookup rows:", len(lookup), "| prediction diff after CSV round trip:", csv_diff)
    assert csv_diff < 1e-6

    # 5. skill table: val from model.txt, test from the 08 results file
    val = pd.read_parquet(config.DATA_PATH, columns=["region", "month", "target_z"] + inputs,
                          filters=[("year", ">=", config.VAL_START), ("year", "<=", config.VAL_END)])
    p_val = tl.predict_z(booster, val, inputs)
    groups = {"overall": np.full(len(val), "overall"),
              "region": ("region_" + val["region"].astype(str)).values,
              "season": ("season_" + val["month"].map(get_season)).values}
    rows = []
    for gtype, labels in groups.items():
        for g in sorted(set(labels)):
            k = labels == g
            rows.append({"group_type": gtype, "group": g,
                         "val_skill": mse_skill(val["target_z"].values[k], p_val[k])})
    skill = pd.DataFrame(rows)
    test_res = pd.read_json(config.TEST_RESULTS_PATH)
    test_res = test_res.rename(columns={"skill": "test_skill", "lower_95": "test_lower_95",
                                        "upper_95": "test_upper_95"})
    skill = skill.merge(test_res[["group", "test_skill", "test_lower_95", "test_upper_95"]],
                        on="group", how="left", validate="one_to_one")
    assert skill.notna().all().all(), "group names do not match the test results"
    skill["has_skill"] = (skill["val_skill"] > 0) & (skill["test_lower_95"] > 0)
    skill["covers"] = [SEASON_COVERS.get(g.replace("season_", ""), "") for g in skill["group"]]
    skill.to_csv(new_file("skill_table.csv"), index=False, float_format="%.4f")
    print(skill.round(4).to_string(index=False))

    # 6. metadata
    overall = skill[skill["group"] == "overall"].iloc[0]
    meta = {
        "registered_name": config.REGISTERED_MODEL,
        "registered_version": None,          # filled in at registration
        "run_id": config.EVAL_RUN_ID,
        "test_run_id": config.TEST_RUN_ID,
        "lightgbm_version": lgb.__version__,
        "training_years": "1901-2002",
        "target": "3-month rainfall total for months t+1..t+3, as z = (total - normal mean) / normal sd",
        "units": {"pred_z": "standard deviations", "pred_anom_mm": "mm", "pred_total_mm": "mm"},
        "normals": "per point and origin month, train years 1901-2002, sd floor 5 mm",
        "season_definition": "by origin month t",
        "grid": "0.25 degree IMD points; cell_id = 1-degree square, floor(value) + 0.5",
        "lookup_origins": [f"{y}-{mo:02d}" for y, mo in LOOKUP_ORIGINS],
        "val_skill": round(float(overall["val_skill"]), 4),
        "test_skill": round(float(overall["test_skill"]), 4),
        "test_range_95": [round(float(overall["test_lower_95"]), 4), round(float(overall["test_upper_95"]), 4)],
        "has_skill_rule": HAS_SKILL_RULE,
        "refit_note": "refit on 1901-2025 tried (run caed54cc...), not served: corr 0.758 < 0.8 rule",
        "dataset_md5": config.file_md5(config.DATA_PATH),
        "git_commit": config.get_git_commit(),
        "n_points": int(len(pts)),
        "created": datetime.date.today().isoformat(),
    }
    with open(new_file("metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 7. size check
    total = 0.0
    for p in sorted(PKG.iterdir()):
        mb = p.stat().st_size / 1e6
        total += mb
        print(f"{p.name:22s} {mb:6.2f} MB")
    print(f"total {total:.2f} MB")
    assert total < 20


if __name__ == "__main__":
    main()