"""Quality gates for rain v2. A model is registered only if every gate passes.

Skill gates use the tested 1901-2002 model on test years 2014-2025.
Technical gates use the served package (rain_package/).
Run from the repo root: python3 precipitation/src/quality/rain_gates.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # precipitation/src

import numpy as np
import pandas as pd

import config
from evaluation.metrics import mse_skill
from features import build_features as bf
from training import reference_models as rm
from training import train_lgbm as tl

# Every threshold in one place, with its reason.
THRESHOLDS = {
    "beats_climatology": (0.0, "test skill and its 95% lower bound > 0: must beat the free normal, and not by luck"),
    "beats_persistence": (0.0, "test skill minus damped persistence skill > 0 points: must beat the simplest real forecast"),
    "finite_outputs": (0, "NaN or infinite predictions must be 0: a broken output must never reach a user"),
    "max_abs_z": (6.0, "|pred_z| <= 6: same cut as extreme rows; the model shrinks to normal, so more means a bug"),
    "negative_mm_share": (0.01, "share of rows with total < 0 before the clip at 0 <= 1%: clipping must be rare"),
    "max_total_mm": (1.0, "predicted total <= 1.0 x largest observed 3-month total in train: nothing beyond the record"),
    "deterministic": (0.0, "same input twice and reloaded model give max difference 0: results must be repeatable"),
    "no_future_data": (0.0, "lookup rebuilt with all months after the origin removed is unchanged: no leakage"),
    "normals_train_only": (1e-3, "package normals equal normals from 1901-2002 only (csv rounding allowed): no leakage"),
    "feature_order": (0, "model feature names equal feature_list.json: wrong order gives silent wrong answers"),
}


def _row(name, passed, value):
    thr, reason = THRESHOLDS[name]
    return {"gate": name, "passed": bool(passed), "value": value, "threshold": thr, "reason": reason}


# ---------- gates (data in, one row out) ----------
def gate_beats_climatology(test_skill, test_lower_95):
    return _row("beats_climatology", test_skill > 0 and test_lower_95 > 0, round(float(test_lower_95), 4))


def gate_beats_persistence(skill_model, skill_persistence):
    diff = 100 * (skill_model - skill_persistence)
    return _row("beats_persistence", diff > 0, round(float(diff), 3))


def gate_finite_outputs(pred_z):
    bad = int((~np.isfinite(pred_z)).sum())
    return _row("finite_outputs", bad == 0, bad)


def gate_max_abs_z(pred_z):
    m = float(np.abs(pred_z).max())
    return _row("max_abs_z", m <= THRESHOLDS["max_abs_z"][0], round(m, 3))


def gate_negative_mm_share(pred_total_mm):
    share = float((pred_total_mm < 0).mean())
    return _row("negative_mm_share", share <= THRESHOLDS["negative_mm_share"][0], round(share, 5))


def gate_max_total_mm(pred_total_mm, observed_max_mm):
    ratio = float(pred_total_mm.max() / observed_max_mm)
    return _row("max_total_mm", ratio <= THRESHOLDS["max_total_mm"][0], round(ratio, 3))


def gate_deterministic(booster, model_path, data, features):
    a = tl.predict_z(booster, data, features)
    b = tl.predict_z(booster, data, features)
    c = tl.predict_z(tl.load_booster(model_path), data, features)
    d = float(max(np.abs(a - b).max(), np.abs(a - c).max()))
    return _row("deterministic", d == 0.0, d)


def gate_no_future_data(X, mu_m, sd_m, nino, dmi, keys, points, lookup, features):
    worst = 0.0
    for (y, m), part in lookup.groupby(["year", "month"]):
        pos = bf.position(int(y), int(m))
        Xc, nc, dc = X.copy(), nino.copy(), dmi.copy()
        Xc[:, pos + 1:] = np.nan
        nc[pos + 1:] = np.nan
        dc[pos + 1:] = np.nan
        rebuilt = bf.rows_at(bf.feature_arrays(Xc, mu_m, sd_m, nc, dc), keys, points, [pos], features)
        merged = part.merge(rebuilt[["key"] + features], on="key", suffixes=("_l", "_r"), validate="one_to_one")
        for f in features:
            worst = max(worst, float(np.abs(merged[f + "_l"].astype(float) - merged[f + "_r"].astype(float)).max()))
    return _row("no_future_data", worst <= THRESHOLDS["no_future_data"][0], worst)


def gate_normals_train_only(pkg_normals, mu_w, sd_w, keys):
    n = pkg_normals.sort_values(["key", "origin_month"])
    assert (np.sort(n["key"].unique()) == keys).all()
    P = len(keys)
    d = float(max(np.abs(n["target_mean_mm"].to_numpy().reshape(P, 12) - mu_w).max(),
                  np.abs(n["target_sd_mm"].to_numpy().reshape(P, 12) - sd_w).max()))
    return _row("normals_train_only", d <= THRESHOLDS["normals_train_only"][0], d)


def gate_feature_order(booster, feature_list):
    ok = booster.feature_name() == list(feature_list)
    return _row("feature_order", ok, 0 if ok else 1)


# ---------- run all ----------
def run_all(pkg_dir):
    pkg_dir = Path(pkg_dir)
    model_path = pkg_dir / "model.txt"
    booster = tl.load_booster(model_path)
    features = json.load(open(pkg_dir / "feature_list.json"))["features"]
    normals = pd.read_csv(pkg_dir / "normals.csv")
    lookup = pd.read_csv(pkg_dir / "lookup_latest.csv")
    points = pd.read_csv(pkg_dir / "points.csv").set_index("key")
    rows = [gate_feature_order(booster, features)]

    # test rows (tested model)
    cols = ["year", "month", "target_z", "target_mm", "target_mean", "target_sd"]
    test = pd.read_parquet(config.DATA_PATH, columns=cols + features,
                           filters=[("year", ">=", config.TEST_START), ("year", "<=", config.TEST_END)])
    p_test = tl.predict_z(booster, test, features)
    skill = mse_skill(test["target_z"], p_test)
    res = pd.read_json(config.PRECIP_DIR / "artifacts" / "test_results_lgbm.json").set_index("group")
    print("test skill now:", skill, "| 08 file:", res.loc["overall", "skill"])
    assert abs(skill - res.loc["overall", "skill"]) < 1e-5, "test skill differs from 08 results"
    rows.append(gate_beats_climatology(skill, res.loc["overall", "lower_95"]))

    pers = rm.predict_persistence(rm.load_persistence(config.PERSISTENCE_PATH), test)
    rows.append(gate_beats_persistence(skill, mse_skill(test["target_z"], pers)))

    # outputs on test rows and lookup rows
    lk = lookup.merge(normals, left_on=["key", "month"], right_on=["key", "origin_month"],
                      how="left", validate="one_to_one")
    assert lk["target_mean_mm"].notna().all()
    p_lk = tl.predict_z(booster, lk, features)
    all_z = np.concatenate([p_test, p_lk])
    all_mm = np.concatenate([test["target_mean"].values + p_test * test["target_sd"].values,
                             lk["target_mean_mm"].values + p_lk * lk["target_sd_mm"].values])
    train_max = float(pd.read_parquet(config.DATA_PATH, columns=["target_mm"],
                                      filters=[("year", "<=", config.TRAIN_END)])["target_mm"].max())
    rows += [gate_finite_outputs(all_z), gate_max_abs_z(all_z), gate_negative_mm_share(all_mm),
             gate_max_total_mm(all_mm, train_max), gate_deterministic(booster, model_path, lk, features)]

    # leakage
    keys, X = bf.monthly_matrix(pd.read_parquet(config.MONTHLY_CLEAN_PATH), config.END_YEAR)
    mu_m, sd_m = bf.monthly_normals(X, config.TRAIN_END, config.SD_FLOOR)
    mu_w, sd_w = bf.target_normals(X, config.TRAIN_END, config.SD_FLOOR)
    ind = bf.load_indices(config.RAW)
    nino = bf.idx_series(ind, "nino34", config.END_YEAR)
    dmi = bf.idx_series(ind, "dmi", config.END_YEAR)
    rows.append(gate_no_future_data(X, mu_m, sd_m, nino, dmi, keys, points, lookup, features))
    rows.append(gate_normals_train_only(normals, mu_w, sd_w, keys))

    return pd.DataFrame(rows)


def main():
    table = run_all(config.SERVE_PACKAGE_DIR)
    pd.set_option("display.max_colwidth", 60)
    print(table[["gate", "passed", "value", "threshold"]].to_string(index=False))
    config.GATES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.GATES_DIR / f"{config.REGISTERED_MODEL}_gates.csv"
    table.to_csv(out, index=False)
    print("\nall passed:", bool(table["passed"].all()), "| saved:", out)


if __name__ == "__main__":
    main()