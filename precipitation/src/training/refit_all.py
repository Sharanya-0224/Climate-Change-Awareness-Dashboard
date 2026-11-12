"""Refit the chosen LightGBM on all years 1901-2025 (production model).

Selection and test were done on the 1901-2002 model (config.EVAL_RUN_ID).
Same params, features and normals (normals stay train-only, 1901-2002).
No held-out years remain, so no skill is claimed for this fit.
Run from the repo root: python3 precipitation/src/training/refit_all.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # precipitation/src

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from evaluation.metrics import mse_skill
from training import train_lgbm as tl

RUN_NAME = "rain_v2_lgbm_final_all"


def main():
    assert lgb.__version__ == tl.LIGHTGBM_VERSION, f"lightgbm {lgb.__version__}, need {tl.LIGHTGBM_VERSION}"
    inputs = json.load(open(config.FEATURE_PATH))["inputs"]
    df = pd.read_parquet(config.DATA_PATH, columns=["year", "target_z"] + inputs)
    print("rows:", len(df), "| years:", int(df["year"].min()), "-", int(df["year"].max()))

    config.SERVE_PACKAGE_DIR.mkdir(exist_ok=True)
    model_txt = config.SERVE_PACKAGE_DIR / "model.txt"
    assert not model_txt.exists(), "model.txt already exists in rain_package, stop"

    mlflow = config.setup_mlflow()
    import mlflow.lightgbm
    mlflow.set_experiment(config.EXPERIMENTS["lightgbm"])

    with mlflow.start_run(run_name=RUN_NAME, log_system_metrics=True) as run:
        mlflow.log_params(tl.DEFAULT_PARAMS)
        mlflow.log_params({"clip_target_z": tl.CLIP_TARGET_Z, "n_features": len(inputs),
                           "sd_floor": config.SD_FLOOR, "train_years": "1901-2025"})
        mlflow.set_tags({
            "git_commit": config.get_git_commit(),
            "dataset_md5": config.file_md5(config.DATA_PATH),
            "lightgbm_version": lgb.__version__,
            "evaluated_run_id": config.EVAL_RUN_ID,
            "test_run_id": config.TEST_RUN_ID,
            "normals": "train years 1901-2002 (unchanged)",
            "season_definition": "origin month t; target = months t+1..t+3",
            "note": "production refit on all years; skill comes from the evaluated 1901-2002 model",
        })

        t0 = time.time()
        model = tl.fit_lgbm(df[inputs], df["target_z"])
        mlflow.log_metric("fit_seconds", time.time() - t0)

        tl.save_booster(model, model_txt)
        booster = tl.load_booster(model_txt)
        tl.check_feature_order(booster, inputs)

        p_mem = model.predict(df[inputs])
        p_txt = tl.predict_z(booster, df, inputs)
        reload_diff = float(np.max(np.abs(p_mem - p_txt)))
        mlflow.log_metric("reload_max_abs_diff", reload_diff)
        mlflow.log_metric("insample_skill_all_years", mse_skill(df["target_z"], p_txt))

        # agreement with the evaluated model on test rows (not a score)
        test = df[df["year"] >= config.TEST_START]
        p_eval = tl.predict_z(tl.load_booster(config.EVAL_MODEL_PATH), test, inputs)
        p_refit = p_txt[(df["year"] >= config.TEST_START).values]
        corr = float(np.corrcoef(p_eval, p_refit)[0, 1])
        mlflow.log_metric("corr_refit_vs_eval_on_test", corr)

        mlflow.log_artifact(str(model_txt))
        mlflow.lightgbm.log_model(model.booster_, name="model")

        print("reload max abs diff:", reload_diff, "(must be below 1e-9)")
        print("in-sample skill (not a test score):", round(mse_skill(df["target_z"], p_txt), 4))
        print("corr refit vs evaluated on test rows:", round(corr, 4))
        print("model.txt MB:", round(model_txt.stat().st_size / 1e6, 2))
        print("run id:", run.info.run_id)


if __name__ == "__main__":
    main()