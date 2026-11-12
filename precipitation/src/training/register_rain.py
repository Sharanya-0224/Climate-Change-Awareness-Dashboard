"""Register the rain v2 champion in the MLflow registry (DagsHub).

1. Runs every gate on rain_package/. Stops if any gate fails.
2. Logs gate_<name> = 1/0, test metrics, the gate table and the package into the
   tested model's run (config.EVAL_RUN_ID).
3. Registers runs:/<EVAL_RUN_ID>/model as config.REGISTERED_MODEL, adds tags,
   sets alias "champion".
4. Writes the version into rain_package/metadata.json.
Commit first, then run from the repo root:
    python3 precipitation/src/training/register_rain.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # precipitation/src

import pandas as pd

import config
from quality import rain_gates

ALIAS = "champion"
CV_NOTE = "-0.0198 mean (fold A -0.0448, fold B +0.0052); no setting had both folds > 0, default kept"
VAL_NOTE = "0.0301 (year-block range 0.0114 to 0.0503, from 06, same params)"


def main():
    # 0. code and package must be committed
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "precipitation/src", "rain_package", "reports/registry"],
        cwd=config.REPO_ROOT, text=True).strip()
    assert not dirty, f"commit these first:\n{dirty}"
    commit = config.get_git_commit()

    meta_path = config.SERVE_PACKAGE_DIR / "metadata.json"
    meta = json.load(open(meta_path))
    assert meta["registered_version"] is None, "metadata already has a version, stop"

    # 1. gates
    table = rain_gates.run_all(config.SERVE_PACKAGE_DIR)
    print(table[["gate", "passed", "value"]].to_string(index=False))
    if not table["passed"].all():
        print("A gate failed: not registering.")
        sys.exit(1)
    gates_csv = config.GATES_DIR / f"{config.REGISTERED_MODEL}_gates.csv"
    table.to_csv(gates_csv, index=False)

    skill = pd.read_csv(config.SERVE_PACKAGE_DIR / "skill_table.csv")
    has_skill_groups = ",".join(skill.loc[skill["has_skill"], "group"])

        # 2-3. registration run: gates, package, model; register + alias (same pattern as register_champions.py)
    mlflow = config.setup_mlflow()
    import mlflow.lightgbm
    import lightgbm as lgb
    from datetime import datetime

    client = mlflow.MlflowClient()
    mlflow.set_experiment(config.EXPERIMENTS["lightgbm"])
    booster = lgb.Booster(model_file=str(config.SERVE_PACKAGE_DIR / "model.txt"))
    features = json.load(open(config.SERVE_PACKAGE_DIR / "feature_list.json"))["features"]
    example = pd.read_csv(config.SERVE_PACKAGE_DIR / "lookup_latest.csv")[features].head(3)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    with mlflow.start_run(run_name=f"register_{config.REGISTERED_MODEL}_{stamp}") as run:
        mlflow.set_tags({
            "kind": "registration", "trained_run_id": config.EVAL_RUN_ID,
            "git_commit": commit, "dataset_md5": meta["dataset_md5"],
            "gates_passed": "true", "served": "yes",
            "mlflow.note.content": "Champion: LightGBM trained 1901-2002 (run f36994a1...), "
                                   "tested once on 2014-2025. All gates passed.",
        })
        mlflow.log_params({"trained_run_id": config.EVAL_RUN_ID, "training_years": "1901-2002",
                           "n_features": len(features)})
        mlflow.log_metrics({f"gate_{r['gate']}": float(r["passed"]) for _, r in table.iterrows()})
        mlflow.log_metrics({"val_skill": meta["val_skill"], "test_skill": meta["test_skill"],
                            "test_skill_lower_95": meta["test_range_95"][0],
                            "test_skill_upper_95": meta["test_range_95"][1]})
        mlflow.log_artifact(str(gates_csv), artifact_path="gates")
        mlflow.log_artifacts(str(config.SERVE_PACKAGE_DIR), artifact_path="rain_package")
        model_info = mlflow.lightgbm.log_model(booster, name="model",
                                               registered_model_name=config.REGISTERED_MODEL,
                                               input_example=example)
        version = model_info.registered_model_version
        reg_run_id = run.info.run_id

    tags = {
        "git_commit": commit,
        "dataset_md5": meta["dataset_md5"],
        "lightgbm_version": meta["lightgbm_version"],
        "feature_list": "rain_v2_3month_feature_list_02 (23 features)",
        "split_rule": "by origin year: train 1901-2002, val 2003-2013, test 2014-2025; Oct-Dec buffer dropped",
        "target": "3-month total, months t+1..t+3, z vs 1901-2002 normal",
        "season_definition": "origin month t",
        "sd_floor_mm": config.SD_FLOOR,
        "clip_rule": "none in training; served pred_total_mm clipped at 0",
        "cv_skill": CV_NOTE,
        "val_skill": VAL_NOTE,
        "test_skill": f"{meta['test_skill']} (range {meta['test_range_95'][0]} to {meta['test_range_95'][1]})",
        "has_skill_groups": has_skill_groups,
        "gates_passed": f"{int(table['passed'].sum())}/{len(table)}",
        "refit_note": meta["refit_note"],
        "trained_run_id": config.EVAL_RUN_ID,
        "registration_run_id": reg_run_id,
    }
    for k, v in tags.items():
        client.set_model_version_tag(config.REGISTERED_MODEL, version, k, str(v))
    client.update_model_version(
        config.REGISTERED_MODEL, version,
        description="LightGBM, 3-month rainfall anomaly (z) for India 0.25 degree grid. "
                    "Trained 1901-2002, tested once on 2014-2025. Skill is small; see tags.")
    try:
        client.set_registered_model_alias(config.REGISTERED_MODEL, ALIAS, version)
        print(f"alias '{ALIAS}' set")
    except Exception as e:   # some servers do not support aliases; fall back to a tag
        client.set_model_version_tag(config.REGISTERED_MODEL, version, "alias", ALIAS)
        print(f"alias not supported ({e.__class__.__name__}); tagged alias={ALIAS} instead")

    # 4. version into the package
    meta["registered_version"] = int(version)
    meta["registered_alias"] = ALIAS
    meta["registration_run_id"] = reg_run_id
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nregistered {config.REGISTERED_MODEL} version {version}, alias '{ALIAS}'")
    print("metadata.json updated: commit it now")


if __name__ == "__main__":
    main()