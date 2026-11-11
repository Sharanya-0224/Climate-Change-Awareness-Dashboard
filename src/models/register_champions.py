"""
register_champions.py  (put it at src/models/register_champions.py)

Step 1 of Phase 2: turn the chosen champions into production models.

For Model 2 (emissions -> CO2 rise) and Model 3 (CO2 -> India temperature):
  1. Read which model the pre-fixed champion rule chose (reports/models/*/summary.json).
  2. Refit that model on years <= cutoff and score the unseen years, for the gates that
     need honest test numbers (hit rate).
  3. Retrain it on ALL years. The exams were trained only up to 2010 so the test was
     honest; the production model should learn from every year we have.
  4. Run the quality gates (src/quality/gates.py).
  5. If every gate passes: log the model to MLflow, register it in the Model Registry
     under a fixed name, and point the alias "champion" at this version.
     If any gate fails: log the gate results and stop with an error (nothing is registered).

Registry names (what the API will load):
  models:/m2_co2_growth@champion
  models:/m3_india_temperature@champion

Run from the repo root:  python3 src/models/register_champions.py [m2|m3|all]
"""
import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv

SRC = Path(__file__).resolve().parents[1]          # .../src
sys.path.insert(0, str(SRC))                        # climate_models, quality
sys.path.insert(0, str(SRC / "models"))             # the Model 2 training script
from climate_models.m3_model import M3Model, MODELS as M3_MODELS, INPUT_COLS, TARGET as M3_TARGET
from quality import gates as G
import train_m2_co2 as m2             # reuses Model 2's model definitions, nothing is retrained there

REGISTRY_DIR = Path("reports/registry")
REGISTERED_NAMES = {"m2": "m2_co2_growth", "m3": "m3_india_temperature"}
EXPERIMENTS = {"m2": "m2_emissions_to_co2", "m3": "m3_india_temperature"}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ====================================================================== Model 2
def prepare_m2():
    data_path = m2.DATA
    df = pd.read_csv(data_path).dropna().sort_values(m2.YEAR).reset_index(drop=True)
    summary = json.loads(Path("reports/models/m2/summary.json").read_text())
    champ = summary["champion"]
    spec = m2.MODELS[champ]
    long = pd.read_csv("reports/models/m2/comparison_long.csv")
    final = long[(long.model == champ) & (long.exam == long.exam.max())].iloc[0]
    persist = df[m2.TARGET].shift(1)

    # honest numbers: fit on train years, predict the unseen years
    cutoff, end = m2.EXAMS[-1]
    train, test = df[df[m2.YEAR] <= cutoff], df[(df[m2.YEAR] > cutoff) & (df[m2.YEAR] <= end)]
    test_pred, _ = m2.predict(champ, spec, train, test, persist)

    # production model: fit on all years
    _, model = m2.predict(champ, spec, df, df, persist)
    all_pred = model.predict(df[spec["features"]])

    checks = [
        G.champion_rule(summary["champion"], champ, summary["reason"]),
        G.beats_baseline(final["skill"] * 100, "same as last year"),
        G.hit_rate(test[m2.TARGET], test_pred, G.TOLERANCE_M2_PPM, "ppm/yr"),
        G.direction_hit_rate(test[m2.TARGET].diff().dropna(), np.diff(test_pred)),
        G.no_missing_outputs(all_pred),
        G.physical_range(all_pred, df[m2.TARGET], G.RANGE_MARGIN_M2_PPM, "ppm/yr"),
        G.deterministic(model, df[spec["features"]]),
    ]
    info = {"champion": champ, "features": spec["features"], "rows_trained": len(df),
            "years": f"{df[m2.YEAR].min()}-{df[m2.YEAR].max()}",
            "final_exam_skill_pct": float(final["skill"] * 100), "final_exam_mae": float(final["mae"]),
            "data_path": str(data_path), "input_example": df[spec["features"]].head(3),
            "code_paths": None}
    return model, checks, info


# ====================================================================== Model 3
def prepare_m3():
    data_path = Path("data/model_ready/m3_india_temp.csv")
    df = pd.read_csv(data_path, usecols=INPUT_COLS + [M3_TARGET])
    df = df.sort_values(["year", "month", "cell_id"]).reset_index(drop=True)
    summary = json.loads(Path("reports/models/m3/summary.json").read_text())
    champ = summary["champion"]
    spec = M3_MODELS[champ]
    res = pd.read_csv("reports/models/m3/comparison.csv")
    final = res[(res.model == champ) & (res.exam == res.exam.max())].iloc[0]
    region = pd.read_csv("reports/models/m3/region_holdout.csv", index_col=0)
    region_skill = float(region.loc[champ, "hidden regions"]) if champ in region.index else np.nan

    # honest numbers: fit up to 2010, predict 2011-2024, then average to all-India yearly
    cutoff = int(res.exam.max())
    train, test = df[df.year <= cutoff], df[df.year > cutoff]
    exam_model = M3Model(**spec).fit(train[INPUT_COLS], train[M3_TARGET])
    yearly = (pd.DataFrame({"year": test.year, "actual": test[M3_TARGET],
                            "pred": exam_model.predict(test[INPUT_COLS])}).groupby("year").mean())

    # production model: fit on all years
    model = M3Model(**spec).fit(df[INPUT_COLS], df[M3_TARGET])
    all_pred = model.predict(df[INPUT_COLS])

    checks = [
        G.champion_rule(summary["champion"], champ, summary["reason"]),
        G.beats_baseline(float(final["skill_vs_zero_pct"]), "every month is normal"),
        G.hit_rate(yearly["actual"], yearly["pred"], G.TOLERANCE_M3_DEGC, "degC",
                   label="India-yearly values"),
        G.direction_hit_rate(yearly["actual"], yearly["pred"]),
        G.no_missing_outputs(all_pred),
        G.physical_range(all_pred, df[M3_TARGET], G.RANGE_MARGIN_M3_DEGC, "degC"),
        G.deterministic(model, df[INPUT_COLS].head(5000)),
    ]
    if not np.isnan(region_skill):
        checks.append(G.hidden_regions(region_skill))
    info = {"champion": champ, "features": INPUT_COLS, "rows_trained": len(df),
            "years": f"{df.year.min()}-{df.year.max()}",
            "final_exam_skill_pct": float(final["skill_vs_zero_pct"]),
            "final_exam_india_yearly_mae": float(final["india_yearly_mae"]),
            "data_path": str(data_path), "input_example": df[INPUT_COLS].head(3),
            "code_paths": [str(SRC / "climate_models")]}   # ship the class with the model
    return model, checks, info


# ====================================================================== register
def register(key, model, checks, info):
    name = REGISTERED_NAMES[key]
    print(f"\n=== {name}: champion '{info['champion']}' retrained on {info['rows_trained']:,} rows "
          f"({info['years']}) ===")
    all_passed, table = G.report(checks)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REGISTRY_DIR / f"{name}_gates.csv", index=False)

    mlflow.set_experiment(EXPERIMENTS[key])
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    with mlflow.start_run(run_name=f"register_{name}_{stamp}") as run:
        mlflow.set_tags({"kind": "registration", "champion": info["champion"],
                         "git_commit": git_commit(), "data_md5": md5(info["data_path"]),
                         "gates_passed": str(all_passed).lower(),
                         "mlflow.note.content":
                             f"Production candidate: {info['champion']} retrained on all years "
                             f"{info['years']}. Gates: {'all passed' if all_passed else 'FAILED'}."})
        mlflow.log_params({"champion": info["champion"], "rows_trained": info["rows_trained"],
                           "years": info["years"], "features": ",".join(info["features"])})
        mlflow.log_metrics({f"gate_{g['gate']}": float(g["passed"]) for g in checks})
        mlflow.log_metrics({k: v for k, v in info.items()
                            if isinstance(v, float) and k.startswith("final_exam")})
        mlflow.log_artifact(str(REGISTRY_DIR / f"{name}_gates.csv"), artifact_path="gates")
        if not all_passed:
            print(f"Not registered. Gate results logged in run {run.info.run_id}")
            return False

        kw = {"code_paths": info["code_paths"]} if info["code_paths"] else {}
        model_info = mlflow.sklearn.log_model(model, name="model", registered_model_name=name,
                                              input_example=info["input_example"], **kw)
        version = model_info.registered_model_version
        client = mlflow.MlflowClient()
        try:
            client.set_registered_model_alias(name, "champion", version)
            print(f"Registered {name} version {version} with alias 'champion'")
        except Exception as e:     # some servers do not support aliases; fall back to a tag
            client.set_model_version_tag(name, version, "alias", "champion")
            print(f"Registered {name} version {version}; alias not supported ({e.__class__.__name__}), "
                  f"tagged alias=champion instead")
        for k, v in [("champion_model", info["champion"]), ("gates", "all passed"),
                     ("git_commit", git_commit()), ("data_md5", md5(info["data_path"])),
                     ("final_exam_skill_pct", f"{info['final_exam_skill_pct']:.2f}")]:
            client.set_model_version_tag(name, version, k, v)
        return True


def main():
    load_dotenv(dotenv_path=Path(".env"))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = {"m2": prepare_m2, "m3": prepare_m3}
    if which != "all":
        todo = {which: todo[which]}
    ok = True
    for key, prep in todo.items():
        model, checks, info = prep()
        ok = register(key, model, checks, info) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
