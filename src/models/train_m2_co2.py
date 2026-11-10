"""Model 2: human emissions -> yearly rise in atmospheric CO2 (ppm).

Run from the repo root:  python src/models/train_m2_co2.py
Reads   data/model_ready/m2_co2_growth.csv   (65 rows, 1960-2024)
Writes  reports/models/m2/  (graphs, comparison.csv, summary.json) and MLflow runs.

Plain-English idea: each year humans emit CO2, part of it stays in the air.
The model learns "ppm rise = share x emissions (+ El Nino effect)".
Judged by three walk-forward exams, as in Model 1.
"""
import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

import datetime
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/model_ready/m2_co2_growth.csv")
OUT = Path("reports/models/m2")
EXPERIMENT = "m2_emissions_to_co2"
TARGET = "growth_ppm"
YEAR = "year"

# Walk-forward exams: (last training year, last test year).
# Third exam = the final holdout (train <= 2009, test 2010-2024).
EXAMS = [(1990, 2002), (2000, 2012), (2009, 2024)]

# Champion rule, fixed in advance (same spirit as Model 1):
REFERENCE = "linear_emissions"
MIN_EXTRA_SKILL = 0.05   # challenger needs average skill > reference + 5 points
MIN_EXAM_WINS = 2        # and must beat the reference in at least 2 of 3 exams

# Unit check: ppm conversion only holds if total_emissions is in GtC/yr.
# VERIFY this in src/features/features_m2_co2.py before quoting the share.
EMISSIONS_UNIT = "GtC/yr (assumed, verify)"
GTC_PER_PPM = 2.124      # from memory in the handoff; matches Mauna Loa closely

MODELS = {
    "same_as_last_year": dict(
        kind="persistence", features=[], eligible=False,
        desc="Baseline. Predicts that this year's CO2 rise equals last year's actual rise."),
    "average_growth": dict(
        kind="mean", features=[], eligible=False,
        desc="Baseline. Predicts the average yearly CO2 rise seen in the training years."),
    "linear_emissions": dict(
        kind="linear", features=["total_emissions"], eligible=True,
        desc="Linear model: CO2 rise = share of emissions that stays in the air. "
             "Reference model for the champion rule."),
    "linear_nino": dict(
        kind="linear", features=["total_emissions", "nino34", "nino34_lag1"], eligible=True,
        desc="Linear model: emissions plus El Nino (same year and year before)."),
    "linear_nino_volcano": dict(
        kind="linear",
        features=["total_emissions", "nino34", "nino34_lag1", "volcano_flag"],
        eligible=False,
        desc="Explains the past only. Adds a volcano-year flag, which cannot be "
             "known in advance, so it is not allowed to be champion."),
    "ridge_nino": dict(
        kind="ridge", features=["total_emissions", "nino34", "nino34_lag1"], eligible=True,
        desc="Same inputs as linear_nino, but Ridge shrinks the weights slightly. "
             "Checks whether that helps with only 65 rows."),
    "gbm_nino": dict(
        kind="gbm", features=["total_emissions", "nino34", "nino34_lag1"], eligible=True,
        desc="Gradient boosting (trees). Trees cannot predict beyond the highest emissions "
             "seen in training, so it is expected to fall behind as emissions grow."),
    "share_no_intercept": dict(
        kind="no_intercept", features=["total_emissions"], eligible=True,
        desc="Physics form: CO2 rise = share x emissions, no constant. Weight x 2.124 "
             "= airborne share, comparable with the Global Carbon Budget."),
}


def load_env():
    """Load MLFLOW_* variables from .env (python-dotenv if present, else by hand)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def scores(y, pred, y_persist):
    mae = mean_absolute_error(y, pred)
    mae_p = mean_absolute_error(y, y_persist)
    return dict(
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        r2=r2_score(y, pred),
        skill=1 - mae / mae_p,   # 1 = perfect, 0 = same as "last year", <0 = worse
    )


def predict(name, spec, train, test, persist_all):
    """Fit on train, predict test. Returns (predictions, fitted model or None)."""
    if spec["kind"] == "persistence":
        return persist_all.loc[test.index].values, None
    if spec["kind"] == "mean":
        return np.full(len(test), train[TARGET].mean()), None
    if spec["kind"] == "ridge":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    elif spec["kind"] == "gbm":
        model = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                          learning_rate=0.05, random_state=0)
    else:
        model = LinearRegression(fit_intercept=spec["kind"] != "no_intercept")
    model.fit(train[spec["features"]], train[TARGET])
    return model.predict(test[spec["features"]]), model


def main():
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    needed = {YEAR, TARGET, "total_emissions", "nino34", "nino34_lag1", "volcano_flag"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Columns not found in {DATA}: {sorted(missing)}. "
                         f"Found: {list(df.columns)}. Fix the names at the top of the script.")
    # Never fill: rows with a missing value in any column we use are dropped.
    before = len(df)
    df = df.dropna(subset=sorted(needed)).sort_values(YEAR).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after dropping rows with missing values")

    persist_all = df[TARGET].shift(1)  # last year's actual rise

    # ---- evaluate every model on every exam --------------------------------
    rows, final_preds, fitted = [], {}, {}
    for name, spec in MODELS.items():
        for i, (train_end, test_end) in enumerate(EXAMS, start=1):
            train = df[df[YEAR] <= train_end]
            test = df[(df[YEAR] > train_end) & (df[YEAR] <= test_end)]
            pred, model = predict(name, spec, train, test, persist_all)
            s = scores(test[TARGET].values, pred, persist_all.loc[test.index].values)
            rows.append(dict(model=name, exam=i, train_end=train_end,
                             test_years=f"{test[YEAR].min()}-{test[YEAR].max()}",
                             n_test=len(test), **s))
            if i == len(EXAMS):
                final_preds[name] = pd.DataFrame(
                    {YEAR: test[YEAR].values, "actual": test[TARGET].values, "predicted": pred})
                fitted[name] = (model, train, spec)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "comparison_long.csv", index=False)

    summary = (results.pivot(index="model", columns="exam", values="skill")
               .rename(columns=lambda c: f"skill_exam{c}"))
    summary["avg_skill"] = summary.mean(axis=1)
    summary = summary.loc[list(MODELS)]
    summary.to_csv(OUT / "comparison.csv")
    print("\nSkill vs 'same as last year' (higher is better, 0 = no better):")
    print((summary * 100).round(1).to_string())

    # ---- champion rule ------------------------------------------------------
    ref = summary.loc[REFERENCE]
    champion, reason = REFERENCE, "no challenger met the rule, so the reference stays"
    best = ref["avg_skill"]
    for name, spec in MODELS.items():
        if not spec["eligible"] or name == REFERENCE:
            continue
        row = summary.loc[name]
        wins = sum(row[f"skill_exam{i}"] > ref[f"skill_exam{i}"] for i in (1, 2, 3))
        ok = row["avg_skill"] > ref["avg_skill"] + MIN_EXTRA_SKILL and wins >= MIN_EXAM_WINS
        if ok and row["avg_skill"] > best:
            champion, best = name, row["avg_skill"]
            reason = (f"average skill {row['avg_skill']*100:.1f}% vs reference "
                      f"{ref['avg_skill']*100:.1f}% (needs > +5 points) and wins {wins} of 3 exams")
    print(f"\nChampion: {champion} ({reason})")

    # ---- fitted coefficients (final exam, train <= 2009) --------------------
    coefs = {}
    for name, (model, train, spec) in fitted.items():
        if model is not None and hasattr(model, "coef_"):
            coefs[name] = dict(zip(spec["features"], map(float, model.coef_)))
            coefs[name]["intercept"] = float(model.intercept_)
    share = coefs[REFERENCE]["total_emissions"]
    implied_airborne = share * GTC_PER_PPM  # only valid if emissions are in GtC/yr
    print(f"Fitted ppm per unit emissions: {share:.4f}; implied airborne share "
          f"if GtC/yr: {implied_airborne:.2f} (compare with the budget's ~0.42-0.44; verify)")
    s0 = coefs["share_no_intercept"]["total_emissions"]
    print(f"No-intercept share: {s0:.4f} ppm per unit -> airborne share if GtC/yr: "
          f"{s0 * GTC_PER_PPM:.2f}")
    

    # ---- graphs -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    base = final_preds[champion]
    ax.plot(base[YEAR], base["actual"], "k-o", label="actual")
    for name in ("same_as_last_year", "average_growth", champion):
        ax.plot(final_preds[name][YEAR], final_preds[name]["predicted"], "--", label=name)
    ax.set_title("Final exam (train <= 2009): yearly CO2 rise, actual vs predicted")
    ax.set_xlabel("year"); ax.set_ylabel("ppm per year"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "m2_final_exam.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    (summary[[f"skill_exam{i}" for i in (1, 2, 3)]] * 100).plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("skill vs 'same as last year' (%)")
    ax.set_title("Skill in each walk-forward exam")
    fig.tight_layout(); fig.savefig(OUT / "m2_skill_by_exam.png", dpi=130); plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(dict(
        champion=champion, reason=reason, emissions_unit=EMISSIONS_UNIT,
        implied_airborne_share_if_gtc=implied_airborne, coefficients=coefs,
        avg_skill={k: float(v) for k, v in summary["avg_skill"].items()},
        exams=EXAMS, note="volcano years are from memory: verify before reporting"),
        indent=2))

    # ---- MLflow -------------------------------------------------------------
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI not set: skipping MLflow logging.")
        return
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    dataset = mlflow.data.from_pandas(df, source=str(DATA), name="m2_co2_growth")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    with mlflow.start_run(run_name=f"m2_session_{stamp}") as parent:
        for name, spec in MODELS.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("mlflow.note.content", spec["desc"])
                mlflow.set_tag("champion", str(name == champion).lower())
                mlflow.log_param("features", ",".join(spec["features"]) or "none")
                mlflow.log_param("exams", str(EXAMS))
                mlflow.log_input(dataset, context="training")
                for i in (1, 2, 3):
                    r = results[(results.model == name) & (results.exam == i)].iloc[0]
                    for m in ("mae", "rmse", "r2", "skill"):
                        mlflow.log_metric(f"{m}_exam{i}", float(r[m]))
                mlflow.log_metric("avg_skill", float(summary.loc[name, "avg_skill"]))
                final_preds[name].to_csv(OUT / f"pred_{name}.csv", index=False)
                mlflow.log_artifact(str(OUT / f"pred_{name}.csv"))
                model, train, _ = fitted[name]
                if model is not None:
                    mlflow.sklearn.log_model(model, name="model",
                                             input_example=train[spec["features"]].head(3))
        mlflow.log_artifacts(str(OUT))
        mlflow.set_tag("champion_model", champion)
        mlflow.set_tag("champion_reason", reason)
        print(f"MLflow parent run: {parent.info.run_id}")


if __name__ == "__main__":
    main()
