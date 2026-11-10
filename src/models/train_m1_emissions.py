"""
train_m1_emissions.py  (put it at src/models/train_m1_emissions.py)

MODEL 1: what drives a country's CO2 emissions?

Plain-English summary
---------------------
We know each country's emissions, population and income per person (GDP per
person) for every year 1965-2022. The "growth" models learn a rule like

    % change in emissions = a * % change in population
                          + b * % change in income per person
                          + drift * number of years

and predict a future year as: the country's LAST KNOWN emissions, grown by that
rule. On the dashboard, the user supplies future population and income paths.

How models are judged
---------------------
Three "exams" (walk-forward testing): train only on years up to 1990, 2000 or
2010, then predict the next 12 years. The main score is SKILL VS STAYS-THE-SAME:
how much smaller a model's error is than simply assuming emissions stay at their
last known level. Positive = the model wins.

Champion rule (fixed in advance)
--------------------------------
The reference model is growth_linear_5yr. A challenger replaces it only if its
average skill is more than 5 percentage points higher AND it wins at least 2 of
the 3 exams. Otherwise the simpler linear model stays.

MLflow layout
-------------
Every run of this script = one SESSION run (parent) with one child run per model.
Each child has a plain-English Description, the dataset, the saved model (with
input example), its predictions and a graph. The session run holds the comparison
graphs, the full score table and the champion decision. The same files are saved
to reports/models/m1/ for the results notebook.

Run from the repo root:  python3 src/models/train_m1_emissions.py
Input: data/model_ready/m1_emissions.csv
"""

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ======================================================================== settings
DATA = Path("data/model_ready/m1_emissions.csv")
REPORT_DIR = Path("reports/models/m1")
EXPERIMENT = "m1_drivers_to_emissions"
TARGET = "log_co2"
POP, INCOME = "log_population", "log_gdp_per_capita"
EXAM_CUTOFFS = [1990, 2000, 2010]      # train on years <= cutoff, test the next 12 years
HORIZON = 12
HORIZON_BANDS = {"1-3 yrs": (1, 3), "4-8 yrs": (4, 8), "9-12 yrs": (9, 12)}
REFERENCE = "growth_linear_5yr"
CHALLENGER_MARGIN = 5.0                # percentage points of average skill
SEED = 42

DESCRIPTIONS = {
    "baseline_stays_same": "Baseline. Each country's emissions stay at their last known level. "
                           "Every model must beat this.",
    "baseline_global_average": "Baseline (floor). One average emissions value for every country.",
    "level_pop_income": "Predicts emissions directly from population and income per person (logs). "
                        "Explains differences BETWEEN countries; does not use a country's own history.",
    "level_pop_income_energy": "Same as level_pop_income plus energy used per unit of GDP. Scores high "
                               "mostly because population x income x energy-per-GDP = total energy, "
                               "and CO2 follows energy almost by accounting.",
    "growth_linear_1yr": "Linear rule learned from 1-year changes: % change in emissions = "
                         "a x % change in population + b x % change in income + drift x years. "
                         "Predicts from each country's last known emissions.",
    "growth_linear_5yr": "Same linear rule, learned from 5-year changes (less year-to-year noise). "
                         "Reference model for the champion rule.",
    "growth_gbm": "Gradient boosting (many small decision trees) learning how emissions change "
                  "from population change, income change, number of years and the country's income "
                  "level, so it can learn that rich and poor countries react differently.",
    "growth_avg_linear_gbm": "Average of growth_linear_5yr and growth_gbm predictions.",
    "growth_linear_plus_gbm_fix": "growth_linear_5yr first; then a small gradient-boosting model "
                                  "learns only the linear rule's typical mistakes and corrects them.",
}
GROWTH_MODELS = ["growth_linear_1yr", "growth_linear_5yr", "growth_gbm",
                 "growth_avg_linear_gbm", "growth_linear_plus_gbm_fix"]


# ======================================================================== helpers
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def typical_pct_error(y, p):
    """Usual size of a miss, in %, on the original (not log) scale."""
    return float(np.exp(mean_absolute_error(y, p)) - 1) * 100


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def describe_run(name, session, kind):
    """Tags shown on every run, including the plain-English Description."""
    mlflow.set_tags({"session": session, "kind": kind, "git_commit": git_commit(),
                     "data_file": str(DATA), "data_md5": hashlib.md5(DATA.read_bytes()).hexdigest(),
                     "mlflow.note.content": DESCRIPTIONS.get(name, "")})


def log_dataset(df):
    ds = mlflow.data.from_pandas(df, source=str(DATA), targets=TARGET, name="m1_emissions")
    mlflow.log_input(ds, context="training")


def metric_key(name):
    return name.replace(" ", "_").replace("-", "_")


# ======================================================================== growth models
def change_pairs(df, spans):
    """For each country, all pairs of years `span` apart: changes in population,
    income and emissions over that span, plus income level at the start."""
    base = df[["country", "year", POP, INCOME, TARGET]]
    out = []
    for s in spans:
        end = base.assign(start_year=base["year"] - s)
        m = end.merge(base, left_on=["country", "start_year"], right_on=["country", "year"],
                      suffixes=("", "_start"))
        out.append(pd.DataFrame({
            "span": s,
            "d_pop": m[POP] - m[f"{POP}_start"],
            "d_income": m[INCOME] - m[f"{INCOME}_start"],
            "income_start": m[f"{INCOME}_start"],
            "d_co2": m[TARGET] - m[f"{TARGET}_start"]}))
    return pd.concat(out, ignore_index=True)


def linear_rule(step, train):
    p = change_pairs(train, [step])
    m = LinearRegression().fit(p[["d_pop", "d_income"]], p["d_co2"])
    return {"model": m, "a_pop": m.coef_[0], "b_income": m.coef_[1], "drift": m.intercept_ / step}


def linear_predict(rule, X):
    return rule["a_pop"] * X["d_pop"] + rule["b_income"] * X["d_income"] + rule["drift"] * X["span"]


def gbm(min_leaf=50, iters=300):
    return HistGradientBoostingRegressor(max_iter=iters, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=min_leaf, random_state=SEED)


GBM_COLS = ["span", "d_pop", "d_income", "income_start"]


def fit_growth(name, train):
    """Fit one growth model on training years. Returns a predict(X) function and parts to save."""
    if name == "growth_linear_1yr":
        r = linear_rule(1, train)
        return (lambda X: linear_predict(r, X)), r["model"], r
    if name == "growth_linear_5yr":
        r = linear_rule(5, train)
        return (lambda X: linear_predict(r, X)), r["model"], r
    pairs = change_pairs(train, range(1, HORIZON + 1))
    if name == "growth_gbm":
        g = gbm().fit(pairs[GBM_COLS], pairs["d_co2"])
        return (lambda X: g.predict(X[GBM_COLS])), g, None
    r = linear_rule(5, train)
    if name == "growth_avg_linear_gbm":
        g = gbm().fit(pairs[GBM_COLS], pairs["d_co2"])
        return (lambda X: 0.5 * linear_predict(r, X) + 0.5 * g.predict(X[GBM_COLS])), g, r
    if name == "growth_linear_plus_gbm_fix":
        residual = pairs["d_co2"] - linear_predict(r, pairs)
        g = gbm(min_leaf=100, iters=150).fit(pairs[GBM_COLS], residual)
        return (lambda X: linear_predict(r, X) + g.predict(X[GBM_COLS])), g, r
    raise ValueError(name)


def exam(name, df, cutoff):
    """Train on years <= cutoff; predict the next HORIZON years from each country's
    last known emissions. Returns predictions with 'stays the same' on the same rows."""
    train = df[df.year <= cutoff]
    test = df[(df.year > cutoff) & (df.year <= cutoff + HORIZON)]
    predict, model, rule = fit_growth(name, train)
    last = train.sort_values("year").groupby("country").last()[[POP, INCOME, TARGET, "year"]]
    t = test.join(last, on="country", rsuffix="_last").dropna(subset=[f"{TARGET}_last"]).copy()
    X = pd.DataFrame({"span": t["year"] - t["year_last"],
                      "d_pop": t[POP] - t[f"{POP}_last"],
                      "d_income": t[INCOME] - t[f"{INCOME}_last"],
                      "income_start": t[f"{INCOME}_last"]}, index=t.index)
    t["years_ahead"] = X["span"]
    t["actual_change"] = t[TARGET] - t[f"{TARGET}_last"]
    t["predicted_change"] = np.asarray(predict(X))
    t["predicted"] = t[f"{TARGET}_last"] + t["predicted_change"]
    t["stays_same"] = t[f"{TARGET}_last"]
    return t, model, rule


def exam_scores(t):
    s = {"rmse": rmse(t[TARGET], t["predicted"]),
         "stays_same_rmse": rmse(t[TARGET], t["stays_same"]),
         "typical_pct_error": typical_pct_error(t[TARGET], t["predicted"]),
         "stays_same_typical_pct_error": typical_pct_error(t[TARGET], t["stays_same"]),
         "change_r2": r2_score(t["actual_change"], t["predicted_change"]),
         "rows": len(t)}
    s["skill_vs_stays_same_pct"] = (1 - s["rmse"] / s["stays_same_rmse"]) * 100
    for band, (lo, hi) in HORIZON_BANDS.items():
        b = t[t["years_ahead"].between(lo, hi)]
        s[f"skill_{band}"] = (1 - rmse(b[TARGET], b["predicted"]) / rmse(b[TARGET], b["stays_same"])) * 100
    return s


def run_growth_model(name, df, session, results, india):
    with mlflow.start_run(run_name=name, nested=True):
        describe_run(name, session, "growth_model")
        mlflow.log_params({"model": name, "exam_cutoffs": ",".join(map(str, EXAM_CUTOFFS)),
                           "years_predicted_per_exam": HORIZON})
        log_dataset(df)
        preds = []
        for cutoff in EXAM_CUTOFFS:
            t, model, rule = exam(name, df, cutoff)
            s = exam_scores(t)
            results.append({"model": name, "exam": cutoff, **s})
            mlflow.log_metrics({f"exam{cutoff}_{metric_key(k)}": v for k, v in s.items()})
            if rule is not None:
                mlflow.log_metrics({f"exam{cutoff}_rule_population_a": float(rule["a_pop"]),
                                    f"exam{cutoff}_rule_income_b": float(rule["b_income"]),
                                    f"exam{cutoff}_rule_drift_pct_per_year":
                                        float(np.exp(rule["drift"]) - 1) * 100})
            preds.append(t.assign(exam=cutoff)[["exam", "country", "year", "years_ahead", TARGET,
                                                "predicted", "stays_same"]])
            if cutoff == EXAM_CUTOFFS[-1]:
                india[name] = t[t.country == "India"].sort_values("year")
                final_model = model
        mean_skill = float(np.mean([r["skill_vs_stays_same_pct"] for r in results if r["model"] == name]))
        mlflow.log_metric("average_skill_vs_stays_same_pct", mean_skill)

        preds = pd.concat(preds).rename(columns={TARGET: "actual"})
        with tempfile.TemporaryDirectory() as d:
            preds.to_csv(f"{d}/predictions_all_exams.csv", index=False)
            last = preds[preds.exam == EXAM_CUTOFFS[-1]]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(last["actual"], last["predicted"], s=5, alpha=0.4)
            lo, hi = last["actual"].min(), last["actual"].max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel("actual log(CO2)")
            ax.set_ylabel("predicted log(CO2)")
            ax.set_title(f"{name}\nexam 2011-2022 (dots on dashed line = perfect)", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{d}/actual_vs_predicted_2011_2022.png", dpi=110)
            plt.close(fig)
            mlflow.log_artifacts(d, artifact_path="results")

        if isinstance(final_model, LinearRegression):
            example = pd.DataFrame({"d_pop": [0.05, 0.10], "d_income": [0.20, 0.35]})
        else:
            example = pd.DataFrame({"span": [5, 10], "d_pop": [0.05, 0.10],
                                    "d_income": [0.20, 0.35], "income_start": [8.9, 8.9]})
        mlflow.sklearn.log_model(final_model, name="model", input_example=example)
        print(f"  {name:27s} skill by exam: "
              + ", ".join(f"{r['exam']}: {r['skill_vs_stays_same_pct']:+.1f}%"
                          for r in results if r["model"] == name)
              + f" | average {mean_skill:+.1f}%")
        return mlflow.active_run().info.run_id


# ======================================================================== level models + baselines
def run_level_and_baselines(df, session, results):
    """Scored on the latest exam only, for context."""
    cutoff = EXAM_CUTOFFS[-1]
    train, test = df[df.year <= cutoff], df[(df.year > cutoff) & (df.year <= cutoff + HORIZON)]
    last = train.sort_values("year").groupby("country")[TARGET].last()
    test = test[test.country.isin(last.index)]
    stays_same = test.country.map(last)
    base_rmse = rmse(test[TARGET], stays_same)
    level_sets = {"level_pop_income": [POP, INCOME],
                  "level_pop_income_energy": [POP, INCOME, "log_energy_per_gdp"]}
    preds = {"baseline_stays_same": stays_same,
             "baseline_global_average": pd.Series(train[TARGET].mean(), index=test.index)}
    fitted = {}
    for name, feats in level_sets.items():
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        m.fit(train[feats], train[TARGET])
        preds[name] = pd.Series(m.predict(test[feats]), index=test.index)
        fitted[name] = (m, feats)
    for name, p in preds.items():
        with mlflow.start_run(run_name=name, nested=True):
            describe_run(name, session, "baseline" if name.startswith("baseline") else "level_model")
            log_dataset(df)
            s = {"rmse": rmse(test[TARGET], p), "r2": r2_score(test[TARGET], p),
                 "typical_pct_error": typical_pct_error(test[TARGET], p)}
            s["skill_vs_stays_same_pct"] = (1 - s["rmse"] / base_rmse) * 100
            mlflow.log_metrics({f"exam{cutoff}_{k}": v for k, v in s.items()})
            if name in fitted:
                m, feats = fitted[name]
                mlflow.log_param("features", ",".join(feats))
                for f, c in zip(feats, m[-1].coef_ / m[0].scale_):
                    mlflow.log_metric(f"effect_of_{f}", float(c))
                mlflow.sklearn.log_model(m, name="model", input_example=train[feats].head(3))
            results.append({"model": name, "exam": cutoff, **s})
            print(f"  {name:27s} exam {cutoff}: R2 {s['r2']:.3f}, "
                  f"skill vs stays-the-same {s['skill_vs_stays_same_pct']:+.1f}%")


# ======================================================================== comparison + champion
def label_bars(ax, bars, fmt):
    ax.bar_label(bars, labels=[fmt.format(v) for v in bars.datavalues], fontsize=7, padding=2)


def make_comparison(df, results, india, champion, reason, out):
    res = pd.DataFrame(results)
    res.to_csv(out / "comparison.csv", index=False)
    g = res[res.model.isin(GROWTH_MODELS)]
    x = np.arange(len(EXAM_CUTOFFS))
    w = 0.8 / len(GROWTH_MODELS)
    exam_labels = [f"exam: train <= {c}\npredict {c + 1}-{c + HORIZON}" for c in EXAM_CUTOFFS]

    for fname, col, ylabel, title, fmt in [
        ("01_skill_by_exam.png", "skill_vs_stays_same_pct", "% smaller error than 'stays the same'",
         "1. Skill in each exam (above 0 = beats 'stays the same')", "{:.0f}"),
        ("03_change_r2.png", "change_r2", "share of real change explained (0 = none)",
         "3. How much of the real rise or fall in emissions each model explains", "{:.2f}")]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for i, m in enumerate(GROWTH_MODELS):
            bars = ax.bar(x + i * w, g[g.model == m][col], w, label=m)
            label_bars(ax, bars, fmt)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x + w * (len(GROWTH_MODELS) - 1) / 2)
        ax.set_xticklabels(exam_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in GROWTH_MODELS:
        r = g[g.model == m]
        ax.plot(list(HORIZON_BANDS), [r[f"skill_{b}"].mean() for b in HORIZON_BANDS], marker="o", label=m)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("how many years ahead the prediction is")
    ax.set_ylabel("% smaller error than 'stays the same'")
    ax.set_title("2. Skill by how far ahead (average of the 3 exams)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out / "02_skill_by_years_ahead.png", dpi=120)
    plt.close(fig)

    actual = df[df.country == "India"].sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(actual["year"], actual["co2"], color="black", lw=2.5, label="actual")
    first = next(iter(india.values()))
    ax.plot(first["year"], np.exp(first["stays_same"]), ls="--", color="grey",
            label="stays the same (2010 level)")
    for m, p in india.items():
        ax.plot(p["year"], np.exp(p["predicted"]), lw=1.5, label=m)
    ax.axvline(EXAM_CUTOFFS[-1] + 0.5, color="grey", ls=":")
    ax.set_xlim(1995, 2023)
    ax.set_ylabel("CO2 emissions (million tonnes)")
    ax.set_title("4. India: predictions for 2011-2022 using data up to 2010")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out / "04_india_2011_2022.png", dpi=120)
    plt.close(fig)

    summary = {"champion": champion, "reason": reason, "reference_model": REFERENCE,
               "rule": f"challenger needs average skill > reference + {CHALLENGER_MARGIN:.0f} points "
                       f"and must win at least 2 of 3 exams",
               "average_skill": {m: float(g[g.model == m]["skill_vs_stays_same_pct"].mean())
                                 for m in GROWTH_MODELS},
               "descriptions": DESCRIPTIONS}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def pick_champion(results):
    r = pd.DataFrame(results)
    skill = r[r.model.isin(GROWTH_MODELS)].pivot(index="exam", columns="model",
                                                 values="skill_vs_stays_same_pct")
    ref = skill[REFERENCE]
    lines, passed = [], []
    for m in GROWTH_MODELS:
        if m in (REFERENCE, "growth_linear_1yr"):
            continue
        margin = skill[m].mean() - ref.mean()
        wins = int((skill[m] > ref).sum())
        ok = margin > CHALLENGER_MARGIN and wins >= 2
        lines.append(f"{m}: {margin:+.1f} pts, wins {wins}/3 -> {'PASSES' if ok else 'fails'}")
        if ok:
            passed.append((margin, m))
    if passed:
        best = max(passed)[1]
        return best, f"{best} passes the rule. " + "; ".join(lines)
    return REFERENCE, f"No challenger passes the rule, so {REFERENCE} stays. " + "; ".join(lines)


# ======================================================================== main
def main():
    load_dotenv(dotenv_path=Path(".env"))
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("WARNING: MLFLOW_TRACKING_URI not set; logging to local ./mlruns instead of DagsHub")
    mlflow.set_experiment(EXPERIMENT)
    session = "m1_session_" + datetime.now().strftime("%Y-%m-%d_%H%M")
    df = pd.read_csv(DATA)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"MLflow: {mlflow.get_tracking_uri()} | experiment {EXPERIMENT} | session {session}")
    print(f"Data: {len(df):,} rows, {df['country'].nunique()} countries\n")

    results, india, run_ids = [], {}, {}
    with mlflow.start_run(run_name=session):
        mlflow.set_tags({"session": session, "kind": "session", "git_commit": git_commit(),
                         "mlflow.note.content": "One training session of Model 1. Expand this row "
                         "to see every model. Comparison graphs and the champion decision are in "
                         "Artifacts > comparison."})
        log_dataset(df)
        mlflow.log_params({"exam_cutoffs": ",".join(map(str, EXAM_CUTOFFS)),
                           "years_predicted_per_exam": HORIZON, "reference_model": REFERENCE,
                           "champion_rule": f"average skill > reference + {CHALLENGER_MARGIN:.0f} pts "
                                            f"and wins >= 2 of 3 exams"})
        print("Baselines and level models (latest exam only):")
        run_level_and_baselines(df, session, results)
        print("\nGrowth models (3 exams):")
        for name in GROWTH_MODELS:
            run_ids[name] = run_growth_model(name, df, session, results, india)

        champion, reason = pick_champion(results)
        summary = make_comparison(df, results, india, champion, reason, REPORT_DIR)
        mlflow.log_params({"champion": champion, "champion_reason": reason[:500]})
        mlflow.log_metrics({f"average_skill_{k}": v for k, v in summary["average_skill"].items()})
        mlflow.log_artifacts(str(REPORT_DIR), artifact_path="comparison")
        mlflow.MlflowClient().set_tag(run_ids[champion], "champion", "true")

    print(f"\nChampion: {champion}\n{reason.replace('; ', chr(10))}")
    print(f"Graphs, comparison.csv and summary.json: {REPORT_DIR}/ (also in the session run)")


if __name__ == "__main__":
    main()
