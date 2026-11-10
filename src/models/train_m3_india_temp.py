"""
train_m3_india_temp.py  (put it at src/models/train_m3_india_temp.py)

MODEL 3: how does rising CO2 change temperature in each part of India?

Plain-English summary
---------------------
One row = one 1-degree square of India in one month (344 squares, 1951-2024).
The target is the temperature anomaly: how much warmer or cooler that square was
than its own 1961-1990 normal for that month (tmean_anom, degC).

Inputs
- natural: El Nino in the previous 1, 3, 6 months and the previous 12-month average
- place and season: latitude, longitude, month (as sin/cos)
- human: CO2 in the air as radiative forcing (co2_forcing, W/m2) or
         cumulative world emissions (cum_emissions_gtco2)

Models
- baseline_zero:          every month is normal (anomaly 0). Score reference.
- baseline_own_trend:     each square's own straight line over time, extended forward.
                          Strong, but cannot answer "what if CO2 changes".
- natural_only:           El Nino + place + season, no CO2. The "no humans" check.
- linear_co2:             + CO2 forcing. Reference model for the champion rule.
- linear_cum_emissions:   + cumulative emissions instead of CO2 forcing.
- linear_co2_regional:    CO2 effect allowed to differ by place and season.
- linear_co2_per_square:  each square gets its own CO2 sensitivity.
- hybrid_regional_gbm:    linear_co2_regional, then gradient boosting learns the
                          leftover El Nino / place / season patterns. CO2 stays in the
                          linear part, because trees cannot go beyond the CO2 seen in
                          training (shown in Model 2).

How models are judged
---------------------
Three walk-forward exams: train on years <= 1990, 2000, 2010 and predict 1991-2002,
2001-2012, 2011-2024. Main score: SKILL VS ZERO = % smaller RMSE than "every month is
normal". Also reported: skill vs each square's own trend, all-India yearly average
error, and the warming trend per decade (actual vs predicted).
Region holdout (final exam period): whole 5x5-degree blocks of squares are hidden
from training, to check the model works in places it has not seen.

Champion rule (fixed in advance)
--------------------------------
Reference: linear_co2. A challenger replaces it only if its average skill is more
than 5 points higher AND it wins at least 2 of 3 exams. Baselines and natural_only
cannot be champion (they cannot run CO2 scenarios).

Run from the repo root:  python3 src/models/train_m3_india_temp.py
Input: data/model_ready/m3_india_temp.csv
"""
import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================================== settings
DATA = Path("data/model_ready/m3_india_temp.csv")
REPORT_DIR = Path("reports/models/m3")
EXPERIMENT = "m3_india_temperature"
TARGET = "tmean_anom"
EXAMS = [(1990, 2002), (2000, 2012), (2010, 2024)]   # (last training year, last test year)
FINAL_CUTOFF = EXAMS[-1][0]
REFERENCE = "linear_co2"
CHALLENGER_MARGIN = 5.0
SEED = 42
N_REGION_FOLDS = 5
BLOCK_DEG = 5

NINO = ["nino_lag1", "nino_lag3", "nino_lag6", "nino_prev12_mean"]
PLACE_SEASON = ["lat", "lon", "month_sin", "month_cos"]
GBM_COLS = NINO + PLACE_SEASON
INPUT_COLS = ["cell_id", "year", "month"] + PLACE_SEASON + NINO + ["co2_forcing", "cum_emissions_gtco2"]

MODELS = {
    "baseline_zero": dict(kind="zero"),
    "baseline_own_trend": dict(kind="own_trend"),
    "natural_only": dict(kind="linear", co2_col=None),
    "linear_co2": dict(kind="linear", co2_col="co2_forcing"),
    "linear_cum_emissions": dict(kind="linear", co2_col="cum_emissions_gtco2"),
    "linear_co2_regional": dict(kind="linear", co2_col="co2_forcing", regional=True),
    "linear_co2_per_square": dict(kind="linear", co2_col="co2_forcing", per_square=True),
    "hybrid_regional_gbm": dict(kind="linear", co2_col="co2_forcing", regional=True,
                                residual_gbm=True),
}
ELIGIBLE = ["linear_cum_emissions", "linear_co2_regional", "linear_co2_per_square",
            "hybrid_regional_gbm"]
# Models that do not need a square's own history can be tested on hidden regions.
REGION_MODELS = ["baseline_zero", "natural_only", "linear_co2", "linear_cum_emissions",
                 "linear_co2_regional", "hybrid_regional_gbm"]

DESCRIPTIONS = {
    "baseline_zero": "Baseline. Predicts every month is normal (anomaly 0). The skill score "
                     "measures how much better than this each model is.",
    "baseline_own_trend": "Baseline. Each square's own straight line over time, fitted on training "
                          "years and extended forward. Strong, but it cannot answer 'what if CO2 "
                          "changes', so it cannot be champion.",
    "natural_only": "Natural factors only: El Nino (previous 1, 3, 6 months and 12-month average), "
                    "place and season. No CO2. If this misses the warming and the CO2 models catch "
                    "it, that is our own evidence of the human cause.",
    "linear_co2": "Natural factors plus CO2 in the air (radiative forcing, 5.35 x ln(CO2/280)). "
                  "One CO2 effect for all of India. Reference model for the champion rule.",
    "linear_cum_emissions": "Same as linear_co2 but uses total human CO2 emitted since 1751 (world, "
                            "GtCO2) instead of CO2 in the air.",
    "linear_co2_regional": "linear_co2, but the CO2 effect may differ by latitude, longitude and "
                           "season (smooth map), because some regions and months warm faster.",
    "linear_co2_per_square": "linear_co2, then each square gets its own extra CO2 sensitivity, "
                             "learned from its own history.",
    "hybrid_regional_gbm": "linear_co2_regional first; then gradient boosting (small decision trees) "
                           "learns the leftover patterns from El Nino, place and season. CO2 stays "
                           "in the linear part because trees cannot go beyond training CO2 levels.",
}


# ======================================================================== the model
def per_square_line(cells, x, y):
    """Straight line y = intercept + slope * x fitted separately for each square."""
    d = pd.DataFrame({"cell": np.asarray(cells), "x": np.asarray(x, float),
                      "y": np.asarray(y, float)})
    g = d.groupby("cell")
    mx, my = g["x"].mean(), g["y"].mean()
    dx = d["x"] - d["cell"].map(mx)
    dy = d["y"] - d["cell"].map(my)
    sxy = (dx * dy).groupby(d["cell"]).sum()
    sxx = (dx * dx).groupby(d["cell"]).sum()
    slope = (sxy / sxx.where(sxx > 0)).fillna(0.0)
    return pd.DataFrame({"slope": slope, "intercept": my - slope * mx})


def apply_square_line(table, cells, x):
    cells = pd.Series(np.asarray(cells))
    a = cells.map(table["intercept"]).fillna(0.0).to_numpy()
    b = cells.map(table["slope"]).fillna(0.0).to_numpy()
    return a + b * np.asarray(x, float)


def decimal_year(X):
    return X["year"].to_numpy() + (X["month"].to_numpy() - 0.5) / 12


class M3Model(BaseEstimator, RegressorMixin):
    """One class for every Model 3 variant, so each saved model takes the same input
    table (INPUT_COLS) and works the same way in the API later."""

    def __init__(self, kind="linear", co2_col="co2_forcing", regional=False,
                 per_square=False, residual_gbm=False, seed=SEED):
        self.kind = kind
        self.co2_col = co2_col
        self.regional = regional
        self.per_square = per_square
        self.residual_gbm = residual_gbm
        self.seed = seed

    def _design(self, X):
        cols = {c: X[c].to_numpy(float) for c in NINO + PLACE_SEASON}
        if self.co2_col:
            c = X[self.co2_col].to_numpy(float)
            cols[self.co2_col] = c
            if self.regional:
                la = (X["lat"].to_numpy(float) - 23.0) / 10.0
                lo = (X["lon"].to_numpy(float) - 80.0) / 10.0
                cols.update({"co2_x_lat": c * la, "co2_x_lon": c * lo, "co2_x_lat_lon": c * la * lo,
                             "co2_x_lat2": c * la ** 2, "co2_x_lon2": c * lo ** 2,
                             "co2_x_month_sin": c * X["month_sin"].to_numpy(float),
                             "co2_x_month_cos": c * X["month_cos"].to_numpy(float)})
        return pd.DataFrame(cols, index=X.index)

    def fit(self, X, y):
        y = np.asarray(y, float)
        if self.kind == "zero":
            return self
        if self.kind == "own_trend":
            self.square_table_ = per_square_line(X["cell_id"], decimal_year(X), y)
            return self
        D = self._design(X)
        self.feature_names_ = list(D.columns)
        self.linear_ = LinearRegression().fit(D, y)
        resid = y - self.linear_.predict(D)
        if self.per_square:
            self.square_table_ = per_square_line(X["cell_id"], X[self.co2_col], resid)
            resid = resid - apply_square_line(self.square_table_, X["cell_id"], X[self.co2_col])
        if self.residual_gbm:
            self.gbm_ = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                l2_regularization=1.0, random_state=self.seed).fit(X[GBM_COLS], resid)
        return self

    def predict(self, X):
        if self.kind == "zero":
            return np.zeros(len(X))
        if self.kind == "own_trend":
            return apply_square_line(self.square_table_, X["cell_id"], decimal_year(X))
        p = self.linear_.predict(self._design(X))
        if self.per_square:
            p = p + apply_square_line(self.square_table_, X["cell_id"], X[self.co2_col])
        if self.residual_gbm:
            p = p + self.gbm_.predict(X[GBM_COLS])
        return p


# ======================================================================== helpers
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def trend_per_decade(years, values):
    return float(np.polyfit(np.asarray(years, float), np.asarray(values, float), 1)[0] * 10)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


DATA_MD5 = None


def describe_run(name, session, kind):
    mlflow.set_tags({"session": session, "kind": kind, "git_commit": git_commit(),
                     "data_file": str(DATA), "data_md5": DATA_MD5,
                     "mlflow.note.content": DESCRIPTIONS.get(name, "")})


def log_dataset(df):
    ds = mlflow.data.from_pandas(df, source=str(DATA), targets=TARGET, name="m3_india_temp")
    mlflow.log_input(ds, context="training")


def evaluate(test, pred, trend_pred):
    """Scores for one exam. India yearly = plain average over squares (not area-weighted)."""
    y = test[TARGET].to_numpy()
    r, r0, rt = rmse(y, pred), rmse(y, np.zeros_like(y)), rmse(y, trend_pred)
    ann = pd.DataFrame({"year": test["year"].to_numpy(), "a": y, "p": pred}).groupby("year").mean()
    return {"rmse": r, "mae": float(mean_absolute_error(y, pred)),
            "skill_vs_zero_pct": (1 - r / r0) * 100,
            "skill_vs_own_trend_pct": (1 - r / rt) * 100,
            "india_yearly_mae": float(mean_absolute_error(ann["a"], ann["p"])),
            "india_yearly_r2": float(r2_score(ann["a"], ann["p"])),
            "trend_actual_per_decade": trend_per_decade(ann.index, ann["a"]),
            "trend_predicted_per_decade": trend_per_decade(ann.index, ann["p"]),
            "rows": int(len(test)), "years": int(len(ann))}


def region_fold_of_cell(df):
    """Group squares into 5x5-degree blocks; blocks are shared out over the folds at random."""
    cells = df.drop_duplicates("cell_id")[["cell_id", "lat", "lon"]]
    block = (np.floor((cells["lat"] - 8) / BLOCK_DEG).astype(int).astype(str) + "_"
             + np.floor((cells["lon"] - 68) / BLOCK_DEG).astype(int).astype(str))
    blocks = sorted(block.unique())
    order = np.random.default_rng(SEED).permutation(len(blocks))
    fold_of_block = {b: int(order[i] % N_REGION_FOLDS) for i, b in enumerate(blocks)}
    return dict(zip(cells["cell_id"], block.map(fold_of_block)))


def region_holdout(name, df, fold_of_cell):
    fold = df["cell_id"].map(fold_of_cell)
    ys, ps = [], []
    for k in range(N_REGION_FOLDS):
        train = df[(df["year"] <= FINAL_CUTOFF) & (fold != k)]
        test = df[(df["year"] > FINAL_CUTOFF) & (fold == k)]
        m = M3Model(**MODELS[name]).fit(train[INPUT_COLS], train[TARGET])
        ps.append(m.predict(test[INPUT_COLS]))
        ys.append(test[TARGET].to_numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    r = rmse(y, p)
    return {"region_rmse": r, "region_skill_vs_zero_pct": (1 - r / rmse(y, np.zeros_like(y))) * 100}


# ======================================================================== one child run per model
def run_model(name, df, session, results, ctx):
    kind = "baseline" if name.startswith("baseline") else "model"
    with mlflow.start_run(run_name=name, nested=True):
        describe_run(name, session, kind)
        mlflow.log_params({"model": name, "exams": str(EXAMS), **{k: str(v) for k, v in MODELS[name].items()}})
        log_dataset(df)
        for cutoff, end in EXAMS:
            train, test = ctx["splits"][cutoff]
            m = M3Model(**MODELS[name]).fit(train[INPUT_COLS], train[TARGET])
            s = evaluate(test, m.predict(test[INPUT_COLS]), ctx["trend_pred"][cutoff])
            results.append({"model": name, "exam": cutoff, "test_years": f"{cutoff + 1}-{end}", **s})
            mlflow.log_metrics({f"exam{cutoff}_{k}": float(v) for k, v in s.items()})
        final = m  # the loop ends on the final exam (train <= 2010)
        mine = [r for r in results if r["model"] == name]
        avg = float(np.mean([r["skill_vs_zero_pct"] for r in mine]))
        mlflow.log_metric("average_skill_vs_zero_pct", avg)

        if name in REGION_MODELS:
            reg = region_holdout(name, df, ctx["fold_of_cell"])
            ctx["region"][name] = reg
            mlflow.log_metrics(reg)

        if hasattr(final, "linear_"):
            mlflow.log_metrics({f"effect_of_{f}": float(c)
                                for f, c in zip(final.feature_names_, final.linear_.coef_)})

        # Predictions of the final-exam model for every row (1951-2024), for graphs
        all_pred = final.predict(df[INPUT_COLS])
        ctx["all_pred"][name] = all_pred
        yearly = (pd.DataFrame({"year": df["year"], "actual": df[TARGET], "predicted": all_pred})
                  .groupby("year").mean())
        with tempfile.TemporaryDirectory() as d:
            yearly.to_csv(f"{d}/india_yearly_actual_vs_predicted.csv")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(yearly.index, yearly["actual"], "k-", lw=2, label="actual (all-India average)")
            ax.plot(yearly.index, yearly["predicted"], lw=2, label=name)
            ax.axvline(FINAL_CUTOFF + 0.5, color="grey", ls=":")
            ax.set_ylabel("temperature anomaly (degC)")
            ax.set_title(f"{name}: trained on 1951-{FINAL_CUTOFF}, right of dotted line = unseen")
            ax.legend()
            fig.tight_layout(); fig.savefig(f"{d}/india_yearly.png", dpi=110); plt.close(fig)
            mlflow.log_artifacts(d, artifact_path="results")
        mlflow.sklearn.log_model(final, name="model", input_example=df[INPUT_COLS].head(3))
        ctx["final_models"][name] = final
        print(f"  {name:24s} skill vs zero by exam: "
              + ", ".join(f"{r['test_years']}: {r['skill_vs_zero_pct']:+.1f}%" for r in mine)
              + f" | average {avg:+.1f}%"
              + (f" | hidden regions {ctx['region'][name]['region_skill_vs_zero_pct']:+.1f}%"
                 if name in ctx["region"] else ""))
        return mlflow.active_run().info.run_id


# ======================================================================== champion + comparison
def pick_champion(results):
    r = pd.DataFrame(results)
    skill = r.pivot(index="exam", columns="model", values="skill_vs_zero_pct")
    ref = skill[REFERENCE]
    lines, passed = [], []
    for m in ELIGIBLE:
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


def make_comparison(df, results, ctx, champion, reason, out):
    res = pd.DataFrame(results)
    res.to_csv(out / "comparison.csv", index=False)
    names = list(MODELS)

    # 1. skill by exam
    skill = res.pivot(index="model", columns="test_years", values="skill_vs_zero_pct").loc[names]
    ax = skill.plot.bar(figsize=(12, 5), width=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("% smaller error than 'every month is normal'")
    ax.set_title("1. Skill in each exam (monthly, every square)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(out / "01_skill_by_exam.png", dpi=120); plt.close()

    # 2. all-India yearly: natural only vs with CO2
    yearly = pd.DataFrame({"year": df["year"], "actual": df[TARGET]})
    for m in names:
        yearly[m] = ctx["all_pred"][m]
    yearly = yearly.groupby("year").mean()
    yearly.to_csv(out / "india_yearly.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(yearly.index, yearly["actual"], "k-", lw=2.5, label="actual")
    for m in dict.fromkeys(["natural_only", "linear_co2", champion, "baseline_own_trend"]):
        ax.plot(yearly.index, yearly[m], lw=2, label=m + (" (champion)" if m == champion else ""))
    ax.axvline(FINAL_CUTOFF + 0.5, color="grey", ls=":")
    ax.text(FINAL_CUTOFF + 0.7, ax.get_ylim()[1] * 0.9, "unseen years")
    ax.set_ylabel("all-India average anomaly (degC)")
    ax.set_title("2. Natural factors alone vs adding CO2 (models trained on 1951-2010)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "02_india_yearly_natural_vs_co2.png", dpi=120)
    plt.close(fig)

    # 3. map: warming trend per square, actual vs champion (1951-2024)
    t = decimal_year(df)
    cells = df.drop_duplicates("cell_id").set_index("cell_id")[["lat", "lon"]]
    sq = cells.join((per_square_line(df["cell_id"], t, df[TARGET])["slope"] * 10).rename("actual"))
    shown = list(dict.fromkeys([champion, "linear_co2_per_square"]))
    for m in shown:
        sq = sq.join((per_square_line(df["cell_id"], t, ctx["all_pred"][m])["slope"] * 10).rename(m))
    sq.to_csv(out / "square_trends.csv")
    cols = ["actual"] + shown
    lim = float(np.nanmax(np.abs(sq[cols].to_numpy())))
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 6.5), sharey=True)
    for ax, col in zip(axes, cols):
        sc = ax.scatter(sq["lon"], sq["lat"], c=sq[col], cmap="RdBu_r", vmin=-lim, vmax=lim,
                        marker="s", s=60)
        label = "actual data" if col == "actual" else col + (" (champion)" if col == champion else "")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("longitude"); ax.set_aspect("equal")
    axes[0].set_ylabel("latitude")
    fig.colorbar(sc, ax=axes, shrink=0.7, label="warming 1951-2024 (degC per decade)")
    fig.suptitle("3. Where India warms fastest: data vs models (models trained on 1951-2010)")
    fig.savefig(out / "03_warming_map.png", dpi=120); plt.close(fig)

    # 4. region holdout vs time-only final exam
    fin = res[res["exam"] == FINAL_CUTOFF].set_index("model")["skill_vs_zero_pct"]
    reg = pd.DataFrame({"time holdout (all squares seen)": fin.loc[REGION_MODELS],
                        "hidden regions": [ctx["region"][m]["region_skill_vs_zero_pct"]
                                           for m in REGION_MODELS]}, index=REGION_MODELS)
    reg.to_csv(out / "region_holdout.csv")
    ax = reg.plot.bar(figsize=(10, 4.5))
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("skill vs zero (%), 2011-2024")
    ax.set_title("4. Does the model work in regions it never saw?")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(out / "04_region_holdout.png", dpi=120); plt.close()

    # human-cause check numbers
    final = res[res["exam"] == FINAL_CUTOFF].set_index("model")
    full_trend = {m: trend_per_decade(yearly.index, yearly[m]) for m in ["actual"] + names}
    summary = {
        "champion": champion, "reason": reason, "reference_model": REFERENCE,
        "rule": f"challenger needs average skill > reference + {CHALLENGER_MARGIN:.0f} points and "
                f"wins at least 2 of 3 exams; eligible: {ELIGIBLE}",
        "average_skill_vs_zero": {m: float(res[res.model == m]["skill_vs_zero_pct"].mean()) for m in names},
        "human_cause_check": {
            "test_2011_2024_trend_per_decade": {
                "actual": float(final.loc[REFERENCE, "trend_actual_per_decade"]),
                **{m: float(final.loc[m, "trend_predicted_per_decade"])
                   for m in ["natural_only", "linear_co2", champion]}},
            "full_1951_2024_trend_per_decade": full_trend},
        "region_holdout": ctx["region"],
        "notes": ["India yearly values are plain averages over squares, not area-weighted",
                  "models predict from same-month CO2 and past El Nino; scenarios use chosen CO2 paths"],
        "descriptions": DESCRIPTIONS}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ======================================================================== main
def main():
    global DATA_MD5
    load_dotenv(dotenv_path=Path(".env"))
    # System metrics (CPU, memory, disk) recorded live for every run; needs psutil
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(1)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("WARNING: MLFLOW_TRACKING_URI not set; logging to local ./mlruns instead of DagsHub")
    mlflow.set_experiment(EXPERIMENT)
    session = "m3_session_" + datetime.now().strftime("%Y-%m-%d_%H%M")
    DATA_MD5 = hashlib.md5(DATA.read_bytes()).hexdigest()

    df = pd.read_csv(DATA, usecols=INPUT_COLS + [TARGET])
    df = df.sort_values(["year", "month", "cell_id"]).reset_index(drop=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"MLflow: {mlflow.get_tracking_uri()} | experiment {EXPERIMENT} | session {session}")
    print(f"Data: {len(df):,} rows, {df['cell_id'].nunique()} squares, "
          f"{df['year'].min()}-{df['year'].max()}\n")

    ctx = {"splits": {}, "trend_pred": {}, "region": {}, "all_pred": {}, "final_models": {},
           "fold_of_cell": region_fold_of_cell(df)}
    for cutoff, end in EXAMS:
        train = df[df["year"] <= cutoff]
        test = df[(df["year"] > cutoff) & (df["year"] <= end)]
        ctx["splits"][cutoff] = (train, test)
        own = M3Model(kind="own_trend").fit(train[INPUT_COLS], train[TARGET])
        ctx["trend_pred"][cutoff] = own.predict(test[INPUT_COLS])

    results, run_ids = [], {}
    with mlflow.start_run(run_name=session):
        mlflow.set_tags({"session": session, "kind": "session", "git_commit": git_commit(),
                         "data_md5": DATA_MD5,
                         "mlflow.note.content": "One training session of Model 3 (India temperature "
                         "anomaly). Expand this row to see every model. Comparison graphs, the "
                         "human-cause check and the champion decision are in Artifacts > comparison."})
        log_dataset(df)
        mlflow.log_params({"exams": str(EXAMS), "reference_model": REFERENCE,
                           "region_holdout": f"{N_REGION_FOLDS} folds of {BLOCK_DEG}x{BLOCK_DEG} degree blocks",
                           "champion_rule": f"average skill > reference + {CHALLENGER_MARGIN:.0f} pts "
                                            f"and wins >= 2 of 3 exams"})
        print("Training (3 exams each; hidden-region test where possible):")
        for name in MODELS:
            run_ids[name] = run_model(name, df, session, results, ctx)

        champion, reason = pick_champion(results)
        summary = make_comparison(df, results, ctx, champion, reason, REPORT_DIR)
        mlflow.log_params({"champion": champion, "champion_reason": reason[:500]})
        mlflow.log_metrics({f"average_skill_{k}": v for k, v in summary["average_skill_vs_zero"].items()})
        mlflow.log_artifacts(str(REPORT_DIR), artifact_path="comparison")
        mlflow.MlflowClient().set_tag(run_ids[champion], "champion", "true")

    hc = summary["human_cause_check"]["test_2011_2024_trend_per_decade"]
    print(f"\nChampion: {champion}\n{reason.replace('; ', chr(10))}")
    print("\nHuman-cause check, warming trend 2011-2024 (degC per decade):")
    for k, v in hc.items():
        print(f"  {k:24s} {v:+.3f}")
    print(f"\nGraphs, comparison.csv and summary.json: {REPORT_DIR}/ (also in the session run)")


if __name__ == "__main__":
    main()
