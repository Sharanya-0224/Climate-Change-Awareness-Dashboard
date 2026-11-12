"""Evaluation helpers for rain v2. Scores are on z (3-month total anomaly / sd)."""
import numpy as np
import pandas as pd

from config import OUTLIER_THRESHOLD, RANDOM_STATE


def _check_split(part, allow_test):
    if not allow_test:
        assert "test" not in part["split"].values, "test rows found, set allow_test=True on purpose"


def make_tercile_thresholds(df):
    """Per key and origin month: 1/3 and 2/3 quantiles of target_z, train rows only."""
    train_df = df.loc[df["split"] == "train"]
    thr = (
        train_df.groupby(["key", "month"])["target_z"]
        .quantile([1 / 3, 2 / 3])
        .unstack()
        .reset_index()
    )
    thr.columns = ["key", "month", "lower_threshold", "upper_threshold"]
    return thr


def metrics(y_true, y_pred, y_true_mm, y_pred_mm):
    mse = np.mean((y_true - y_pred) ** 2)
    mse_clim = np.mean(y_true ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "mse": mse,
        "mse_skill": 1 - mse / mse_clim if mse_clim > 0 else np.nan,
        "rmse_z": np.sqrt(mse),
        "mae_z": np.mean(np.abs(y_true - y_pred)),
        "bias_z": np.mean(y_pred - y_true),
        "acc": np.corrcoef(y_true, y_pred)[0, 1]
        if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan,
        "r2": 1 - np.sum((y_true - y_pred) ** 2) / ss_tot if ss_tot > 0 else np.nan,
        "rmse_mm": np.sqrt(np.mean((y_true_mm - y_pred_mm) ** 2)),
        "mae_mm": np.mean(np.abs(y_true_mm - y_pred_mm)),
        "bias_mm": np.mean(y_pred_mm - y_true_mm),
    }

def mse_skill(y, p):
    """MSE skill vs climatology (climatology predicts z = 0)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return 1.0 - np.sum((y - p) ** 2) / np.sum(y ** 2)

def categorical_metrics(y_true, y_pred, lower, upper):
    def classify(values):
        return np.where(values < lower, 0, np.where(values > upper, 2, 1))

    observed = classify(y_true)
    predicted = classify(y_pred)
    confusion = pd.crosstab(pd.Series(observed), pd.Series(predicted)).reindex(
        index=[0, 1, 2], columns=[0, 1, 2], fill_value=0
    ).values
    n = confusion.sum()
    accuracy = np.trace(confusion) / n
    expected = np.sum(confusion.sum(axis=0) * confusion.sum(axis=1)) / n ** 2
    hss = (accuracy - expected) / (1 - expected) if expected < 1 else np.nan
    return {"tercile_accuracy": accuracy, "hss": hss}


def _with_thresholds(part, pred_z, tercile_thresholds, cols):
    data = part[cols].copy()
    data["pred_z"] = np.asarray(pred_z)
    data = data.merge(tercile_thresholds, on=["key", "month"], how="left", validate="many_to_one")
    assert not data[["lower_threshold", "upper_threshold"]].isna().any().any(), "missing thresholds"
    data["pred_anom_mm"] = data["pred_z"] * data["target_sd"]
    return data


def _calculate(data):
    return {
        **metrics(data["target_z"].values, data["pred_z"].values,
                  data["target_anom_mm"].values, data["pred_anom_mm"].values),
        **categorical_metrics(data["target_z"].values, data["pred_z"].values,
                              data["lower_threshold"].values, data["upper_threshold"].values),
    }


def evaluate_model(part, pred_z, tercile_thresholds,
                   outlier_threshold=OUTLIER_THRESHOLD, allow_test=False):
    """Metrics on all rows and with |target_z| > outlier_threshold removed."""
    assert part["split"].nunique() == 1
    _check_split(part, allow_test)
    assert len(part) == len(pred_z)

    data = _with_thresholds(
        part, pred_z, tercile_thresholds,
        ["key", "month", "target_z", "target_anom_mm", "target_sd"],
    )
    filtered = data[data["target_z"].abs() <= outlier_threshold]
    return {
        "all": {**_calculate(data), "n": len(data)},
        "no_extremes": {**_calculate(filtered), "n": len(filtered)},
    }


def get_season(month):
    """Season of the ORIGIN month t (JJA = forecast made Jun-Aug, covering Jul-Nov)."""
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[month]


def evaluate_by_group(part, pred_z, tercile_thresholds, allow_test=False):
    """evaluate_model per region and per origin-month season."""
    pred_z = np.asarray(pred_z)
    groups = {"region": part["region"], "season": part["month"].map(get_season)}
    results = {}
    for group_type, values in groups.items():
        for name in sorted(values.unique()):
            mask = (values == name).values
            results[f"{group_type}_{name}"] = evaluate_model(
                part.iloc[np.where(mask)[0]], pred_z[mask], tercile_thresholds,
                allow_test=allow_test,
            )
    return results


def bootstrap_ci(part, pred_z, tercile_thresholds, n_boot=200,
                 seed=RANDOM_STATE, allow_test=False):
    """Year-block bootstrap: 95% range of every metric."""
    _check_split(part, allow_test)
    rng = np.random.default_rng(seed)
    years = part["year"].unique()
    data = _with_thresholds(
        part, pred_z, tercile_thresholds,
        ["key", "month", "year", "target_z", "target_anom_mm", "target_sd"],
    )
    year_values = data["year"].values
    year_index = {y: np.flatnonzero(year_values == y) for y in years}

    scores = []
    for _ in range(n_boot):
        sampled = rng.choice(years, size=len(years), replace=True)
        idx = np.concatenate([year_index[y] for y in sampled])
        scores.append(_calculate(data.iloc[idx]))
    scores = pd.DataFrame(scores)
    return pd.DataFrame({"lower_95": scores.quantile(0.025), "upper_95": scores.quantile(0.975)})


def paired_year_bootstrap(year, y, p_a, p_b, n_boot=1000, seed=RANDOM_STATE, drop_extreme=None):
    """MSE skill of model A minus model B, years resampled together.

    Returns (summary dict, per-year DataFrame). Skill = 1 - sum((y-p)^2) / sum(y^2).
    """
    year = np.asarray(year)
    y = np.asarray(y, float)
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)
    assert len(year) == len(y) == len(p_a) == len(p_b)

    if drop_extreme is not None:
        keep = np.abs(y) <= drop_extreme
        year, y, p_a, p_b = year[keep], y[keep], p_a[keep], p_b[keep]

    # per-year sums, so each resample is fast
    df = pd.DataFrame({"year": year, "clim": y ** 2,
                       "err_a": (y - p_a) ** 2, "err_b": (y - p_b) ** 2})
    per_year = df.groupby("year")[["clim", "err_a", "err_b"]].sum()

    def skill_diff(t):
        clim = t["clim"].sum()
        return (1 - t["err_a"].sum() / clim) - (1 - t["err_b"].sum() / clim)

    point = skill_diff(per_year)
    rng = np.random.default_rng(seed)
    years = per_year.index.values
    diffs = np.array([
        skill_diff(per_year.loc[rng.choice(years, size=len(years), replace=True)])
        for _ in range(n_boot)
    ])

    per_year_out = pd.DataFrame({
        "skill_a": 1 - per_year["err_a"] / per_year["clim"],
        "skill_b": 1 - per_year["err_b"] / per_year["clim"],
    })
    per_year_out["diff_points"] = 100 * (per_year_out["skill_a"] - per_year_out["skill_b"])

    summary = {
        "diff_points": 100 * point,
        "lower_95": 100 * np.quantile(diffs, 0.025),
        "upper_95": 100 * np.quantile(diffs, 0.975),
        "share_a_better": float(np.mean(diffs > 0)),
        "n_years": len(years),
        "n_rows": len(y),
    }
    return summary, per_year_out