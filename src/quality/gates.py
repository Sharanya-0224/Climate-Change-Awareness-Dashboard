"""
quality/gates.py

Quality gates: simple pass/fail checks a model must clear before it is allowed
into the Model Registry. Each gate returns a dict {gate, passed, detail, threshold}.

Why gates?  A model can score well in an experiment and still be unsafe to serve
(for example if it produces NaN, or impossible values for some inputs). Gates
are the automatic "release checklist". They are logged to MLflow as metrics
(1 = pass, 0 = fail) so every registered model carries proof that it passed.

Every threshold is a named constant with its justification next to it, so the
choice can be explained and changed in one place.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- thresholds
# A model must beat a free, simple guess, otherwise it adds nothing. 0 = "at least as good".
MIN_SKILL_PCT = 0.0

# Hit rate: at least half the unseen years must land within the tolerance.
# 0.5 means "right more often than wrong", the weakest sensible bar for a climate signal
# that is buried in weather noise.
MIN_HIT_SHARE = 0.5
TOLERANCE_M3_DEGC = 0.2    # yearly India anomaly within 0.2 degC (final-exam MAE was ~0.13)
TOLERANCE_M2_PPM = 0.3     # yearly CO2 rise within 0.3 ppm (final-exam MAE was ~0.22)

# Physical range = observed data range plus a safety margin. Anything outside signals
# a bug or an impossible input, not a real climate value.
RANGE_MARGIN_M3_DEGC = 5.0  # observed anomalies span about -4.3 to +5.4 degC
RANGE_MARGIN_M2_PPM = 1.5   # observed yearly rises are a few ppm; 1.5 ppm beyond is generous


# ---------------------------------------------------------------- gates
def _gate(name, passed, detail, threshold=None):
    return {"gate": name, "passed": bool(passed), "detail": detail, "threshold": threshold}


def beats_baseline(skill_pct, baseline_name, min_skill=MIN_SKILL_PCT):
    """Final-exam skill against the baseline must be above the minimum."""
    return _gate("beats_baseline", skill_pct > min_skill,
                 f"skill vs {baseline_name} on unseen years = {skill_pct:+.2f}%",
                 f"> {min_skill}%")


def champion_rule(champion, expected, reason):
    """The model being registered must be the one the pre-fixed rule chose."""
    return _gate("champion_rule", champion == expected,
                 f"rule picked '{champion}', registering '{expected}'. {reason[:120]}", "equal")


def no_missing_outputs(pred):
    """No NaN or infinite predictions."""
    bad = int((~np.isfinite(np.asarray(pred, float))).sum())
    return _gate("no_missing_outputs", bad == 0, f"{bad} bad values in {len(pred)} predictions", "0")


def physical_range(pred, observed, margin, unit):
    """Predictions must stay inside the observed range plus a margin."""
    lo, hi = float(np.min(observed)) - margin, float(np.max(observed)) + margin
    p = np.asarray(pred, float)
    outside = int(((p < lo) | (p > hi)).sum())
    return _gate("physical_range", outside == 0,
                 f"{outside} predictions outside [{lo:.2f}, {hi:.2f}] {unit}; "
                 f"predicted range [{p.min():.2f}, {p.max():.2f}]", f"[{lo:.2f}, {hi:.2f}]")


def deterministic(model, X):
    """The same input must give the same output twice."""
    a, b = model.predict(X), model.predict(X)
    same = bool(np.allclose(a, b))
    return _gate("deterministic", same, "identical on repeat" if same else "outputs differ", "identical")


def hit_rate(actual, pred, tolerance, unit, min_share=MIN_HIT_SHARE, label="years"):
    """Accuracy-style check: share of unseen points within +/- tolerance of the truth."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    within = float(np.mean(np.abs(actual - pred) <= tolerance))
    return _gate("hit_rate", within >= min_share,
                 f"{within:.0%} of {len(actual)} unseen {label} within {tolerance} {unit}",
                 f">= {min_share:.0%}")


def direction_hit_rate(actual, pred):
    """Information only (never fails): how often the model gets the direction
    (warmer/cooler than normal, or rise bigger/smaller than last year) right."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    share = float(np.mean(np.sign(actual) == np.sign(pred)))
    return _gate("direction_hit_rate_info", True, f"{share:.0%} correct direction", "info only")


def hidden_regions(skill_pct, min_skill=MIN_SKILL_PCT):
    """Model 3 only: must not be worse than 'normal' in regions it never saw."""
    return _gate("hidden_regions", skill_pct > min_skill,
                 f"skill on hidden regions = {skill_pct:+.2f}%", f"> {min_skill}%")


# ---------------------------------------------------------------- reporting
def report(gates):
    """Print a table and return (all_passed, DataFrame)."""
    df = pd.DataFrame(gates)[["gate", "passed", "threshold", "detail"]]
    print(df.to_string(index=False))
    all_passed = bool(df["passed"].all())
    print("\nALL GATES PASSED" if all_passed else "\nSOME GATES FAILED: model will NOT be registered")
    return all_passed, df
