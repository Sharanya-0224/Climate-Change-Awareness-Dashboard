"""Reference models for rain v2 (fixed, trained on 1901-2002, from 06_baselines_v2).
Loaded from JSON coefficients. No scikit-learn needed."""
import json
import numpy as np


def load_persistence(path):
    with open(path) as f:
        d = json.load(f)
    return {int(k): float(v) for k, v in d["coefficients"].items()}


def predict_persistence(coefs, part):
    """Damped persistence: pred_z = rain_z_t * coefficient[origin month]."""
    a = part["month"].astype(int).map(coefs).to_numpy("float64")
    return a * part["rain_z_t"].to_numpy("float64")


def load_ridge(path):
    with open(path) as f:
        r = json.load(f)
    assert r["feature_order"] == r["continuous_features"] + r["region_features"]
    return r


def predict_ridge(ridge, part):
    """Ridge: continuous features scaled (train mean and scale), region one-hots unscaled."""
    cont = part[ridge["continuous_features"]].to_numpy(np.float32)
    mean = np.asarray(ridge["scaler_mean"], np.float32)
    scale = np.asarray(ridge["scaler_scale"], np.float32)
    x_cont = ((cont - mean) / scale).astype(np.float32)
    x_reg = part[ridge["region_features"]].to_numpy(np.float32)
    X = np.column_stack([x_cont, x_reg]).astype(np.float64)
    return X @ np.asarray(ridge["coef"], np.float64) + float(ridge["intercept"])