"""LightGBM code for rain v2, copied from 07_lgbm_cv_tuning_v2.

Final model = DEFAULT_PARAMS, no clipping, 23 features, trained on 1901-2002.
Selection: rolling CV, no setting had both folds above 0, so the default was kept.
"""
import numpy as np
import lightgbm as lgb

from config import RANDOM_STATE

LIGHTGBM_VERSION = "4.7.0"
CLIP_TARGET_Z = None
BUFFER_FROM_MONTH = 10   # train origins Oct-Dec before a val block are dropped

DEFAULT_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=RANDOM_STATE,
)

FOLDS = {
    "A": dict(train_start=1901, train_end=1950, val_start=1951, val_end=1975),
    "B": dict(train_start=1901, train_end=1975, val_start=1976, val_end=2002),
}


def make_split(data, train_start, train_end, val_start, val_end, buffer=True):
    """Train and val by origin year. Buffer drops train origins Oct-Dec of the
    year before val_start (their 3-month target crosses into val)."""
    tr = data[data["year"].between(train_start, train_end)]
    va = data[data["year"].between(val_start, val_end)]
    if buffer:
        cross = (tr["year"] == val_start - 1) & (tr["month"] >= BUFFER_FROM_MONTH)
        tr = tr[~cross]
    return tr, va


def fit_lgbm(X, y, params=None):
    model = lgb.LGBMRegressor(**(params or DEFAULT_PARAMS))
    model.fit(X, y)
    return model


def save_booster(model, path):
    model.booster_.save_model(str(path))


def load_booster(path):
    return lgb.Booster(model_file=str(path))


def check_feature_order(booster, features):
    """The booster must expect exactly these features in this order."""
    assert booster.feature_name() == list(features), "feature order mismatch"


def predict_z(booster, data, features):
    """Predicted target_z. Columns are taken in model order."""
    return np.asarray(booster.predict(data[list(features)]), dtype="float64")