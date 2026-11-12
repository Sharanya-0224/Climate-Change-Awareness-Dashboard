"""Feature code for rain v2, copied from 05_model_dataset_finalisation_v2.

Target = 3-month total (months t+1..t+3). Normals from train years only.
Time axis: position p = (year - START_YEAR) * 12 + (month - 1).
"""
import numpy as np
import pandas as pd

from config import REGIONS, START_YEAR, MIN_VALID_TRAIN_YEARS

NINO_FILE = "nino34.long.anom.csv"
DMI_FILE = "dmi.had.long.csv"


# ---------- ids and inputs ----------
def point_key(lat, lon):
    """Integer id for a 0.25 degree point."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    return np.round(lat * 100).astype("int64") * 10000 + np.round(lon * 100).astype("int64")


def load_index(path, out_name):
    """Nino 3.4 or DMI file. Values <= -99 become NaN."""
    d = pd.read_csv(path).iloc[:, :2].copy()
    d.columns = ["date", out_name]
    d["date"] = pd.to_datetime(d["date"])
    d.loc[d[out_name] <= -99, out_name] = np.nan
    return d


def load_indices(raw_dir):
    """Nino 3.4 and DMI merged by month."""
    n = load_index(raw_dir / NINO_FILE, "nino34")
    m = load_index(raw_dir / DMI_FILE, "dmi")
    return n.merge(m, on="date", how="outer").sort_values("date").reset_index(drop=True)


def get_points(rain_2025_path, point_region_path):
    """All 0.25 degree points with key, latitude, longitude, state, region."""
    p = pd.read_csv(rain_2025_path, usecols=["latitude", "longitude"]).drop_duplicates()
    p["latitude"] = p.latitude.round(2)
    p["longitude"] = p.longitude.round(2)
    p = p.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    p["key"] = point_key(p.latitude, p.longitude)
    r = pd.read_csv(point_region_path, usecols=["key", "state", "region"])
    assert r.key.is_unique, "duplicate keys in point_region file"
    p = p.merge(r, on="key", how="left", validate="one_to_one")
    assert p.region.notna().all(), "points without a region"
    assert set(p.region.unique()) <= set(REGIONS), "unknown region name"
    return p


# ---------- cleaning (dataset_02 rule) ----------
def clean_zero_fill(mdf, train_end, min_valid_years=MIN_VALID_TRAIN_YEARS):
    """Mask all 12 months of any point-year with annual total exactly 0 (fill values),
    then drop points with fewer than min_valid_years non-zero train years.
    Returns (cleaned monthly frame, dropped keys)."""
    mdf = mdf.sort_values(["key", "year", "month"]).reset_index(drop=True)
    annual = (mdf.groupby(["key", "year"], observed=True)["rain_mm"]
              .sum(min_count=12).rename("annual").reset_index())
    zero = annual.loc[annual["annual"] == 0, ["key", "year"]]

    tr = annual[annual["year"].between(START_YEAR, train_end)]
    valid = (tr[tr["annual"] > 0].groupby("key")["year"].nunique()
             .reindex(mdf["key"].unique(), fill_value=0))
    keep_keys = valid[valid >= min_valid_years].index
    drop_keys = valid[valid < min_valid_years].index

    mdf = mdf.merge(zero.assign(zero_fill=True), on=["key", "year"],
                    how="left", validate="many_to_one")
    mdf.loc[mdf["zero_fill"].eq(True), "rain_mm"] = np.nan
    mdf = mdf.drop(columns="zero_fill")
    mdf = mdf[mdf["key"].isin(keep_keys)].copy()
    return mdf, drop_keys


# ---------- arrays ----------
def monthly_matrix(mdf, end_year):
    """Monthly rain as (points, months) array. Returns (keys, X)."""
    n_months = (end_year - START_YEAR + 1) * 12
    mdf = mdf.sort_values(["key", "year", "month"]).reset_index(drop=True)
    keys = np.sort(mdf["key"].unique())
    P = len(keys)
    assert len(mdf) == P * n_months
    assert (mdf["year"].to_numpy().reshape(P, n_months)[:, 0] == START_YEAR).all()
    assert (mdf["month"].to_numpy().reshape(P, n_months)[:, :12] == np.arange(1, 13)).all()
    X = mdf["rain_mm"].to_numpy("float64").reshape(P, n_months)
    return keys, X


def idx_series(indices, col, end_year):
    """One index as a 1D array over START_YEAR-01 .. end_year-12. No gaps allowed."""
    s = indices.set_index("date")[col]
    s.index = s.index.to_period("M")
    s = s[~s.index.duplicated()]
    s = s.reindex(pd.period_range(f"{START_YEAR}-01", f"{end_year}-12", freq="M"))
    assert s.isna().sum() == 0, f"{col}: missing months"
    return s.to_numpy("float64")


def monthly_normals(X, train_end, sd_floor):
    """Per point and calendar month: mean and sd of monthly rain, train years only."""
    P = X.shape[0]
    ntr = train_end - START_YEAR + 1
    Xy = X.reshape(P, -1, 12)
    mu_m = np.nanmean(Xy[:, :ntr, :], axis=1)
    sd_m = np.maximum(np.nanstd(Xy[:, :ntr, :], axis=1, ddof=1), sd_floor)
    assert np.isfinite(mu_m).all() and np.isfinite(sd_m).all()
    return mu_m, sd_m


def target_window(X):
    """3-month total of months t+1..t+3 (NaN for the last 3 origins)."""
    W = np.full(X.shape, np.nan)
    W[:, :-3] = X[:, 1:-2] + X[:, 2:-1] + X[:, 3:]
    return W


def target_normals(X, train_end, sd_floor):
    """Per point and origin month: mean and sd of the 3-month total, train years only.
    Train origins Oct-Dec of train_end are excluded (window crosses into val)."""
    P = X.shape[0]
    n_years = X.shape[1] // 12
    W3 = target_window(X).reshape(P, n_years, 12)
    years = np.arange(START_YEAR, START_YEAR + n_years)[:, None]
    mos = np.arange(12)[None, :]
    train_m = np.broadcast_to((years <= train_end) & ~((years == train_end) & (mos >= 9)),
                              (n_years, 12))
    mu_w = np.full((P, 12), np.nan)
    sd_raw = np.full((P, 12), np.nan)
    for m in range(12):
        v = W3[:, train_m[:, m], m]
        mu_w[:, m] = np.nanmean(v, axis=1)
        sd_raw[:, m] = np.nanstd(v, axis=1, ddof=1)
    sd_w = np.maximum(sd_raw, sd_floor)
    assert np.isfinite(mu_w).all() and np.isfinite(sd_w).all()
    return mu_w, sd_w


def shift(A, k):
    """Shift a (points, months) array right by k months."""
    if k == 0:
        return A
    B = np.full_like(A, np.nan)
    B[:, k:] = A[:, :-k]
    return B


def shift1(a, k):
    """Shift a 1D array right by k months."""
    b = np.full_like(a, np.nan)
    if k == 0:
        return a.copy()
    b[k:] = a[:-k]
    return b


def idx_feats(a, prefix, with_chg=False):
    f = {
        f"{prefix}_t": a,
        f"{prefix}_lag1": shift1(a, 1),
        f"{prefix}_lag2": shift1(a, 2),
        f"{prefix}_lag3": shift1(a, 3),
        f"{prefix}_mean3": (a + shift1(a, 1) + shift1(a, 2)) / 3,
    }
    if with_chg:
        f[f"{prefix}_chg3"] = a - shift1(a, 3)
    return f


def feature_arrays(X, mu_m, sd_m, nino, dmi):
    """All time-varying features as (points, months) float32 arrays.
    Only month t and earlier are used."""
    P, T = X.shape
    Z = ((X.reshape(P, -1, 12) - mu_m[:, None, :]) / sd_m[:, None, :]).reshape(P, T).astype("float32")
    S = Z.copy()
    roll3 = None
    for k in range(1, 6):
        S = S + shift(Z, k)
        if k == 2:
            roll3 = S / 3
    roll6 = S / 6

    arrays = {"rain_z_t": Z, "rain_z_lag1": shift(Z, 1),
              "rain_z_roll3": roll3, "rain_z_roll6": roll6}
    idx = {**idx_feats(nino, "nino34", True), **idx_feats(dmi, "dmi")}
    for k, a in idx.items():
        arrays[k] = np.broadcast_to(a.astype("float32")[None, :], (P, T))
    return arrays


def rows_at(arrays, keys, points, positions, inputs):
    """Feature rows for the chosen time positions, all points.
    points: DataFrame indexed by key with latitude, longitude, state, region.
    inputs: feature names in model order (from the feature list file)."""
    pts = points.loc[keys]
    lat = pts["latitude"].to_numpy("float32")
    lon = pts["longitude"].to_numpy("float32")
    frames = []
    for p in positions:
        month = p % 12 + 1
        d = {
            "key": keys,
            "state": pts["state"].to_numpy(),
            "region": pts["region"].to_numpy(),
            "year": np.full(len(keys), START_YEAR + p // 12, dtype="int16"),
            "month": np.full(len(keys), month, dtype="int8"),
            "lat": lat,
            "lon": lon,
            "month_sin": np.full(len(keys), np.sin(2 * np.pi * month / 12), dtype="float32"),
            "month_cos": np.full(len(keys), np.cos(2 * np.pi * month / 12), dtype="float32"),
        }
        for r in REGIONS:
            d[f"region_{r}"] = (pts["region"].to_numpy() == r).astype("int8")
        for name, a in arrays.items():
            d[name] = a[:, p].astype("float32")
        frames.append(pd.DataFrame(d))
    out = pd.concat(frames, ignore_index=True)
    return out[["key", "state", "region", "year", "month"] + list(inputs)]


def position(year, month):
    return (year - START_YEAR) * 12 + (month - 1)