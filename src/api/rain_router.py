"""
api/rain_router.py

Rain v2 forecasts: 3-month rainfall total (months t+1..t+3 after origin month t)
for the India 0.25 degree grid, as an anomaly from the 1901-2002 normal.
Model: LightGBM, registered as rain_v2_3month_lgbm (alias champion).
Included by main.py with prefix "/rain".

Routes
  GET  /rain/forecast    lat, lon, optional year + month (default: latest origin in the package)
  POST /rain/predict     lat, lon, origin month + the 15 time features
  GET  /rain/model-info  metadata and skill table

Functions for main.py
  load_rain_model()  call once at start-up; never raises (routes give 503 if loading failed)
  rain_status()      {"name", "version", "loaded"} for the shared /health
  rain_smoke()       one real prediction, returns a float; raises if the model cannot load

Rules
  - Run time uses ONLY the package folder (env RAIN_MODEL_DIR, default "rain_package").
    No internet, no MLflow, no DVC, nothing from data/ or precipitation/.
  - Imports only: lightgbm, numpy, pandas, fastapi, pydantic, standard library. Python 3.9 compatible.
  - Snapping: nearest of the package grid points. If the nearest point is more than
    0.25 degree away, the place is outside the grid: 422.
  - /forecast serves only the origins in lookup_latest.csv; any other year/month: 422.
  - /predict: the caller sends only lat, lon, month and the 15 time features. month_sin/cos,
    region one-hots and lat/lon features are built here from the month and the snapped point,
    so they cannot be inconsistent. Features are then ordered by feature_list.json.
  - Input limits = observed 1901-2025 range plus a margin. Outside them the model has never
    seen such values, so the request is rejected (422).
  - pred_total_mm = normal mean + pred_z x normal sd, clipped at 0 (rain cannot be negative);
    clipped_at_zero says when that happened (about 0.1% of test rows).
  - has_skill_region / has_skill_season come from skill_table.csv
    (val skill > 0 and test 95% range above 0); has_skill = both true.
  - beyond_test_period = true when the origin is after the last tested origin (Sep 2025).
  - Errors: FastAPI default {"detail": ...}; 422 bad input; 503 model not loaded.
  - cell_id = 1-degree square of the REQUESTED lat/lon (friend's rule), so both models share one map.
  - lookup CSV is read as float32 (the training type): a 64-bit re-read can cross a tree split.
"""
import calendar
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["rain"])

DEFAULT_DIR = "rain_package"
DEFAULT_NAME = "rain_v2_3month_lgbm"
MAX_SNAP_DEG = 0.25                 # nearest grid point must be within this distance
LAST_TESTED_ORIGIN = (2025, 9)      # last origin with an observed target (test ends here)
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

_STATE: Dict[str, Any] = {"loaded": False, "error": "not loaded yet"}


class RainNotLoaded(RuntimeError):
    pass


# ---------------------------------------------------------------- loading
def package_dir() -> Path:
    return Path(os.getenv("RAIN_MODEL_DIR", DEFAULT_DIR))


def load_rain_model(pkg_dir: Optional[str] = None) -> bool:
    """Load the package once. Never raises: on failure the routes return 503."""
    _STATE.clear()
    d = Path(pkg_dir) if pkg_dir else package_dir()
    try:
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        with open(d / "feature_list.json") as f:
            features = json.load(f)["features"]
        if booster.feature_name() != features:
            raise ValueError("model features differ from feature_list.json")
        with open(d / "metadata.json") as f:
            meta = json.load(f)
        points = pd.read_csv(d / "points.csv")
        normals = pd.read_csv(d / "normals.csv").set_index(["key", "origin_month"])
        lookup = pd.read_csv(d / "lookup_latest.csv", dtype={f: "float32" for f in features}).set_index(["key", "year", "month"])
        skill = pd.read_csv(d / "skill_table.csv").fillna({"covers": ""}).set_index("group")
        _STATE.update(
            loaded=True, error=None, dir=str(d), booster=booster, features=features, meta=meta,
            points=points, normals=normals, lookup=lookup, skill=skill,
            plat=points["latitude"].to_numpy(float), plon=points["longitude"].to_numpy(float),
            origins=sorted({(int(y), int(m)) for _, y, m in lookup.index}),
        )
    except Exception as e:
        _STATE.update(loaded=False, error=f"{type(e).__name__}: {e}")
    return _STATE["loaded"]


def _require() -> Dict[str, Any]:
    if not _STATE.get("loaded"):
        raise RainNotLoaded(f"rain model is not loaded ({_STATE.get('error')})")
    return _STATE


def rain_status() -> Dict[str, Any]:
    meta = _STATE.get("meta") or {}
    return {"name": meta.get("registered_name", DEFAULT_NAME),
            "version": meta.get("registered_version"),
            "loaded": bool(_STATE.get("loaded"))}


def rain_smoke() -> float:
    """One real prediction for the Docker build. Raises if the model cannot load."""
    if not _STATE.get("loaded") and not load_rain_model():
        raise RainNotLoaded(_STATE.get("error"))
    p = _STATE["points"].iloc[0]
    value = forecast(float(p["latitude"]), float(p["longitude"]))["pred_z"]
    if not np.isfinite(value):
        raise ValueError("smoke prediction is not finite")
    return float(value)

def cell_id(lat: float, lon: float) -> str:
    """Friend's 1-degree square of the requested place: centre = floor(value) + 0.5."""
    return f"{np.floor(lat) + 0.5:.1f}_{np.floor(lon) + 0.5:.1f}"

# ---------------------------------------------------------------- logic (plain functions)
def snap(lat: float, lon: float) -> Dict[str, Any]:
    s = _require()
    dist = np.sqrt((s["plat"] - lat) ** 2 + (s["plon"] - lon) ** 2)
    i = int(np.argmin(dist))
    if dist[i] > MAX_SNAP_DEG:
        raise ValueError(f"({lat}, {lon}) is outside the rain grid (nearest point {dist[i]:.2f} degree away)")
    p = s["points"].iloc[i]
    return {"key": int(p["key"]), "point_lat": float(p["latitude"]), "point_lon": float(p["longitude"]),
            "distance_deg": round(float(dist[i]), 4), "cell_id": str(p["cell_id"]),
            "state": str(p["state"]), "region": str(p["region"])}


def _months_covered(month: int, year: Optional[int] = None) -> str:
    end = month + 3
    m1, m3 = month % 12 + 1, (end - 1) % 12 + 1
    if year is None:
        return f"{calendar.month_abbr[m1]}-{calendar.month_abbr[m3]}"
    y1, y3 = year + (1 if month == 12 else 0), year + (end - 1) // 12
    return f"{y1}-{m1:02d} to {y3}-{m3:02d}"


def _result(point: Dict[str, Any], month: int, year: Optional[int], pred_z: float) -> Dict[str, Any]:
    s = _require()
    if not np.isfinite(pred_z):
        raise ValueError("prediction is not finite")
    nrm = s["normals"].loc[(point["key"], month)]
    mean, sd = float(nrm["target_mean_mm"]), float(nrm["target_sd_mm"])
    anom = pred_z * sd
    raw_total = mean + anom
    season = SEASONS[month]
    skill = s["skill"]
    sk_region = bool(skill.loc["region_" + point["region"], "has_skill"])
    sk_season = bool(skill.loc["season_" + season, "has_skill"])
    covered = _months_covered(month, year)
    note = (f"Forecast of total rainfall for {covered}, made in origin month {month}"
            f"{'' if year is None else f' of {year}'}. pred_z is the anomaly in standard deviations, "
            f"pred_anom_mm and pred_total_mm are in mm, normal = 1901-2002. "
            f"Skill is small (test MSE skill {s['meta'].get('test_skill')}); "
            f"has_skill=false means do not rely on this forecast.")
    if raw_total < 0:
        note += " pred_total_mm was below 0 and was set to 0."
    return {
        **point,
        "origin_year": year, "origin_month": month, "season": season,
        "season_label": f"{season} by origin month", "months_covered": covered,
        "pred_z": round(float(pred_z), 4), "pred_anom_mm": round(float(anom), 2),
        "pred_total_mm": round(float(max(raw_total, 0.0)), 2),
        "normal_total_mm": round(mean, 2), "clipped_at_zero": bool(raw_total < 0),
        "has_skill_region": sk_region, "has_skill_season": sk_season,
        "has_skill": sk_region and sk_season,
        "beyond_test_period": None if year is None else (year, month) > LAST_TESTED_ORIGIN,
        "note": note,
        "served_model": rain_status()["name"], "served_version": rain_status()["version"],
    }


def forecast(lat: float, lon: float, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
    s = _require()
    if (year is None) != (month is None):
        raise ValueError("give both year and month, or neither")
    if year is None:
        year, month = s["origins"][-1]
    if (year, month) not in s["origins"]:
        raise ValueError(f"origin {year}-{month:02d} not available; available: "
                         + ", ".join(f"{y}-{m:02d}" for y, m in s["origins"]))
    point = snap(lat, lon)
    point["cell_id"] = cell_id(lat, lon)
    x = s["lookup"].loc[[(point["key"], year, month)], s["features"]]
    pred_z = float(s["booster"].predict(x)[0])
    return _result(point, month, year, pred_z)


def predict(lat: float, lon: float, month: int, time_features: Dict[str, float]) -> Dict[str, Any]:
    s = _require()
    point = snap(lat, lon)
    point["cell_id"] = cell_id(lat, lon)
    row = dict(time_features)
    row.update(lat=point["point_lat"], lon=point["point_lon"],
               month_sin=float(np.sin(2 * np.pi * month / 12)),
               month_cos=float(np.cos(2 * np.pi * month / 12)))
    for f in s["features"]:
        if f.startswith("region_"):
            row[f] = 1.0 if f == "region_" + point["region"] else 0.0
    missing = [f for f in s["features"] if f not in row]
    if missing:
        raise ValueError(f"missing features: {missing}")
    x = pd.DataFrame([row])[s["features"]].astype("float32")
    pred_z = float(s["booster"].predict(x)[0])
    return _result(point, month, None, pred_z)


# ---------------------------------------------------------------- input schema
# Limits = observed range 1901-2025 (dataset_02) plus a margin. Observed:
# rain_z_t/lag1 -3.81..75.39, roll3 -2.83..27.03, roll6 -2.42..13.42 (large highs come from
# dry-season points with the 5 mm sd floor); nino34 -2.18..2.57, chg3 -1.88..1.65;
# dmi -1.63..1.28, mean3 -1.24..1.01.
class RainPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # unknown or misspelled fields give 422

    lat: float = Field(..., ge=6, le=38)
    lon: float = Field(..., ge=66, le=100)
    month: int = Field(..., ge=1, le=12, description="origin month t")
    rain_z_t: float = Field(..., ge=-5, le=80)
    rain_z_lag1: float = Field(..., ge=-5, le=80)
    rain_z_roll3: float = Field(..., ge=-4, le=30)
    rain_z_roll6: float = Field(..., ge=-4, le=15)
    nino34_t: float = Field(..., ge=-3.5, le=3.5)
    nino34_lag1: float = Field(..., ge=-3.5, le=3.5)
    nino34_lag2: float = Field(..., ge=-3.5, le=3.5)
    nino34_lag3: float = Field(..., ge=-3.5, le=3.5)
    nino34_mean3: float = Field(..., ge=-3.5, le=3.5)
    nino34_chg3: float = Field(..., ge=-3, le=3)
    dmi_t: float = Field(..., ge=-2.5, le=2.5)
    dmi_lag1: float = Field(..., ge=-2.5, le=2.5)
    dmi_lag2: float = Field(..., ge=-2.5, le=2.5)
    dmi_lag3: float = Field(..., ge=-2.5, le=2.5)
    dmi_mean3: float = Field(..., ge=-2.5, le=2.5)


# ---------------------------------------------------------------- routes (thin)
def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RainNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/forecast")
def forecast_route(lat: float = Query(..., ge=6, le=38), lon: float = Query(..., ge=66, le=100),
                   year: Optional[int] = Query(None, ge=1901, le=2100),
                   month: Optional[int] = Query(None, ge=1, le=12)):
    return _call(forecast, lat, lon, year, month)


@router.post("/predict")
def predict_route(req: RainPredictRequest):
    body = req.model_dump()
    lat, lon, month = body.pop("lat"), body.pop("lon"), body.pop("month")
    return _call(predict, lat, lon, month, body)


@router.get("/model-info")
def model_info_route():
    def info():
        s = _require()
        return {**s["meta"], "loaded": True, "package_dir": s["dir"],
                "available_origins": [f"{y}-{m:02d}" for y, m in s["origins"]],
                "skill_table": s["skill"].reset_index().to_dict(orient="records"),
                "served_model": rain_status()["name"], "served_version": rain_status()["version"]}
    return _call(info)