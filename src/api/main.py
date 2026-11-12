"""
api/main.py

The inference service: a small web server that answers prediction requests.

Run on your laptop from the repo root:
    uvicorn api.main:app --app-dir src --port 8000
then open http://127.0.0.1:8000/docs for the interactive page.

Endpoints
  GET  /health                       is the service up, which model versions are loaded
  POST /predict/india_temperature    Model 3: temperature anomaly for a place, month and CO2 level
  POST /predict/co2_growth           Model 2: yearly CO2 rise from emissions and El Nino
  POST /predict/country_emissions    Model 1: a country's emissions from population and income change

Every answer says which model and version produced it (traceability).
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../src, so climate_models imports

from api import model_store, service

# The rain model is a separate part of the app. If its code cannot even be imported
# (for example LightGBM is not installed), the three climate endpoints must still work,
# so the import is guarded. The Docker build is strict instead: its smoke test fails
# when the rain model is missing, so an image without it can never be built.
try:
    from api import rain_router as rain
    RAIN_IMPORT_ERROR = None
except Exception as e:
    rain = None
    RAIN_IMPORT_ERROR = f"{e.__class__.__name__}: {e}"
from api.schemas import (CO2GrowthRequest, CO2GrowthResponse, CountryEmissionsRequest,
                         CountryEmissionsResponse, HealthResponse, IndiaTemperatureRequest,
                         IndiaTemperatureResponse)


@asynccontextmanager
async def lifespan(app):
    # Runs once when the server starts: load the champion models.
    if os.getenv("API_SKIP_MODEL_LOAD") != "1":
        model_store.load_all()
        if rain is None:
            print(f"could NOT import the rain router: {RAIN_IMPORT_ERROR}")
        elif rain.load_rain_model():        # never raises; returns True or False
            status = rain.rain_status()
            print(f"loaded {status['name']} v{status['version']} from rain package")
        else:
            print("could NOT load the rain model (its routes will answer 503)")
    yield


app = FastAPI(
    title="Climate Change Awareness API",
    description="Serves the three champion models: country emissions, CO2 growth, India temperature.",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS: lets a dashboard running in a browser on another address call this API.
# ALLOWED_ORIGINS is a comma-separated list; "*" (any site) is fine for a public,
# read-only API with no logins. Set it to the dashboard's address to tighten it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Rain routes: /rain/forecast, /rain/predict, /rain/model-info
if rain is not None:
    app.include_router(rain.router, prefix="/rain")


def _rain_info():
    """The rain model in the same shape as the other models in /health."""
    if rain is None:
        return {"name": "rain_v2_3month_lgbm", "version": None, "source": "not loaded"}
    status = rain.rain_status()
    version = None if status["version"] is None else str(status["version"])
    return {"name": status["name"], "version": version,
            "source": "rain package" if status["loaded"] else "not loaded"}


def _model(key):
    """The loaded model, or error 503 (service unavailable) if it is not loaded."""
    try:
        return model_store.get(key)
    except KeyError:
        raise HTTPException(status_code=503, detail=f"model '{key}' is not loaded")


def _served(info):
    return {"served_model": info["name"], "served_version": info["version"]}


@app.get("/", include_in_schema=False)
def root():
    """Opening the main address sends you to the interactive docs page."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    models = {**model_store.info(), "rain": _rain_info()}
    all_ok = model_store.all_loaded() and models["rain"]["source"] != "not loaded"
    return {"status": "ok" if all_ok else "degraded", "models": models}


@app.post("/predict/india_temperature", response_model=IndiaTemperatureResponse)
def india_temperature(req: IndiaTemperatureRequest):
    model, info = _model("m3")
    try:
        result = service.predict_india_temperature(model, **req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {**result, **_served(info)}


@app.post("/predict/co2_growth", response_model=CO2GrowthResponse)
def co2_growth(req: CO2GrowthRequest):
    model, info = _model("m2")
    return {**service.predict_co2_growth(model, **req.model_dump()), **_served(info)}


@app.post("/predict/country_emissions", response_model=CountryEmissionsResponse)
def country_emissions(req: CountryEmissionsRequest):
    model, info = _model("m1")
    return {**service.predict_country_emissions(model, **req.model_dump()), **_served(info)}
