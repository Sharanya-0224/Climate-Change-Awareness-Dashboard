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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # .../src, so climate_models imports

from api import model_store, service
from api.schemas import (CO2GrowthRequest, CO2GrowthResponse, CountryEmissionsRequest,
                         CountryEmissionsResponse, HealthResponse, IndiaTemperatureRequest,
                         IndiaTemperatureResponse)


@asynccontextmanager
async def lifespan(app):
    # Runs once when the server starts: load the champion models.
    if os.getenv("API_SKIP_MODEL_LOAD") != "1":
        model_store.load_all()
    yield


app = FastAPI(
    title="Climate Change Awareness API",
    description="Serves the three champion models: country emissions, CO2 growth, India temperature.",
    version="1.0.0",
    lifespan=lifespan,
)


def _model(key):
    """The loaded model, or error 503 (service unavailable) if it is not loaded."""
    try:
        return model_store.get(key)
    except KeyError:
        raise HTTPException(status_code=503, detail=f"model '{key}' is not loaded")


def _served(info):
    return {"served_model": info["name"], "served_version": info["version"]}


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok" if model_store.all_loaded() else "degraded", "models": model_store.info()}


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
