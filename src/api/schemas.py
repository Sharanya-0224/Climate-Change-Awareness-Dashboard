"""
api/schemas.py

The shape of every request and response. FastAPI uses these (Pydantic models) to
  - reject bad input automatically with error 422 and a clear message,
  - build the interactive documentation page at /docs.
The limits come from limits.py, where each one is justified.
"""
from typing import Dict, Optional

from pydantic import BaseModel, Field

from api import limits as L


# ---------------------------------------------------------------- Model 3
class IndiaTemperatureRequest(BaseModel):
    lat: float = Field(..., ge=L.LAT_MIN, le=L.LAT_MAX, description="Latitude in degrees north (India)")
    lon: float = Field(..., ge=L.LON_MIN, le=L.LON_MAX, description="Longitude in degrees east (India)")
    year: int = Field(..., ge=L.YEAR_MIN, le=L.YEAR_MAX)
    month: int = Field(..., ge=1, le=12)
    co2_ppm: float = Field(..., ge=L.CO2_PPM_MIN, le=L.CO2_PPM_MAX,
                           description="CO2 in the air that year, in ppm (2024 was about 424.6)")
    nino_lag1: float = Field(0.0, ge=L.NINO_MIN, le=L.NINO_MAX,
                             description="El Nino index 1 month before (0 = neutral)")
    nino_lag3: float = Field(0.0, ge=L.NINO_MIN, le=L.NINO_MAX, description="El Nino index 3 months before")
    nino_lag6: float = Field(0.0, ge=L.NINO_MIN, le=L.NINO_MAX, description="El Nino index 6 months before")
    nino_prev12_mean: float = Field(0.0, ge=L.NINO_MIN, le=L.NINO_MAX,
                                    description="Average El Nino index over the previous 12 months")
    cum_emissions_gtco2: Optional[float] = Field(None, ge=0,
                                                 description="Only needed if the champion is the emissions-based model")

    model_config = {"json_schema_extra": {"example": {
        "lat": 26.9, "lon": 75.8, "year": 2030, "month": 5, "co2_ppm": 440.0}}}


class IndiaTemperatureResponse(BaseModel):
    anomaly_degc: float
    grid_square: str
    co2_forcing_wm2: float
    beyond_training_range: bool
    note: str
    served_model: str
    served_version: str


# ---------------------------------------------------------------- Model 2
class CO2GrowthRequest(BaseModel):
    total_emissions_gtc: float = Field(..., ge=L.EMISSIONS_GTC_MIN, le=L.EMISSIONS_GTC_MAX,
                                       description="Total human CO2 emissions that year, same unit as the training table")
    nino34: float = Field(0.0, ge=L.NINO_MIN, le=L.NINO_MAX, description="El Nino index this year (0 = neutral)")
    nino34_lag1: float = Field(0.0, ge=L.NINO_MIN, le=L.NINO_MAX, description="El Nino index last year")

    model_config = {"json_schema_extra": {"example": {
        "total_emissions_gtc": 11.0, "nino34": 0.5, "nino34_lag1": 0.0}}}


class CO2GrowthResponse(BaseModel):
    co2_growth_ppm: float
    note: str
    served_model: str
    served_version: str


# ---------------------------------------------------------------- Model 1
class CountryEmissionsRequest(BaseModel):
    co2_last_mt: float = Field(..., ge=L.CO2_MT_MIN, le=L.CO2_MT_MAX,
                               description="The country's last known yearly CO2 emissions, million tonnes")
    population_change_pct: float = Field(..., ge=L.PCT_CHANGE_MIN, le=L.PCT_CHANGE_MAX,
                                         description="Total % change in population over the period")
    income_change_pct: float = Field(..., ge=L.PCT_CHANGE_MIN, le=L.PCT_CHANGE_MAX,
                                     description="Total % change in income per person over the period")
    years_ahead: int = Field(..., ge=L.YEARS_AHEAD_MIN, le=L.YEARS_AHEAD_MAX)

    model_config = {"json_schema_extra": {"example": {
        "co2_last_mt": 2800.0, "population_change_pct": 8.0, "income_change_pct": 60.0, "years_ahead": 10}}}


class CountryEmissionsResponse(BaseModel):
    co2_mt: float
    change_pct: float
    beyond_tested_horizon: bool
    note: str
    served_model: str
    served_version: str


# ---------------------------------------------------------------- health
class HealthResponse(BaseModel):
    status: str
    models: Dict[str, Dict[str, Optional[str]]]
