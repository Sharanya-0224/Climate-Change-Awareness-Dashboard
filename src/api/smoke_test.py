"""
api/smoke_test.py

A quick end-to-end check: load the three models from the local folder and ask
each one for a prediction. Exits with an error if anything is wrong.

The Docker build runs this (RUN python -m api.smoke_test). If a model is missing
or cannot be loaded with the image's library versions, the BUILD fails, so a
broken image can never be produced or deployed.

Run by hand from the repo root:  PYTHONPATH=src python3 -m api.smoke_test
"""
import sys

from api import model_store, service


def main():
    model_store.clear()
    model_store.load_all()
    missing = [k for k, v in model_store.info().items() if v["source"] != "local folder"]
    if missing:
        print(f"SMOKE TEST FAILED: not loaded from the local folder: {missing}")
        return 1

    m1, _ = model_store.get("m1")
    m2, _ = model_store.get("m2")
    m3, _ = model_store.get("m3")
    results = {
        "country_emissions": service.predict_country_emissions(m1, 2800.0, 8.0, 60.0, 10)["co2_mt"],
        "co2_growth": service.predict_co2_growth(m2, 11.0, 0.5, 0.0)["co2_growth_ppm"],
        "india_temperature": service.predict_india_temperature(m3, 26.9, 75.8, 2030, 5, 440.0)["anomaly_degc"],
    }
    for name, value in results.items():
        print(f"  {name}: {value}")
        if not isinstance(value, float) or value != value:      # value != value is true only for NaN
            print(f"SMOKE TEST FAILED: {name} did not return a number")
            return 1
    try:
        from api import rain_router
        rain_value = rain_router.rain_smoke()     # loads the package and makes one real forecast
    except Exception as e:
        print(f"SMOKE TEST FAILED: rain model: {e.__class__.__name__}: {e}")
        return 1
    print(f"  rain_forecast_z: {rain_value}")
    print("SMOKE TEST PASSED: 4 models loaded and predicting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
