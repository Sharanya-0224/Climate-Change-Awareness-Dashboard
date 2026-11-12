"""
api/model_store.py

Loads the three champion models once, when the API starts, and hands them to the
endpoints.

Two ways to load, chosen automatically:
  1. From a local folder (MODEL_DIR, default ./model_store) if it has the models.
     This is what the Docker image uses: the models are downloaded at build time
     by download_models.py, so the running container needs no DagsHub password.
  2. From the MLflow Model Registry on DagsHub, using the alias "champion"
     (models:/<name>@champion). This is what you use on your laptop; it needs the
     MLFLOW_* variables from .env.
"""
import json
import os
from pathlib import Path

REGISTERED = {"m1": "m1_country_emissions", "m2": "m2_co2_growth", "m3": "m3_india_temperature"}
ALIAS = "champion"

_models = {}
_info = {}


def set_model(key, model, version="test", source="injected"):
    """Used by tests (and by load_all) to place a model in the store."""
    _models[key] = model
    _info[key] = {"name": REGISTERED[key], "version": str(version), "source": source}


def get(key):
    """Returns (model, info). Raises KeyError if that model is not loaded."""
    return _models[key], _info[key]


def info():
    return {key: _info.get(key, {"name": name, "version": None, "source": "not loaded"})
            for key, name in REGISTERED.items()}


def all_loaded():
    return all(key in _models for key in REGISTERED)


def clear():
    _models.clear()
    _info.clear()


def load_all():
    import mlflow
    import mlflow.sklearn
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(".env"))
    except ImportError:
        pass

    model_dir = Path(os.getenv("MODEL_DIR", "model_store"))
    versions_file = model_dir / "versions.json"
    versions = json.loads(versions_file.read_text()) if versions_file.exists() else {}

    for key, name in REGISTERED.items():
        local = model_dir / name
        # the model folder is wherever the MLmodel file sits (directly inside, or one level down)
        mlmodel = next(iter(sorted(local.rglob("MLmodel"))), None) if local.exists() else None
        try:
            if mlmodel is not None:
                model = mlflow.sklearn.load_model(str(mlmodel.parent))
                set_model(key, model, versions.get(name, {}).get("version", "unknown"), "local folder")
            else:
                model = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS}")
                version = mlflow.MlflowClient().get_model_version_by_alias(name, ALIAS).version
                set_model(key, model, version, "mlflow registry")
            print(f"loaded {name} v{_info[key]['version']} from {_info[key]['source']}")
        except Exception as e:      # one model failing must not stop the others
            print(f"could NOT load {name}: {e.__class__.__name__}: {e}")
