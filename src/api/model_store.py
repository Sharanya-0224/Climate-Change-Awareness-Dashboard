"""
api/model_store.py

Loads the three champion models once, when the API starts, and hands them to the
endpoints.

Two ways to load, chosen automatically:
  1. From a local folder (MODEL_DIR, default ./model_store) if it has the models.
     This is what the Docker image uses. The folder is filled beforehand by
     download_models.py, so the running container needs no DagsHub password and
     no internet. Loading from a folder also needs no MLflow installed, which
     keeps the image small.
  2. From the MLflow Model Registry on DagsHub, using the alias "champion"
     (models:/<name>@champion). This is what you use on your laptop when there is
     no local folder; it needs MLflow and the MLFLOW_* variables from .env.
     Setting MODEL_REGISTRY_FALLBACK=0 switches this off (the container does that:
     it must never reach out to the internet for a model).
"""
import json
import os
import pickle
import sys
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


def find_model_folder(folder):
    """The saved model is wherever the MLmodel file sits (directly inside, or deeper)."""
    folder = Path(folder)
    if not folder.exists():
        return None
    found = sorted(folder.rglob("MLmodel"))
    return found[0].parent if found else None


def load_local(model_folder):
    """Load a model that MLflow saved in the scikit-learn format, without needing MLflow.

    MLflow stores such a model as a folder with a small text file 'MLmodel' (which says
    where the pickled model is) and the pickle itself. If the model was logged with
    code_paths, its helper code sits in a 'code' subfolder, which is put on the import path.
    """
    import yaml
    model_folder = Path(model_folder)
    conf = yaml.safe_load((model_folder / "MLmodel").read_text())
    flavor = (conf.get("flavors") or {}).get("sklearn") or {}
    code_dir = model_folder / (flavor.get("code") or "code")
    if code_dir.exists() and str(code_dir) not in sys.path:
        sys.path.append(str(code_dir))          # appended: the app's own src/ stays first
    with open(model_folder / flavor.get("pickled_model", "model.pkl"), "rb") as f:
        return pickle.load(f)


def load_all():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(".env"))
    except ImportError:
        pass

    model_dir = Path(os.getenv("MODEL_DIR", "model_store"))
    versions_file = model_dir / "versions.json"
    versions = json.loads(versions_file.read_text()) if versions_file.exists() else {}

    for key, name in REGISTERED.items():
        try:
            folder = find_model_folder(model_dir / name)
            if folder is not None:
                model = load_local(folder)
                set_model(key, model, versions.get(name, {}).get("version", "unknown"), "local folder")
            elif os.getenv("MODEL_REGISTRY_FALLBACK", "1") != "1":
                raise FileNotFoundError(f"no saved model under {model_dir / name}")
            else:
                import mlflow
                import mlflow.sklearn
                model = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS}")
                version = mlflow.MlflowClient().get_model_version_by_alias(name, ALIAS).version
                set_model(key, model, version, "mlflow registry")
            print(f"loaded {name} v{_info[key]['version']} from {_info[key]['source']}")
        except Exception as e:      # one model failing must not stop the others
            print(f"could NOT load {name}: {e.__class__.__name__}: {e}")
