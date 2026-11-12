"""
tests/test_model_store.py

Tests the part of the API that loads models from a local folder (what the Docker
container does). A small folder in MLflow's layout is created on the fly, so no
registry and no MLflow are needed.
"""
import pickle

import pytest
from sklearn.linear_model import LinearRegression

from api import model_store, service
from climate_models.m1_model import M1GrowthModel
from climate_models.m3_model import M3Model, MODELS, INPUT_COLS, TARGET


def save_like_mlflow(model, folder, nested=False):
    """Write a model the way MLflow's scikit-learn format does: an MLmodel text file + model.pkl."""
    target = folder / "artifacts" if nested else folder
    target.mkdir(parents=True, exist_ok=True)
    (target / "MLmodel").write_text(
        "flavors:\n  sklearn:\n    pickled_model: model.pkl\n    serialization_format: cloudpickle\n")
    with open(target / "model.pkl", "wb") as f:
        pickle.dump(model, f)


@pytest.fixture()
def store_dir(tmp_path, m1, m2, m3, monkeypatch):
    save_like_mlflow(M1GrowthModel(step=5).fit(m1), tmp_path / "m1_country_emissions")
    save_like_mlflow(LinearRegression().fit(m2[service.M2_FEATURES], m2["growth_ppm"]),
                     tmp_path / "m2_co2_growth")
    save_like_mlflow(M3Model(**MODELS["linear_co2"]).fit(m3[INPUT_COLS], m3[TARGET]),
                     tmp_path / "m3_india_temperature", nested=True)
    (tmp_path / "versions.json").write_text(
        '{"m1_country_emissions": {"version": "7"}, "m2_co2_growth": {"version": "8"}, '
        '"m3_india_temperature": {"version": "9"}}')
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    model_store.clear()
    yield tmp_path
    model_store.clear()


def test_loads_all_three_from_local_folder_with_versions(store_dir):
    model_store.load_all()
    assert model_store.all_loaded()
    info = model_store.info()
    assert {v["source"] for v in info.values()} == {"local folder"}
    assert info["m1"]["version"] == "7" and info["m3"]["version"] == "9"


def test_finds_model_one_level_down(store_dir):
    assert model_store.find_model_folder(store_dir / "m3_india_temperature").name == "artifacts"
    assert model_store.find_model_folder(store_dir / "does_not_exist") is None


def test_loaded_models_predict(store_dir):
    model_store.load_all()
    m3_model, _ = model_store.get("m3")
    out = service.predict_india_temperature(m3_model, 26.9, 75.8, 2030, 5, 440.0)
    assert -10 < out["anomaly_degc"] < 10


def test_smoke_test_passes_and_fails_correctly(store_dir, monkeypatch):
    from api import smoke_test
    assert smoke_test.main() == 0
    monkeypatch.setenv("MODEL_DIR", str(store_dir / "empty"))
    monkeypatch.setenv("MODEL_REGISTRY_FALLBACK", "0")      # as in the container: never try the registry
    assert smoke_test.main() == 1
