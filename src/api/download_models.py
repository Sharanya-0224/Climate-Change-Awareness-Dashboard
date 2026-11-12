"""
api/download_models.py

Downloads the three champion models from the MLflow Model Registry into a local
folder (default ./model_store), and records which version of each was downloaded.

Why: the Docker image gets the models baked in at build time, so the running
container needs no DagsHub password and starts quickly. CI/CD runs this script
before building the image.

Run from the repo root (needs the MLFLOW_* variables, from .env or the environment):
    python3 src/api/download_models.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.model_store import ALIAS, REGISTERED

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".env"))
except ImportError:
    pass


def main():
    out = Path(os.getenv("MODEL_DIR", "model_store"))
    out.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient()
    versions = {}
    for name in REGISTERED.values():
        mv = client.get_model_version_by_alias(name, ALIAS)
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{ALIAS}", dst_path=str(target))
        versions[name] = {"version": str(mv.version), "run_id": mv.run_id, "alias": ALIAS}
        print(f"downloaded {name} version {mv.version} -> {target}")
    (out / "versions.json").write_text(json.dumps(versions, indent=2))
    print(f"wrote {out / 'versions.json'}")


if __name__ == "__main__":
    main()
