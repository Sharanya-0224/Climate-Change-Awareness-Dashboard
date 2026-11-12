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


# register_champions.py writes this tag on a model version only when every gate passed
GATES_TAG, GATES_PASSED = "gates", "all passed"
RAIN_METADATA = Path("rain_package/metadata.json")


def check_rain_package(client):
    """The rain model ships as a package in the repo. It must be the version the
    registry calls champion. Returns a problem message, or None if all is well."""
    if not RAIN_METADATA.exists():
        print("registry check skipped: no rain_package/metadata.json")
        return None
    meta = json.loads(RAIN_METADATA.read_text())
    name, packaged = meta["registered_name"], str(meta["registered_version"])
    try:
        champion = str(client.get_model_version_by_alias(name, ALIAS).version)
    except Exception as e:
        return f"{name}: no '{ALIAS}' version found in the registry ({e.__class__.__name__})"
    if champion != packaged:
        return (f"{name}: rain_package holds version {packaged} but the registry champion "
                f"is version {champion}; rebuild the package from the champion")
    print(f"registry check passed: {name} package is version {packaged}, the registry champion")
    return None


def main():
    out = Path(os.getenv("MODEL_DIR", "model_store"))
    out.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient()
    versions, problems = {}, []
    for name in REGISTERED.values():
        mv = client.get_model_version_by_alias(name, ALIAS)
        tags = dict(client.get_model_version(name, mv.version).tags or {})
        if tags.get(GATES_TAG) != GATES_PASSED:
            problems.append(f"{name} version {mv.version}: tag {GATES_TAG}={tags.get(GATES_TAG)!r}, "
                            f"expected {GATES_PASSED!r}")
            continue
        print(f"registry check passed: {name} version {mv.version} has {GATES_TAG} = {GATES_PASSED}")
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{ALIAS}", dst_path=str(target))
        versions[name] = {"version": str(mv.version), "run_id": mv.run_id, "alias": ALIAS,
                          "gates": tags.get(GATES_TAG), "git_commit": tags.get("git_commit")}
        print(f"downloaded {name} version {mv.version} -> {target}")
    (out / "versions.json").write_text(json.dumps(versions, indent=2))
    print(f"wrote {out / 'versions.json'}")
    rain_problem = check_rain_package(client)
    if rain_problem:
        problems.append(rain_problem)
    if problems:
        print("\nREGISTRY CHECK FAILED: the image must not be built")
        for p in problems:
            print("  -", p)
        sys.exit(1)


if __name__ == "__main__":
    main()
