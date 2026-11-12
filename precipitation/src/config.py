"""Shared configuration for the rain v2 model (3-month total, origin month t)."""
import os
import subprocess
from pathlib import Path

# Folders
SRC_DIR = Path(__file__).resolve().parent          # precipitation/src
PRECIP_DIR = SRC_DIR.parent                         # precipitation
REPO_ROOT = PRECIP_DIR.parent                       # Climate-Change-Awareness-Dashboard
PROCESSED = PRECIP_DIR / "data" / "processed"

INTERIM = PRECIP_DIR / "data" / "interim"
RAW = PRECIP_DIR / "data" / "raw"
MONTHLY_RAW_PATH = INTERIM / "monthly_rain_v2.parquet"
MONTHLY_CLEAN_PATH = INTERIM / "monthly_rain_v2_02.parquet"   # check folder
NORMALS_PATH = PROCESSED / "rain_v2_3month_normals_02.parquet"
POINT_REGION_PATH = PROCESSED / "point_region_v2.csv"
RAIN_2025_PATH = INTERIM / "imd_rainfall_2025.csv"
REFERENCE_DIR = PRECIP_DIR / "inference" / "reference_models"
TEST_PRED_PATH = PROCESSED / "rain_v2_3month_test_predictions.parquet"
START_YEAR = 1901
END_YEAR = 2025
REGIONS = ["northwest", "central", "south_peninsula", "east_northeast"]
MIN_VALID_TRAIN_YEARS = 50   # points with fewer non-zero train years are dropped

# Data files
DATA_PATH = PROCESSED / "rain_v2_3month_dataset_02.parquet"
FEATURE_PATH = PROCESSED / "rain_v2_3month_feature_list_02.json"
VAL_PRED_PATH = PROCESSED / "rain_v2_3month_val_predictions.parquet"
# TEST_PRED_PATH: add after you confirm the file name from 08

# Package output (copied to repo-root rain_package/ at the end)
PACKAGE_DIR = PRECIP_DIR / "inference" / "rain_package"

# Splits (by year of the origin month)
TRAIN_END = 2002
VAL_START = 2003
VAL_END = 2013
TEST_START = 2014
TEST_END = 2025

# Constants
OUTLIER_THRESHOLD = 6   # |z| above this counts as extreme (reported separately)
SD_FLOOR = 5            # mm, minimum sd in normals
RANDOM_STATE = 42

# MLflow
MLFLOW_URI = "https://dagshub.com/Sharanya-0224/Climate-Change-Awareness-Dashboard.mlflow"
MLFLOW_USER = "Sharanya-0224"
EXPERIMENTS = {
    "baselines": "rain_v2_3month_baselines",
    "ridge": "rain_v2_3month_ridge",
    "lightgbm": "rain_v2_3month_lightgbm",
    "lgbm_cv": "rain_v2_3month_lgbm_cv",
}
REGISTERED_MODEL = "rain_v2_3month_lgbm"
REGISTERED_REFERENCE = "rain_v2_3month_ridge"


def get_git_commit():
    """Short git commit of the repo."""
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def setup_mlflow():
    """Connect to DagsHub MLflow. Token is read from the repo-root .env."""
    import mlflow
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_USER_TOKEN"]
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_system_metrics_sampling_interval(1)
    return mlflow