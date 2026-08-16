import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
REGISTRY_DIR = BASE_DIR / "registry"

# Splitting Configuration
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Paths to raw files
HEART_CSV_PATH = DATA_DIR / "heart.csv"
CLEVELAND_CSV_PATH = DATA_DIR / "heart_cleveland_upload.csv"

# Quality Gate Thresholds
MIN_RECALL = 0.85
MIN_F1 = 0.80
MIN_ROC_AUC = 0.85
