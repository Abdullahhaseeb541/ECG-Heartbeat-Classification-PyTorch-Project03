"""Shared project configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

TRAIN_FILE = "mitbih_train.csv"
TEST_FILE = "mitbih_test.csv"
PREPARED_FILE = RESULTS_DIR / "prepared_data.npz"

N_SAMPLES = 187
N_CLASSES = 5
CLASS_CODES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = [
    "N - Normal beat",
    "S - Supraventricular ectopic beat",
    "V - Ventricular ectopic beat",
    "F - Fusion beat",
    "Q - Unknown / paced beat",
]

# Values and training settings specified by Project 03.
HIDDEN_UNITS = (128, 64)
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATES = (0.01, 0.1, 0.5)
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SEED = 0
VALIDATION_SPLIT_SEED = 42
VALIDATION_FRACTION = 0.10
