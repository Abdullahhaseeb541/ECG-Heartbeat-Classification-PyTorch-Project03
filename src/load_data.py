"""Load the two original Kaggle MIT-BIH CSV files without changing them."""
from pathlib import Path
import pandas as pd

from .config import DATA_DIR, TRAIN_FILE, TEST_FILE

FILES = {"train": TRAIN_FILE, "test": TEST_FILE}
EXPECTED_COLUMNS = 188


def load_raw(data_dir=DATA_DIR):
    """Return {'train': DataFrame, 'test': DataFrame} exactly as stored.

    The files have no header row. Columns 0..186 contain one ECG heartbeat
    and column 187 contains its integer class label. This loader intentionally
    does not split, scale, normalise, or otherwise modify the data.
    """
    data_dir = Path(data_dir)
    frames = {}

    for split, filename in FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download {filename} from the Kaggle dataset "
                "and place it in the project's data/ folder. See README.md."
            )
        frame = pd.read_csv(path, header=None, low_memory=False)
        if frame.shape[1] != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path.name} must contain 188 columns (187 signal values + 1 label); "
                f"found {frame.shape[1]}. Check that the correct CSV was downloaded."
            )
        frames[split] = frame

    return frames


if __name__ == "__main__":
    raw = load_raw()
    for split, frame in raw.items():
        print(f"{split}: {frame.shape[0]:,} rows, {frame.shape[1]} columns")
    print("\nFirst three rows (selected columns):")
    print(raw["train"].iloc[:3, [0, 1, 2, 185, 186, 187]].to_string(index=False))
