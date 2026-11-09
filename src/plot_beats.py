"""Plot one example heartbeat for each of the five classes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS_DIR, CLASS_NAMES
from .load_data import load_raw

FS = 125  # samples per second


def plot_one_per_class(frame, out_path=RESULTS_DIR / "example_beats.png"):
    """Save a five-panel plot using the first example of each class."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_ms = 1000.0 * np.arange(187) / FS
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)

    for class_id, axis in enumerate(axes):
        matching = frame[frame[187].astype(int) == class_id]
        if matching.empty:
            raise ValueError(f"No examples of class {class_id} were found.")
        signal = matching.iloc[0, :187].to_numpy(dtype=np.float32)
        axis.plot(time_ms, signal, linewidth=1.25)
        axis.set_title(CLASS_NAMES[class_id], fontsize=8)
        axis.set_xlabel("Time (ms)")
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Scaled ECG amplitude")
    fig.suptitle("Example ECG Heartbeats by Class", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    raw = load_raw()
    plot_one_per_class(raw["train"])
