"""Task 5: load the selected checkpoint and evaluate the official test set once."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import PREPARED_FILE, RESULTS_DIR, CLASS_CODES, CLASS_NAMES
from .model import ECGNet
from .metrics import accuracy, confusion_matrix, per_class_accuracy, balanced_accuracy
from .prepare import prepare_data
from .train import choose_device
from .reporting import generate_reports


def run_test(device_name="auto"):
    checkpoint = RESULTS_DIR / "best_model.pt"
    training_summary_path = RESULTS_DIR / "training_summary.json"
    if not PREPARED_FILE.exists():
        prepare_data()
    if not checkpoint.exists() or not training_summary_path.exists():
        raise FileNotFoundError(
            "Trained checkpoint not found. Run `python -m src.train` before the test step."
        )

    device = choose_device(device_name)
    with np.load(PREPARED_FILE) as prepared:
        X_test = prepared["X_test"].copy()
        y_test = prepared["y_test"].astype(np.int64, copy=True)

    x_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    model = ECGNet().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    probability_parts = []
    prediction_parts = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), 4096):
            logits = model(x_tensor[start:start + 4096])
            probability_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
            prediction_parts.append(logits.argmax(dim=1).cpu().numpy())

    probabilities = np.concatenate(probability_parts, axis=0)
    predictions = np.concatenate(prediction_parts, axis=0).astype(np.int64)
    overall_accuracy = accuracy(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, num_classes=5)
    class_accuracies = per_class_accuracy(matrix)
    balanced = balanced_accuracy(matrix)
    always_n = float(np.mean(y_test == 0))

    metrics = {
        "test_rows": int(len(y_test)),
        "test_accuracy": float(overall_accuracy),
        "balanced_accuracy": float(balanced),
        "always_N_accuracy": always_n,
        "per_class_accuracy": [float(value) for value in class_accuracies],
        "class_counts": [int(value) for value in matrix.sum(axis=1)],
        "confusion_matrix": matrix.tolist(),
        "class_codes": CLASS_CODES,
        "class_names": CLASS_NAMES,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Confusion matrix CSV: true classes are rows; predicted classes are columns.
    confusion_df = pd.DataFrame(matrix, index=CLASS_CODES, columns=CLASS_CODES)
    confusion_df.index.name = "True \\ Predicted"
    confusion_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    per_class_df = pd.DataFrame({
        "class_code": CLASS_CODES,
        "class_name": CLASS_NAMES,
        "test_examples": matrix.sum(axis=1),
        "correct_predictions": np.diag(matrix),
        "per_class_accuracy": class_accuracies,
    })
    per_class_df.to_csv(RESULTS_DIR / "per_class_accuracy.csv", index=False)

    # Table 4 values in a compact machine-readable file.
    with (RESULTS_DIR / "table4_test_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_accuracy", overall_accuracy])
        writer.writerow(["balanced_accuracy", balanced])
        writer.writerow(["always_N_accuracy", always_n])
        for code, value in zip(CLASS_CODES, class_accuracies):
            writer.writerow([f"per_class_accuracy_{code}", value])

    # Row-numbered test data/predictions make the complete held-out set accessible.
    prediction_frame = pd.DataFrame({
        "test_row_number_1_based": np.arange(1, len(y_test) + 1),
        "true_class_id": y_test,
        "true_class": [CLASS_CODES[value] for value in y_test],
        "predicted_class_id": predictions,
        "predicted_class": [CLASS_CODES[value] for value in predictions],
    })
    for class_id, code in enumerate(CLASS_CODES):
        prediction_frame[f"probability_{code}"] = probabilities[:, class_id]
    prediction_frame.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Number of beats")
    axis.set(
        xticks=np.arange(5), yticks=np.arange(5),
        xticklabels=CLASS_CODES, yticklabels=CLASS_CODES,
        xlabel="Predicted class", ylabel="True class",
        title="Test Confusion Matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(5):
        for column in range(5):
            axis.text(column, row, f"{matrix[row, column]:,}",
                      ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)

    with training_summary_path.open("r", encoding="utf-8") as file:
        training_summary = json.load(file)
    generate_reports(training_summary, metrics, matrix)

    print("Task 5 - Official test evaluation complete")
    print(f"Test rows:         {len(y_test):,}")
    print(f"Test accuracy:     {overall_accuracy:.4%}")
    print(f"Balanced accuracy: {balanced:.4%}")
    print(f"Always-N baseline: {always_n:.4%}")
    print("Per-class accuracy:")
    for code, value in zip(CLASS_CODES, class_accuracies):
        print(f"  {code}: {value:.4%}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print(f"Saved predictions: {RESULTS_DIR / 'test_predictions.csv'}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the selected ECG model on the official test file.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    run_test(args.device)
