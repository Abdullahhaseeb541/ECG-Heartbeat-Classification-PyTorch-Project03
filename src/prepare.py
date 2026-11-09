"""Task 1: inspect, split and convert the ECG data."""
import json
import numpy as np
import torch

from .config import (
    DATA_DIR, RESULTS_DIR, PREPARED_FILE, CLASS_CODES,
    VALIDATION_SPLIT_SEED, VALIDATION_FRACTION,
)
from .load_data import load_raw


def _class_counts(labels):
    counts = np.bincount(labels.astype(np.int64), minlength=5)
    return {CLASS_CODES[i]: int(counts[i]) for i in range(5)}


def prepare_data(data_dir=DATA_DIR, results_dir=RESULTS_DIR, save=True):
    """Prepare train/validation/test arrays using the assignment's exact split.

    The official Kaggle test CSV remains untouched. Exactly 10% of the
    original training CSV is assigned to validation using the supplied
    NumPy seed-42 permutation code.
    """
    raw = load_raw(data_dir)
    train_df = raw["train"]
    test_df = raw["test"]

    # Task 1.2: separate signal and target; keep the already-scaled signals.
    X_all = train_df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_all_raw = train_df.iloc[:, 187]
    label_dtype_before = str(y_all_raw.dtype)
    y_all = y_all_raw.to_numpy(dtype=np.int64)

    X_test = test_df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_test_raw = test_df.iloc[:, 187]
    test_label_dtype_before = str(y_test_raw.dtype)
    y_test = y_test_raw.to_numpy(dtype=np.int64)

    for name, labels in (("training", y_all), ("test", y_test)):
        if not np.isin(labels, np.arange(5)).all():
            raise ValueError(f"The {name} labels must be integers from 0 through 4.")

    if not np.isfinite(X_all).all() or not np.isfinite(X_test).all():
        raise ValueError("The ECG signal data contains NaN or infinite values.")

    # Task 1.6: preserve the exact split code required in the assignment.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Task 1.7: convert every split to the required PyTorch dtypes.
    X_train_tensor = torch.from_numpy(X_train).to(dtype=torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(dtype=torch.long)
    X_val_tensor = torch.from_numpy(X_val).to(dtype=torch.float32)
    y_val_tensor = torch.from_numpy(y_val).to(dtype=torch.long)
    X_test_tensor = torch.from_numpy(X_test).to(dtype=torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(dtype=torch.long)
    tensor_dtype_check = {
        "signals": str(X_train_tensor.dtype),
        "labels": str(y_train_tensor.dtype),
        "validation_signals": str(X_val_tensor.dtype),
        "test_labels": str(y_test_tensor.dtype),
    }

    expected_train_counts = {"N": 72471, "S": 2223, "V": 5788, "F": 641, "Q": 6431}
    expected_test_counts = {"N": 18118, "S": 556, "V": 1448, "F": 162, "Q": 1608}
    actual_train_counts = _class_counts(y_all)
    actual_test_counts = _class_counts(y_test)

    summary = {
        "source": "Kaggle ECG Heartbeat Categorization Dataset (MIT-BIH files)",
        "train_file_shape": [int(v) for v in train_df.shape],
        "test_file_shape": [int(v) for v in test_df.shape],
        "label_dtype_before_conversion": label_dtype_before,
        "test_label_dtype_before_conversion": test_label_dtype_before,
        "signal_minimum_training_file": float(X_all.min()),
        "signal_maximum_training_file": float(X_all.max()),
        "rows": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
            "original_training_file": int(len(train_df)),
        },
        "class_counts_original_train_file": actual_train_counts,
        "class_counts_test_file": actual_test_counts,
        "matches_assignment_class_count_table": (actual_train_counts == expected_train_counts and actual_test_counts == expected_test_counts),
        "expected_class_counts_train_file": expected_train_counts,
        "expected_class_counts_test_file": expected_test_counts,
        "class_counts_train_split": _class_counts(y_train),
        "class_counts_validation_split": _class_counts(y_val),
        "validation_fraction": VALIDATION_FRACTION,
        "validation_split_seed": VALIDATION_SPLIT_SEED,
        "tensor_dtypes": tensor_dtype_check,
        "signal_columns": 187,
        "label_column_index": 187,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    if save:
        np.savez_compressed(
            results_dir / "prepared_data.npz",
            X_train=X_train_tensor.numpy(), y_train=y_train_tensor.numpy(),
            X_val=X_val_tensor.numpy(), y_val=y_val_tensor.numpy(),
            X_test=X_test_tensor.numpy(), y_test=y_test_tensor.numpy(),
            train_indices=train_idx, validation_indices=val_idx,
        )
        with (results_dir / "data_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        import csv
        with (results_dir / "table1_data.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["item", "value"])
            writer.writerow(["training_rows_after_split", len(X_train)])
            writer.writerow(["validation_rows", len(X_val)])
            writer.writerow(["official_test_rows", len(X_test)])
            writer.writerow(["training_signal_minimum", float(X_all.min())])
            writer.writerow(["training_signal_maximum", float(X_all.max())])
            for code, count in actual_train_counts.items():
                writer.writerow([f"original_training_file_class_{code}", count])
            for code, count in actual_test_counts.items():
                writer.writerow([f"official_test_file_class_{code}", count])

    print("Task 1 - Data preparation complete")
    print(f"Original train file: {train_df.shape}")
    print(f"Official test file:  {test_df.shape}")
    print(f"Training rows:       {len(X_train):,}")
    print(f"Validation rows:     {len(X_val):,}")
    print(f"Test rows:           {len(X_test):,}")
    print(f"Training signal min/max: {X_all.min():.6f} / {X_all.max():.6f}")
    print(f"Training label dtype before conversion: {label_dtype_before}")
    print("Class counts in original training file:", summary["class_counts_original_train_file"])
    print("Class counts in official test file:   ", summary["class_counts_test_file"])
    print("Class counts match assignment table:  ", summary["matches_assignment_class_count_table"])
    if save:
        print(f"Saved prepared arrays to: {results_dir / 'prepared_data.npz'}")
    return summary


if __name__ == "__main__":
    prepare_data()
