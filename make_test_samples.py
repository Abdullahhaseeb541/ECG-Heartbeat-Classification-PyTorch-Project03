"""Create smaller correctly formatted ECG test CSVs from mitbih_test.csv.

These generated files are SUBSETS of the official Kaggle test set, not new
independent test data. Use the full official test file for the assignment's
final reported test accuracy.

Format: no header, 187 ECG signal columns + final class label (0..4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/mitbih_test.csv")
    parser.add_argument("--output-dir", default="data/test_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-rows", type=int, default=1000)
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output_dir)
    if not source.is_file():
        raise FileNotFoundError(
            f"Cannot find {source}. Place mitbih_test.csv in the project's data folder "
            "or pass its location with --input."
        )
    if args.random_rows < 1 or args.per_class < 1:
        raise ValueError("Sample sizes must be positive integers.")

    df = pd.read_csv(source, header=None)
    if df.shape[1] != 188:
        raise ValueError(
            f"Expected 188 columns (187 signals + label), received {df.shape[1]}."
        )
    df = df.apply(pd.to_numeric, errors="raise")
    values = df.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("The source CSV contains missing or infinite values.")

    raw_labels = values[:, 187]
    labels = raw_labels.astype(np.int64)
    if not np.array_equal(raw_labels, labels) or not set(np.unique(labels)).issubset(CLASS_NAMES):
        raise ValueError("Labels in column 188 must be integer values 0, 1, 2, 3, or 4.")

    if args.random_rows > len(df):
        raise ValueError(f"Requested {args.random_rows} rows, but source has only {len(df)}.")

    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Balanced sample: the same count from each of the five labelled classes.
    pieces = []
    for class_id in range(5):
        positions = np.flatnonzero(labels == class_id)
        if len(positions) < args.per_class:
            raise ValueError(
                f"Class {CLASS_NAMES[class_id]} has only {len(positions)} rows; "
                f"cannot sample {args.per_class}."
            )
        selected = rng.choice(positions, size=args.per_class, replace=False)
        pieces.append(df.iloc[selected])
    balanced = pd.concat(pieces, ignore_index=True)
    balanced = balanced.iloc[rng.permutation(len(balanced))]
    balanced_path = out_dir / f"balanced_{5 * args.per_class}.csv"
    balanced.to_csv(balanced_path, header=False, index=False)

    # Unstratified random sample preserves approximately the official test distribution.
    random_positions = rng.choice(len(df), size=args.random_rows, replace=False)
    random_sample = df.iloc[random_positions]
    random_path = out_dir / f"random_{args.random_rows}.csv"
    random_sample.to_csv(random_path, header=False, index=False)

    manifest = {
        "source_file": str(source),
        "source_rows": int(len(df)),
        "seed": args.seed,
        "column_format": "187 signal values followed by class label 0-4; no header",
        "warning": "Both outputs are subsets of the official test set, not independent new data.",
        "outputs": {},
    }
    for path, sample in ((balanced_path, balanced), (random_path, random_sample)):
        sample_labels = sample.iloc[:, 187].astype(int)
        counts = {CLASS_NAMES[k]: int((sample_labels == k).sum()) for k in range(5)}
        manifest["outputs"][path.name] = {
            "rows": int(len(sample)),
            "columns": int(sample.shape[1]),
            "class_counts": counts,
        }
        print(f"Created {path}: {len(sample)} rows x {sample.shape[1]} columns")
        print(f"  Class counts: {counts}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}")
    print("Use mitbih_test.csv (all official test rows) for the assignment's final result.")


if __name__ == "__main__":
    main()
