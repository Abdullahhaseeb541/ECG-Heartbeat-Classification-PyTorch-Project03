"""Input validation and batched inference helpers for the Streamlit app."""
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import N_CLASSES, N_SAMPLES


def _read_numeric_csv(contents: bytes, expected_columns: int, description: str) -> np.ndarray:
    if not contents:
        raise ValueError(f"{description} is empty.")

    try:
        frame = pd.read_csv(BytesIO(contents), header=None, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"{description} must be a readable CSV file.") from error

    if frame.empty:
        raise ValueError(f"{description} must contain at least one row.")
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"{description} must contain exactly {expected_columns} columns; "
            f"found {frame.shape[1]}."
        )

    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must contain numeric values and no header row.") from error

    if not np.isfinite(values).all():
        raise ValueError(f"{description} contains missing or non-finite values.")
    return values


def read_signal_csv(contents: bytes) -> np.ndarray:
    """Read exactly one headerless CSV heartbeat with 187 numeric values."""
    values = _read_numeric_csv(contents, N_SAMPLES, "The heartbeat CSV")
    if values.shape[0] != 1:
        raise ValueError(f"The heartbeat CSV must contain exactly one row; found {values.shape[0]}.")
    return values[0].astype(np.float32)


def read_labeled_csv(contents: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read a headerless test CSV with 187 signals followed by a class label."""
    values = _read_numeric_csv(contents, N_SAMPLES + 1, "The labeled test CSV")
    labels = values[:, N_SAMPLES]
    rounded_labels = np.rint(labels)
    if not np.equal(labels, rounded_labels).all():
        raise ValueError("The final column must contain whole-number class labels.")
    if not np.isin(rounded_labels, np.arange(N_CLASSES)).all():
        raise ValueError(f"Class labels must be integers from 0 through {N_CLASSES - 1}.")

    signals = values[:, :N_SAMPLES].astype(np.float32)
    return signals, rounded_labels.astype(np.int64)


def predict_batch(
    model: nn.Module,
    signals: np.ndarray,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class IDs and class probabilities for each signal."""
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != N_SAMPLES or values.shape[0] == 0:
        raise ValueError(f"Signals must have shape (rows, {N_SAMPLES}) with at least one row.")
    if not np.isfinite(values).all():
        raise ValueError("Signals contain missing or non-finite values.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    device = next(model.parameters()).device
    predicted_parts = []
    probability_parts = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits = model(batch)
            probability_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
            predicted_parts.append(logits.argmax(dim=1).cpu().numpy())

    return (
        np.concatenate(predicted_parts).astype(np.int64, copy=False),
        np.concatenate(probability_parts),
    )
