"""Task 5.2: metric functions implemented without scikit-learn."""
import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of all predictions that are correct."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays of the same shape.")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes=5):
    """Return C[j, k] = number of true class-j rows predicted as class k."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays of the same shape.")
    if not (np.isin(y_true, np.arange(num_classes)).all() and
            np.isin(y_pred, np.arange(num_classes)).all()):
        raise ValueError("Labels and predictions must be valid class indices.")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_accuracy(matrix):
    """Return the diagonal / row sum for each true class."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    row_totals = matrix.sum(axis=1)
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    non_empty = row_totals > 0
    result[non_empty] = np.diag(matrix)[non_empty] / row_totals[non_empty]
    return result


def balanced_accuracy(matrix):
    """Mean of the per-class accuracies (each class has equal weight)."""
    matrix = np.asarray(matrix, dtype=np.int64)
    values = per_class_accuracy(matrix)
    represented = matrix.sum(axis=1) > 0
    if not represented.any():
        raise ValueError("Balanced accuracy requires at least one true class example.")
    return float(values[represented].mean())
