from __future__ import annotations

import numpy as np

from .validation import check_binary_array, check_same_length


def _safe_divide(numerator: float, denominator: float, zero_division: float = 0.0) -> float:
    if denominator == 0:
        return float(zero_division)
    return float(numerator / denominator)


def point_precision(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    y_true = check_binary_array(y_true, name="y_true")
    y_pred = check_binary_array(y_pred, name="y_pred")
    check_same_length(y_true, y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return _safe_divide(tp, tp + fp, zero_division)


def point_recall(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    y_true = check_binary_array(y_true, name="y_true")
    y_pred = check_binary_array(y_pred, name="y_pred")
    check_same_length(y_true, y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return _safe_divide(tp, tp + fn, zero_division)


def point_f1(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    precision = point_precision(y_true, y_pred, zero_division=zero_division)
    recall = point_recall(y_true, y_pred, zero_division=zero_division)
    if precision + recall == 0:
        return float(zero_division)
    return 2 * precision * recall / (precision + recall)
