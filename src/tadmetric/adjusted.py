from __future__ import annotations

import numpy as np

from .converters import binary_to_intervals
from .point import point_precision, point_recall, point_f1
from .validation import check_binary_array, check_same_length


def _adjust_prediction_to_events(y_true, y_pred):
    y_true = check_binary_array(y_true, name="y_true")
    y_pred = check_binary_array(y_pred, name="y_pred")
    check_same_length(y_true, y_pred)

    adjusted = y_pred.copy()
    for start, end in binary_to_intervals(y_true):
        if np.any(y_pred[start:end] == 1):
            adjusted[start:end] = 1
    return y_true, adjusted


def _adjust_prediction_to_k_events(y_true, y_pred, *, k: float):
    if not 0.0 <= k <= 100.0:
        raise ValueError("k must be in [0, 100]")

    y_true = check_binary_array(y_true, name="y_true")
    y_pred = check_binary_array(y_pred, name="y_pred")
    check_same_length(y_true, y_pred)

    adjusted = y_pred.copy()
    for start, end in binary_to_intervals(y_true):
        event_length = end - start
        positives = int(np.sum(y_pred[start:end] == 1))
        required = int(np.ceil(event_length * (k / 100.0)))

        if k == 0.0:
            detected = positives > 0
        else:
            detected = positives >= required

        if detected:
            adjusted[start:end] = 1
    return y_true, adjusted


def point_adjusted_precision(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    _, adjusted = _adjust_prediction_to_events(y_true, y_pred)
    return point_precision(y_true, adjusted, zero_division=zero_division)


def point_adjusted_recall(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    _, adjusted = _adjust_prediction_to_events(y_true, y_pred)
    return point_recall(y_true, adjusted, zero_division=zero_division)


def point_adjusted_f1(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    _, adjusted = _adjust_prediction_to_events(y_true, y_pred)
    return point_f1(y_true, adjusted, zero_division=zero_division)


def point_adjusted_k_precision(y_true, y_pred, *, k: float, zero_division: float = 0.0) -> float:
    _, adjusted = _adjust_prediction_to_k_events(y_true, y_pred, k=k)
    return point_precision(y_true, adjusted, zero_division=zero_division)


def point_adjusted_k_recall(y_true, y_pred, *, k: float, zero_division: float = 0.0) -> float:
    _, adjusted = _adjust_prediction_to_k_events(y_true, y_pred, k=k)
    return point_recall(y_true, adjusted, zero_division=zero_division)


def point_adjusted_k_f1(y_true, y_pred, *, k: float, zero_division: float = 0.0) -> float:
    _, adjusted = _adjust_prediction_to_k_events(y_true, y_pred, k=k)
    return point_f1(y_true, adjusted, zero_division=zero_division)
