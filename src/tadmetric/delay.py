from __future__ import annotations

from typing import List, Optional

import numpy as np

from .converters import binary_to_intervals
from .validation import check_binary_array, check_same_length


def time_to_detect(y_true, y_pred) -> List[Optional[int]]:
    y_true = check_binary_array(y_true, name="y_true")
    y_pred = check_binary_array(y_pred, name="y_pred")
    check_same_length(y_true, y_pred)

    delays: List[Optional[int]] = []
    for start, end in binary_to_intervals(y_true):
        positive_indices = np.flatnonzero(y_pred[start:end] == 1)
        if positive_indices.size == 0:
            delays.append(None)
        else:
            delays.append(int(positive_indices[0]))
    return delays


def mean_time_to_detect(y_true, y_pred) -> float:
    delays = [d for d in time_to_detect(y_true, y_pred) if d is not None]
    if not delays:
        return float("nan")
    return float(np.mean(delays))


def median_time_to_detect(y_true, y_pred) -> float:
    delays = [d for d in time_to_detect(y_true, y_pred) if d is not None]
    if not delays:
        return float("nan")
    return float(np.median(delays))


def missed_detection_rate(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    delays = time_to_detect(y_true, y_pred)
    if not delays:
        return float(zero_division)
    misses = sum(delay is None for delay in delays)
    return misses / len(delays)
