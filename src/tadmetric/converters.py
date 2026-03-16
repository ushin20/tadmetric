from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .intervals import merge_intervals, validate_interval
from .validation import check_binary_array, check_score_array

Interval = Tuple[int, int]


def binary_to_intervals(y: Sequence[int]) -> List[Interval]:
    arr = check_binary_array(y, name="y")
    intervals: List[Interval] = []
    start: Optional[int] = None

    for i, value in enumerate(arr):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            intervals.append((start, i))
            start = None

    if start is not None:
        intervals.append((start, len(arr)))

    return intervals


def intervals_to_binary(intervals: Iterable[Interval], length: int) -> np.ndarray:
    if length < 0:
        raise ValueError("length must be non-negative")
    out = np.zeros(length, dtype=int)
    for start, end in intervals:
        validate_interval((start, end))
        if end > length:
            raise ValueError("Interval exceeds target length")
        out[start:end] = 1
    return out


def scores_to_binary(y_score, threshold: float) -> np.ndarray:
    scores = check_score_array(y_score, name="y_score")
    return (scores >= threshold).astype(int)


def events_from_binary(y: Sequence[int], *, merge_adjacent_events: bool = True) -> List[Interval]:
    intervals = binary_to_intervals(y)
    return merge_intervals(intervals, merge_adjacent=merge_adjacent_events)
