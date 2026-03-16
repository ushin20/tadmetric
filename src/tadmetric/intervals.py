from __future__ import annotations

from typing import Iterable, List, Tuple

Interval = Tuple[int, int]


def validate_interval(interval: Interval) -> None:
    start, end = interval
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("Intervals must contain integer indices")
    if start < 0 or end < 0:
        raise ValueError("Interval indices must be non-negative")
    if end < start:
        raise ValueError("Interval end must be greater than or equal to start")


def interval_overlap(a: Interval, b: Interval) -> int:
    validate_interval(a)
    validate_interval(b)
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_intersection(a: Interval, b: Interval):
    overlap = interval_overlap(a, b)
    if overlap == 0:
        return None
    return max(a[0], b[0]), min(a[1], b[1])


def interval_union_length(a: Interval, b: Interval) -> int:
    validate_interval(a)
    validate_interval(b)
    return (a[1] - a[0]) + (b[1] - b[0]) - interval_overlap(a, b)


def interval_iou(a: Interval, b: Interval) -> float:
    union = interval_union_length(a, b)
    if union == 0:
        return 0.0
    return interval_overlap(a, b) / union


def contains(interval: Interval, index: int) -> bool:
    validate_interval(interval)
    return interval[0] <= index < interval[1]


def merge_intervals(intervals: Iterable[Interval], *, merge_adjacent: bool = True) -> List[Interval]:
    intervals = list(intervals)
    if not intervals:
        return []

    for interval in intervals:
        validate_interval(interval)

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[Interval] = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        condition = start <= last_end if merge_adjacent else start < last_end
        if condition:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged
