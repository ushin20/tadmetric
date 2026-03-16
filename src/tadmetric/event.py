from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .converters import binary_to_intervals
from .intervals import interval_iou, interval_overlap

Interval = Tuple[int, int]


def _prepare_events(y_or_events) -> List[Interval]:
    if len(y_or_events) == 0:
        return []
    first = y_or_events[0]
    if isinstance(first, tuple):
        return list(y_or_events)
    return binary_to_intervals(y_or_events)


def _event_match(true_event: Interval, pred_event: Interval, match: str = "overlap", iou_threshold: float = 0.0) -> bool:
    if match == "overlap":
        return interval_overlap(true_event, pred_event) > 0
    if match == "iou":
        return interval_iou(true_event, pred_event) >= iou_threshold
    raise ValueError("match must be 'overlap' or 'iou'")


def event_recall(y_true, y_pred, *, match: str = "overlap", iou_threshold: float = 0.0, zero_division: float = 0.0) -> float:
    true_events = _prepare_events(y_true)
    pred_events = _prepare_events(y_pred)
    if not true_events:
        return float(zero_division)
    hits = 0
    for true_event in true_events:
        if any(_event_match(true_event, pred_event, match, iou_threshold) for pred_event in pred_events):
            hits += 1
    return hits / len(true_events)


def event_precision(y_true, y_pred, *, match: str = "overlap", iou_threshold: float = 0.0, zero_division: float = 0.0) -> float:
    true_events = _prepare_events(y_true)
    pred_events = _prepare_events(y_pred)
    if not pred_events:
        return float(zero_division)
    hits = 0
    for pred_event in pred_events:
        if any(_event_match(true_event, pred_event, match, iou_threshold) for true_event in true_events):
            hits += 1
    return hits / len(pred_events)


def event_f1(y_true, y_pred, *, match: str = "overlap", iou_threshold: float = 0.0, zero_division: float = 0.0) -> float:
    precision = event_precision(y_true, y_pred, match=match, iou_threshold=iou_threshold, zero_division=zero_division)
    recall = event_recall(y_true, y_pred, match=match, iou_threshold=iou_threshold, zero_division=zero_division)
    if precision + recall == 0:
        return float(zero_division)
    return 2 * precision * recall / (precision + recall)
