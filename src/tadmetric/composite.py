from __future__ import annotations

from .event import event_recall
from .point import _safe_divide, point_precision


def composite_precision(y_true, y_pred, *, zero_division: float = 0.0) -> float:
    return point_precision(y_true, y_pred, zero_division=zero_division)


def composite_recall(
    y_true,
    y_pred,
    *,
    match: str = "overlap",
    iou_threshold: float = 0.0,
    zero_division: float = 0.0,
) -> float:
    return event_recall(
        y_true,
        y_pred,
        match=match,
        iou_threshold=iou_threshold,
        zero_division=zero_division,
    )


def composite_f1(
    y_true,
    y_pred,
    *,
    match: str = "overlap",
    iou_threshold: float = 0.0,
    zero_division: float = 0.0,
) -> float:
    precision = composite_precision(y_true, y_pred, zero_division=zero_division)
    recall = composite_recall(
        y_true,
        y_pred,
        match=match,
        iou_threshold=iou_threshold,
        zero_division=zero_division,
    )
    return _safe_divide(2 * precision * recall, precision + recall, zero_division)
