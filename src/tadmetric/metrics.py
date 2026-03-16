from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np

from .adjusted import (
    point_adjusted_f1,
    point_adjusted_k_f1,
    point_adjusted_k_precision,
    point_adjusted_k_recall,
    point_adjusted_precision,
    point_adjusted_recall,
)
from .composite import composite_f1, composite_precision, composite_recall
from .converters import scores_to_binary
from .point import point_f1, point_precision, point_recall


@dataclass(frozen=True)
class PRFScore:
    precision: float
    recall: float
    f1: float

    def asdict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class EvaluationResult:
    scores: Dict[str, PRFScore]
    y_pred: np.ndarray
    threshold: Optional[float] = None

    def __getitem__(self, metric_name: str) -> PRFScore:
        return self.scores[metric_name]

    def keys(self):
        return self.scores.keys()

    def items(self):
        return self.scores.items()

    def values(self):
        return self.scores.values()

    def asdict(self) -> Dict[str, Dict[str, float]]:
        return {name: value.asdict() for name, value in self.scores.items()}


MetricFunction = Callable[..., PRFScore]


def _point_metric(y_true, y_pred, *, zero_division: float = 0.0, **_: object) -> PRFScore:
    return PRFScore(
        precision=point_precision(y_true, y_pred, zero_division=zero_division),
        recall=point_recall(y_true, y_pred, zero_division=zero_division),
        f1=point_f1(y_true, y_pred, zero_division=zero_division),
    )


def _point_adjusted_metric(y_true, y_pred, *, zero_division: float = 0.0, **_: object) -> PRFScore:
    return PRFScore(
        precision=point_adjusted_precision(y_true, y_pred, zero_division=zero_division),
        recall=point_adjusted_recall(y_true, y_pred, zero_division=zero_division),
        f1=point_adjusted_f1(y_true, y_pred, zero_division=zero_division),
    )


def _point_adjusted_k_metric(
    y_true,
    y_pred,
    *,
    k: float = 50.0,
    zero_division: float = 0.0,
    **_: object,
) -> PRFScore:
    return PRFScore(
        precision=point_adjusted_k_precision(y_true, y_pred, k=k, zero_division=zero_division),
        recall=point_adjusted_k_recall(y_true, y_pred, k=k, zero_division=zero_division),
        f1=point_adjusted_k_f1(y_true, y_pred, k=k, zero_division=zero_division),
    )


def _composite_metric(
    y_true,
    y_pred,
    *,
    match: str = "overlap",
    iou_threshold: float = 0.0,
    zero_division: float = 0.0,
    **_: object,
) -> PRFScore:
    return PRFScore(
        precision=composite_precision(y_true, y_pred, zero_division=zero_division),
        recall=composite_recall(
            y_true,
            y_pred,
            match=match,
            iou_threshold=iou_threshold,
            zero_division=zero_division,
        ),
        f1=composite_f1(
            y_true,
            y_pred,
            match=match,
            iou_threshold=iou_threshold,
            zero_division=zero_division,
        ),
    )


_METRIC_REGISTRY: MutableMapping[str, MetricFunction] = {
    "point": _point_metric,
    "point_adjusted": _point_adjusted_metric,
    "point_adjusted_k": _point_adjusted_k_metric,
    "composite": _composite_metric,
}


def register_metric(name: str, metric_fn: MetricFunction, *, overwrite: bool = False) -> None:
    if not name:
        raise ValueError("name must be a non-empty string")
    if name in _METRIC_REGISTRY and not overwrite:
        raise ValueError(f"Metric '{name}' is already registered")
    _METRIC_REGISTRY[name] = metric_fn


def available_metrics() -> tuple[str, ...]:
    return tuple(sorted(_METRIC_REGISTRY))


def evaluate(
    y_true,
    y_pred,
    *,
    metrics: Optional[Iterable[str]] = None,
    metric_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    zero_division: float = 0.0,
) -> EvaluationResult:
    metric_names = tuple(metrics or ("point", "point_adjusted", "point_adjusted_k", "composite"))
    metric_kwargs = metric_kwargs or {}

    results: Dict[str, PRFScore] = {}
    for metric_name in metric_names:
        if metric_name not in _METRIC_REGISTRY:
            available = ", ".join(available_metrics())
            raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available}")

        kwargs = dict(metric_kwargs.get(metric_name, {}))
        kwargs.setdefault("zero_division", zero_division)
        results[metric_name] = _METRIC_REGISTRY[metric_name](y_true, y_pred, **kwargs)

    return EvaluationResult(scores=results, y_pred=np.asarray(y_pred, dtype=int))


def evaluate_scores(
    y_true,
    y_score,
    *,
    threshold: float = 0.5,
    metrics: Optional[Iterable[str]] = None,
    metric_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    zero_division: float = 0.0,
) -> EvaluationResult:
    y_pred = scores_to_binary(y_score, threshold)
    result = evaluate(
        y_true,
        y_pred,
        metrics=metrics,
        metric_kwargs=metric_kwargs,
        zero_division=zero_division,
    )
    return EvaluationResult(scores=result.scores, y_pred=result.y_pred, threshold=threshold)
