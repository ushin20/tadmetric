from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np

from .converters import binary_to_intervals
from .intervals import interval_iou, interval_overlap
from .point import _safe_divide
from .validation import check_binary_array, check_same_length, check_score_array

DEFAULT_METRICS = ("point", "point_adjusted", "point_adjusted_k", "composite")


@dataclass(frozen=True)
class PRFScore:
    precision: float
    recall: float
    f1: float

    def __iter__(self):
        yield self.precision
        yield self.recall
        yield self.f1

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


@dataclass(frozen=True)
class MetricSpec:
    name: str
    category: str
    summary: str
    tunable_kwargs: tuple[str, ...] = ()

    def asdict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "summary": self.summary,
            "tunable_kwargs": self.tunable_kwargs,
        }


MetricFunction = Callable[..., PRFScore]


@dataclass
class BinaryEvaluationContext:
    y_true: np.ndarray
    y_pred: np.ndarray
    _true_events: list[tuple[int, int]] | None = None
    _pred_events: list[tuple[int, int]] | None = None
    _point_counts: tuple[int, int, int] | None = None
    _adjusted_predictions: dict[float, np.ndarray] | None = None

    @classmethod
    def from_arrays(cls, y_true, y_pred) -> "BinaryEvaluationContext":
        true_arr = check_binary_array(y_true, name="y_true")
        pred_arr = check_binary_array(y_pred, name="y_pred")
        check_same_length(true_arr, pred_arr)
        return cls(
            y_true=true_arr,
            y_pred=pred_arr,
            _adjusted_predictions={},
        )

    @property
    def true_events(self) -> list[tuple[int, int]]:
        if self._true_events is None:
            self._true_events = binary_to_intervals(self.y_true)
        return self._true_events

    @property
    def pred_events(self) -> list[tuple[int, int]]:
        if self._pred_events is None:
            self._pred_events = binary_to_intervals(self.y_pred)
        return self._pred_events

    def point_counts(self) -> tuple[int, int, int]:
        if self._point_counts is None:
            tp = int(np.sum((self.y_true == 1) & (self.y_pred == 1)))
            fp = int(np.sum((self.y_true == 0) & (self.y_pred == 1)))
            fn = int(np.sum((self.y_true == 1) & (self.y_pred == 0)))
            self._point_counts = (tp, fp, fn)
        return self._point_counts

    def point_score(self, *, zero_division: float = 0.0) -> PRFScore:
        tp, fp, fn = self.point_counts()
        precision = _safe_divide(tp, tp + fp, zero_division)
        recall = _safe_divide(tp, tp + fn, zero_division)
        if precision + recall == 0:
            f1 = float(zero_division)
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return PRFScore(precision=precision, recall=recall, f1=f1)

    def adjusted_prediction(self, *, k: float | None = None) -> np.ndarray:
        cache_key = -1.0 if k is None else float(k)
        adjusted_cache = self._adjusted_predictions
        assert adjusted_cache is not None
        if cache_key in adjusted_cache:
            return adjusted_cache[cache_key]

        if k is not None and not 0.0 <= k <= 100.0:
            raise ValueError("k must be in [0, 100]")

        adjusted = self.y_pred.copy()
        cumulative_pred = np.concatenate(([0], np.cumsum(self.y_pred, dtype=int)))
        for start, end in self.true_events:
            positives = int(cumulative_pred[end] - cumulative_pred[start])
            event_length = end - start
            if k is None:
                detected = positives > 0
            elif k == 0.0:
                detected = positives > 0
            else:
                required = int(np.ceil(event_length * (k / 100.0)))
                detected = positives >= required

            if detected:
                adjusted[start:end] = 1

        adjusted_cache[cache_key] = adjusted
        return adjusted

    def adjusted_score(self, *, k: float | None = None, zero_division: float = 0.0) -> PRFScore:
        adjusted_context = BinaryEvaluationContext(
            y_true=self.y_true,
            y_pred=self.adjusted_prediction(k=k),
            _true_events=self.true_events,
            _adjusted_predictions={},
        )
        return adjusted_context.point_score(zero_division=zero_division)

    def event_score(
        self,
        *,
        match: str = "overlap",
        iou_threshold: float = 0.0,
        zero_division: float = 0.0,
    ) -> PRFScore:
        true_events = self.true_events
        pred_events = self.pred_events

        if not pred_events:
            precision = float(zero_division)
        else:
            hits = sum(
                any(_event_match(true_event, pred_event, match, iou_threshold) for true_event in true_events)
                for pred_event in pred_events
            )
            precision = hits / len(pred_events)

        if not true_events:
            recall = float(zero_division)
        else:
            hits = sum(
                any(_event_match(true_event, pred_event, match, iou_threshold) for pred_event in pred_events)
                for true_event in true_events
            )
            recall = hits / len(true_events)

        if precision + recall == 0:
            f1 = float(zero_division)
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return PRFScore(precision=precision, recall=recall, f1=f1)


_METRIC_ALIASES = {
    "p": "point_wise",
    "point": "point_wise",
    "point_wise": "point_wise",
    "pw": "point_wise",
    "pa": "point_adjusted",
    "point_adjusted": "point_adjusted",
    "pak": "point_adjusted",
    "point_adjusted_k": "point_adjusted",
    "composite": "composite",
    "comp": "composite",
    "c": "composite",
}


def _freeze_metric_kwargs(metric_kwargs: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in metric_kwargs.items()))


def _threshold_candidates_from_scores(y_score: np.ndarray) -> np.ndarray:
    if y_score.size == 0:
        raise ValueError("y_score must not be empty")
    unique_scores = np.unique(y_score)
    all_negative_threshold = np.nextafter(np.max(unique_scores), np.inf)
    candidates = np.concatenate(([all_negative_threshold], unique_scores[::-1]))
    return candidates.astype(float)


class Evaluator:
    def __init__(self, y_true, y_score) -> None:
        self.reevaluate(y_true, y_score)

    def reevaluate(self, y_true, y_score) -> "Evaluator":
        true_arr = check_binary_array(y_true, name="y_true")
        score_arr = check_score_array(y_score, name="y_score")
        check_same_length(true_arr, score_arr)

        self.y_true = true_arr
        self.y_score = score_arr
        self._binary_context_cache: dict[float, BinaryEvaluationContext] = {}
        self._metric_cache: dict[tuple[object, ...], PRFScore] = {}
        self._best_cache: dict[tuple[object, ...], object] = {}
        return self

    def _normalize_metric(self, metric: str) -> str:
        normalized = _METRIC_ALIASES.get(metric.lower(), metric)
        if normalized not in {"point_wise", "point_adjusted", "composite"}:
            available = ", ".join(sorted(_METRIC_ALIASES))
            raise ValueError(f"Unknown metric '{metric}'. Available aliases: {available}")
        return normalized

    def _binary_context(self, threshold: float) -> BinaryEvaluationContext:
        threshold = float(threshold)
        if threshold not in self._binary_context_cache:
            y_pred = (self.y_score >= threshold).astype(int)
            self._binary_context_cache[threshold] = BinaryEvaluationContext(
                y_true=self.y_true,
                y_pred=y_pred,
                _adjusted_predictions={},
            )
        return self._binary_context_cache[threshold]

    def _metric_score(
        self,
        *,
        metric: str,
        thr: float | str | None = None,
        zero_division: float = 0.0,
        **metric_kwargs: object,
    ) -> PRFScore:
        metric_name = self._normalize_metric(metric)
        threshold = self._resolve_threshold(
            metric=metric_name,
            thr=thr,
            zero_division=zero_division,
            **metric_kwargs,
        )
        cache_key = (
            metric_name,
            threshold,
            float(zero_division),
            _freeze_metric_kwargs(metric_kwargs),
        )
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]

        context = self._binary_context(threshold)
        if metric_name == "point_wise":
            score = context.point_score(zero_division=zero_division)
        elif metric_name == "point_adjusted":
            score = context.adjusted_score(
                k=float(metric_kwargs.get("k", 100.0)),
                zero_division=zero_division,
            )
        elif metric_name == "composite":
            point_score = context.point_score(zero_division=zero_division)
            event_score = context.event_score(
                match=str(metric_kwargs.get("match", "overlap")),
                iou_threshold=float(metric_kwargs.get("iou_threshold", 0.0)),
                zero_division=zero_division,
            )
            score = PRFScore(
                precision=point_score.precision,
                recall=event_score.recall,
                f1=_safe_divide(
                    2 * point_score.precision * event_score.recall,
                    point_score.precision + event_score.recall,
                    zero_division,
                ),
            )
        else:
            score = _METRIC_REGISTRY[metric_name](
                context.y_true,
                context.y_pred,
                zero_division=zero_division,
                **metric_kwargs,
            )

        self._metric_cache[cache_key] = score
        return score

    def _resolve_threshold(
        self,
        *,
        metric: str,
        thr: float | str | None,
        zero_division: float,
        **metric_kwargs: object,
    ) -> float:
        if thr is None or thr == "best":
            return float(
                self.best(
                    metric=metric,
                    zero_division=zero_division,
                    **metric_kwargs,
                ).threshold
            )
        return float(thr)

    def point_wise(
        self,
        *,
        thr: float | str | None = "best",
        zero_division: float = 0.0,
    ) -> PRFScore:
        return self.metric(
            metric="point_wise",
            thr=thr,
            zero_division=zero_division,
        )

    def point_adjusted(
        self,
        *,
        thr: float | str | None = "best",
        k: float = 100.0,
        zero_division: float = 0.0,
    ) -> PRFScore:
        return self.metric(
            metric="point_adjusted",
            thr=thr,
            k=k,
            zero_division=zero_division,
        )

    def composite(
        self,
        *,
        thr: float | str | None = "best",
        zero_division: float = 0.0,
        **metric_kwargs: object,
    ) -> PRFScore:
        return self.metric(
            metric="composite",
            thr=thr,
            zero_division=zero_division,
            **metric_kwargs,
        )

    def metric(
        self,
        *,
        metric: str = "point_wise",
        thr: float | str | None = None,
        zero_division: float = 0.0,
        k: float = 100.0,
        **metric_kwargs: object,
    ) -> PRFScore:
        if self._normalize_metric(metric) == "point_adjusted":
            metric_kwargs.setdefault("k", k)
        return self._metric_score(
            metric=metric,
            thr=thr,
            zero_division=zero_division,
            **metric_kwargs,
        )

    def precision(
        self,
        *,
        metric: str = "point_wise",
        thr: float | str | None = None,
        zero_division: float = 0.0,
        k: float = 100.0,
        **metric_kwargs: object,
    ) -> float:
        return self.metric(
            metric=metric,
            thr=thr,
            zero_division=zero_division,
            k=k,
            **metric_kwargs,
        ).precision

    def recall(
        self,
        *,
        metric: str = "point_wise",
        thr: float | str | None = None,
        zero_division: float = 0.0,
        k: float = 100.0,
        **metric_kwargs: object,
    ) -> float:
        return self.metric(
            metric=metric,
            thr=thr,
            zero_division=zero_division,
            k=k,
            **metric_kwargs,
        ).recall

    def f1(
        self,
        *,
        metric: str = "point_wise",
        thr: float | str | None = None,
        zero_division: float = 0.0,
        k: float = 100.0,
        **metric_kwargs: object,
    ) -> float:
        return self.metric(
            metric=metric,
            thr=thr,
            zero_division=zero_division,
            k=k,
            **metric_kwargs,
        ).f1

    def score(
        self,
        *,
        metric: str = "point_wise",
        thr: float | str | None = None,
        zero_division: float = 0.0,
        k: float = 100.0,
        **metric_kwargs: object,
    ) -> PRFScore:
        return self.metric(
            metric=metric,
            thr=thr,
            zero_division=zero_division,
            k=k,
            **metric_kwargs,
        )

    def evaluate(
        self,
        *,
        thr: float,
        metrics: Optional[Iterable[str]] = None,
        metric_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
        zero_division: float = 0.0,
    ) -> EvaluationResult:
        threshold = float(thr)
        context = self._binary_context(threshold)
        return _evaluate_prevalidated(
            self.y_true,
            context.y_pred,
            metrics=metrics,
            metric_kwargs=metric_kwargs,
            zero_division=zero_division,
        )

    def best(
        self,
        *,
        metric: str = "point_wise",
        zero_division: float = 0.0,
        k: float = 100.0,
        **metric_kwargs: object,
    ):
        from .threshold import ThresholdSearchResult

        metric_name = self._normalize_metric(metric)
        if metric_name == "point_adjusted":
            metric_kwargs.setdefault("k", k)
        cache_key = (
            metric_name,
            float(zero_division),
            _freeze_metric_kwargs(metric_kwargs),
        )
        if cache_key in self._best_cache:
            return self._best_cache[cache_key]

        best_result: ThresholdSearchResult | None = None
        for threshold in _threshold_candidates_from_scores(self.y_score):
            score = self._metric_score(
                metric=metric_name,
                thr=float(threshold),
                zero_division=zero_division,
                **metric_kwargs,
            )
            current = ThresholdSearchResult(
                threshold=float(threshold),
                metric_name=metric_name,
                f1=score.f1,
                evaluation=EvaluationResult(
                    scores={metric_name: score},
                    y_pred=self._binary_context(float(threshold)).y_pred,
                    threshold=float(threshold),
                ),
            )
            if best_result is None or current.f1 > best_result.f1:
                best_result = current
                continue
            if current.f1 == best_result.f1 and current.threshold > best_result.threshold:
                best_result = current

        self._best_cache[cache_key] = best_result
        return best_result


def evaluator(y_true, y_score) -> Evaluator:
    return Evaluator(y_true, y_score)


def _event_match(true_event, pred_event, match: str = "overlap", iou_threshold: float = 0.0) -> bool:
    if match == "overlap":
        return interval_overlap(true_event, pred_event) > 0
    if match == "iou":
        return interval_iou(true_event, pred_event) >= iou_threshold
    raise ValueError("match must be 'overlap' or 'iou'")


def _point_metric(y_true, y_pred, *, zero_division: float = 0.0, **_: object) -> PRFScore:
    context = BinaryEvaluationContext.from_arrays(y_true, y_pred)
    return context.point_score(zero_division=zero_division)


def _point_adjusted_metric(y_true, y_pred, *, zero_division: float = 0.0, **_: object) -> PRFScore:
    context = BinaryEvaluationContext.from_arrays(y_true, y_pred)
    return context.adjusted_score(zero_division=zero_division)


def _point_adjusted_k_metric(
    y_true,
    y_pred,
    *,
    k: float = 100.0,
    zero_division: float = 0.0,
    **_: object,
) -> PRFScore:
    context = BinaryEvaluationContext.from_arrays(y_true, y_pred)
    return context.adjusted_score(k=k, zero_division=zero_division)


def _composite_metric(
    y_true,
    y_pred,
    *,
    match: str = "overlap",
    iou_threshold: float = 0.0,
    zero_division: float = 0.0,
    **_: object,
) -> PRFScore:
    context = BinaryEvaluationContext.from_arrays(y_true, y_pred)
    point_score = context.point_score(zero_division=zero_division)
    event_score = context.event_score(
        match=match,
        iou_threshold=iou_threshold,
        zero_division=zero_division,
    )
    f1 = _safe_divide(
        2 * point_score.precision * event_score.recall,
        point_score.precision + event_score.recall,
        zero_division,
    )
    return PRFScore(
        precision=point_score.precision,
        recall=event_score.recall,
        f1=f1,
    )


_METRIC_REGISTRY: MutableMapping[str, MetricFunction] = {
    "point": _point_metric,
    "point_adjusted": _point_adjusted_metric,
    "point_adjusted_k": _point_adjusted_k_metric,
    "composite": _composite_metric,
}
_BUILTIN_METRIC_FUNCTIONS: Dict[str, MetricFunction] = dict(_METRIC_REGISTRY)

_METRIC_SPECS: Dict[str, MetricSpec] = {
    "point": MetricSpec("point", "core", "Timestamp-wise precision/recall/F1.", ()),
    "point_adjusted": MetricSpec("point_adjusted", "adjusted", "Credits a whole true event once any point is detected.", ()),
    "point_adjusted_k": MetricSpec("point_adjusted_k", "adjusted", "Credits a true event when at least K percent is detected.", ("k",)),
    "composite": MetricSpec("composite", "hybrid", "Uses point precision with event-based recall.", ("match", "iou_threshold")),
}


def register_metric(name: str, metric_fn: MetricFunction, *, overwrite: bool = False) -> None:
    if not name:
        raise ValueError("name must be a non-empty string")
    if name in _METRIC_REGISTRY and not overwrite:
        raise ValueError(f"Metric '{name}' is already registered")
    _METRIC_REGISTRY[name] = metric_fn


def available_metrics() -> tuple[str, ...]:
    return tuple(sorted(_METRIC_REGISTRY))


def describe_metrics() -> tuple[MetricSpec, ...]:
    return tuple(_METRIC_SPECS[name] for name in available_metrics() if name in _METRIC_SPECS)


def api_overview() -> str:
    lines = [
        "tadmetric API overview",
        "",
        "High-level entry points:",
        "- evaluator(y_true, y_score): reusable evaluator with cached metric-thr results.",
        "- pre, rec, f1 = e.point_wise(thr='best')",
        "- pre, rec, f1 = e.point_adjusted(k=30)",
        "- pre, rec, f1 = e.composite()",
        "- e.best(metric='composite'): search the best threshold on the current scores.",
        "- evaluate(y_true, y_pred, ...): evaluate binary predictions.",
        "- evaluate_scores(y_true, y_score, threshold=...): threshold scores and evaluate them.",
        "",
        "Metric discovery:",
        f"- available_metrics(): {', '.join(available_metrics())}",
        "- describe_metrics(): metric descriptions and tunable kwargs.",
        "",
        "Threshold utilities:",
        "- search_best_f1_threshold(...)",
        "- threshold_by_best_f1(...)",
        "- threshold_by_quantile(...)",
        "- threshold_by_topk(...)",
        "",
        "Conversion utilities:",
        "- scores_to_binary(...)",
        "- binary_to_intervals(...)",
        "- intervals_to_binary(...)",
    ]
    return "\n".join(lines)


def _evaluate_prevalidated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: Optional[Iterable[str]] = None,
    metric_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    zero_division: float = 0.0,
) -> EvaluationResult:
    metric_names = tuple(metrics or DEFAULT_METRICS)
    metric_kwargs = metric_kwargs or {}

    context = BinaryEvaluationContext(
        y_true=y_true,
        y_pred=y_pred,
        _adjusted_predictions={},
    )
    results: Dict[str, PRFScore] = {}
    for metric_name in metric_names:
        if metric_name not in _METRIC_REGISTRY:
            available = ", ".join(available_metrics())
            raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available}")

        kwargs = dict(metric_kwargs.get(metric_name, {}))
        kwargs.setdefault("zero_division", zero_division)

        registry_fn = _METRIC_REGISTRY[metric_name]
        is_builtin = _BUILTIN_METRIC_FUNCTIONS.get(metric_name) is registry_fn

        if metric_name == "point" and is_builtin:
            results[metric_name] = context.point_score(zero_division=kwargs["zero_division"])
        elif metric_name == "point_adjusted" and is_builtin:
            results[metric_name] = context.adjusted_score(zero_division=kwargs["zero_division"])
        elif metric_name == "point_adjusted_k" and is_builtin:
            results[metric_name] = context.adjusted_score(
                k=float(kwargs.get("k", 50.0)),
                zero_division=kwargs["zero_division"],
            )
        elif metric_name == "composite" and is_builtin:
            event_score = context.event_score(
                match=str(kwargs.get("match", "overlap")),
                iou_threshold=float(kwargs.get("iou_threshold", 0.0)),
                zero_division=kwargs["zero_division"],
            )
            point_score = context.point_score(zero_division=kwargs["zero_division"])
            f1 = _safe_divide(
                2 * point_score.precision * event_score.recall,
                point_score.precision + event_score.recall,
                kwargs["zero_division"],
            )
            results[metric_name] = PRFScore(
                precision=point_score.precision,
                recall=event_score.recall,
                f1=f1,
            )
        else:
            results[metric_name] = registry_fn(context.y_true, context.y_pred, **kwargs)

    return EvaluationResult(scores=results, y_pred=np.asarray(y_pred, dtype=int))


def evaluate(
    y_true,
    y_pred,
    *,
    metrics: Optional[Iterable[str]] = None,
    metric_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    zero_division: float = 0.0,
) -> EvaluationResult:
    context = BinaryEvaluationContext.from_arrays(y_true, y_pred)
    return _evaluate_prevalidated(
        context.y_true,
        context.y_pred,
        metrics=metrics,
        metric_kwargs=metric_kwargs,
        zero_division=zero_division,
    )


def evaluate_scores(
    y_true,
    y_score,
    *,
    threshold: float = 0.5,
    metrics: Optional[Iterable[str]] = None,
    metric_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    zero_division: float = 0.0,
) -> EvaluationResult:
    return Evaluator(y_true, y_score).evaluate(
        thr=threshold,
        metrics=metrics,
        metric_kwargs=metric_kwargs,
        zero_division=zero_division,
    )
