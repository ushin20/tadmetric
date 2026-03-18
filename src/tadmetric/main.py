from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._typing import ArrayLike, Mode

_MODE_ALIASES = {
    "point": "point-wise",
    "point-wise": "point-wise",
    "point_wise": "point-wise",
    "pointwise": "point-wise",
    "pw": "point-wise",
    "point-adjusted": "point-adjusted",
    "point_adjusted": "point-adjusted",
    "pointadjusted": "point-adjusted",
    "pa": "point-adjusted",
    "composite": "composite",
    "comp": "composite",
    "c": "composite",
}


@dataclass(frozen=True)
class MetricResult:
    mode: Mode
    threshold: float | None
    f1: float
    precision: float
    recall: float
    tp: int
    tn: int
    fp: int
    fn: int
    latency: float | None = None
    detected_events: int | None = None
    total_events: int | None = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    mode: Mode
    threshold: float
    result: MetricResult
    candidates_evaluated: int

    @property
    def f1(self) -> float:
        return self.result.f1

    @property
    def precision(self) -> float:
        return self.result.precision

    @property
    def recall(self) -> float:
        return self.result.recall

    def asdict(self) -> dict[str, Any]:
        data = asdict(self)
        data["f1"] = self.f1
        data["precision"] = self.precision
        data["recall"] = self.recall
        return data


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _normalize_mode(mode: str) -> Mode:
    normalized = _MODE_ALIASES.get(mode.lower())
    if normalized is None:
        choices = ", ".join(sorted(_MODE_ALIASES))
        raise ValueError(f"Invalid mode '{mode}'. Available aliases: {choices}")
    return normalized  # type: ignore[return-value]


def _as_1d_array(values: ArrayLike, *, name: str, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like object")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return array


def _as_score_array(values: ArrayLike, *, name: str = "score") -> npt.NDArray[np.float64]:
    return _as_1d_array(values, name=name, dtype=float).astype(np.float64)


def _as_label_array(values: ArrayLike, *, name: str = "label") -> npt.NDArray[np.int_]:
    array = _as_1d_array(values, name=name, dtype=float)
    unique = np.unique(array)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"{name} must contain only binary values 0 or 1")
    return array.astype(int)


class Tadmetric:
    """Simple TSAD evaluator for point-wise, point-adjusted, and composite F1."""

    def __init__(self, score: ArrayLike, label: ArrayLike):
        self.predict: npt.NDArray[np.int_] | None = None
        self.update(score, label)

    def update(self, score: ArrayLike, label: ArrayLike) -> None:
        score_array = _as_score_array(score, name="score")
        label_array = _as_label_array(label, name="label")
        if score_array.shape[0] != label_array.shape[0]:
            raise ValueError("score and label must have the same length")

        self.score = score_array
        self.label = label_array
        self.predict = None

    def _exact_thresholds(self) -> npt.NDArray[np.float64]:
        unique_scores = np.unique(self.score)
        all_negative = np.nextafter(unique_scores.max(), np.inf)
        return np.concatenate(([all_negative], unique_scores[::-1])).astype(float)

    def _predict_from_threshold(self, threshold: float) -> npt.NDArray[np.int_]:
        return (self.score >= float(threshold)).astype(int)

    def _calc_point2point(self, pred: ArrayLike | None = None) -> tuple[float, float, float, int, int, int, int]:
        prediction = self._resolve_prediction(pred)
        tp = int(np.sum((prediction == 1) & (self.label == 1)))
        tn = int(np.sum((prediction == 0) & (self.label == 0)))
        fp = int(np.sum((prediction == 1) & (self.label == 0)))
        fn = int(np.sum((prediction == 0) & (self.label == 1)))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        return f1, precision, recall, tp, tn, fp, fn

    def _get_event_indices(self) -> list[tuple[int, int]]:
        events: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(self.label):
            if value == 1 and start is None:
                start = index
            elif value == 0 and start is not None:
                events.append((start, index))
                start = None
        if start is not None:
            events.append((start, len(self.label)))
        return events

    def _adjust_predicts(
        self,
        threshold: float | None = None,
        pred: ArrayLike | None = None,
        calc_latency: bool = False,
    ) -> npt.NDArray[np.int_] | tuple[npt.NDArray[np.int_], float | None]:
        prediction = self._resolve_prediction(pred, threshold=threshold)
        adjusted = prediction.copy()
        latencies: list[int] = []

        for start, end in self._get_event_indices():
            hit_indices = np.flatnonzero(prediction[start:end] == 1)
            if hit_indices.size > 0:
                adjusted[start:end] = 1
                latencies.append(int(hit_indices[0]))

        if not calc_latency:
            return adjusted

        latency = float(np.mean(latencies)) if latencies else None
        return adjusted, latency

    def _event_detection_stats(self, pred: ArrayLike | None = None) -> tuple[int, int]:
        prediction = self._resolve_prediction(pred)
        events = self._get_event_indices()
        detected_events = sum(int(np.any(prediction[start:end])) for start, end in events)
        return detected_events, len(events)

    def calc_composite_f1(
        self,
        pred: ArrayLike | None = None,
        threshold: float | None = None,
    ) -> tuple[float, float, float, int, int, int, int]:
        prediction = self._resolve_prediction(pred, threshold=threshold)
        _, precision, _, tp, _, fp, _ = self._calc_point2point(pred=prediction)
        detected_events, total_events = self._event_detection_stats(pred=prediction)
        fn_events = total_events - detected_events
        recall = _safe_divide(detected_events, total_events)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        return f1, precision, recall, tp, fp, detected_events, fn_events

    def evaluate(
        self,
        threshold: float,
        *,
        mode: str = "point-wise",
        calc_latency: bool = False,
    ) -> MetricResult:
        normalized_mode = _normalize_mode(mode)
        raw_prediction = self._predict_from_threshold(float(threshold))

        if normalized_mode == "point-wise":
            self.predict = raw_prediction
            f1, precision, recall, tp, tn, fp, fn = self._calc_point2point(pred=raw_prediction)
            return MetricResult(
                mode=normalized_mode,
                threshold=float(threshold),
                f1=f1,
                precision=precision,
                recall=recall,
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
            )

        if normalized_mode == "point-adjusted":
            latency: float | None = None
            adjusted_prediction: npt.NDArray[np.int_]
            adjusted_output = self._adjust_predicts(
                threshold=float(threshold),
                pred=raw_prediction,
                calc_latency=calc_latency,
            )
            if calc_latency:
                adjusted_prediction, latency = adjusted_output  # type: ignore[misc]
            else:
                adjusted_prediction = adjusted_output  # type: ignore[assignment]

            self.predict = adjusted_prediction
            f1, precision, recall, tp, tn, fp, fn = self._calc_point2point(pred=adjusted_prediction)
            detected_events, total_events = self._event_detection_stats(pred=raw_prediction)
            return MetricResult(
                mode=normalized_mode,
                threshold=float(threshold),
                f1=f1,
                precision=precision,
                recall=recall,
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                latency=latency,
                detected_events=detected_events,
                total_events=total_events,
            )

        self.predict = raw_prediction
        f1, precision, recall, tp, fp, detected_events, fn_events = self.calc_composite_f1(pred=raw_prediction)
        _, _, _, _, tn, _, fn = self._calc_point2point(pred=raw_prediction)
        return MetricResult(
            mode=normalized_mode,
            threshold=float(threshold),
            f1=f1,
            precision=precision,
            recall=recall,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            detected_events=detected_events,
            total_events=detected_events + fn_events,
        )

    def search(
        self,
        *,
        mode: str = "point-wise",
        start: float | None = None,
        end: float | None = None,
        steps: int = 100,
        verbose: bool = False,
    ) -> SearchResult:
        normalized_mode = _normalize_mode(mode)
        thresholds = self._search_thresholds(start=start, end=end, steps=steps)
        best_result: MetricResult | None = None
        best_threshold: float | None = None

        for index, threshold in enumerate(thresholds, start=1):
            current = self.evaluate(float(threshold), mode=normalized_mode)
            if self._is_better(current=current, best=best_result):
                best_result = current
                best_threshold = float(threshold)
                if verbose:
                    print(
                        f"[{index}/{len(thresholds)}] "
                        f"{normalized_mode} f1={current.f1:.4f}, "
                        f"precision={current.precision:.4f}, recall={current.recall:.4f}, "
                        f"threshold={threshold:.6f}"
                    )

        assert best_result is not None
        assert best_threshold is not None
        return SearchResult(
            mode=normalized_mode,
            threshold=best_threshold,
            result=best_result,
            candidates_evaluated=len(thresholds),
        )

    def _search_thresholds(
        self,
        *,
        start: float | None,
        end: float | None,
        steps: int,
    ) -> npt.NDArray[np.float64]:
        if start is None and end is None:
            return self._exact_thresholds()
        if start is None or end is None:
            raise ValueError("start and end must either both be provided or both be omitted")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if start == end or steps == 1:
            return np.asarray([float(start)], dtype=float)
        return np.linspace(float(start), float(end), num=int(steps), dtype=float)

    def _resolve_prediction(
        self,
        pred: ArrayLike | None = None,
        *,
        threshold: float | None = None,
    ) -> npt.NDArray[np.int_]:
        if pred is not None:
            prediction = _as_label_array(pred, name="pred")
        elif threshold is not None:
            prediction = self._predict_from_threshold(float(threshold))
        elif self.predict is not None:
            prediction = self.predict.astype(int)
        else:
            raise ValueError("Either pred or threshold must be provided")

        if prediction.shape[0] != self.label.shape[0]:
            raise ValueError("pred and label must have the same length")
        return prediction

    @staticmethod
    def _is_better(current: MetricResult, best: MetricResult | None) -> bool:
        if best is None:
            return True
        if current.f1 > best.f1:
            return True
        if current.f1 < best.f1:
            return False
        if current.precision > best.precision:
            return True
        if current.precision < best.precision:
            return False
        return (current.threshold or float("-inf")) > (best.threshold or float("-inf"))


__all__ = ["MetricResult", "SearchResult", "Tadmetric"]
