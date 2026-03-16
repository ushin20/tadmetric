from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import EvaluationResult, evaluate_scores
from .validation import check_binary_array
from .validation import check_score_array


def threshold_by_quantile(y_score, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    scores = check_score_array(y_score, name="y_score")
    return float(np.quantile(scores, quantile))


def threshold_by_topk(y_score, k: int) -> float:
    scores = check_score_array(y_score, name="y_score")
    n = len(scores)
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and len(y_score)")
    sorted_scores = np.sort(scores)
    return float(sorted_scores[-k])


def apply_hysteresis(y_score, high: float, low: float):
    if low > high:
        raise ValueError("low threshold must be <= high threshold")
    scores = check_score_array(y_score, name="y_score")
    output = np.zeros_like(scores, dtype=int)
    active = False
    for i, score in enumerate(scores):
        if not active and score >= high:
            active = True
        elif active and score < low:
            active = False
        output[i] = int(active)
    return output


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    metric_name: str
    f1: float
    evaluation: EvaluationResult


def _threshold_candidates(y_score) -> np.ndarray:
    scores = check_score_array(y_score, name="y_score")
    if scores.size == 0:
        raise ValueError("y_score must not be empty")

    unique_scores = np.unique(scores)
    all_negative_threshold = np.nextafter(np.max(unique_scores), np.inf)
    candidates = np.concatenate(([all_negative_threshold], unique_scores[::-1]))
    return candidates.astype(float)


def search_best_f1_threshold(
    y_true,
    y_score,
    *,
    metric: str = "point",
    metric_kwargs: dict[str, object] | None = None,
    zero_division: float = 0.0,
) -> ThresholdSearchResult:
    check_binary_array(y_true, name="y_true")
    candidates = _threshold_candidates(y_score)

    best_result: ThresholdSearchResult | None = None
    for threshold in candidates:
        evaluation = evaluate_scores(
            y_true,
            y_score,
            threshold=float(threshold),
            metrics=(metric,),
            metric_kwargs={metric: dict(metric_kwargs or {})},
            zero_division=zero_division,
        )
        current_f1 = evaluation[metric].f1
        current = ThresholdSearchResult(
            threshold=float(threshold),
            metric_name=metric,
            f1=current_f1,
            evaluation=evaluation,
        )
        if best_result is None:
            best_result = current
            continue

        if current.f1 > best_result.f1:
            best_result = current
            continue

        if current.f1 == best_result.f1 and current.threshold > best_result.threshold:
            best_result = current

    return best_result


def threshold_by_best_f1(
    y_true,
    y_score,
    *,
    metric: str = "point",
    metric_kwargs: dict[str, object] | None = None,
    zero_division: float = 0.0,
) -> float:
    result = search_best_f1_threshold(
        y_true,
        y_score,
        metric=metric,
        metric_kwargs=metric_kwargs,
        zero_division=zero_division,
    )
    return result.threshold
