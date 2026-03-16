from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import EvaluationResult, Evaluator
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
    return Evaluator(y_true, y_score).best(
        metric=metric,
        zero_division=zero_division,
        **dict(metric_kwargs or {}),
    )


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
