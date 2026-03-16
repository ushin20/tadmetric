# tadmetric

Lightweight evaluation metrics for time-series anomaly detection.

`tadmetric` helps you evaluate anomaly scores or binary predictions with a small,
simple API. It includes point-wise metrics, point-adjusted metrics, point-adjusted
`%K`, composite metrics, event-wise metrics, delay-aware metrics, and threshold search.

## Installation

```bash
pip install tadmetric
```

## Why tadmetric?

- Minimal input: use `y_true` with either `y_pred` or `y_score`
- Practical TSAD metrics in one place
- Fast threshold search for best F1
- Easy high-level API for benchmarking multiple metrics at once
- Extensible metric registry for adding new custom metrics

## Quick start

### Evaluate from anomaly scores

```python
from tadmetric import evaluate_scores

y_true = [0, 0, 1, 1, 1, 0, 0]
y_score = [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0]

result = evaluate_scores(
    y_true,
    y_score,
    threshold=0.5,
    metric_kwargs={"point_adjusted_k": {"k": 50}},
)

print(result["point"].asdict())
print(result["point_adjusted"].asdict())
print(result["point_adjusted_k"].asdict())
print(result["composite"].asdict())
```

### Find the best threshold

```python
from tadmetric import search_best_f1_threshold

y_true = [0, 0, 1, 1, 1, 0, 0]
y_score = [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0]

best = search_best_f1_threshold(y_true, y_score, metric="composite")

print(best.threshold)
print(best.f1)
print(best.evaluation["composite"].asdict())
```

## Included metrics

### Core precision / recall / F1

- Point-wise: `point_precision`, `point_recall`, `point_f1`
- Point-adjusted: `point_adjusted_precision`, `point_adjusted_recall`, `point_adjusted_f1`
- Point-adjusted `%K`: `point_adjusted_k_precision`, `point_adjusted_k_recall`, `point_adjusted_k_f1`
- Composite: `composite_precision`, `composite_recall`, `composite_f1`
- Event-wise: `event_precision`, `event_recall`, `event_f1`

### Delay-aware metrics

- `time_to_detect`
- `mean_time_to_detect`
- `median_time_to_detect`
- `missed_detection_rate`

### Thresholding and utilities

- `threshold_by_quantile`
- `threshold_by_topk`
- `threshold_by_best_f1`
- `search_best_f1_threshold`
- `apply_hysteresis`
- `binary_to_intervals`
- `intervals_to_binary`
- `merge_intervals`
- `scores_to_binary`

### Curve-based metrics

- `precision_recall_curve`
- `roc_curve`
- `auc_pr`
- `auc_roc`

## High-level API

For most workflows, these are the main entry points:

- `evaluate`
- `evaluate_scores`
- `available_metrics`
- `register_metric`

## Metric semantics

- Point-wise metrics treat each timestamp independently.
- Event-wise metrics treat each contiguous anomaly region as one event.
- Point-adjusted metrics credit an event if it is detected at least once.
- Point-adjusted `%K` metrics credit an event when at least `K%` of the event is detected.
- Composite metrics use time-wise precision and event-based recall.
- Interval semantics are half-open: `[start, end)`.
- Event matching defaults to `overlap > 0`.
- Zero division returns `0.0` by default.

## Development

```bash
pip install -e .[dev]
pytest
```

## Build

```bash
python -m build --no-isolation
python -m twine check dist/*
```
