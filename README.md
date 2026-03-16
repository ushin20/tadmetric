# tadmetric

Lightweight evaluation metrics for time-series anomaly detection.

`tadmetric` helps you evaluate anomaly scores or binary predictions with a small,
simple API. The primary workflow is centered on `evaluator(...)`, where you ask
for a metric result or the best threshold directly.

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

### Discover the API quickly

```python
import tadmetric as tm

print(tm.api_overview())
e = tm.evaluator(y_true, y_score)
pre, rec, f1 = e.point_adjusted(thr=0.3)
pre_k, rec_k, f1_k = e.point_adjusted(k=30)
print(e.best(metric="composite").threshold)
print(tm.available_metrics())
print([spec.asdict() for spec in tm.describe_metrics()])
```

### Evaluate with a reusable evaluator

```python
from tadmetric import evaluator

e = evaluator(y_true, y_score)

point_pre, point_rec, point_f1 = e.point_wise(thr=0.5)
pa_pre, pa_rec, pa_f1 = e.point_adjusted(thr=0.3)
pa_default_pre, pa_default_rec, pa_default_f1 = e.point_adjusted()
comp_pre, comp_rec, comp_f1 = e.composite()
best = e.best(metric="composite")

print(best.threshold)
print(best.f1)
print(best.evaluation["composite"].asdict())
print((pa_pre, pa_rec, pa_f1))

e.reevaluate(other_y_true, other_y_score)
```

Public evaluator metrics are:
- `point_wise`
- `point_adjusted` with `k=100` by default
- `composite`

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

- `evaluator`
- `e.point_wise(thr=...)`
- `e.point_adjusted(thr=..., k=100)`
- `e.composite(thr=...)`
- `e.best(metric=...)`
- `evaluate`
- `evaluate_scores`
- `available_metrics`
- `describe_metrics`
- `api_overview`
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
