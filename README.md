# tadmetric

A lightweight evaluation toolkit for time-series anomaly detection.

`tadmetric` provides point-wise, event-wise, point-adjusted, composite, and delay-aware
metrics, along with utilities for thresholding and interval conversion.

## Installation

```bash
pip install tadmetric
```

## Quick example

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

```python
from tadmetric import threshold_by_best_f1

best_threshold = threshold_by_best_f1(y_true, y_score, metric="composite")
```

## Concepts

- **Point-wise metrics** treat each timestamp independently.
- **Event-wise metrics** treat each contiguous anomaly region as one event.
- **Point-adjusted metrics** credit an anomaly event if it is detected at least once.
- **Point-adjusted %K metrics** credit an anomaly event when at least `K%` of the event is detected.
- **Composite metrics** combine time-wise precision with event-based recall.
- **Delay-aware metrics** measure how long it takes to first detect each event.

## Design choices

- Interval semantics are half-open: `[start, end)`.
- Event matching defaults to `overlap > 0`.
- Zero division returns `0.0` by default.
- Internally, event logic uses interval lists like `[(start, end), ...]`.

## API overview

### Point-wise
- `point_precision`
- `point_recall`
- `point_f1`

### Event-wise
- `event_precision`
- `event_recall`
- `event_f1`

### Point-adjusted
- `point_adjusted_precision`
- `point_adjusted_recall`
- `point_adjusted_f1`
- `point_adjusted_k_precision`
- `point_adjusted_k_recall`
- `point_adjusted_k_f1`

### Composite
- `composite_precision`
- `composite_recall`
- `composite_f1`

### Delay-aware
- `time_to_detect`
- `mean_time_to_detect`
- `median_time_to_detect`
- `missed_detection_rate`

### Thresholding and conversion
- `threshold_by_quantile`
- `threshold_by_topk`
- `threshold_by_best_f1`
- `search_best_f1_threshold`
- `apply_hysteresis`
- `binary_to_intervals`
- `intervals_to_binary`
- `merge_intervals`

### Curve-based
- `precision_recall_curve`
- `roc_curve`
- `auc_pr`
- `auc_roc`

### High-level API
- `evaluate`
- `evaluate_scores`
- `register_metric`
- `available_metrics`

## Development

```bash
pip install -e .[dev]
pytest
```

## Packaging

```bash
python -m build
twine check dist/*
```
