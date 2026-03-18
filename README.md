# tadmetric

Lightweight evaluation utilities for time-series anomaly detection.

`tadmetric` is now centered on one small class, `Tadmetric`, for the workflow:

1. Load anomaly scores and binary labels
2. Evaluate a threshold with one of the built-in modes
3. Search for the best threshold when needed

## Installation

```bash
pip install tadmetric
```

Python `3.9+` is supported.

## Quick Start

```python
from tadmetric import Tadmetric

score = [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0]
label = [0, 0, 1, 1, 1, 0, 0]

tm = Tadmetric(score, label)

point = tm.evaluate(0.5, mode="point-wise")
adjusted = tm.evaluate(0.5, mode="point-adjusted", calc_latency=True)
composite = tm.evaluate(0.5, mode="composite")

print(point.asdict())
print(adjusted.latency)
print(composite.f1)
```

## Modes

- `point-wise`: regular point-level precision, recall, and F1
- `point-adjusted`: if an anomaly event is detected once, the whole event is credited
- `composite`: point-wise precision with event-wise recall

The mode argument accepts a few aliases such as `point`, `point_wise`, `point_adjusted`, and `comp`.

## Threshold Search

Use `search()` when you want the best threshold over a threshold grid:

```python
from tadmetric import Tadmetric

tm = Tadmetric(score, label)
best = tm.search(mode="point-adjusted")

print(best.threshold)
print(best.f1)
print(best.result.asdict())
```

By default, `search()` scans `steps=100` thresholds from `score.min()` to `score.max()`. If multiple thresholds tie, it prefers the higher threshold.

```python
best = tm.search(start=0.0, end=1.0, steps=101, mode="point-wise", verbose=False)
```

## Returned Objects

`evaluate()` returns a `MetricResult` with:

- `f1`, `precision`, `recall`
- `tp`, `tn`, `fp`, `fn`
- `latency` for point-adjusted evaluation when requested
- `detected_events`, `total_events` for event-aware modes

`search()` returns a `SearchResult` with the best threshold and the corresponding `MetricResult`.

## Validation Rules

- `score` and `label` must be 1D and non-empty
- `score` must be finite numeric values
- `label` must be binary values `0` or `1`
- `score` and `label` must have the same length

## Development

```bash
pip install -e .[dev]
pytest -q
```
