# API

## Main Class

```python
from tadmetric import Tadmetric

tm = Tadmetric(score, label)
```

## Evaluation

```python
point = tm.evaluate(0.5, mode="point-wise")
adjusted = tm.evaluate(0.5, mode="point-adjusted", calc_latency=True)
composite = tm.evaluate(0.5, mode="composite")
```

Each call returns a `MetricResult`.

Important fields:

- `f1`
- `precision`
- `recall`
- `tp`, `tn`, `fp`, `fn`
- `latency`
- `detected_events`, `total_events`

## Search

```python
best = tm.search(mode="point-adjusted")
print(best.threshold)
print(best.result.asdict())
```

`search()` evaluates the exact unique score thresholds.

If you want a grid search over a fixed range:

```python
best = tm.search(start=0.0, end=1.0, steps=101, mode="point-wise", verbose=False)
```

## Modes

- `point-wise`
- `point-adjusted`
- `composite`

Common aliases such as `point`, `point_wise`, `point_adjusted`, and `comp` are also accepted.
