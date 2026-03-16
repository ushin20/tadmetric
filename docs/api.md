# API

## Quick discovery

```python
import tadmetric as tm

print(tm.api_overview())
e = tm.evaluator(y_true, y_score)
print(e.point_adjusted(thr=0.3).f1)
print(e.point_adjusted(k=30).precision)
print(e.best(metric="composite").threshold)
print(tm.available_metrics())
print([spec.asdict() for spec in tm.describe_metrics()])
```

## Main entry points

- `evaluator(y_true, y_score)`
- `e.point_wise(thr=...)`
- `e.point_adjusted(thr=..., k=100)`
- `e.composite(thr=...)`
- `e.best(metric="composite")`
- `evaluate(y_true, y_pred, ...)`
- `evaluate_scores(y_true, y_score, threshold=...)`
- `search_best_f1_threshold(y_true, y_score, ...)`

## Reusable evaluator

```python
from tadmetric import evaluator

e = evaluator(y_true, y_score)

pre, rec, f1 = e.point_wise(thr=0.5)
pre, rec, f1 = e.point_adjusted(thr=0.5, k=30)
pre, rec, f1 = e.point_adjusted()
pre, rec, f1 = e.composite()
e.best(metric="composite")

e.reevaluate(next_y_true, next_y_score)
```

Evaluator metric names:
- `point_wise`
- `point_adjusted` with `k=100` by default
- `composite`
