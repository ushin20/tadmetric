# Examples

## Basic Evaluation

```python
from tadmetric import Tadmetric

score = [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0]
label = [0, 0, 1, 1, 1, 0, 0]

tm = Tadmetric(score, label)

print(tm.evaluate(0.5, mode="point-wise").asdict())
print(tm.evaluate(0.5, mode="point-adjusted", calc_latency=True).asdict())
print(tm.evaluate(0.5, mode="composite").asdict())
```

## Best Threshold Search

```python
from tadmetric import Tadmetric

tm = Tadmetric(score, label)
best = tm.search(mode="point-adjusted")

print(best.threshold)
print(best.f1)
```
