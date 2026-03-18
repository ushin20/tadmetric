# tadmetric

`tadmetric` is a lightweight evaluation toolkit for time-series anomaly detection.

The current package is intentionally small and built around one class:

```python
from tadmetric import Tadmetric
```

Use it to evaluate a fixed threshold or search for a good one with point-wise,
point-adjusted, and composite metrics.
