# Concepts

## Point-wise

Each timestamp is treated independently.

## Point-adjusted

If any timestamp inside a true anomaly event is detected, the whole event is
credited for point-adjusted evaluation.

## Composite

Composite F1 combines point-wise precision with event-wise recall.

## Search Behavior

`search()` evaluates a threshold grid from `score.min()` to `score.max()` by
default. If several thresholds have the same score, the higher threshold is
preferred.
