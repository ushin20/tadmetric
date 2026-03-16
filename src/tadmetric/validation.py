from __future__ import annotations

import numpy as np


def as_numpy_1d(x, *, name: str, dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like object")
    if arr.size == 0:
        return arr
    if np.isnan(arr).any():
        raise ValueError(f"{name} must not contain NaN values")
    return arr


def check_same_length(y_true, y_pred) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have the same length")


def check_binary_array(x, *, name: str) -> np.ndarray:
    arr = as_numpy_1d(x, name=name)
    unique = np.unique(arr)
    if not np.all(np.isin(unique, [0, 1, False, True])):
        raise ValueError(f"{name} must contain only binary values 0/1")
    return arr.astype(int)


def check_score_array(x, *, name: str) -> np.ndarray:
    arr = as_numpy_1d(x, name=name, dtype=float)
    return arr.astype(float)
