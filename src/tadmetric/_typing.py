from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[int], Sequence[float], np.ndarray]
Interval = Tuple[int, int]
IntervalList = List[Interval]
