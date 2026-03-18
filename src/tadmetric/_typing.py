from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
import numpy.typing as npt

ArrayLike = Union[Sequence[int], Sequence[float], npt.NDArray[np.generic]]
Mode = Literal["point-wise", "point-adjusted", "composite"]
