"""Tensor-product friendly levels where each level selects a single index."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Levels


class TensorProductLevels(Levels):
    """Levels where each level corresponds to exactly one basis/grid index.

    This is useful for full tensor-product constructions where level ``k`` simply
    refers to the ``k``-th point (or basis function) in 1D. Stacking these levels
    across dimensions via :class:`equilibrium.approx.core.index.Index` yields full tensor
    grids/bases when paired with ``max_levels`` equal to the number of 1D points.
    """

    def __init__(self, n_points: int) -> None:
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        # Each level indexes one point, so highest level is n_points - 1
        super().__init__(level=n_points - 1)
        self._n_points = n_points

    def _validate_level(self, level: int) -> None:
        if level < 0 or level >= self._n_points:
            raise ValueError(f"level must be in [0, {self._n_points - 1}], got {level}")

    def level_size(self, level: int) -> int:
        self._validate_level(level)
        return 1

    def grid_indices(self, level: int) -> NDArray[np.int64]:
        self._validate_level(level)
        # Each level corresponds to the matching grid index
        return np.array([level], dtype=np.int64)

    def __repr__(self) -> str:
        return f"TensorProductLevels(n_points={self._n_points})"
