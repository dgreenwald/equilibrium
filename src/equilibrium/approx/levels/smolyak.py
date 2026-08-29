"""Smolyak-style 1D indexing by levels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Levels


class SmolyakLevels(Levels):
    """Smolyak indexing for selecting indices by levels (1D)."""

    def level_size(self, level: int) -> int:
        if level < 0:
            raise ValueError("level must be >= 0")
        if level == 0:
            return 1
        if level in (1, 2):
            return 2
        return 2 ** (level - 1)

    def grid_indices(self, level: int) -> NDArray[np.int64]:
        if level < 0:
            raise ValueError("level must be >= 0")
        n_points = self.n_points()

        if level == 0:
            return np.array([(n_points - 1) // 2], dtype=np.int64)
        if level == 1:
            return np.array([0, n_points - 1], dtype=np.int64)

        size = self.level_size(level)
        stride = n_points // size
        if stride < 1:
            raise ValueError("n_points too small for level size")
        indices = stride // 2 + stride * np.arange(size, dtype=np.int64)
        return indices


class SmolyakInteriorLevels(Levels):
    """Smolyak-style indexing skipping boundary level points."""

    def level_size(self, level: int) -> int:
        if level < 0:
            raise ValueError("level must be >= 0")
        if level == 0:
            return 1
        return 2**level

    def grid_indices(self, level: int) -> NDArray[np.int64]:
        if level < 0:
            raise ValueError("level must be >= 0")
        n_points = self.n_points()

        if level == 0:
            return np.array([(n_points - 1) // 2], dtype=np.int64)

        max_level = self.level
        stride = 2 ** (max_level - level)
        size = self.level_size(level)
        k_values = stride * (2 * np.arange(size, dtype=np.int64) + 1)
        return k_values - 1
