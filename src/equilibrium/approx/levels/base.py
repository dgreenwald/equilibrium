"""Indexing levels for 1D basis and grid selections."""

from __future__ import annotations

import abc

import numpy as np
from numpy.typing import NDArray


class Levels(abc.ABC):
    """Base class for 1D level indexing (basis and grid)."""

    def __init__(self, level: int) -> None:
        if level < 0:
            raise ValueError("level must be >= 0")
        self._level = level

    @property
    def level(self) -> int:
        return self._level

    @abc.abstractmethod
    def level_size(self, level: int) -> int:
        """Return the number of indices at a given level."""

    def n_points(self, level: int | None = None) -> int:
        """Return total number of grid points up to the given level.

        Note: For these levels, the number of grid points equals the number
        of basis indices.
        """
        level = self._level if level is None else level
        if level < 0:
            raise ValueError("level must be >= 0")
        return sum(self.level_size(lv) for lv in range(level + 1))

    def basis_indices(self, level: int) -> NDArray[np.int64]:
        """Return the basis indices associated with a given level."""
        if level < 0:
            raise ValueError("level must be >= 0")
        start = sum(self.level_size(lv) for lv in range(level))
        size = self.level_size(level)
        return np.arange(start, start + size, dtype=np.int64)

    @abc.abstractmethod
    def grid_indices(self, level: int) -> NDArray[np.int64]:
        """Return grid indices associated with a given level."""
