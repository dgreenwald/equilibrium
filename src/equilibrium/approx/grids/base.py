"""Abstract base class for 1D grids."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Grid1d(ABC):
    """Abstract base class for 1D grids used in function approximation.

    Subclasses must implement:
    - grid_type: String identifier for the grid type
    - lb: Lower bound of the grid domain
    - ub: Upper bound of the grid domain
    - make_grid: Generate grid points
    """

    @property
    @abstractmethod
    def grid_type(self) -> str:
        """Return the grid type identifier."""
        ...

    @property
    @abstractmethod
    def lb(self) -> float:
        """Lower bound of the grid domain."""
        ...

    @property
    @abstractmethod
    def ub(self) -> float:
        """Upper bound of the grid domain."""
        ...

    @abstractmethod
    def make_grid(self, n_points: int) -> NDArray[np.float64]:
        """Generate grid points.

        Args:
            n_points: Number of grid points to generate (must be >= 1)

        Returns:
            1D array of grid points in ascending order
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lb={self.lb}, ub={self.ub})"
