"""Uniform grid implementations."""

import numpy as np
from numpy.typing import NDArray

from .base import Grid1d


class UniformGrid1d(Grid1d):
    """Uniform grid on (0, 1) excluding boundary points.

    Generates n evenly-spaced interior points, excluding 0 and 1.
    For n points, the spacing is 1/(n+1) and points are at
    k/(n+1) for k = 1, ..., n.

    Example:
        >>> grid = UniformGrid1d()
        >>> grid.make_grid(3)
        array([0.25, 0.5, 0.75])
    """

    @property
    def grid_type(self) -> str:
        return "uniform"

    @property
    def lb(self) -> float:
        return 0.0

    @property
    def ub(self) -> float:
        return 1.0

    def make_grid(self, n_points: int) -> NDArray[np.float64]:
        """Generate uniform interior grid points.

        Args:
            n_points: Number of grid points (must be >= 1)

        Returns:
            Array of n_points values in ascending order on (0, 1)
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        if n_points == 1:
            return np.array([0.5])
        return np.linspace(0.0, 1.0, n_points + 2)[1:-1]

    def __repr__(self) -> str:
        return "UniformGrid1d()"


class UniformGridWithBoundary1d(Grid1d):
    """Uniform grid on [0, 1] including boundary points.

    Generates n evenly-spaced points including the endpoints 0 and 1.
    For n points, the spacing is 1/(n-1) and points are at
    k/(n-1) for k = 0, ..., n-1.

    Example:
        >>> grid = UniformGridWithBoundary1d()
        >>> grid.make_grid(5)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """

    @property
    def grid_type(self) -> str:
        return "uniform_with_boundary"

    @property
    def lb(self) -> float:
        return 0.0

    @property
    def ub(self) -> float:
        return 1.0

    def make_grid(self, n_points: int) -> NDArray[np.float64]:
        """Generate uniform grid points including boundaries.

        Args:
            n_points: Number of grid points (must be >= 1)

        Returns:
            Array of n_points values in ascending order on [0, 1]
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        if n_points == 1:
            return np.array([0.5])
        return np.linspace(0.0, 1.0, n_points)

    def __repr__(self) -> str:
        return "UniformGridWithBoundary1d()"
