"""Chebyshev grid implementation."""

import numpy as np
from numpy.typing import NDArray

from .base import Grid1d


class ChebyshevLobattoGrid1d(Grid1d):
    """1D grid using Chebyshev-Gauss-Lobatto (extrema) points.

    Generates n points on [-1, 1] at the extrema of the Chebyshev polynomial
    T_{n-1}(x). These are optimal nodes for polynomial interpolation,
    minimizing the Runge phenomenon.

    The points are: x_k = cos(k * pi / (n-1)) for k = 0, 1, ..., n-1

    Returned in ascending order: [-1, ..., 0, ..., 1]
    """

    @property
    def grid_type(self) -> str:
        return "chebyshev_lobatto"

    @property
    def lb(self) -> float:
        return -1.0

    @property
    def ub(self) -> float:
        return 1.0

    def make_grid(self, n_points: int) -> NDArray[np.float64]:
        """Generate Chebyshev-Gauss-Lobatto grid points.

        Args:
            n_points: Number of grid points (must be >= 1)

        Returns:
            Array of n_points values in ascending order on [-1, 1]
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1")

        if n_points == 1:
            return np.array([0.0])

        # Chebyshev extrema: x_k = cos(k * pi / (n-1)) for k = 0, ..., n-1
        # Use linspace from pi to 0 to get ascending order directly
        return np.cos(np.linspace(np.pi, 0.0, n_points))

    def __repr__(self) -> str:
        return "ChebyshevLobattoGrid1d()"
