"""Abstract base class for 1D basis functions."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Basis1d(ABC):
    """Abstract base class for 1D basis functions.

    Bases are purely mathematical function families used for approximation.
    They are independent of grids, which define spatial points.

    Subclasses must implement:
    - basis_type: String identifier for the basis type
    - evaluate: Evaluate basis functions at given points
    """

    @property
    @abstractmethod
    def basis_type(self) -> str:
        """Identifier for the basis type."""
        ...

    @abstractmethod
    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        """Evaluate basis functions at given points.

        Args:
            x: Points at which to evaluate (shape: (n_points,))
            n_basis: Number of basis functions to evaluate

        Returns:
            Basis matrix (shape: (n_points, n_basis))
        """
        ...

    @abstractmethod
    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate basis functions and their derivatives.

        Args:
            x: Points at which to evaluate (shape: (n_points,))
            n_basis: Number of basis functions to evaluate

        Returns:
            Tuple of (basis_values, basis_gradients), each of shape (n_points, n_basis)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
