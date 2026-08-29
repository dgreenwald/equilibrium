"""Chebyshev polynomial basis implementation."""

import numpy as np
from numpy.typing import NDArray

from .base import Basis1d


class ChebyshevBasis1d(Basis1d):
    """Chebyshev polynomial basis on [-1, 1].

    Uses Chebyshev polynomials of the first kind T_k(x) evaluated via
    the recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)

    This basis is typically paired with Chebyshev-Gauss-Lobatto grid points
    (ChebyshevLobattoGrid1d) for optimal numerical properties and to minimize
    the Runge phenomenon in polynomial interpolation.
    """

    @property
    def basis_type(self) -> str:
        return "chebyshev"

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        """Evaluate Chebyshev polynomials T_0, T_1, ..., T_{n_basis-1} at x.

        Args:
            x: Points at which to evaluate (shape: (n_points,))
            n_basis: Number of basis functions (polynomials) to evaluate

        Returns:
            Basis matrix (shape: (n_points, n_basis)) where entry [i, k]
            is T_k(x[i])
        """
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        x = np.asarray(x)
        n_points = len(x)

        basis = np.ones((n_points, n_basis))

        if n_basis > 1:
            basis[:, 1] = x

        for k in range(2, n_basis):
            basis[:, k] = 2 * x * basis[:, k - 1] - basis[:, k - 2]

        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate Chebyshev polynomials and their derivatives."""
        basis = self.evaluate(x, n_basis)
        gradients = np.zeros_like(basis)

        if n_basis > 1:
            gradients[:, 1] = 1.0

        if n_basis > 2:
            gradients[:, 2] = 4.0 * np.asarray(x)
            for k in range(3, n_basis):
                gradients[:, k] = k * (
                    2.0 * basis[:, k - 1] + gradients[:, k - 2] / (k - 2)
                )

        return basis, gradients

    def __repr__(self) -> str:
        return "ChebyshevBasis1d()"
