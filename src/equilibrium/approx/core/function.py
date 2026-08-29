"""Function approximation using sparse grids and collocation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .scheme import Scheme


class Function:
    """Sparse grid function approximation via collocation.

    This class provides a high-level interface for approximating functions using
    sparse grids. It wraps a Scheme object and handles coordinate transformation,
    coefficient fitting via collocation, and function evaluation.

    Attributes:
        scheme: The underlying Scheme defining sparse grid structure
        lb: Lower bounds in user coordinates (shape: dimension)
        ub: Upper bounds in user coordinates (shape: dimension)
        coefficients: Fitted coefficients (None until fit() is called)
                     Shape: (n_basis,) or (n_basis, n_outputs)
    """

    def __init__(
        self,
        scheme: Scheme,
        lb: NDArray[np.float64],
        ub: NDArray[np.float64],
        *,
        auto_construct: bool = True,
    ) -> None:
        """Initialize function with a scheme and user coordinate bounds.

        Args:
            scheme: Scheme object (must have construct() called already)
            lb: Lower bounds for each dimension
            ub: Upper bounds for each dimension

        Raises:
            ValueError: If lb/ub shapes don't match scheme dimension
            RuntimeError: If scheme.construct() hasn't been called
        """
        # Validate inputs
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        if lb.shape != (scheme.dimension,):
            raise ValueError(
                f"lb must have shape ({scheme.dimension},), got {lb.shape}"
            )
        if ub.shape != (scheme.dimension,):
            raise ValueError(
                f"ub must have shape ({scheme.dimension},), got {ub.shape}"
            )
        if np.any(lb >= ub):
            raise ValueError("lb must be < ub for all dimensions")

        # Ensure scheme is constructed if requested
        if auto_construct:
            try:
                _ = scheme.grid()
            except RuntimeError:
                scheme.construct()
        else:
            try:
                _ = scheme.grid()
            except RuntimeError as exc:
                raise RuntimeError(
                    "scheme.construct() must be called before creating Function"
                ) from exc

        self._scheme = scheme
        self._lb = lb
        self._ub = ub

        # Compute scaling factors for coordinate transformation
        grid_lb = np.array([scheme.grids[0].lb for _ in range(scheme.dimension)])
        grid_ub = np.array([scheme.grids[0].ub for _ in range(scheme.dimension)])
        self._scale = (ub - lb) / (grid_ub - grid_lb)
        self._grid_lb = grid_lb
        self._grid_ub = grid_ub

        # Transform grid points to user coordinates and cache
        grid_points_grid = scheme.grid()
        self._grid_points_user = self._transform_to_user(grid_points_grid)

        # Coefficients (None until fit() is called)
        self.coefficients: NDArray[np.float64] | None = None

    @property
    def scheme(self) -> Scheme:
        """The underlying Scheme object."""
        return self._scheme

    @property
    def lb(self) -> NDArray[np.float64]:
        """Lower bounds in user coordinates."""
        return self._lb

    @property
    def ub(self) -> NDArray[np.float64]:
        """Upper bounds in user coordinates."""
        return self._ub

    def get_grid_points(self) -> NDArray[np.float64]:
        """Return all sparse grid points in user coordinates.

        Returns:
            Array of shape (n_points, dimension)
        """
        return self._grid_points_user

    def get_n_points(self) -> int:
        """Return total number of grid points."""
        return self._grid_points_user.shape[0]

    def fit(self, values: NDArray[np.float64]) -> None:
        """Fit coefficients using collocation.

        Args:
            values: Function values at grid points (from get_grid_points())
                   Shape: (n_points,) for scalar or (n_points, n_outputs) for vector

        Raises:
            ValueError: If values.shape[0] != n_points
        """
        values = np.asarray(values, dtype=np.float64)

        n_points = self.get_n_points()

        # Validate shape
        if values.ndim == 1:
            if values.shape[0] != n_points:
                raise ValueError(
                    f"values must have length {n_points}, got {values.shape[0]}"
                )
            # Reshape to column vector for matrix multiply
            values_2d = values.reshape(-1, 1)
        elif values.ndim == 2:
            if values.shape[0] != n_points:
                raise ValueError(
                    f"values must have shape ({n_points}, n_outputs), got {values.shape}"
                )
            values_2d = values
        else:
            raise ValueError(f"values must be 1D or 2D array, got {values.ndim}D")

        # Fit using precomputed basis inverse: coeffs = B^{-1} @ values
        self.coefficients = self._scheme.basis_inverse @ values_2d

        # If single output, flatten back to 1D
        if values.ndim == 1:
            self.coefficients = self.coefficients.ravel()

    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate approximation at points.

        Args:
            points: Shape (dimension,) for single point or (n_eval, dimension) for multiple

        Returns:
            Shape: scalar for single point, (n_eval,) or (n_eval, n_outputs) for multiple

        Raises:
            RuntimeError: If fit() hasn't been called yet
            ValueError: If points shape is incorrect
        """
        if self.coefficients is None:
            raise RuntimeError("Must call fit() before evaluate()")

        points = np.asarray(points, dtype=np.float64)

        # Handle shape
        single_point = points.ndim == 1
        if single_point:
            if points.shape[0] != self._scheme.dimension:
                raise ValueError(
                    f"points must have shape ({self._scheme.dimension},), got {points.shape}"
                )
            points = points.reshape(1, -1)
        else:
            if points.shape[1] != self._scheme.dimension:
                raise ValueError(
                    f"points must have shape (n_eval, {self._scheme.dimension}), got {points.shape}"
                )

        # Transform to grid coordinates
        points_grid = self._transform_to_grid(points)

        # Evaluate basis functions
        basis_matrix = self._scheme.evaluate_bases(points_grid)

        # Compute function values: f = B @ coeffs
        result = basis_matrix @ self.coefficients

        # Handle single point case
        if single_point:
            if self.coefficients.ndim == 1:
                return float(result[0])
            else:
                return result[0, :]

        return result

    def evaluate_with_gradient(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate approximation and its gradient at points."""
        values, gradients, _, _ = self.evaluate_with_gradient_and_basis(points)
        return values, gradients

    def evaluate_with_basis(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate approximation and return basis functions.

        Args:
            points: Shape (dimension,) for single point or (n_eval, dimension) for multiple

        Returns:
            (values, basis_matrix) tuple

        Raises:
            RuntimeError: If fit() hasn't been called yet
            ValueError: If points shape is incorrect
        """
        if self.coefficients is None:
            raise RuntimeError("Must call fit() before evaluate_with_basis()")

        points = np.asarray(points, dtype=np.float64)

        # Handle shape
        single_point = points.ndim == 1
        if single_point:
            if points.shape[0] != self._scheme.dimension:
                raise ValueError(
                    f"points must have shape ({self._scheme.dimension},), got {points.shape}"
                )
            points = points.reshape(1, -1)
        else:
            if points.shape[1] != self._scheme.dimension:
                raise ValueError(
                    f"points must have shape (n_eval, {self._scheme.dimension}), got {points.shape}"
                )

        # Transform to grid coordinates
        points_grid = self._transform_to_grid(points)

        # Evaluate basis functions
        basis_matrix = self._scheme.evaluate_bases(points_grid)

        # Compute function values
        if self.coefficients.ndim == 1:
            result = basis_matrix @ self.coefficients
        else:
            result = basis_matrix @ self.coefficients

        # Handle single point case
        if single_point:
            if self.coefficients.ndim == 1:
                result = float(result[0])
            else:
                result = result[0, :]
            basis_matrix = basis_matrix.ravel()

        return result, basis_matrix

    def evaluate_with_gradient_and_basis(self, points: NDArray[np.float64]) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Evaluate approximation, gradients, and basis data at points."""
        if self.coefficients is None:
            raise RuntimeError(
                "Must call fit() before evaluate_with_gradient_and_basis()"
            )

        points = np.asarray(points, dtype=np.float64)
        single_point = points.ndim == 1
        if single_point:
            if points.shape[0] != self._scheme.dimension:
                raise ValueError(
                    f"points must have shape ({self._scheme.dimension},), got {points.shape}"
                )
            points = points.reshape(1, -1)
        else:
            if points.shape[1] != self._scheme.dimension:
                raise ValueError(
                    f"points must have shape (n_eval, {self._scheme.dimension}), got {points.shape}"
                )

        points_grid = self._transform_to_grid(points)
        basis_matrix, basis_gradients = self._scheme.evaluate_bases_with_gradients(
            points_grid
        )

        if self.coefficients.ndim == 1:
            values = basis_matrix @ self.coefficients
            grad_grid = np.einsum("idb,b->id", basis_gradients, self.coefficients)
        else:
            values = basis_matrix @ self.coefficients
            grad_grid = np.einsum("idb,bk->idk", basis_gradients, self.coefficients)

        gradients = self._rescale_gradients_to_user(grad_grid)

        if single_point:
            if self.coefficients.ndim == 1:
                values_out: NDArray[np.float64] | float = float(values[0])
                gradients_out = gradients[0, :]
            else:
                values_out = values[0, :]
                gradients_out = gradients[0, :, :]
            return (
                values_out,
                gradients_out,
                basis_matrix.ravel(),
                basis_gradients.reshape(self._scheme.dimension, -1),
            )

        return values, gradients, basis_matrix, basis_gradients

    def _transform_to_grid(
        self, points_user: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform points from user coordinates to grid coordinates.

        Args:
            points_user: Points in user coordinate system
                        Shape: (n_points, dimension)

        Returns:
            Points in grid coordinate system (same shape)
        """
        # points_grid = grid_lb + (points_user - user_lb) / scale
        return self._grid_lb + (points_user - self._lb) / self._scale

    def _transform_to_user(
        self, points_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform points from grid coordinates to user coordinates.

        Args:
            points_grid: Points in grid coordinate system
                        Shape: (n_points, dimension)

        Returns:
            Points in user coordinate system (same shape)
        """
        # points_user = user_lb + (points_grid - grid_lb) * scale
        return self._lb + (points_grid - self._grid_lb) * self._scale

    def _rescale_gradients_to_user(
        self, gradients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Scale gradients from grid coordinates back to user coordinates."""
        if gradients.ndim == 2:
            return gradients / self._scale.reshape(1, -1)
        return gradients / self._scale.reshape(1, -1, 1)

    def __repr__(self) -> str:
        return (
            f"Function(dimension={self._scheme.dimension}, "
            f"n_points={self.get_n_points()}, "
            f"fitted={self.coefficients is not None})"
        )
