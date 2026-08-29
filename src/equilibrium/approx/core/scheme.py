"""Scheme for constructing multidimensional grids and bases."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..bases import Basis1d
from ..grids import Grid1d
from ..levels import Levels
from .index import Index


class Scheme:
    """Construct multidimensional grids and bases from 1D components.

    The Scheme class combines 1D grids, 1D bases, and level information to construct
    multidimensional approximation schemes. It uses an internally created Index to
    determine which level combinations are admissible and constructs the corresponding
    multidimensional grids and basis functions.

    Grids and bases are now separate concepts - grids define spatial points,
    while bases are mathematical function families used for approximation.
    """

    def __init__(
        self,
        grid: Grid1d,
        basis: Basis1d,
        levels: Levels,
        *,
        max_total_level: int | None,
        max_levels: Sequence[int] | int | None = None,
        dimension: int | None = None,
        auto_construct: bool = True,
    ) -> None:
        """Initialize the scheme with 1D grid, basis, and level information.

        Args:
            grid: 1D grid to be used for all dimensions
            basis: 1D basis to be used for all dimensions
            levels: Levels object to be used for all dimensions
            max_total_level: Maximum sum of levels across dimensions
            max_levels: Maximum level per dimension (int or sequence). If sequence,
                       dimension is inferred from its length. If int or None,
                       dimension parameter must be provided.
            dimension: Number of dimensions. Required if max_levels is int or None.

        Raises:
            ValueError: If dimension cannot be determined from parameters
            ValueError: If both max_levels sequence and dimension are provided but conflict
        """
        # Determine dimension from max_levels if it's a sequence
        if isinstance(max_levels, Sequence) and not isinstance(max_levels, str):
            inferred_dim = len(max_levels)
            if dimension is not None and dimension != inferred_dim:
                raise ValueError(
                    f"dimension={dimension} conflicts with max_levels length={inferred_dim}"
                )
            dimension = inferred_dim
        elif dimension is None:
            raise ValueError(
                "dimension must be provided when max_levels is not a sequence"
            )

        if dimension < 1:
            raise ValueError("dimension must be >= 1")

        # Create levels tuple by repeating the single Levels object
        levels_tuple = tuple(levels for _ in range(dimension))

        # Create Index internally
        self._index = Index(
            levels=levels_tuple,
            max_total_level=max_total_level,
            max_levels=max_levels,
        )

        # Store grids and bases (repeated for each dimension)
        self._grids = tuple(grid for _ in range(dimension))
        self._bases = tuple(basis for _ in range(dimension))

        # Sparse grid and basis inverse (computed by construct())
        self._sparse_grid: NDArray[np.float64] | None = None
        self._basis_inverse: NDArray[np.float64] | None = None

        if auto_construct:
            self.construct()

    @property
    def grids(self) -> tuple[Grid1d, ...]:
        """The 1D grids for each dimension."""
        return self._grids

    @property
    def bases(self) -> tuple[Basis1d, ...]:
        """The 1D bases for each dimension."""
        return self._bases

    @property
    def index(self) -> Index:
        """The index defining admissible level combinations."""
        return self._index

    @property
    def dimension(self) -> int:
        """Number of dimensions."""
        return self._index.dimension

    @property
    def basis_inverse(self) -> NDArray[np.float64]:
        """The inverse of the basis matrix for collocation.

        Returns:
            Basis inverse matrix of shape (n_points, n_points)

        Raises:
            RuntimeError: If construct() hasn't been called
        """
        if self._basis_inverse is None:
            raise RuntimeError("Must call construct() before accessing basis_inverse")
        return self._basis_inverse

    def construct(self) -> None:
        """Build the sparse grid and basis inverse matrix.

        This method must be called before using grid() or evaluate_bases().
        It computes and caches the sparse grid points and the basis inverse
        matrix for collocation.
        """
        # 1. Create complete 1D grid with all points up to max level
        max_level = self._index.levels[0].level
        # Get number of points needed for max level from the Levels object
        n_points_1d = self._index.levels[0].n_points(max_level)
        grid_1d = self._grids[0].make_grid(n_points_1d)

        # 2. Build sparse grid using Index
        self._sparse_grid = self._make_sparse_grid(grid_1d)

        # 3. Build basis matrix at grid points and invert for collocation
        basis_matrix = self._build_basis_matrix_at_grid()
        self._basis_inverse = np.linalg.inv(basis_matrix)

    def grid(self) -> NDArray[np.float64]:
        """Return the sparse grid points.

        Returns:
            Array of shape (n_points, dimension) where each row is a grid point

        Raises:
            RuntimeError: If construct() hasn't been called
        """
        if self._sparse_grid is None:
            raise RuntimeError("Must call construct() before accessing grid")
        return self._sparse_grid

    def _make_sparse_grid(self, grid_1d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build multidimensional sparse grid from complete 1D grid.

        Args:
            grid_1d: Complete 1D grid containing all points for max level

        Returns:
            Sparse grid array of shape (size, dimension)
        """
        size = self._index.size
        if size == 0:
            return np.empty((0, self.dimension), dtype=np.float64)

        sparse_grid = np.empty((size, self.dimension), dtype=np.float64)
        grid_ix = self._index.grid_ix

        for i in range(size):
            for d in range(self.dimension):
                grid_idx = grid_ix[self.dimension * i + d]
                sparse_grid[i, d] = grid_1d[grid_idx]

        return sparse_grid

    def _make_sparse_basis(self, basis_1d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build sparse basis tensor products from 1D basis evaluations.

        Args:
            basis_1d: 1D basis matrices stacked vertically, one per dimension
                     Shape: (n_eval * dimension, max_n_basis_1d)

        Returns:
            Sparse basis matrix of shape (n_eval, size)
        """
        size = self._index.size
        if size == 0:
            n_eval = basis_1d.shape[0] // self.dimension if basis_1d.shape[0] > 0 else 0
            return np.empty((n_eval, 0), dtype=np.float64)

        n_eval = basis_1d.shape[0] // self.dimension
        sparse_basis = np.ones((n_eval, size), dtype=np.float64)
        basis_ix = self._index.basis_ix

        for j in range(size):
            for d in range(self.dimension):
                basis_idx = basis_ix[self.dimension * j + d]
                sparse_basis[:, j] *= basis_1d[d * n_eval : (d + 1) * n_eval, basis_idx]

        return sparse_basis

    def _make_sparse_basis_with_gradients(
        self,
        basis_1d: NDArray[np.float64],
        grad_1d: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combine 1D basis values and gradients into sparse representations."""
        size = self._index.size
        if size == 0:
            n_eval = basis_1d.shape[0] // self.dimension if basis_1d.size else 0
            empty_basis = np.empty((n_eval, 0), dtype=np.float64)
            empty_grad = np.empty((n_eval, self.dimension, 0), dtype=np.float64)
            return empty_basis, empty_grad

        n_eval = basis_1d.shape[0] // self.dimension
        sparse_basis = np.ones((n_eval, size), dtype=np.float64)
        sparse_grad = np.zeros((n_eval, self.dimension, size), dtype=np.float64)

        basis_ix = self._index.basis_ix
        values_by_dim = [
            basis_1d[d * n_eval : (d + 1) * n_eval, :] for d in range(self.dimension)
        ]
        grads_by_dim = [
            grad_1d[d * n_eval : (d + 1) * n_eval, :] for d in range(self.dimension)
        ]

        for j in range(size):
            values_per_dim = []
            grads_per_dim = []
            for d in range(self.dimension):
                idx = basis_ix[self.dimension * j + d]
                values_per_dim.append(values_by_dim[d][:, idx])
                grads_per_dim.append(grads_by_dim[d][:, idx])

            basis_col = np.ones(n_eval, dtype=np.float64)
            for values in values_per_dim:
                basis_col *= values
            sparse_basis[:, j] = basis_col

            for d in range(self.dimension):
                values_d = values_per_dim[d]
                grads_d = grads_per_dim[d]
                grad_col = np.zeros(n_eval, dtype=np.float64)
                mask = np.abs(values_d) > 1e-12
                grad_col[mask] = grads_d[mask] * basis_col[mask] / values_d[mask]

                if np.any(~mask):
                    product = np.ones(n_eval, dtype=np.float64)
                    for d_other in range(self.dimension):
                        if d_other == d:
                            product *= grads_per_dim[d_other]
                        else:
                            product *= values_per_dim[d_other]
                    grad_col[~mask] = product[~mask]

                sparse_grad[:, d, j] = grad_col

        return sparse_basis, sparse_grad

    def _build_basis_matrix_at_grid(self) -> NDArray[np.float64]:
        """Build the full basis matrix evaluated at the sparse grid points.

        Returns:
            Basis matrix of shape (n_points, n_points) where n_points = index.size
        """
        if self._sparse_grid is None:
            raise RuntimeError("Sparse grid must be built first")

        # Determine max number of basis functions needed
        max_level = self._index.levels[0].level
        max_n_basis_1d = self._index.levels[0].n_points(max_level)

        # Evaluate 1D bases at each dimension of the grid points
        basis_1d_list = []
        for d in range(self.dimension):
            basis_1d = self._bases[d].evaluate(self._sparse_grid[:, d], max_n_basis_1d)
            basis_1d_list.append(basis_1d)

        # Stack vertically: shape (n_points * dimension, max_n_basis_1d)
        basis_1d_stacked = np.vstack(basis_1d_list)

        # Build sparse basis matrix
        basis_matrix = self._make_sparse_basis(basis_1d_stacked)

        return basis_matrix

    def make_grids(self) -> tuple[tuple[NDArray[np.float64], ...], ...]:
        """Construct multidimensional grids for all level combinations.

        Returns:
            Tuple of grid tuples, one for each level combination in the index.
            Each inner tuple contains 1D grid arrays for each dimension.
        """
        grids_by_level = []
        for block in self._index.iter_blocks():
            level_grids = tuple(
                grid.make_grid(len(grid_indices))
                for grid, grid_indices in zip(self._grids, block.grid_indices)
            )
            grids_by_level.append(level_grids)
        return tuple(grids_by_level)

    def evaluate_bases(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate sparse basis functions at given points.

        Args:
            points: Evaluation points
                   Shape: (dimension,) for single point
                          (n_eval, dimension) for multiple points

        Returns:
            Basis matrix of shape (n_eval, n_basis) or (n_basis,) for single point
            Each column is one basis function evaluated at all points

        Raises:
            ValueError: If points shape doesn't match dimension
            RuntimeError: If construct() hasn't been called
        """
        if self._sparse_grid is None:
            raise RuntimeError("Must call construct() before evaluating bases")

        # Handle shape: 1D → 2D
        single_point = points.ndim == 1
        if single_point:
            if points.shape[0] != self.dimension:
                raise ValueError(
                    f"points must have shape ({self.dimension},), got {points.shape}"
                )
            points = points.reshape(1, -1)
        else:
            if points.shape[1] != self.dimension:
                raise ValueError(
                    f"points must have shape (n_eval, {self.dimension}), got {points.shape}"
                )

        # Determine max number of basis functions needed
        max_level = self._index.levels[0].level
        max_n_basis_1d = self._index.levels[0].n_points(max_level)

        # Evaluate 1D bases for each dimension
        basis_1d_list = []
        for d in range(self.dimension):
            basis_1d = self._bases[d].evaluate(points[:, d], max_n_basis_1d)
            basis_1d_list.append(basis_1d)

        # Stack vertically: shape (n_eval * dimension, max_n_basis_1d)
        basis_1d_stacked = np.vstack(basis_1d_list)

        # Build sparse basis matrix
        sparse_basis = self._make_sparse_basis(basis_1d_stacked)

        if single_point:
            return sparse_basis.ravel()
        return sparse_basis

    def evaluate_bases_with_gradients(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate sparse basis functions and gradients at points."""
        if self._sparse_grid is None:
            raise RuntimeError("Must call construct() before evaluating bases")

        single_point = points.ndim == 1
        if single_point:
            if points.shape[0] != self.dimension:
                raise ValueError(
                    f"points must have shape ({self.dimension},), got {points.shape}"
                )
            points = points.reshape(1, -1)
        else:
            if points.shape[1] != self.dimension:
                raise ValueError(
                    f"points must have shape (n_eval, {self.dimension}), got {points.shape}"
                )

        max_level = max(self._index.max_levels)
        max_n_basis_1d = self._index.levels[0].n_points(max_level)

        basis_values = []
        basis_grads = []
        for d in range(self.dimension):
            values, grads = self._bases[d].evaluate_with_gradients(
                points[:, d], max_n_basis_1d
            )
            basis_values.append(values)
            basis_grads.append(grads)

        basis_1d_stacked = np.vstack(basis_values)
        grad_1d_stacked = np.vstack(basis_grads)

        sparse_basis, sparse_grad = self._make_sparse_basis_with_gradients(
            basis_1d_stacked, grad_1d_stacked
        )

        if single_point:
            return sparse_basis.ravel(), sparse_grad.reshape(self.dimension, -1)
        return sparse_basis, sparse_grad

    def __repr__(self) -> str:
        return (
            f"Scheme(dimension={self.dimension}, "
            f"max_total_level={self._index.max_total_level})"
        )
