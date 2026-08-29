"""Hat (piecewise linear) basis implementations."""

import numpy as np
from numpy.typing import NDArray

from .base import Basis1d


def _evaluate_hat_family(
    x: NDArray[np.float64],
    centers: NDArray[np.float64],
    widths: NDArray[np.float64],
    *,
    include_constant: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate hats with predefined centers and widths."""
    x = np.asarray(x, dtype=np.float64)
    n_points = x.shape[0]
    n_hats = centers.shape[0]

    columns = n_hats + (1 if include_constant else 0)
    basis = np.zeros((n_points, columns), dtype=np.float64)
    gradients = np.zeros_like(basis)

    offset = 0
    if include_constant:
        basis[:, 0] = 1.0
        offset = 1

    if n_hats == 0:
        return basis, gradients

    diff = x.reshape(-1, 1) - centers.reshape(1, -1)
    abs_diff = np.abs(diff)
    widths_row = widths.reshape(1, -1)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = 1.0 - abs_diff / widths_row
    basis[:, offset:] = np.maximum(0.0, normalized)

    support = abs_diff < widths_row
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(diff < 0.0, 1.0 / widths_row, -1.0 / widths_row)
    slopes[~support] = 0.0
    slopes[diff == 0.0] = 0.0
    gradients[:, offset:] = slopes

    return basis, gradients


def _evaluate_modified_hierarchical_hat_family(
    x: NDArray[np.float64],
    centers: NDArray[np.float64],
    widths: NDArray[np.float64],
    edge_flags: NDArray[np.int8],
    *,
    include_constant: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate modified hierarchical hats with linear edge extrapolation."""
    x = np.asarray(x, dtype=np.float64)
    n_points = x.shape[0]
    n_hats = centers.shape[0]

    columns = n_hats + (1 if include_constant else 0)
    basis = np.zeros((n_points, columns), dtype=np.float64)
    gradients = np.zeros_like(basis)

    offset = 0
    if include_constant:
        basis[:, 0] = 1.0
        offset = 1

    if n_hats == 0:
        return basis, gradients

    diff = x.reshape(-1, 1) - centers.reshape(1, -1)
    abs_diff = np.abs(diff)
    widths_row = widths.reshape(1, -1)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = 1.0 - abs_diff / widths_row
    values = np.maximum(0.0, normalized)

    support = abs_diff < widths_row
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(diff < 0.0, 1.0 / widths_row, -1.0 / widths_row)
    slopes[~support] = 0.0
    slopes[diff == 0.0] = 0.0

    edge_left = edge_flags == -1
    if np.any(edge_left):
        diff_left = diff[:, edge_left]
        widths_left = widths_row[:, edge_left]
        with np.errstate(divide="ignore", invalid="ignore"):
            values_left = 1.0 - diff_left / widths_left
        values_left = np.where(
            diff_left >= 0.0, np.maximum(0.0, values_left), values_left
        )
        values[:, edge_left] = values_left

        with np.errstate(divide="ignore", invalid="ignore"):
            slopes_left = -np.ones_like(diff_left) / widths_left
        slopes_left = np.where(diff_left >= widths_left, 0.0, slopes_left)
        slopes_left[diff_left == 0.0] = 0.0
        slopes[:, edge_left] = slopes_left

    edge_right = edge_flags == 1
    if np.any(edge_right):
        diff_right = diff[:, edge_right]
        widths_right = widths_row[:, edge_right]
        with np.errstate(divide="ignore", invalid="ignore"):
            values_right = 1.0 + diff_right / widths_right
        values_right = np.where(
            diff_right <= 0.0, np.maximum(0.0, values_right), values_right
        )
        values[:, edge_right] = values_right

        with np.errstate(divide="ignore", invalid="ignore"):
            slopes_right = np.ones_like(diff_right) / widths_right
        slopes_right = np.where(diff_right <= -widths_right, 0.0, slopes_right)
        slopes_right[diff_right == 0.0] = 0.0
        slopes[:, edge_right] = slopes_right

    basis[:, offset:] = values
    gradients[:, offset:] = slopes

    return basis, gradients


class HierarchicalHatBasisInterior1d(Basis1d):
    """Hierarchical hats on (0, 1) without boundary functions."""

    @property
    def basis_type(self) -> str:
        return "hierarchical_hat_interior"

    def _centers_and_widths(
        self, n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        n_hats = n_basis - 1
        if n_hats <= 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        max_level = int(np.ceil(np.log2(n_basis + 1))) - 1
        centers = np.zeros(n_hats, dtype=np.float64)
        widths = np.zeros(n_hats, dtype=np.float64)

        idx = 0
        for level in range(1, max_level + 2):
            n_at_level = 2**level
            width = 1.0 / (2 * n_at_level)
            for k in range(n_at_level):
                if idx >= n_hats:
                    break
                centers[idx] = (2 * k + 1) / (2 * n_at_level)
                widths[idx] = width
                idx += 1
            if idx >= n_hats:
                break

        return centers, widths

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        centers, widths = self._centers_and_widths(n_basis)
        basis, _ = _evaluate_hat_family(x, centers, widths, include_constant=True)
        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centers, widths = self._centers_and_widths(n_basis)
        return _evaluate_hat_family(x, centers, widths, include_constant=True)

    def __repr__(self) -> str:
        return "HierarchicalHatBasisInterior1d()"


class HierarchicalHatBasis1d(Basis1d):
    """Hierarchical hats on [0, 1] including boundary functions."""

    @property
    def basis_type(self) -> str:
        return "hierarchical_hat"

    def _centers_and_widths(
        self, n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        n_hats = n_basis - 1
        if n_hats <= 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        centers = np.zeros(n_hats, dtype=np.float64)
        widths = np.zeros(n_hats, dtype=np.float64)

        idx = 0
        if n_hats >= 1:
            centers[idx] = 0.0
            widths[idx] = 0.5
            idx += 1
        if n_hats >= 2:
            centers[idx] = 1.0
            widths[idx] = 0.5
            idx += 1

        level = 2
        while idx < n_hats:
            n_at_level = 2 ** (level - 1)
            width = 1.0 / (2**level)
            for k in range(n_at_level):
                if idx >= n_hats:
                    break
                centers[idx] = (1.0 / (2**level)) + k / (2 ** (level - 1))
                widths[idx] = width
                idx += 1
            level += 1

        return centers, widths

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        centers, widths = self._centers_and_widths(n_basis)
        basis, _ = _evaluate_hat_family(x, centers, widths, include_constant=True)
        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centers, widths = self._centers_and_widths(n_basis)
        return _evaluate_hat_family(x, centers, widths, include_constant=True)

    def __repr__(self) -> str:
        return "HierarchicalHatBasis1d()"


class ModifiedHierarchicalHatBasis1d(Basis1d):
    """Hierarchical hats with linear edge extrapolation (no boundary hats)."""

    @property
    def basis_type(self) -> str:
        return "modified_hierarchical_hat"

    def _centers_widths_edges(
        self, n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        n_hats = n_basis - 1
        if n_hats < 1:
            raise ValueError("n_basis must be >= 2 for modified uniform hats")

        max_level = int(np.ceil(np.log2(n_basis + 1))) - 1
        centers = np.zeros(n_hats, dtype=np.float64)
        widths = np.zeros(n_hats, dtype=np.float64)
        edge_flags = np.zeros(n_hats, dtype=np.int8)

        idx = 0
        for level in range(1, max_level + 2):
            n_at_level = 2**level
            width = 1.0 / (2 * n_at_level)
            for k in range(n_at_level):
                if idx >= n_hats:
                    break
                centers[idx] = (2 * k + 1) / (2 * n_at_level)
                widths[idx] = width
                if k == 0:
                    edge_flags[idx] = -1
                elif k == n_at_level - 1:
                    edge_flags[idx] = 1
                idx += 1
            if idx >= n_hats:
                break

        return centers, widths, edge_flags

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        centers, widths, edge_flags = self._centers_widths_edges(n_basis)
        basis, _ = _evaluate_modified_hierarchical_hat_family(
            x, centers, widths, edge_flags, include_constant=True
        )
        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centers, widths, edge_flags = self._centers_widths_edges(n_basis)
        return _evaluate_modified_hierarchical_hat_family(
            x, centers, widths, edge_flags, include_constant=True
        )

    def __repr__(self) -> str:
        return "ModifiedHierarchicalHatBasis1d()"


class ModifiedUniformHatBasis1d(Basis1d):
    """Uniform-width hats with linear edge extrapolation (no boundary hats)."""

    @property
    def basis_type(self) -> str:
        return "modified_uniform_hat"

    def _centers_widths_edges(
        self, n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        n_hats = n_basis - 1
        if n_hats <= 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int8),
            )

        spacing = 1.0 / (n_hats + 1)
        centers = spacing * np.arange(1, n_hats + 1, dtype=np.float64)
        widths = np.full(n_hats, spacing, dtype=np.float64)
        edge_flags = np.zeros(n_hats, dtype=np.int8)

        if n_hats > 1:
            edge_flags[0] = -1
            edge_flags[-1] = 1

        return centers, widths, edge_flags

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        centers, widths, edge_flags = self._centers_widths_edges(n_basis)
        basis, _ = _evaluate_modified_hierarchical_hat_family(
            x, centers, widths, edge_flags, include_constant=True
        )
        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centers, widths, edge_flags = self._centers_widths_edges(n_basis)
        return _evaluate_modified_hierarchical_hat_family(
            x, centers, widths, edge_flags, include_constant=True
        )

    def __repr__(self) -> str:
        return "ModifiedUniformHatBasis1d()"


class UniformHatBasisInterior1d(Basis1d):
    """Uniform-width hats on (0, 1) without boundary hats."""

    @property
    def basis_type(self) -> str:
        return "uniform_hat_interior"

    def _centers_and_widths(
        self, n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        n_hats = n_basis - 1
        if n_hats <= 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        spacing = 1.0 / (n_hats + 1)
        centers = spacing * np.arange(1, n_hats + 1, dtype=np.float64)
        widths = np.full(n_hats, spacing, dtype=np.float64)
        return centers, widths

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        centers, widths = self._centers_and_widths(n_basis)
        basis, _ = _evaluate_hat_family(x, centers, widths, include_constant=True)
        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centers, widths = self._centers_and_widths(n_basis)
        return _evaluate_hat_family(x, centers, widths, include_constant=True)

    def __repr__(self) -> str:
        return "UniformHatBasisInterior1d()"


class UniformHatBasis1d(Basis1d):
    """Uniform-width hats on [0, 1] including boundary hats."""

    @property
    def basis_type(self) -> str:
        return "uniform_hat"

    def _centers_and_widths(
        self, n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute hat function centers and widths aligned with grid points.

        For n_basis points on [0, 1] at positions 0, 1/(n-1), ..., 1, we place
        n_basis - 1 hat functions at positions 1/(n-1), 2/(n-1), ..., 1
        (skipping the first grid point at 0, which is covered by the constant).
        Each hat has width 1/(n-1) so it equals 1 at its center grid point
        and 0 at adjacent grid points.
        """
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")

        n_hats = n_basis - 1
        if n_hats <= 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        # Grid spacing for n_basis points on [0, 1]
        spacing = 1.0 / (n_basis - 1) if n_basis > 1 else 1.0

        # Place hats at grid points 1, 2, ..., n_basis-1 (indices 1 to n-1)
        # These are at positions spacing, 2*spacing, ..., (n-1)*spacing = 1
        centers = spacing * np.arange(1, n_basis, dtype=np.float64)
        widths = np.full(n_hats, spacing, dtype=np.float64)

        return centers, widths

    def evaluate(self, x: NDArray[np.float64], n_basis: int) -> NDArray[np.float64]:
        centers, widths = self._centers_and_widths(n_basis)
        basis, _ = _evaluate_hat_family(x, centers, widths, include_constant=True)
        return basis

    def evaluate_with_gradients(
        self, x: NDArray[np.float64], n_basis: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centers, widths = self._centers_and_widths(n_basis)
        return _evaluate_hat_family(x, centers, widths, include_constant=True)

    def __repr__(self) -> str:
        return "UniformHatBasis1d()"
