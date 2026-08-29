"""Stateless JAX evaluation for Chebyshev approximations.

Grid construction and coefficient fitting remain NumPy setup operations. This
module converts the immutable data needed for evaluation once, then accepts
coefficients explicitly so callers can differentiate with respect to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .bases import ChebyshevBasis1d
from .core.function import Function


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxApproximationData:
    """Immutable arrays and static metadata for JAX approximation evaluation."""

    basis_indices: jax.Array
    lower_bounds: jax.Array
    upper_bounds: jax.Array
    canonical_lower: jax.Array
    canonical_upper: jax.Array
    n_basis_1d: int
    dimension: int

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], tuple[int, int]]:
        """Separate dynamic arrays from shape-defining static metadata."""
        children = (
            self.basis_indices,
            self.lower_bounds,
            self.upper_bounds,
            self.canonical_lower,
            self.canonical_upper,
        )
        return children, (self.n_basis_1d, self.dimension)

    @classmethod
    def tree_unflatten(
        cls,
        auxiliary_data: tuple[int, int],
        children: tuple[jax.Array, ...],
    ) -> JaxApproximationData:
        """Reconstruct data after a JAX tree transformation."""
        n_basis_1d, dimension = auxiliary_data
        return cls(*children, n_basis_1d=n_basis_1d, dimension=dimension)


def make_jax_data(function: Function) -> JaxApproximationData:
    """Create immutable JAX evaluation data from a constructed function.

    Phase 1 supports Chebyshev bases only. Hat and mixed-basis schemes retain
    their NumPy evaluation API and are rejected here explicitly.

    Parameters
    ----------
    function
        Approximation function whose scheme supplies bounds and sparse basis
        indices. The function does not need to have fitted coefficients.

    Returns
    -------
    JaxApproximationData
        Data that can be passed through ``jax.jit`` and ``jax.vmap``.
    """
    scheme = function.scheme

    # Accessing the grid provides the public constructed-state validation.
    scheme.grid()

    if not all(isinstance(basis, ChebyshevBasis1d) for basis in scheme.bases):
        basis_types = ", ".join(basis.basis_type for basis in scheme.bases)
        raise NotImplementedError(
            "JAX evaluation currently supports ChebyshevBasis1d only; "
            f"received: {basis_types}"
        )

    dimension = scheme.dimension
    basis_indices = np.asarray(scheme.index.basis_ix, dtype=np.int64).reshape(
        -1, dimension
    )
    max_level = scheme.index.levels[0].level
    n_basis_1d = scheme.index.levels[0].n_points(max_level)
    canonical_lower = np.asarray([grid.lb for grid in scheme.grids], dtype=np.float64)
    canonical_upper = np.asarray([grid.ub for grid in scheme.grids], dtype=np.float64)

    return JaxApproximationData(
        basis_indices=jnp.asarray(basis_indices),
        lower_bounds=jnp.asarray(function.lb),
        upper_bounds=jnp.asarray(function.ub),
        canonical_lower=jnp.asarray(canonical_lower),
        canonical_upper=jnp.asarray(canonical_upper),
        n_basis_1d=n_basis_1d,
        dimension=dimension,
    )


def _validate_and_batch_points(
    data: JaxApproximationData, points: Any
) -> tuple[jax.Array, bool]:
    points_array = jnp.asarray(points)
    if points_array.ndim == 1:
        if points_array.shape != (data.dimension,):
            raise ValueError(
                f"points must have shape ({data.dimension},), got "
                f"{points_array.shape}"
            )
        return points_array[None, :], True
    if points_array.ndim == 2:
        if points_array.shape[1] != data.dimension:
            raise ValueError(
                f"points must have shape (n_eval, {data.dimension}), got "
                f"{points_array.shape}"
            )
        return points_array, False
    raise ValueError(
        f"points must be one- or two-dimensional, got {points_array.ndim}D"
    )


def _transform_to_canonical(data: JaxApproximationData, points: jax.Array) -> jax.Array:
    user_scale = (data.upper_bounds - data.lower_bounds) / (
        data.canonical_upper - data.canonical_lower
    )
    return data.canonical_lower + (points - data.lower_bounds) / user_scale


def _evaluate_chebyshev_1d(x: jax.Array, n_basis: int) -> jax.Array:
    """Evaluate ``T_0`` through ``T_{n_basis-1}`` on the final array axis."""
    values = jnp.ones((*x.shape, n_basis), dtype=x.dtype)
    if n_basis == 1:
        return values

    values = values.at[..., 1].set(x)

    def recurrence(k: int, current: jax.Array) -> jax.Array:
        next_value = 2.0 * x * current[..., k - 1] - current[..., k - 2]
        return current.at[..., k].set(next_value)

    return jax.lax.fori_loop(2, n_basis, recurrence, values)


def evaluate_bases_jax(data: JaxApproximationData, points: Any) -> jax.Array:
    """Evaluate the sparse Chebyshev basis at one point or a batch of points.

    Points outside the user domain are evaluated by polynomial extrapolation.
    Later solver phases are responsible for detecting or controlling such
    evaluations.
    """
    batched_points, single_point = _validate_and_batch_points(data, points)
    canonical_points = _transform_to_canonical(data, batched_points)
    values_1d = _evaluate_chebyshev_1d(canonical_points, data.n_basis_1d)

    # Expand to (n_eval, n_sparse_basis, dimension, n_basis_1d), gather the
    # selected one-dimensional polynomial in every dimension, then form tensor
    # products across dimensions.
    selected = jnp.take_along_axis(
        values_1d[:, None, :, :],
        data.basis_indices[None, :, :, None],
        axis=-1,
    )[..., 0]
    basis_matrix = jnp.prod(selected, axis=-1)

    if single_point:
        return basis_matrix[0]
    return basis_matrix


def evaluate_jax(
    data: JaxApproximationData,
    coefficients: Any,
    points: Any,
) -> jax.Array:
    """Evaluate scalar- or vector-valued coefficients with a sparse basis."""
    coefficient_array = jnp.asarray(coefficients)
    if coefficient_array.ndim not in (1, 2):
        raise ValueError(
            "coefficients must have shape (n_basis,) or "
            f"(n_basis, n_outputs), got {coefficient_array.shape}"
        )
    if coefficient_array.shape[0] != data.basis_indices.shape[0]:
        raise ValueError(
            f"coefficients must have leading dimension "
            f"{data.basis_indices.shape[0]}, got {coefficient_array.shape[0]}"
        )

    basis = evaluate_bases_jax(data, points)
    return basis @ coefficient_array


__all__ = [
    "JaxApproximationData",
    "evaluate_bases_jax",
    "evaluate_jax",
    "make_jax_data",
]
