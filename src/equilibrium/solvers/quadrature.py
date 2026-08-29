"""Quadrature and exogenous-process data containers.

Rule construction is a NumPy setup-time operation.  The lightweight JAX
containers returned by :meth:`QuadratureRule.as_jax` and
:meth:`ExogenousProcess.as_jax` contain arrays only, so they can be passed
through JAX transformations without making static metadata part of a trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterator, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

_WEIGHT_SUM_TOLERANCE = 1e-12
_RULE_KINDS = frozenset({"deterministic", "tensor", "smolyak"})


class JaxQuadratureRule(NamedTuple):
    """JAX arrays for a quadrature rule."""

    nodes: jax.Array
    weights: jax.Array


@dataclass(frozen=True)
class QuadratureRule:
    """An immutable, row-oriented quadrature rule.

    ``nodes`` has shape ``(n_nodes, dimension)`` and ``weights`` has shape
    ``(n_nodes,)``.  Weights must sum to one within ``1e-12``.  Signed weights
    are permitted only for Smolyak rules.
    """

    nodes: np.ndarray
    weights: np.ndarray
    kind: str
    orders: tuple[int, ...] | None
    level: int | None = None

    def __post_init__(self) -> None:
        nodes = _immutable_float_array(self.nodes, "nodes")
        weights = _immutable_float_array(self.weights, "weights")

        if nodes.ndim != 2:
            raise ValueError("nodes must be a two-dimensional array")
        if weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional array")
        if nodes.shape[0] == 0:
            raise ValueError("a quadrature rule must contain at least one node")
        if nodes.shape[0] != weights.shape[0]:
            raise ValueError("nodes and weights must contain the same number of rows")
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=_WEIGHT_SUM_TOLERANCE):
            raise ValueError("weights must sum to one")

        if self.kind not in _RULE_KINDS:
            raise ValueError(f"unknown quadrature rule kind: {self.kind!r}")
        if self.kind != "smolyak" and np.any(weights < 0.0):
            raise ValueError("negative weights are permitted only for Smolyak rules")

        orders = self.orders
        if self.kind == "deterministic":
            if orders != ():
                raise ValueError("a deterministic rule must have empty orders")
            if self.level is not None:
                raise ValueError("a deterministic rule cannot have a level")
            if nodes.shape != (1, 0):
                raise ValueError("a deterministic rule must have node shape (1, 0)")
        elif self.kind == "tensor":
            if not isinstance(orders, tuple) or len(orders) != nodes.shape[1]:
                raise ValueError("tensor orders must have one entry per dimension")
            if any(
                not isinstance(order, int) or isinstance(order, bool) or order < 1
                for order in orders
            ):
                raise ValueError("tensor orders must be positive integers")
            if self.level is not None:
                raise ValueError("a tensor rule cannot have a level")
        else:
            if orders is not None:
                raise ValueError("a Smolyak rule must have orders=None")
            if (
                not isinstance(self.level, int)
                or isinstance(self.level, bool)
                or self.level < 0
            ):
                raise ValueError("a Smolyak rule must have a nonnegative integer level")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def dimension(self) -> int:
        """Number of node coordinates."""

        return self.nodes.shape[1]

    @property
    def n_nodes(self) -> int:
        """Number of quadrature nodes."""

        return self.nodes.shape[0]

    def integrate(self, values: np.ndarray, axis: int = 0) -> np.ndarray:
        """Contract ``values`` with the rule weights along ``axis``."""

        values_array = np.asarray(values)
        if values_array.ndim == 0:
            raise ValueError("values must have at least one dimension")
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise ValueError(f"invalid integration axis {axis!r}")
        normalized_axis = axis if axis >= 0 else axis + values_array.ndim
        if normalized_axis < 0 or normalized_axis >= values_array.ndim:
            raise ValueError(f"invalid integration axis {axis!r}")
        if values_array.shape[normalized_axis] != self.n_nodes:
            raise ValueError(
                "the integration axis length must equal the number of nodes"
            )
        return np.tensordot(self.weights, values_array, axes=(0, normalized_axis))

    def as_jax(self) -> JaxQuadratureRule:
        """Copy the rule data into JAX arrays."""

        return JaxQuadratureRule(jnp.asarray(self.nodes), jnp.asarray(self.weights))


class JaxExogenousProcess(NamedTuple):
    """JAX matrices defining an exogenous process."""

    persistence: jax.Array
    innovation_impact: jax.Array


@dataclass(frozen=True)
class ExogenousProcess:
    """Immutable matrices for ``z_next = persistence @ z + impact @ epsilon``."""

    names: tuple[str, ...]
    persistence: np.ndarray
    innovation_impact: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple) or not all(
            isinstance(name, str) for name in self.names
        ):
            raise ValueError("names must be a tuple of strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("exogenous names must be unique")

        persistence = _immutable_float_array(self.persistence, "persistence")
        innovation_impact = _immutable_float_array(
            self.innovation_impact, "innovation_impact"
        )
        dimension = len(self.names)
        if persistence.shape != (dimension, dimension):
            raise ValueError("persistence must have shape (n_exogenous, n_exogenous)")
        if innovation_impact.ndim != 2 or innovation_impact.shape[0] != dimension:
            raise ValueError(
                "innovation_impact must have shape (n_exogenous, n_innovations)"
            )

        object.__setattr__(self, "persistence", persistence)
        object.__setattr__(self, "innovation_impact", innovation_impact)

    def as_jax(self) -> JaxExogenousProcess:
        """Copy the process matrices into JAX arrays."""

        return JaxExogenousProcess(
            jnp.asarray(self.persistence), jnp.asarray(self.innovation_impact)
        )


def deterministic_quadrature() -> QuadratureRule:
    """Return the one-node, zero-dimensional deterministic rule."""

    return QuadratureRule(
        nodes=np.empty((1, 0), dtype=float),
        weights=np.ones(1, dtype=float),
        kind="deterministic",
        orders=(),
    )


def gauss_hermite_normal(
    degree: int,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> QuadratureRule:
    """Construct a Gauss-Hermite rule for ``N(mu, sigma**2)``.

    A rule with ``degree`` nodes integrates polynomials through degree
    ``2 * degree - 1`` exactly, up to floating-point roundoff.  NumPy's
    physicists' Hermite rule is normalized and its nodes are multiplied by
    ``sqrt(2) * sigma`` to convert its ``exp(-x**2)`` weighting to a normal
    probability distribution.
    """

    if not isinstance(degree, Integral) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be a positive integer")
    degree = int(degree)
    mu = _finite_scalar(mu, "mu")
    sigma = _finite_scalar(sigma, "sigma")
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive")

    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    weights = weights / weights.sum()
    nodes = mu + np.sqrt(2.0) * sigma * nodes
    return QuadratureRule(
        nodes=nodes[:, np.newaxis],
        weights=weights,
        kind="tensor",
        orders=(degree,),
    )


def tensor_gauss_hermite(
    degrees: int | Sequence[int],
    *,
    dimension: int | None = None,
    mu: float | Sequence[float] = 0.0,
    sigma: float | Sequence[float] = 1.0,
    max_nodes: int | None = 100_000,
) -> QuadratureRule:
    """Construct a tensor-product rule for independent normal variables.

    A scalar degree is used in every dimension and defaults to one dimension
    when ``dimension`` is omitted.  A degree sequence determines the dimension
    when ``dimension`` is omitted.  Nodes are ordered as if produced by
    ``numpy.meshgrid(..., indexing="ij")`` and flattened in C order.

    Parameters
    ----------
    max_nodes
        Optional setup-time allocation guard.  The default rejects rules with
        more than 100,000 nodes; pass ``None`` to disable the guard.
    """

    normalized_dimension = _normalize_dimension(dimension)
    normalized_degrees, normalized_dimension = _normalize_degrees(
        degrees, normalized_dimension
    )
    means = _normalize_distribution_parameter(mu, normalized_dimension, "mu")
    standard_deviations = _normalize_distribution_parameter(
        sigma, normalized_dimension, "sigma"
    )
    if any(value <= 0.0 for value in standard_deviations):
        raise ValueError("sigma values must be strictly positive")
    normalized_max_nodes = _normalize_max_nodes(max_nodes)

    if normalized_dimension == 0:
        return deterministic_quadrature()

    n_nodes = math.prod(normalized_degrees)
    if normalized_max_nodes is not None and n_nodes > normalized_max_nodes:
        raise ValueError(
            f"tensor rule requires {n_nodes} nodes, exceeding max_nodes="
            f"{normalized_max_nodes}"
        )

    one_dimensional_rules = [
        gauss_hermite_normal(degree, mu=mean, sigma=standard_deviation)
        for degree, mean, standard_deviation in zip(
            normalized_degrees, means, standard_deviations
        )
    ]
    node_meshes = np.meshgrid(
        *(rule.nodes[:, 0] for rule in one_dimensional_rules), indexing="ij"
    )
    weight_meshes = np.meshgrid(
        *(rule.weights for rule in one_dimensional_rules), indexing="ij"
    )
    nodes = np.stack([mesh.reshape(-1) for mesh in node_meshes], axis=1)
    weights = np.prod(np.stack(weight_meshes, axis=0), axis=0).reshape(-1)

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        kind="tensor",
        orders=normalized_degrees,
    )


def smolyak_gauss_hermite(
    dimension: int,
    level: int,
    *,
    mu: float | Sequence[float] = 0.0,
    sigma: float | Sequence[float] = 1.0,
    merge_tolerance: float = 1e-14,
    weight_tolerance: float = 1e-15,
    max_nodes: int | None = 1_000_000,
) -> QuadratureRule:
    """Construct a non-nested Smolyak rule for independent normal variables.

    One-dimensional level ``i >= 1`` uses a normalized degree-``i``
    Gauss-Hermite rule.  For user level ``L`` and dimension ``d``, this
    combines positive level vectors with sums from ``max(d, L + 1)`` through
    ``L + d``.  A vector with sum ``s`` has coefficient
    ``(-1)**(L + d - s) * comb(d - 1, L + d - s)``.

    This zero-based level convention maps to the legacy ``nwspgr`` convention
    as ``K = L + 1``; in particular, the old default ``K=2`` is ``level=1``.
    Level zero is the single node at the vector of marginal means.  A monomial
    with coordinate powers ``p_j`` is exact when
    ``sum(ceil((p_j + 1) / 2)) <= L + d``.

    Smolyak combination weights can be negative and are retained.  The
    ``max_nodes`` guard applies to the raw candidate count before coincident
    nodes are merged, because that count determines peak setup allocation.
    """

    normalized_dimension = _normalize_sparse_dimension(dimension)
    normalized_level = _normalize_level(level)
    means = _normalize_distribution_parameter(mu, normalized_dimension, "mu")
    standard_deviations = _normalize_distribution_parameter(
        sigma, normalized_dimension, "sigma"
    )
    if any(value <= 0.0 for value in standard_deviations):
        raise ValueError("sigma values must be strictly positive")
    normalized_merge_tolerance = _finite_nonnegative_scalar(
        merge_tolerance, "merge_tolerance"
    )
    normalized_weight_tolerance = _finite_nonnegative_scalar(
        weight_tolerance, "weight_tolerance"
    )
    normalized_max_nodes = _normalize_max_nodes(max_nodes)

    q = normalized_level + normalized_dimension
    level_vectors = [
        level_vector
        for level_sum in range(max(normalized_dimension, normalized_level + 1), q + 1)
        for level_vector in _positive_compositions(level_sum, normalized_dimension)
    ]
    raw_node_count = sum(math.prod(level_vector) for level_vector in level_vectors)
    if normalized_max_nodes is not None and raw_node_count > normalized_max_nodes:
        raise ValueError(
            f"Smolyak rule requires {raw_node_count} candidate nodes, exceeding "
            f"max_nodes={normalized_max_nodes}"
        )

    component_nodes: list[np.ndarray] = []
    component_weights: list[np.ndarray] = []
    for level_vector in level_vectors:
        level_sum = sum(level_vector)
        coefficient = (-1) ** (q - level_sum) * math.comb(
            normalized_dimension - 1, q - level_sum
        )
        component = tensor_gauss_hermite(
            level_vector,
            mu=means,
            sigma=standard_deviations,
            max_nodes=None,
        )
        component_nodes.append(component.nodes)
        component_weights.append(coefficient * component.weights)

    nodes = np.concatenate(component_nodes, axis=0)
    weights = np.concatenate(component_weights)
    nodes, weights = _merge_nodes(nodes, weights, tolerance=normalized_merge_tolerance)
    retained = np.abs(weights) > normalized_weight_tolerance
    nodes = nodes[retained]
    weights = weights[retained]
    if weights.size == 0:
        raise ValueError("weight_tolerance removed every Smolyak node")

    weight_sum = float(weights.sum())
    if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=_WEIGHT_SUM_TOLERANCE):
        raise ValueError(
            "Smolyak weights do not sum to one after duplicate merging and "
            "weight filtering"
        )
    weights = weights / weight_sum

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        kind="smolyak",
        orders=None,
        level=normalized_level,
    )


def _positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield positive integer compositions in lexicographic order."""

    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for remainder in _positive_compositions(total - first, parts - 1):
            yield (first, *remainder)


def _merge_nodes(
    nodes: np.ndarray, weights: np.ndarray, *, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lexicographically sort and merge coordinatewise-close nodes."""

    sort_keys = tuple(
        nodes[:, coordinate] for coordinate in range(nodes.shape[1] - 1, -1, -1)
    )
    ordering = np.lexsort(sort_keys)
    sorted_nodes = nodes[ordering]
    sorted_weights = weights[ordering]

    merged_nodes: list[np.ndarray] = []
    merged_weights: list[float] = []
    first_active = 0
    for node, weight in zip(sorted_nodes, sorted_weights):
        while (
            first_active < len(merged_nodes)
            and node[0] - merged_nodes[first_active][0] > tolerance
        ):
            first_active += 1

        match = None
        for candidate in range(first_active, len(merged_nodes)):
            if np.all(np.abs(node - merged_nodes[candidate]) <= tolerance):
                match = candidate
                break
        if match is None:
            merged_nodes.append(node.copy())
            merged_weights.append(float(weight))
        else:
            merged_weights[match] += float(weight)

    return np.stack(merged_nodes), np.asarray(merged_weights)


def _normalize_sparse_dimension(dimension: int) -> int:
    if (
        not isinstance(dimension, Integral)
        or isinstance(dimension, bool)
        or dimension < 1
    ):
        raise ValueError("dimension must be a positive integer")
    return int(dimension)


def _normalize_level(level: int) -> int:
    if not isinstance(level, Integral) or isinstance(level, bool) or level < 0:
        raise ValueError("level must be a nonnegative integer")
    return int(level)


def _normalize_dimension(dimension: int | None) -> int | None:
    if dimension is None:
        return None
    if (
        not isinstance(dimension, Integral)
        or isinstance(dimension, bool)
        or dimension < 0
    ):
        raise ValueError("dimension must be a nonnegative integer")
    return int(dimension)


def _normalize_degrees(
    degrees: int | Sequence[int], dimension: int | None
) -> tuple[tuple[int, ...], int]:
    if isinstance(degrees, Integral) and not isinstance(degrees, bool):
        if degrees < 1:
            raise ValueError("degrees must contain positive integers")
        resolved_dimension = 1 if dimension is None else dimension
        return (int(degrees),) * resolved_dimension, resolved_dimension
    if isinstance(degrees, (str, bytes)):
        raise ValueError("degrees must be an integer or a sequence of integers")
    try:
        degree_values = tuple(degrees)
    except TypeError as error:
        raise ValueError(
            "degrees must be an integer or a sequence of integers"
        ) from error
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or value < 1
        for value in degree_values
    ):
        raise ValueError("degrees must contain positive integers")
    resolved_dimension = len(degree_values) if dimension is None else dimension
    if len(degree_values) != resolved_dimension:
        raise ValueError("degrees must have one entry per dimension")
    return tuple(int(value) for value in degree_values), resolved_dimension


def _normalize_distribution_parameter(
    values: float | Sequence[float], dimension: int, name: str
) -> tuple[float, ...]:
    if isinstance(values, Real) and not isinstance(values, bool):
        value = _finite_scalar(values, name)
        return (value,) * dimension
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a scalar or a sequence of scalars")
    try:
        parameter_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a scalar or a sequence of scalars") from error
    if len(parameter_values) != dimension:
        raise ValueError(f"{name} must have one entry per dimension")
    return tuple(_finite_scalar(value, name) for value in parameter_values)


def _normalize_max_nodes(max_nodes: int | None) -> int | None:
    if max_nodes is None:
        return None
    if (
        not isinstance(max_nodes, Integral)
        or isinstance(max_nodes, bool)
        or max_nodes < 1
    ):
        raise ValueError("max_nodes must be a positive integer or None")
    return int(max_nodes)


def _finite_nonnegative_scalar(value: object, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _finite_scalar(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _immutable_float_array(value: object, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
