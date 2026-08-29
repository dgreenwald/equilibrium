"""Quadrature and exogenous-process data containers.

Rule construction is a NumPy setup-time operation.  The lightweight JAX
containers returned by :meth:`QuadratureRule.as_jax` and
:meth:`ExogenousProcess.as_jax` contain arrays only, so they can be passed
through JAX transformations without making static metadata part of a trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import NamedTuple

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
