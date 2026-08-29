"""Preset helpers for common Function configurations."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import (
    ChebyshevBasis1d,
    ChebyshevLobattoGrid1d,
    Function,
    HierarchicalHatBasis1d,
    ModifiedHierarchicalHatBasis1d,
    ModifiedUniformHatBasis1d,
    UniformGridWithBoundary1d,
)
from .bases import UniformHatBasis1d
from .core import Scheme
from .grids import UniformGrid1d
from .levels import SmolyakInteriorLevels, SmolyakLevels, TensorProductLevels


def _normalize_levels(value: Sequence[int] | int, dimension: int) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != dimension:
            raise ValueError(
                f"max_levels must have length {dimension}, got {len(value)}"
            )
        levels = tuple(int(v) for v in value)
    else:
        levels = tuple(int(value) for _ in range(dimension))
    if any(v < 0 for v in levels):
        raise ValueError("max_levels entries must be non-negative")
    return levels


def _build_function(
    *,
    dimension: int,
    lb: ArrayLike,
    ub: ArrayLike,
    scheme: Scheme,
    auto_construct: bool,
) -> Function:
    lb_arr = np.asarray(lb, dtype=np.float64)
    ub_arr = np.asarray(ub, dtype=np.float64)
    if lb_arr.shape != (dimension,) or ub_arr.shape != (dimension,):
        raise ValueError(f"lb/ub must have shape ({dimension},)")
    return Function(scheme, lb_arr, ub_arr, auto_construct=auto_construct)


def make_smolyak_chebyshev(
    *,
    dimension: int,
    max_levels: Sequence[int] | int,
    max_total_level: int,
    lb: ArrayLike,
    ub: ArrayLike,
    auto_construct: bool = True,
    level: int | None = None,
) -> Function:
    """Smolyak scheme with Chebyshev grids/basis."""
    levels_tuple = _normalize_levels(max_levels, dimension)
    max_level_needed = max(levels_tuple)
    if level is not None and level < max_level_needed:
        raise ValueError("level must be >= max(max_levels)")
    level_value = max_level_needed if level is None else int(level)
    levels_obj = SmolyakLevels(level=level_value)
    scheme = Scheme(
        grid=ChebyshevLobattoGrid1d(),
        basis=ChebyshevBasis1d(),
        levels=levels_obj,
        max_total_level=max_total_level,
        max_levels=levels_tuple,
        dimension=dimension,
        auto_construct=auto_construct,
    )
    return _build_function(
        dimension=dimension, lb=lb, ub=ub, scheme=scheme, auto_construct=auto_construct
    )


def make_smolyak_hierarchical_hat(
    *,
    dimension: int,
    max_levels: Sequence[int] | int,
    max_total_level: int,
    lb: ArrayLike,
    ub: ArrayLike,
    auto_construct: bool = True,
    level: int | None = None,
) -> Function:
    levels_tuple = _normalize_levels(max_levels, dimension)
    max_level_needed = max(levels_tuple)
    if level is not None and level < max_level_needed:
        raise ValueError("level must be >= max(max_levels)")
    level_value = max_level_needed if level is None else int(level)
    levels_obj = SmolyakLevels(level=level_value)
    scheme = Scheme(
        grid=UniformGridWithBoundary1d(),
        basis=HierarchicalHatBasis1d(),
        levels=levels_obj,
        max_total_level=max_total_level,
        max_levels=levels_tuple,
        dimension=dimension,
        auto_construct=auto_construct,
    )
    return _build_function(
        dimension=dimension, lb=lb, ub=ub, scheme=scheme, auto_construct=auto_construct
    )


def make_smolyak_hat(**kwargs) -> Function:
    """Alias for make_smolyak_hierarchical_hat."""
    return make_smolyak_hierarchical_hat(**kwargs)


def make_smolyak_modified_hat(
    *,
    dimension: int,
    max_levels: Sequence[int] | int,
    max_total_level: int,
    lb: ArrayLike,
    ub: ArrayLike,
    auto_construct: bool = True,
    level: int | None = None,
) -> Function:
    levels_tuple = _normalize_levels(max_levels, dimension)
    max_level_needed = max(levels_tuple)
    if level is not None and level < max_level_needed:
        raise ValueError("level must be >= max(max_levels)")
    level_value = max_level_needed if level is None else int(level)
    levels_obj = SmolyakInteriorLevels(level=level_value)
    scheme = Scheme(
        grid=UniformGrid1d(),
        basis=ModifiedHierarchicalHatBasis1d(),
        levels=levels_obj,
        max_total_level=max_total_level,
        max_levels=levels_tuple,
        dimension=dimension,
        auto_construct=auto_construct,
    )
    return _build_function(
        dimension=dimension, lb=lb, ub=ub, scheme=scheme, auto_construct=auto_construct
    )


def make_tensor_chebyshev(
    *,
    dimension: int,
    n_points: Sequence[int] | int,
    lb: ArrayLike,
    ub: ArrayLike,
    auto_construct: bool = True,
) -> Function:
    raw_points = _normalize_levels(n_points, dimension)
    if any(p < 1 for p in raw_points):
        raise ValueError("n_points must be >= 1")
    levels_tuple = tuple(p - 1 for p in raw_points)
    level_value = max(raw_points)
    levels_obj = TensorProductLevels(level_value)
    scheme = Scheme(
        grid=ChebyshevLobattoGrid1d(),
        basis=ChebyshevBasis1d(),
        levels=levels_obj,
        max_total_level=sum(levels_tuple),
        max_levels=levels_tuple,
        dimension=dimension,
        auto_construct=auto_construct,
    )
    return _build_function(
        dimension=dimension, lb=lb, ub=ub, scheme=scheme, auto_construct=auto_construct
    )


def make_tensor_uniform_hat(
    *,
    dimension: int,
    n_points: Sequence[int] | int,
    lb: ArrayLike,
    ub: ArrayLike,
    auto_construct: bool = True,
) -> Function:
    raw_points = _normalize_levels(n_points, dimension)
    if any(p < 1 for p in raw_points):
        raise ValueError("n_points must be >= 1")
    levels_tuple = tuple(p - 1 for p in raw_points)
    level_value = max(raw_points)
    levels_obj = TensorProductLevels(level_value)
    scheme = Scheme(
        grid=UniformGridWithBoundary1d(),
        basis=UniformHatBasis1d(),
        levels=levels_obj,
        max_total_level=sum(levels_tuple),
        max_levels=levels_tuple,
        dimension=dimension,
        auto_construct=auto_construct,
    )
    return _build_function(
        dimension=dimension, lb=lb, ub=ub, scheme=scheme, auto_construct=auto_construct
    )


def make_tensor_hat(**kwargs) -> Function:
    """Alias for make_tensor_uniform_hat."""
    return make_tensor_uniform_hat(**kwargs)


def make_tensor_modified_hat(
    *,
    dimension: int,
    n_points: Sequence[int] | int,
    lb: ArrayLike,
    ub: ArrayLike,
    auto_construct: bool = True,
) -> Function:
    raw_points = _normalize_levels(n_points, dimension)
    if any(p < 1 for p in raw_points):
        raise ValueError("n_points must be >= 1")
    levels_tuple = tuple(p - 1 for p in raw_points)
    level_value = max(raw_points)
    levels_obj = TensorProductLevels(level_value)
    scheme = Scheme(
        grid=UniformGrid1d(),
        basis=ModifiedUniformHatBasis1d(),
        levels=levels_obj,
        max_total_level=sum(levels_tuple),
        max_levels=levels_tuple,
        dimension=dimension,
        auto_construct=auto_construct,
    )
    return _build_function(
        dimension=dimension, lb=lb, ub=ub, scheme=scheme, auto_construct=auto_construct
    )


FUNCAPPROX_BUILDERS: dict[str, Callable[..., Function]] = {
    "smolyak_chebyshev": make_smolyak_chebyshev,
    "smolyak_hierarchical_hat": make_smolyak_hierarchical_hat,
    "smolyak_hat": make_smolyak_hat,
    "smolyak_modified_hat": make_smolyak_modified_hat,
    "tensor_chebyshev": make_tensor_chebyshev,
    "tensor_uniform_hat": make_tensor_uniform_hat,
    "tensor_hat": make_tensor_hat,
    "tensor_modified_hat": make_tensor_modified_hat,
}
VALID_FUNCAPPROX_NAMES = tuple(FUNCAPPROX_BUILDERS.keys())


def normalize_funcapprox_name(name: str) -> str:
    """Normalize and validate a preset name."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = name.strip().lower()
    if normalized not in VALID_FUNCAPPROX_NAMES:
        valid = ", ".join(VALID_FUNCAPPROX_NAMES)
        raise ValueError(f"unknown funcapprox preset '{name}'. Valid options: {valid}")
    return normalized


def make_funcapprox(name: str, /, *args, **kwargs) -> Function:
    """Dispatch to a preset helper by name."""
    normalized = normalize_funcapprox_name(name)
    builder = FUNCAPPROX_BUILDERS[normalized]
    return builder(*args, **kwargs)


def _get_smolyak_n_points(
    *, dimension: int, max_level: int, lb: np.ndarray, ub: np.ndarray
) -> int:
    func = make_smolyak_chebyshev(
        dimension=dimension,
        max_levels=max_level,
        max_total_level=max_level,
        lb=lb,
        ub=ub,
    )
    return func.get_n_points()


def _compute_tensor_n_points_1d(target_n_points: int, dimension: int) -> int:
    n_points_1d = int(np.ceil(target_n_points ** (1.0 / dimension)))
    return max(2, n_points_1d)


def create_approximation(
    name: str,
    *,
    dimension: int,
    max_level: int,
    lb: ArrayLike,
    ub: ArrayLike,
    smolyak_n_points: int | None = None,
) -> Function:
    """Create a Function approximation with consistent tensor sizing."""
    normalized = normalize_funcapprox_name(name)
    lb_arr = np.asarray(lb, dtype=np.float64)
    ub_arr = np.asarray(ub, dtype=np.float64)
    if lb_arr.shape != (dimension,) or ub_arr.shape != (dimension,):
        raise ValueError(f"lb/ub must have shape ({dimension},)")

    if normalized.startswith("tensor_"):
        if smolyak_n_points is None:
            smolyak_n_points = _get_smolyak_n_points(
                dimension=dimension,
                max_level=max_level,
                lb=lb_arr,
                ub=ub_arr,
            )
        n_points = _compute_tensor_n_points_1d(smolyak_n_points, dimension)
        return make_funcapprox(
            normalized,
            dimension=dimension,
            n_points=n_points,
            lb=lb_arr,
            ub=ub_arr,
        )
    return make_funcapprox(
        normalized,
        dimension=dimension,
        max_levels=max_level,
        max_total_level=max_level,
        lb=lb_arr,
        ub=ub_arr,
    )
