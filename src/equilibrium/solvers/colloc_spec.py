"""Configuration and setup-time domain helpers for collocation solves."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov

from .quadrature import ExogenousProcess, exogenous_process_from_model

if TYPE_CHECKING:
    from ..approx import Function
    from ..model import Model


ApproximationKind = Literal["smolyak_chebyshev", "tensor_chebyshev"]
QuadratureKind = Literal["tensor", "smolyak"]
CollocationAlgorithm = Literal["time_iteration", "newton", "hybrid"]
InitializationKind = Literal["linear", "steady"]
ExtrapolationKind = Literal["allow", "error"]
DomainSource = Literal["explicit", "approximation", "automatic"]

_APPROXIMATION_KINDS = frozenset({"smolyak_chebyshev", "tensor_chebyshev"})
_QUADRATURE_KINDS = frozenset({"tensor", "smolyak"})
_ALGORITHMS = frozenset({"time_iteration", "newton", "hybrid"})
_INITIALIZATIONS = frozenset({"linear", "steady"})
_EXTRAPOLATION_KINDS = frozenset({"allow", "error"})
_STABILITY_MARGIN = 1e-10
_COVARIANCE_TOLERANCE = 1e-12


def _immutable_float_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _validate_choice(value: str, choices: frozenset[str], name: str) -> None:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(repr(choice) for choice in choices))
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")


def _validate_positive_real(value: Real, name: str) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _validate_integer(value: Integral, name: str, *, minimum: int) -> None:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < minimum
    ):
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {qualifier}")


def _validate_scalar_or_tuple(
    value: int | tuple[int, ...], name: str, *, minimum: int
) -> None:
    values = value if isinstance(value, tuple) else (value,)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for entry in values:
        _validate_integer(entry, f"{name} entries", minimum=minimum)


def _normalize_dimension_integers(
    value: int | tuple[int, ...],
    dimension: int,
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    """Broadcast an integer or validate an exact per-dimension tuple."""

    _validate_scalar_or_tuple(value, name, minimum=minimum)
    if isinstance(value, tuple):
        if len(value) != dimension:
            raise ValueError(f"{name} must have length {dimension}, got {len(value)}")
        return tuple(int(entry) for entry in value)
    return tuple(int(value) for _ in range(dimension))


@dataclass(frozen=True)
class CollocationSpec:
    """Validated configuration for a nonlinear collocation solve."""

    approximation: ApproximationKind = "smolyak_chebyshev"
    max_levels: int | tuple[int, ...] = 3
    max_total_level: int = 3
    tensor_points: int | tuple[int, ...] = 5

    domain: tuple[ArrayLike, ArrayLike] | None = None
    domain_stddevs: float = 3.0
    domain_min_half_width: float = 1e-4

    quadrature_kind: QuadratureKind = "tensor"
    quadrature_degree: int | tuple[int, ...] = 5
    quadrature_level: int = 2
    quadrature_max_nodes: int | None = 100_000

    algorithm: CollocationAlgorithm = "hybrid"
    initialization: InitializationKind = "linear"
    tolerance: float = 1e-8
    inner_tolerance: float = 1e-10
    max_time_iterations: int = 500
    max_inner_iterations: int = 30
    max_newton_iterations: int = 50
    max_backtracks: int = 12
    damping: float = 1.0
    hybrid_switch_tolerance: float = 1e-4
    hybrid_max_time_iterations: int = 50
    max_newton_unknowns: int | None = 2_000

    extrapolation: ExtrapolationKind = "allow"
    verbose: bool = False

    def __post_init__(self) -> None:
        _validate_choice(self.approximation, _APPROXIMATION_KINDS, "approximation")
        _validate_choice(self.quadrature_kind, _QUADRATURE_KINDS, "quadrature_kind")
        _validate_choice(self.algorithm, _ALGORITHMS, "algorithm")
        _validate_choice(self.initialization, _INITIALIZATIONS, "initialization")
        _validate_choice(self.extrapolation, _EXTRAPOLATION_KINDS, "extrapolation")

        _validate_scalar_or_tuple(self.max_levels, "max_levels", minimum=0)
        _validate_integer(self.max_total_level, "max_total_level", minimum=0)
        _validate_scalar_or_tuple(self.tensor_points, "tensor_points", minimum=1)
        _validate_scalar_or_tuple(
            self.quadrature_degree, "quadrature_degree", minimum=1
        )
        _validate_integer(self.quadrature_level, "quadrature_level", minimum=0)

        _validate_positive_real(self.domain_stddevs, "domain_stddevs")
        _validate_positive_real(self.domain_min_half_width, "domain_min_half_width")
        _validate_positive_real(self.tolerance, "tolerance")
        _validate_positive_real(self.inner_tolerance, "inner_tolerance")
        _validate_positive_real(self.hybrid_switch_tolerance, "hybrid_switch_tolerance")

        if self.hybrid_switch_tolerance < self.tolerance:
            raise ValueError(
                "hybrid_switch_tolerance must be greater than or equal to tolerance"
            )

        _validate_integer(self.max_time_iterations, "max_time_iterations", minimum=1)
        _validate_integer(self.max_inner_iterations, "max_inner_iterations", minimum=1)
        _validate_integer(
            self.max_newton_iterations, "max_newton_iterations", minimum=1
        )
        _validate_integer(self.max_backtracks, "max_backtracks", minimum=0)
        _validate_integer(
            self.hybrid_max_time_iterations,
            "hybrid_max_time_iterations",
            minimum=1,
        )

        if (
            not isinstance(self.damping, Real)
            or isinstance(self.damping, bool)
            or not np.isfinite(self.damping)
            or not 0.0 < self.damping <= 1.0
        ):
            raise ValueError("damping must be finite and satisfy 0 < damping <= 1")

        if self.quadrature_max_nodes is not None:
            _validate_integer(
                self.quadrature_max_nodes, "quadrature_max_nodes", minimum=1
            )
        if self.max_newton_unknowns is not None:
            _validate_integer(
                self.max_newton_unknowns, "max_newton_unknowns", minimum=1
            )
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean")

        if self.domain is not None:
            if not isinstance(self.domain, tuple) or len(self.domain) != 2:
                raise ValueError("domain must be a (lower, upper) tuple")
            lower = _immutable_float_array(self.domain[0], "domain lower bounds")
            upper = _immutable_float_array(self.domain[1], "domain upper bounds")
            if lower.ndim != 1 or upper.ndim != 1:
                raise ValueError(
                    "domain lower and upper bounds must be one-dimensional"
                )
            if lower.shape != upper.shape:
                raise ValueError(
                    "domain lower and upper bounds must have matching shapes"
                )
            if not np.all(lower < upper):
                raise ValueError("domain lower bounds must be less than upper bounds")
            object.__setattr__(self, "domain", (lower, upper))


@dataclass(frozen=True)
class _CollocationDimensions:
    n_controls: int
    n_x: int
    n_z: int
    control_names: tuple[str, ...]
    x_names: tuple[str, ...]
    z_names: tuple[str, ...]

    @property
    def n_states(self) -> int:
        return self.n_x + self.n_z

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.x_names + self.z_names


@dataclass(frozen=True)
class _CollocationDomain:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    source: DomainSource
    covariance: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        lower = _immutable_float_array(self.lower, "domain lower bounds")
        upper = _immutable_float_array(self.upper, "domain upper bounds")
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("resolved domain bounds must be matching vectors")
        if not np.all(lower < upper):
            raise ValueError(
                "resolved domain lower bounds must be less than upper bounds"
            )

        covariance = self.covariance
        if covariance is not None:
            covariance = _immutable_float_array(covariance, "domain covariance")
            if covariance.shape != (lower.size, lower.size):
                raise ValueError(
                    "domain covariance must have shape (n_states, n_states)"
                )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class _CollocationPrerequisites:
    dimensions: _CollocationDimensions
    process: ExogenousProcess
    domain: _CollocationDomain
    linearized: bool


def _validate_collocation_model(model: Model) -> _CollocationDimensions:
    """Validate model lifecycle and return ordered collocation dimensions."""

    if (
        getattr(model, "inner_functions", None) is None
        or getattr(model, "var_lists", None) is None
        or getattr(model, "linear_mod", None) is None
        or not hasattr(model, "N")
    ):
        raise RuntimeError("model must be finalized before collocation setup")
    if not getattr(model, "_steady_solved", False):
        raise RuntimeError("model steady state must be solved before collocation setup")

    try:
        control_names = tuple(model.var_lists["u"])
        x_names = tuple(model.var_lists["x"])
        z_names = tuple(model.var_lists["z"])
        n_controls = int(model.N["u"])
        n_x = int(model.N["x"])
        n_z = int(model.N["z"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("model has invalid collocation variable metadata") from error

    metadata = (
        ("controls", control_names, n_controls),
        ("endogenous states", x_names, n_x),
        ("exogenous states", z_names, n_z),
    )
    for label, names, size in metadata:
        if size < 0 or len(names) != size or len(set(names)) != len(names):
            raise RuntimeError(f"model has inconsistent {label} metadata")

    try:
        exogenous_names = tuple(model.exog_list)
    except (AttributeError, TypeError) as error:
        raise RuntimeError("model must provide an iterable exog_list") from error
    if z_names != exogenous_names:
        raise RuntimeError("model exogenous variable order must match model.exog_list")
    if n_controls == 0:
        raise ValueError("collocation requires at least one control variable")
    if n_x + n_z == 0:
        raise ValueError(
            "collocation requires at least one dynamic state; use solve_steady() "
            "for a static model"
        )

    steady_components = getattr(model, "steady_components", None)
    if not isinstance(steady_components, dict):
        raise RuntimeError("model steady-state components are unavailable")
    for name, size in (("u", n_controls), ("x", n_x), ("z", n_z), ("params", None)):
        if name not in steady_components:
            raise RuntimeError(f"model steady-state component {name!r} is unavailable")
        array = np.asarray(steady_components[name])
        if array.ndim != 1 or (size is not None and array.shape != (size,)):
            raise RuntimeError(
                f"model steady-state component {name!r} has an invalid shape"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"model steady-state component {name!r} must be finite")

    return _CollocationDimensions(
        n_controls=n_controls,
        n_x=n_x,
        n_z=n_z,
        control_names=control_names,
        x_names=x_names,
        z_names=z_names,
    )


def _resolve_exogenous_process(
    model: Model,
    dimensions: _CollocationDimensions,
    process: ExogenousProcess | None,
) -> ExogenousProcess:
    """Resolve process data and enforce the model's exogenous ordering."""

    resolved = exogenous_process_from_model(model) if process is None else process
    if not isinstance(resolved, ExogenousProcess):
        raise TypeError("process must be an ExogenousProcess")
    if resolved.names != dimensions.z_names:
        raise ValueError("process names must match model.exog_list in the same order")
    return resolved


def _validate_domain_dimension(
    lower: ArrayLike,
    upper: ArrayLike,
    dimension: int,
    source: DomainSource,
) -> _CollocationDomain:
    domain = _CollocationDomain(lower=lower, upper=upper, source=source)
    if domain.lower.shape != (dimension,):
        raise ValueError(
            f"domain bounds must have shape ({dimension},), got "
            f"{domain.lower.shape}"
        )
    return domain


def _linearize_for_collocation(model: Model, process: ExogenousProcess) -> None:
    """Linearize with the exact process convention used by collocation."""

    model.linearize(
        Phi=np.asarray(process.persistence),
        impact_matrix=np.asarray(process.innovation_impact),
    )


def _automatic_collocation_domain(
    model: Model,
    spec: CollocationSpec,
    process: ExogenousProcess,
    dimensions: _CollocationDimensions,
) -> _CollocationDomain:
    """Construct bounds from the stationary joint linear state covariance."""

    linear_model = model.linear_mod
    h_x = np.asarray(linear_model.H_x, dtype=np.float64)
    h_z = np.asarray(linear_model.H_z, dtype=np.float64)
    expected_h_x_shape = (dimensions.n_x, dimensions.n_x)
    expected_h_z_shape = (dimensions.n_x, dimensions.n_z)
    if h_x.shape != expected_h_x_shape or h_z.shape != expected_h_z_shape:
        raise RuntimeError(
            "linearized state policy matrices have incompatible shapes: "
            f"H_x={h_x.shape}, H_z={h_z.shape}"
        )

    phi = np.asarray(process.persistence, dtype=np.float64)
    impact = np.asarray(process.innovation_impact, dtype=np.float64)
    state_matrix = np.block(
        [
            [h_x, h_z],
            [np.zeros((dimensions.n_z, dimensions.n_x)), phi],
        ]
    )
    shock_matrix = np.vstack((np.zeros((dimensions.n_x, impact.shape[1])), impact))

    if not np.all(np.isfinite(state_matrix)):
        raise ValueError(
            "automatic collocation domain requires finite linear state dynamics"
        )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(state_matrix))))
    if spectral_radius >= 1.0 - _STABILITY_MARGIN:
        raise ValueError(
            "automatic collocation domain requires stable linear state dynamics; "
            f"spectral radius is {spectral_radius:.6g}. Pass an explicit domain."
        )

    innovation_covariance = shock_matrix @ shock_matrix.T
    try:
        covariance = solve_discrete_lyapunov(state_matrix, innovation_covariance)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "could not compute a stationary covariance; pass an explicit domain"
        ) from error
    covariance = np.asarray(covariance, dtype=np.float64)
    expected_covariance_shape = (dimensions.n_states, dimensions.n_states)
    if covariance.shape != expected_covariance_shape:
        raise ValueError(
            "stationary covariance has an invalid shape; pass an explicit domain"
        )
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("stationary covariance is nonfinite; pass an explicit domain")

    variances = np.diag(covariance).copy()
    variance_scale = max(1.0, float(np.max(np.abs(variances))))
    tolerance = _COVARIANCE_TOLERANCE * variance_scale
    if np.any(variances < -tolerance):
        raise ValueError(
            "stationary covariance has negative variances; pass an explicit domain"
        )
    variances = np.maximum(variances, 0.0)
    covariance = covariance.copy()
    np.fill_diagonal(covariance, variances)

    center = np.concatenate(
        (
            np.asarray(model.steady_components["x"], dtype=np.float64),
            np.asarray(model.steady_components["z"], dtype=np.float64),
        )
    )
    stochastic_width = spec.domain_stddevs * np.sqrt(variances)
    minimum_width = spec.domain_min_half_width * np.maximum(1.0, np.abs(center))
    half_width = np.maximum(stochastic_width, minimum_width)

    return _CollocationDomain(
        lower=center - half_width,
        upper=center + half_width,
        source="automatic",
        covariance=covariance,
    )


def _resolve_collocation_prerequisites(
    model: Model,
    spec: CollocationSpec,
    *,
    process: ExogenousProcess | None = None,
    approximation: Function | None = None,
    initial_coefficients_provided: bool = False,
) -> _CollocationPrerequisites:
    """Resolve lifecycle, dimensions, process, linearization, and domain."""

    if not isinstance(spec, CollocationSpec):
        raise TypeError("spec must be a CollocationSpec")
    dimensions = _validate_collocation_model(model)
    resolved_process = _resolve_exogenous_process(model, dimensions, process)

    if spec.domain is not None:
        lower, upper = spec.domain
        domain = _validate_domain_dimension(
            lower, upper, dimensions.n_states, "explicit"
        )
        automatic_domain = False
    elif approximation is not None:
        try:
            lower = approximation.lb
            upper = approximation.ub
        except AttributeError as error:
            raise TypeError(
                "approximation must be an equilibrium.approx.Function"
            ) from error
        domain = _validate_domain_dimension(
            lower, upper, dimensions.n_states, "approximation"
        )
        automatic_domain = False
    else:
        domain = None
        automatic_domain = True

    needs_linear_initialization = (
        spec.initialization == "linear" and not initial_coefficients_provided
    )
    needs_linearization = automatic_domain or needs_linear_initialization
    if needs_linearization:
        _linearize_for_collocation(model, resolved_process)
    if automatic_domain:
        domain = _automatic_collocation_domain(
            model, spec, resolved_process, dimensions
        )

    if domain is None:  # pragma: no cover - guarded by the branches above
        raise RuntimeError("collocation domain resolution failed")
    return _CollocationPrerequisites(
        dimensions=dimensions,
        process=resolved_process,
        domain=domain,
        linearized=needs_linearization,
    )


__all__ = ["CollocationSpec"]
