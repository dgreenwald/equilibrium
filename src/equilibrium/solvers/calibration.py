#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified calibration interface for deterministic and linear path/IRF matching.

This module provides a unified API for calibrating model parameters and/or shocks
to specified target outcomes. It supports:
- Deterministic path matching
- Linear IRF matching
- Linear sequence matching
- Just-identified cases (root solving)
- Over-identified cases (minimization)
- Scalar and vector parameter special cases
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import scipy.optimize as opt

from .det_spec import DetSpec
from .linear_spec import LinearSpec
from .results import (
    DeterministicResult,
    IrfResult,
    PathResult,
    SequenceResult,
    SeriesTransform,
)

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_solver_output(enabled: bool):
    if not enabled:
        yield
        return
    loggers = [
        logging.getLogger("equilibrium.solvers.deterministic"),
        logging.getLogger("equilibrium.solvers.linear"),
        logging.getLogger("equilibrium.model.model"),
        logging.getLogger("equilibrium.solvers.newton"),
    ]
    previous_levels = [(lg, lg.level) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.WARNING)
        yield
    finally:
        for lg, level in previous_levels:
            lg.setLevel(level)


@dataclass
class PointTarget:
    """
    Specifies that a variable's value at a specific time should match a target.

    This is suitable for both deterministic path calibration and individual
    points along an IRF.

    Parameters
    ----------
    variable : str
        Name of the variable to target.
    time : int
        Time index at which to evaluate the variable.
    value : float
        Target value for the variable at the specified time.
    shock : str, optional
        For IRF matching, the name of the shock that generates the IRF.
        If None, applies to deterministic paths.
    weight : float, default 1.0
        Weight for this target in over-identified optimization problems.
        Higher weights give more importance to matching this target.

    Examples
    --------
    >>> # Match output at time 10 to value 1.05
    >>> target = PointTarget(variable="output", time=10, value=1.05)
    >>>
    >>> # Match consumption at time 5 in response to TFP shock with high weight
    >>> target = PointTarget(variable="c", time=5, value=0.98, shock="tfp", weight=2.0)
    """

    variable: str
    time: int
    value: float
    shock: Optional[str] = None
    weight: float = 1.0

    def __post_init__(self):
        """Validate target specification."""
        if self.time < 0:
            raise ValueError(f"time must be non-negative, got {self.time}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")


@dataclass
class FunctionalTarget:
    """
    Specifies an arbitrary loss function over a solution object.

    This is useful for complex criteria such as weighted sums of deviations,
    matching moments, or advanced features of the solved path/IRF.

    Parameters
    ----------
    func : callable
        Function that takes a solution object (DeterministicResult, IrfResult,
        or SequenceResult) and returns a vector (or scalar) of target errors.
        The function should return 0 for perfect match.
    description : str, optional
        Human-readable description of what this target represents.
    weights : np.ndarray or list, optional
        Weights for vector-valued functional targets in over-identified problems.
        If None, defaults to ones. Should match the length of the vector returned
        by func. For scalar functions, this is treated as a single weight.

    Examples
    --------
    >>> # Match the average of consumption over periods 0-10
    >>> def avg_consumption_error(result):
    ...     c_idx = result.var_names.index("c")
    ...     avg_c = np.mean(result.UX[:11, c_idx])
    ...     return avg_c - 0.95
    >>> target = FunctionalTarget(
    ...     func=avg_consumption_error,
    ...     description="Average consumption over first 10 periods = 0.95"
    ... )
    >>>
    >>> # Vector target with custom weights
    >>> def multi_moment_error(result):
    ...     return np.array([mean_error, std_error, skew_error])
    >>> target = FunctionalTarget(
    ...     func=multi_moment_error,
    ...     weights=[1.0, 2.0, 0.5],  # Weight std more heavily
    ...     description="Match multiple moments"
    ... )
    """

    func: Callable[
        [Union[PathResult, DeterministicResult, IrfResult, SequenceResult]],
        Union[float, np.ndarray],
    ]
    description: str = ""
    weights: Optional[Union[np.ndarray, list]] = None

    def __post_init__(self):
        """Validate and convert weights."""
        if self.weights is not None:
            self.weights = np.atleast_1d(self.weights)
            if np.any(self.weights <= 0):
                raise ValueError("All weights must be positive")


@dataclass
class ModelParam:
    """
    Declares a base-model parameter to calibrate.

    Changing a ``ModelParam`` triggers ``update_copy`` + ``solve_steady`` +
    ``linearize`` (cached: only re-solved when the value actually changes).

    Parameters
    ----------
    name : str
        Key in ``model.params``.
    initial : float
        Starting value for the optimiser.
    bounds : tuple of (float, float)
        (lower, upper) bounds.

    Examples
    --------
    >>> ModelParam("bet", initial=0.95, bounds=(0.90, 0.99))
    """

    name: str
    initial: float
    bounds: tuple[float, float]

    def __post_init__(self):
        if not self.name:
            raise ValueError("ModelParam name must be non-empty")
        lo, hi = self.bounds
        if lo > hi:
            raise ValueError(
                f"ModelParam bounds are inverted: lower ({lo}) > upper ({hi})"
            )
        if not (lo <= self.initial <= hi):
            raise ValueError(
                f"ModelParam initial value {self.initial} is outside "
                f"bounds ({lo}, {hi})"
            )


@dataclass
class ShockParam:
    """
    Declares a shock size to calibrate.

    Changing a ``ShockParam`` only modifies the spec (no model re-solve).

    Parameters
    ----------
    name : str
        Exogenous variable name (must be in ``model.exog_list``).
    initial : float
        Starting value for the optimiser.
    bounds : tuple of (float, float)
        (lower, upper) bounds.
    regime : int
        Regime index in ``DetSpec`` (or ignored for ``LinearSpec``).
    period : int
        Period at which the shock hits (maps to ``shock_per`` in ``DetSpec``).

    Examples
    --------
    >>> ShockParam("Z_til", initial=0.01, bounds=(0.0, 0.1), regime=0, period=0)
    """

    name: str
    initial: float
    bounds: tuple[float, float]
    regime: int = 0
    period: int = 0

    def __post_init__(self):
        if not self.name:
            raise ValueError("ShockParam name must be non-empty")
        if self.regime < 0:
            raise ValueError(
                f"ShockParam regime must be non-negative, got {self.regime}"
            )
        if self.period < 0:
            raise ValueError(
                f"ShockParam period must be non-negative, got {self.period}"
            )
        lo, hi = self.bounds
        if lo > hi:
            raise ValueError(
                f"ShockParam bounds are inverted: lower ({lo}) > upper ({hi})"
            )


@dataclass
class RegimeParam:
    """
    Declares a regime-specific parameter override to calibrate.

    Changing a ``RegimeParam`` only modifies the spec (no model re-solve).

    Parameters
    ----------
    name : str
        Parameter name to override in ``DetSpec.preset_par_list[regime]``.
    regime : int or list of int
        Regime index (or indices for persistent changes across regimes).
    initial : float
        Starting value for the optimiser.
    bounds : tuple of (float, float)
        (lower, upper) bounds.

    Examples
    --------
    >>> RegimeParam("tau", regime=1, initial=0.35, bounds=(0.2, 0.5))
    >>> RegimeParam("tau", regime=[1, 2, 3], initial=0.35, bounds=(0.2, 0.5))
    """

    name: str
    regime: int | list[int]
    initial: float
    bounds: tuple[float, float]

    def __post_init__(self):
        if not self.name:
            raise ValueError("RegimeParam name must be non-empty")
        # Normalise regime to list
        if isinstance(self.regime, int):
            self.regime = [self.regime]
        for r in self.regime:
            if r < 0:
                raise ValueError(f"RegimeParam regime must be non-negative, got {r}")
        lo, hi = self.bounds
        if lo > hi:
            raise ValueError(
                f"RegimeParam bounds are inverted: lower ({lo}) > upper ({hi})"
            )
        if not (lo <= self.initial <= hi):
            raise ValueError(
                f"RegimeParam initial value {self.initial} is outside "
                f"bounds ({lo}, {hi})"
            )


@dataclass
class CalibrationResult:
    """
    Container for calibration results.

    Attributes
    ----------
    parameters : dict
        Fitted parameter values (parameter name -> value).
    parameters_array : np.ndarray
        Fitted parameter values as array.
    success : bool
        Whether calibration succeeded.
    residual : float
        Final residual norm.
    iterations : int
        Number of iterations taken.
    message : str
        Solver message.
    solution : Union[DeterministicResult, IrfResult, SequenceResult]
        The solved path/IRF using the fitted parameters.
    model : object
        The final model instance with fitted parameters.
    method : str
        Calibration method used ('root_scalar', 'root', 'minimize', 'minimize_scalar').
    """

    parameters: Dict[str, float] = field(default_factory=dict)
    parameters_array: np.ndarray = field(default_factory=lambda: np.array([]))
    success: bool = False
    residual: float = np.inf
    iterations: int = 0
    message: str = ""
    solution: Optional[Union[DeterministicResult, IrfResult, SequenceResult]] = None
    model: Optional[Any] = None
    method: str = ""

    def save(self, label: str, save_dir: Optional[Union[Path, str]] = None) -> Any:
        """
        Save calibrated parameters to disk.

        Parameters
        ----------
        label : str
            Label for the calibration file.
        save_dir : Path | str | None, optional
            Target directory. Defaults to settings.paths.save_dir.

        Returns
        -------
        Path
            Path to the saved file.
        """
        from ..utils.io import save_calibrated_params

        return save_calibrated_params(self.parameters, label, save_dir=save_dir)


# ---------------------------------------------------------------------------
# Internal helpers for declarative calib_params
# ---------------------------------------------------------------------------


def _build_spec(
    template_spec: Union[DetSpec, LinearSpec],
    regime_params: list[RegimeParam],
    shock_params: list[ShockParam],
    regime_vals: np.ndarray,
    shock_vals: np.ndarray,
) -> Union[DetSpec, LinearSpec]:
    """
    Build a concrete spec from a template by applying current regime/shock values.

    For ``DetSpec``: deep-copies, sets regime param overrides, deduplicates
    shocks, then adds calibrated shocks.

    For ``LinearSpec``: creates a new ``LinearSpec`` with the shock size from
    the single ``ShockParam``.
    """
    if isinstance(template_spec, LinearSpec):
        # LinearSpec: at most one ShockParam, no RegimeParam (validated earlier)
        if len(shock_params) == 1:
            return LinearSpec(
                shock_name=shock_params[0].name,
                shock_size=float(shock_vals[0]),
                Nt=template_spec.Nt,
            )
        # No shock params → return template as-is
        return template_spec

    # DetSpec path
    spec = copy.deepcopy(template_spec)

    # Apply regime param overrides
    for rp, val in zip(regime_params, regime_vals):
        for r in rp.regime:
            spec.update_n_regimes(r + 1)
            spec.preset_par_list[r][rp.name] = float(val)

    # Apply shock params (with deduplication)
    for sp, val in zip(shock_params, shock_vals):
        spec.update_n_regimes(sp.regime + 1)
        # Remove any existing shock matching (name, period) in this regime
        spec.shocks[sp.regime] = [
            s
            for s in spec.shocks[sp.regime]
            if not (s[0] == sp.name and int(s[1]) == sp.period)
        ]
        spec.add_shock(sp.regime, sp.name, shock_per=sp.period, shock_val=float(val))

    return spec


def _build_param_to_model(
    model,
    calib_params: list[Union[ModelParam, ShockParam, RegimeParam]],
    template_spec: Union[DetSpec, LinearSpec],
    suppress_solver_output: bool,
) -> tuple[Callable, np.ndarray, list[tuple], list[str]]:
    """
    Build an efficient internal ``param_to_model`` callback from declarative
    ``calib_params``.

    Returns
    -------
    param_to_model : callable
        ``(params_array) -> (model, spec)``
    initial_params : np.ndarray
        Concatenated initial values.
    bounds : list of tuple
        Concatenated bounds.
    param_names : list of str
        Human-readable names (same order as array).
    """
    model_params: list[ModelParam] = []
    regime_params: list[RegimeParam] = []
    shock_params: list[ShockParam] = []

    for p in calib_params:
        if isinstance(p, ModelParam):
            model_params.append(p)
        elif isinstance(p, RegimeParam):
            regime_params.append(p)
        elif isinstance(p, ShockParam):
            shock_params.append(p)
        else:
            raise TypeError(f"Unknown calib_param type: {type(p)}")

    if len(calib_params) == 0:
        raise ValueError("calib_params must not be empty")

    # Validate names against model
    for mp in model_params:
        if mp.name not in model.params:
            raise ValueError(
                f"ModelParam name '{mp.name}' not found in model.params. "
                f"Available: {sorted(model.params.keys())}"
            )
    for sp in shock_params:
        if sp.name not in model.exog_list:
            raise ValueError(
                f"ShockParam name '{sp.name}' not found in model.exog_list. "
                f"Available: {model.exog_list}"
            )

    # LinearSpec-specific validation
    if isinstance(template_spec, LinearSpec):
        if len(shock_params) > 1:
            raise ValueError(
                "LinearSpec supports at most one ShockParam, "
                f"got {len(shock_params)}"
            )
        if len(regime_params) > 0:
            raise ValueError("LinearSpec does not support RegimeParam")

    # Layout: [model_params..., regime_params..., shock_params...]
    n_mp = len(model_params)
    n_rp = len(regime_params)

    initial = np.array(
        [p.initial for p in model_params]
        + [p.initial for p in regime_params]
        + [p.initial for p in shock_params]
    )
    bounds = (
        [p.bounds for p in model_params]
        + [p.bounds for p in regime_params]
        + [p.bounds for p in shock_params]
    )

    # Build human-readable names
    param_names: list[str] = []
    for mp in model_params:
        param_names.append(mp.name)
    for rp in regime_params:
        regime_str = (
            f"r{rp.regime[0]}"
            if len(rp.regime) == 1
            else "r" + "_".join(str(r) for r in rp.regime)
        )
        param_names.append(f"regime_{rp.name}_{regime_str}")
    for sp in shock_params:
        param_names.append(f"shock_{sp.name}_r{sp.regime}_t{sp.period}")

    # Caching closure
    _cache: dict[str, Any] = {
        "model": None,
        "model_param_vals": None,
    }

    # If no model params, solve the base model once
    if n_mp == 0:
        # Ensure base model is solved and linearised
        if not getattr(model, "_linearized", False):
            with _suppress_solver_output(suppress_solver_output):
                model.solve_steady(calibrate=False, display=False)
                model.linearize()
        _cache["model"] = model

    def param_to_model(params: np.ndarray) -> tuple:
        params = np.atleast_1d(params)
        mp_vals = params[:n_mp]
        rp_vals = params[n_mp : n_mp + n_rp]
        sp_vals = params[n_mp + n_rp :]

        # --- Model (re)solve ---
        if n_mp > 0:
            cached_vals = _cache["model_param_vals"]
            if cached_vals is None or not np.array_equal(mp_vals, cached_vals):
                param_dict = {mp.name: float(v) for mp, v in zip(model_params, mp_vals)}
                new_model = model.update_copy(params=param_dict)
                with _suppress_solver_output(suppress_solver_output):
                    new_model.solve_steady(calibrate=False, display=False)
                    new_model.linearize()
                _cache["model"] = new_model
                _cache["model_param_vals"] = mp_vals.copy()
            mod = _cache["model"]
        else:
            mod = _cache["model"]

        # --- Spec ---
        spec = _build_spec(template_spec, regime_params, shock_params, rp_vals, sp_vals)

        return mod, spec

    return param_to_model, initial, bounds, param_names


def calibrate(
    model,
    targets: List[Union[PointTarget, FunctionalTarget]],
    calib_params: list[Union[ModelParam, ShockParam, RegimeParam]],
    solver: str = "deterministic",
    spec: Union[DetSpec, LinearSpec] = None,
    bounds: Optional[List[tuple]] = None,
    method: Optional[str] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
    suppress_solver_output: bool = True,
    series_transforms: Optional[
        Mapping[str, SeriesTransform | Mapping[str, Any]]
    ] = None,
    default_transform: Optional[SeriesTransform | Mapping[str, Any]] = None,
    return_solution: bool = False,
    progress_every: int = 1,
    label: Optional[str] = None,
    save_dir: Optional[Union[Path, str]] = None,
    **solver_kwargs,
) -> CalibrationResult:
    """
    Unified calibration function for fitting parameters to target outcomes.

    Uses declarative parameter specifications (``ModelParam``, ``ShockParam``,
    ``RegimeParam``) to automatically build an efficient internal callback with
    model-level caching. Only ``ModelParam`` changes trigger model re-solving;
    ``ShockParam`` and ``RegimeParam`` changes only modify the spec.

    Parameters
    ----------
    model : Model
        Base model instance to calibrate.
    targets : list of PointTarget or FunctionalTarget
        Target specifications to match.
    calib_params : list of ModelParam, ShockParam, or RegimeParam
        Declarative parameter specifications. Array layout:
        ``[...model_params, ...regime_params, ...shock_params]``.
    solver : str, default "deterministic"
        Solver to use: "deterministic", "linear_irf", or "linear_sequence".
    spec : DetSpec or LinearSpec
        Template specification. For deterministic/linear_sequence: ``DetSpec``.
        For linear_irf: ``LinearSpec``. Shock values from ``ShockParam`` and
        regime overrides from ``RegimeParam`` are applied on top of this template.
    bounds : list of tuples, optional
        Parameter bounds. If None, uses bounds from ``calib_params``.
    method : str, optional
        Optimization method. If None, automatically selected based on problem
        structure.
    tol : float, default 1e-6
        Convergence tolerance.
    maxiter : int, default 100
        Maximum number of iterations.
    suppress_solver_output : bool, default True
        If True, suppress logging from steady-state, deterministic, and linear
        solvers during calibration evaluations.
    series_transforms : dict[str, SeriesTransform or dict], optional
        Per-series transform specifications keyed by series name. Applied to
        solution data before evaluating targets. The transform base_index
        refers to the time index in the solved path (t=0 is the initial
        condition).
    default_transform : SeriesTransform or dict, optional
        Transform to apply to any series without an explicit entry in
        series_transforms.
    return_solution : bool, default False
        If True, return the (transformed) solution and model in the result.
    progress_every : int, default 1
        Log calibration progress every N evaluations. Set to 0 to disable.
    label : str, optional
        Label for saving results automatically. If provided:
        - On success: saves parameters to disk.
        - On failure: raises RuntimeError.
    save_dir : Path | str, optional
        Directory to save results to when label is provided.
    **solver_kwargs
        Additional keyword arguments passed to the solver.

    Returns
    -------
    CalibrationResult
        Result object with fitted parameters, diagnostics, and solution.

    Examples
    --------
    >>> # Calibrate a shock size using linear IRF
    >>> result = calibrate(
    ...     model=base_model,
    ...     targets=[PointTarget(variable="I", time=5, value=0.01)],
    ...     calib_params=[ShockParam("Z_til", initial=0.005, bounds=(0.001, 0.1))],
    ...     solver="linear_irf",
    ...     spec=LinearSpec(shock_name="Z_til", shock_size=0.01, Nt=50),
    ... )

    >>> # Calibrate a model parameter with deterministic solver
    >>> result = calibrate(
    ...     model=base_model,
    ...     targets=[PointTarget(variable="I", time=10, value=0.55)],
    ...     calib_params=[ModelParam("bet", initial=0.95, bounds=(0.90, 0.99))],
    ...     solver="deterministic",
    ...     spec=spec,
    ... )

    >>> # Mixed: model param + shock param
    >>> result = calibrate(
    ...     model=base_model,
    ...     targets=[
    ...         PointTarget(variable="I", time=10, value=0.55),
    ...         PointTarget(variable="I", time=20, value=0.56),
    ...     ],
    ...     calib_params=[
    ...         ModelParam("bet", initial=0.95, bounds=(0.90, 0.99)),
    ...         ShockParam("Z_til", initial=0.01, bounds=(0.0, 0.1)),
    ...     ],
    ...     solver="deterministic",
    ...     spec=spec,
    ... )
    """
    # Validate inputs
    if len(targets) == 0:
        raise ValueError("At least one target must be specified")

    if spec is None:
        raise ValueError("spec is required (DetSpec or LinearSpec)")

    # Validate solver choice
    valid_solvers = ["deterministic", "linear_irf", "linear_sequence"]
    if solver not in valid_solvers:
        raise ValueError(f"Unknown solver: {solver}. Must be one of {valid_solvers}.")

    # Build internal callback from declarative calib_params
    param_to_model, initial_params, auto_bounds, param_names = _build_param_to_model(
        model, calib_params, spec, suppress_solver_output
    )

    # Use auto-bounds from calib_params if user didn't pass explicit bounds
    if bounds is None:
        bounds = auto_bounds

    n_params = len(initial_params)

    # For functional targets, we can't know the dimensionality until runtime
    # So we do a conservative check here
    min_targets = sum(
        1 for t in targets if isinstance(t, (PointTarget, FunctionalTarget))
    )

    if n_params > min_targets and not any(
        isinstance(t, FunctionalTarget) for t in targets
    ):
        # Only error if we have no functional targets (which could be vector-valued)
        raise ValueError(
            f"Problem is under-identified: {n_params} parameters "
            f"but only {min_targets} targets. Add more targets or reduce parameters."
        )

    # Determine problem type
    # We need to evaluate targets once to know the true dimensionality
    is_scalar = n_params == 1

    def _apply_transforms(solution: PathResult) -> PathResult:
        if series_transforms is None and default_transform is None:
            return solution
        return solution.transform(
            series_transforms=series_transforms,
            default_transform=default_transform,
        )

    # Test evaluation to determine actual number of targets
    try:
        test_mod, test_spec = param_to_model(initial_params)

        # Extract Nt from the spec
        if isinstance(test_spec, DetSpec):
            Nt = test_spec.Nt
        elif isinstance(test_spec, LinearSpec):
            Nt = test_spec.Nt
        else:
            raise ValueError(
                "spec must be DetSpec or LinearSpec with Nt attribute. "
                f"Got {type(test_spec)}"
            )

        # Quick solve to get dimensionality
        with _suppress_solver_output(suppress_solver_output):
            if solver == "linear_irf":
                test_solution = _compute_irf_from_linear_model(test_mod, test_spec)
            elif solver == "linear_sequence":
                from .linear import solve_sequence_linear

                seq_result = solve_sequence_linear(
                    test_spec, test_mod, Nt, **solver_kwargs
                )
                test_solution = (
                    seq_result.splice(Nt)
                    if seq_result.n_regimes > 1
                    else seq_result.regimes[0]
                )
            else:  # deterministic
                from .deterministic import solve, solve_sequence

                if isinstance(test_spec, DetSpec):
                    seq_result = solve_sequence(
                        test_spec,
                        test_mod,
                        Nt,
                        tol=solver_kwargs.get("tol", 1e-8),
                        save_results=False,
                        **{
                            k: v
                            for k, v in solver_kwargs.items()
                            if k not in {"tol", "save_results"}
                        },
                    )
                    test_solution = (
                        seq_result.splice(Nt)
                        if seq_result.n_regimes > 1
                        else seq_result.regimes[0]
                    )
                else:
                    Z_path = np.zeros((Nt, len(test_mod.exog_list)))
                    test_solution = solve(
                        test_mod,
                        Z_path,
                        tol=solver_kwargs.get("tol", 1e-8),
                        **{k: v for k, v in solver_kwargs.items() if k != "tol"},
                    )

        test_solution = _apply_transforms(test_solution)
        test_errors, _ = _evaluate_targets(targets, test_solution)
        n_targets = len(test_errors)

    except Exception as e:
        logger.warning(
            "Could not determine target dimensionality from test evaluation: "
            "%s. Assuming minimal targets.",
            str(e),
        )
        n_targets = min_targets

    # Check for under-identification
    if n_params > n_targets:
        raise ValueError(
            f"Problem is under-identified: {n_params} parameters "
            f"but {n_targets} target values. Add more targets or reduce parameters."
        )

    is_just_identified = n_params == n_targets

    logger.info(
        "Calibration problem: %d params, %d targets, %s, %s",
        n_params,
        n_targets,
        "just-identified" if is_just_identified else "over-identified",
        "scalar" if is_scalar else "vector",
    )

    eval_count = 0

    # Build unified objective function
    def _solve_and_evaluate(params, return_weights=False):
        """
        Compute target errors (and optionally weights) for given parameters.
        """
        params = np.atleast_1d(params)

        try:
            # Get model and spec from parameters
            mod, spec_from_params = param_to_model(params)

            # Extract Nt from the spec
            if isinstance(spec_from_params, DetSpec):
                Nt_local = spec_from_params.Nt
            elif isinstance(spec_from_params, LinearSpec):
                Nt_local = spec_from_params.Nt
            else:
                raise ValueError("spec must be DetSpec or LinearSpec with Nt attribute")

            # Solve using specified solver
            with _suppress_solver_output(suppress_solver_output):
                if solver == "deterministic":
                    from .deterministic import solve, solve_sequence

                    if isinstance(spec_from_params, DetSpec):
                        # Use sequence solver if DetSpec provided
                        seq_result = solve_sequence(
                            spec_from_params,
                            mod,
                            Nt_local,
                            tol=solver_kwargs.get("tol", 1e-8),
                            save_results=False,
                            **{
                                k: v
                                for k, v in solver_kwargs.items()
                                if k not in {"tol", "save_results"}
                            },
                        )
                        solution = (
                            seq_result.splice(Nt_local)
                            if seq_result.n_regimes > 1
                            else seq_result.regimes[0]
                        )
                    else:
                        # Simple deterministic solve
                        Z_path = np.zeros((Nt_local, len(mod.exog_list)))
                        solution = solve(
                            mod,
                            Z_path,
                            tol=solver_kwargs.get("tol", 1e-8),
                            **{k: v for k, v in solver_kwargs.items() if k != "tol"},
                        )

                elif solver == "linear_irf":
                    # Compute IRF using existing LinearModel machinery
                    solution = _compute_irf_from_linear_model(mod, spec_from_params)

                elif solver == "linear_sequence":
                    from .linear import solve_sequence_linear

                    if not isinstance(spec_from_params, DetSpec):
                        raise ValueError(
                            "linear_sequence solver requires DetSpec specification"
                        )

                    seq_result = solve_sequence_linear(
                        spec_from_params,
                        mod,
                        Nt_local,
                        **solver_kwargs,
                    )
                    solution = (
                        seq_result.splice(Nt_local)
                        if seq_result.n_regimes > 1
                        else seq_result.regimes[0]
                    )

                else:
                    raise ValueError(
                        f"Unknown solver: {solver}. Must be 'deterministic', "
                        "'linear_irf', or 'linear_sequence'."
                    )

            # Evaluate targets - returns both errors and weights
            transformed_solution = _apply_transforms(solution)
            errors, target_weights = _evaluate_targets(targets, transformed_solution)

            nonlocal eval_count
            eval_count += 1
            if progress_every and eval_count % progress_every == 0:
                params_array = np.atleast_1d(params)
                params_dict = {
                    name: float(val)
                    for name, val in zip(param_names, params_array, strict=False)
                }
                residual = np.linalg.norm(errors)
                if return_weights:
                    weighted_errors = target_weights * (np.asarray(errors) ** 2)
                    residual = float(np.sqrt(np.sum(weighted_errors)))
                logger.info(
                    "Calibration eval %d: params=%s residual=%g",
                    eval_count,
                    params_dict,
                    residual,
                )

            # Return based on what's requested
            if return_weights:
                return errors, target_weights
            else:
                # For root-finding, return scalar for scalar problems
                if is_scalar and len(errors) == 1:
                    return float(errors[0])
                return errors

        except Exception as e:
            logger.error("Error in objective function: %s", str(e))
            # Return large error on failure
            if return_weights:
                return np.full(n_targets, 1e10), np.ones(n_targets)
            else:
                if is_scalar and n_targets == 1:
                    return 1e10
                return np.full(n_targets, 1e10)

    # Wrapper for root-finding (no weights)
    def objective(params):
        """Compute target errors for root-finding."""
        return _solve_and_evaluate(params, return_weights=False)

    # Wrapper for minimization (with weights)
    def objective_with_weights(params):
        """Compute target errors and weights for minimization."""
        return _solve_and_evaluate(params, return_weights=True)

    # Select and run optimization method
    if is_just_identified:
        if is_scalar:
            # Scalar root finding
            result = _solve_scalar_root(
                objective, initial_params[0], bounds, method, tol, maxiter
            )
        else:
            # Vector root finding
            result = _solve_vector_root(objective, initial_params, method, tol, maxiter)
    else:
        # Over-identified: use minimization with weights
        if is_scalar:
            # Scalar minimization
            result = _solve_scalar_minimize(
                objective_with_weights,
                initial_params[0],
                bounds,
                method,
                tol,
                maxiter,
            )
        else:
            # Vector minimization
            result = _solve_vector_minimize(
                objective_with_weights,
                initial_params,
                bounds,
                method,
                tol,
                maxiter,
            )

    # Post-process: replace generic param names with descriptive names
    result.parameters = {
        name: float(val) for name, val in zip(param_names, result.parameters_array)
    }

    # Extract final solution using the same unified evaluation function
    try:
        final_model, final_spec = param_to_model(result.parameters_array)

        # Extract Nt from the spec
        if isinstance(final_spec, DetSpec):
            Nt_final = final_spec.Nt
        elif isinstance(final_spec, LinearSpec):
            Nt_final = final_spec.Nt
        else:
            raise ValueError("spec must be DetSpec or LinearSpec with Nt attribute")

        if return_solution:
            with _suppress_solver_output(suppress_solver_output):
                if solver == "deterministic":
                    from .deterministic import solve, solve_sequence

                    if isinstance(final_spec, DetSpec):
                        seq_result = solve_sequence(
                            final_spec,
                            final_model,
                            Nt_final,
                            tol=solver_kwargs.get("tol", 1e-8),
                            save_results=False,
                            **{
                                k: v
                                for k, v in solver_kwargs.items()
                                if k not in {"tol", "save_results"}
                            },
                        )
                        final_solution = (
                            seq_result.splice(Nt_final)
                            if seq_result.n_regimes > 1
                            else seq_result.regimes[0]
                        )
                    else:
                        Z_path = np.zeros((Nt_final, len(final_model.exog_list)))
                        final_solution = solve(
                            final_model,
                            Z_path,
                            tol=solver_kwargs.get("tol", 1e-8),
                            **{k: v for k, v in solver_kwargs.items() if k != "tol"},
                        )

                elif solver == "linear_irf":
                    final_solution = _compute_irf_from_linear_model(
                        final_model, final_spec
                    )

                elif solver == "linear_sequence":
                    from .linear import solve_sequence_linear

                    seq_result = solve_sequence_linear(
                        final_spec,
                        final_model,
                        Nt_final,
                        **solver_kwargs,
                    )
                    final_solution = (
                        seq_result.splice(Nt_final)
                        if seq_result.n_regimes > 1
                        else seq_result.regimes[0]
                    )

            result.solution = _apply_transforms(final_solution)
            result.model = final_model
        else:
            result.solution = None
            result.model = None

    except Exception as e:
        logger.error("Error computing final solution: %s", str(e))
        result.solution = None
        result.model = None

    # Handle automatic saving or error reporting
    if label is not None:
        if result.success:
            result.save(label, save_dir=save_dir)
        else:
            raise RuntimeError(
                f"Calibration failed for label '{label}': {result.message}. "
                "Parameters were not saved."
            )

    return result


def _count_targets(targets: List[Union[PointTarget, FunctionalTarget]]) -> int:
    """Count the total number of target values."""
    count = 0
    for target in targets:
        if isinstance(target, PointTarget):
            count += 1
        elif isinstance(target, FunctionalTarget):
            # For functional targets, we need to know dimensionality
            # For now, assume scalar unless we can infer otherwise
            # This will be determined at runtime
            count += 1
        else:
            raise TypeError(f"Unknown target type: {type(target)}")
    return count


def _evaluate_targets(
    targets: List[Union[PointTarget, FunctionalTarget]],
    solution: Union[PathResult, DeterministicResult, IrfResult, SequenceResult],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate all targets against a solution and return errors and weights.

    Returns
    -------
    errors : np.ndarray
        Array of target errors.
    weights : np.ndarray
        Array of target weights (same length as errors).
    """
    errors = []
    weights = []

    for target in targets:
        if isinstance(target, PointTarget):
            # Extract variable value at specified time
            try:
                var_idx = solution.var_names.index(target.variable)
            except ValueError:
                # Variable not found, check intermediate variables
                if solution.Y is not None and target.variable in solution.y_names:
                    var_idx = solution.y_names.index(target.variable)
                    if target.time >= solution.Y.shape[0]:
                        raise ValueError(
                            f"Time {target.time} out of range for solution "
                            f"with {solution.Y.shape[0]} periods"
                        )
                    actual_value = solution.Y[target.time, var_idx]
                else:
                    raise ValueError(
                        f"Variable '{target.variable}' not found in solution"
                    )
            else:
                if target.time >= solution.UX.shape[0]:
                    raise ValueError(
                        f"Time {target.time} out of range for solution "
                        f"with {solution.UX.shape[0]} periods"
                    )
                actual_value = solution.UX[target.time, var_idx]

            error = actual_value - target.value
            errors.append(error)
            weights.append(target.weight)

        elif isinstance(target, FunctionalTarget):
            # Call user-defined function
            func_result = target.func(solution)
            func_errors = np.atleast_1d(func_result)
            errors.extend(func_errors)

            # Handle weights for functional target
            if target.weights is not None:
                func_weights = np.atleast_1d(target.weights)
                if len(func_weights) != len(func_errors):
                    raise ValueError(
                        f"FunctionalTarget weights length ({len(func_weights)}) "
                        f"must match function output length ({len(func_errors)})"
                    )
                weights.extend(func_weights)
            else:
                # Default to weight of 1.0 for each output
                weights.extend([1.0] * len(func_errors))

        else:
            raise TypeError(f"Unknown target type: {type(target)}")

    return np.array(errors), np.array(weights)


def _compute_irf_from_linear_model(
    model,
    shock_spec: LinearSpec,
) -> IrfResult:
    """
    Compute impulse response function using existing LinearModel machinery.

    This is a thin wrapper around LinearModel.compute_irfs() that extracts
    the IRF for a single shock as specified by LinearSpec.

    Parameters
    ----------
    model : Model
        Model with linearization already computed (model.linear_mod exists).
    shock_spec : LinearSpec
        Shock specification with shock_name, shock_size, and Nt.

    Returns
    -------
    IrfResult
        IRF result container for the specified shock.
    """
    if not isinstance(shock_spec, LinearSpec):
        raise TypeError(
            f"shock_spec must be LinearSpec, got {type(shock_spec).__name__}"
        )

    # Use existing LinearModel.compute_irfs() method
    if not hasattr(model, "linear_mod") or model.linear_mod is None:
        raise ValueError("Model must be linearized before computing IRFs")

    # Compute IRFs for all shocks using existing machinery
    irf_dict = model.linear_mod.compute_irfs(shock_spec.Nt)

    # Extract the IRF for the requested shock
    if shock_spec.shock_name not in irf_dict:
        raise ValueError(
            f"Shock '{shock_spec.shock_name}' not found. "
            f"Available shocks: {list(irf_dict.keys())}"
        )

    irf_result = irf_dict[shock_spec.shock_name]

    # Scale the IRF by the requested shock size
    # The compute_irfs() method returns unit-sized shocks, so we need to scale
    if shock_spec.shock_size != 1.0:
        irf_result = IrfResult(
            UX=irf_result.UX * shock_spec.shock_size,
            Z=irf_result.Z * shock_spec.shock_size,
            Y=(
                irf_result.Y * shock_spec.shock_size
                if irf_result.Y is not None
                else None
            ),
            model_label=irf_result.model_label,
            var_names=irf_result.var_names,
            exog_names=irf_result.exog_names,
            y_names=irf_result.y_names,
            shock_name=shock_spec.shock_name,
            shock_size=shock_spec.shock_size,
        )

    return irf_result


def _solve_scalar_root(
    func: Callable,
    x0: float,
    bounds: Optional[List[tuple]],
    method: Optional[str],
    tol: float,
    maxiter: int,
) -> CalibrationResult:
    """Solve scalar root-finding problem."""
    # Set up bounds
    if bounds is not None and len(bounds) > 0:
        bracket = bounds[0]
    else:
        # Use wide default bracket
        bracket = (x0 - 10.0, x0 + 10.0)

    # Choose method
    if method is None:
        method = "brentq"  # Robust bracketing method

    def _safe_residual(x):
        try:
            return float(abs(func(x)))
        except Exception:
            return np.inf

    try:
        sol = opt.root_scalar(
            func,
            method=method,
            bracket=bracket,
            xtol=tol,
            maxiter=maxiter,
        )

        return CalibrationResult(
            parameters={"param_0": sol.root},
            parameters_array=np.array([sol.root]),
            success=sol.converged,
            residual=_safe_residual(sol.root),
            iterations=sol.iterations if hasattr(sol, "iterations") else 0,
            message=sol.flag if hasattr(sol, "flag") else "",
            method="root_scalar",
        )

    except ValueError as e:
        # Brent-style methods require a sign change; fall back to secant if absent.
        msg = str(e)
        if "different signs" in msg or "sign" in msg:
            x1 = bracket[1] if x0 != bracket[1] else bracket[0]
            try:
                sol = opt.root_scalar(
                    func,
                    method="secant",
                    x0=x0,
                    x1=x1,
                    xtol=tol,
                    maxiter=maxiter,
                )
                return CalibrationResult(
                    parameters={"param_0": sol.root},
                    parameters_array=np.array([sol.root]),
                    success=sol.converged,
                    residual=_safe_residual(sol.root),
                    iterations=sol.iterations if hasattr(sol, "iterations") else 0,
                    message=sol.flag if hasattr(sol, "flag") else msg,
                    method="root_scalar",
                )
            except Exception as secant_err:
                logger.error(
                    "Scalar root finding failed after secant fallback: %s",
                    str(secant_err),
                )
                return CalibrationResult(
                    parameters={"param_0": x0},
                    parameters_array=np.array([x0]),
                    success=False,
                    residual=_safe_residual(x0),
                    message=str(secant_err),
                    method="root_scalar",
                )
        logger.error("Scalar root finding failed: %s", msg)
        return CalibrationResult(
            parameters={"param_0": x0},
            parameters_array=np.array([x0]),
            success=False,
            residual=_safe_residual(x0),
            message=msg,
            method="root_scalar",
        )

    except Exception as e:
        logger.error("Scalar root finding failed: %s", str(e))
        return CalibrationResult(
            parameters={"param_0": x0},
            parameters_array=np.array([x0]),
            success=False,
            residual=_safe_residual(x0),
            message=str(e),
            method="root_scalar",
        )


def _solve_vector_root(
    func: Callable,
    x0: np.ndarray,
    method: Optional[str],
    tol: float,
    maxiter: int,
) -> CalibrationResult:
    """Solve vector root-finding problem."""
    if method is None:
        method = "hybr"  # Hybrid Powell method

    try:
        sol = opt.root(
            func,
            x0,
            method=method,
            tol=tol,
            options={"maxiter": maxiter},
        )

        # Build parameter dict
        params = {f"param_{i}": val for i, val in enumerate(sol.x)}

        # Compute residual norm
        residual = np.linalg.norm(sol.fun) if hasattr(sol, "fun") else 0.0

        return CalibrationResult(
            parameters=params,
            parameters_array=sol.x,
            success=sol.success,
            residual=residual,
            iterations=sol.nfev if hasattr(sol, "nfev") else 0,
            message=sol.message,
            method="root",
        )

    except Exception as e:
        logger.error("Vector root finding failed: %s", str(e))
        params = {f"param_{i}": val for i, val in enumerate(x0)}
        return CalibrationResult(
            parameters=params,
            parameters_array=x0,
            success=False,
            message=str(e),
            method="root",
        )


def _solve_scalar_minimize(
    func_with_weights: Callable,
    x0: float,
    bounds: Optional[List[tuple]],
    method: Optional[str],
    tol: float,
    maxiter: int,
) -> CalibrationResult:
    """Solve scalar minimization problem with weighted objectives."""

    # Wrap function to return weighted squared error
    def objective_squared(x):
        errors, weights = func_with_weights(x)
        # Apply weights to squared errors
        weighted_errors = weights * (np.asarray(errors) ** 2)
        return np.sum(weighted_errors)

    # Set up bounds
    if bounds is not None and len(bounds) > 0:
        bracket = bounds[0]
    else:
        bracket = (x0 - 10.0, x0 + 10.0)

    # Choose method
    if method is None:
        method = "bounded"

    try:
        sol = opt.minimize_scalar(
            objective_squared,
            method=method,
            bounds=bracket,
            options={"xatol": tol, "maxiter": maxiter},
        )

        return CalibrationResult(
            parameters={"param_0": sol.x},
            parameters_array=np.array([sol.x]),
            success=sol.success,
            residual=np.sqrt(sol.fun),  # Convert back from weighted squared
            iterations=sol.nfev if hasattr(sol, "nfev") else 0,
            message=sol.message,
            method="minimize_scalar",
        )

    except Exception as e:
        logger.error("Scalar minimization failed: %s", str(e))
        return CalibrationResult(
            parameters={"param_0": x0},
            parameters_array=np.array([x0]),
            success=False,
            message=str(e),
            method="minimize_scalar",
        )


def _solve_vector_minimize(
    func_with_weights: Callable,
    x0: np.ndarray,
    bounds: Optional[List[tuple]],
    method: Optional[str],
    tol: float,
    maxiter: int,
) -> CalibrationResult:
    """Solve vector minimization problem with weighted objectives."""

    # Wrap function to return weighted sum of squared errors
    def objective_squared(x):
        errors, weights = func_with_weights(x)
        # Apply weights to squared errors
        weighted_errors = weights * (np.asarray(errors) ** 2)
        return np.sum(weighted_errors)

    # Choose method
    if method is None:
        method = "L-BFGS-B" if bounds is not None else "Nelder-Mead"

    try:
        sol = opt.minimize(
            objective_squared,
            x0,
            method=method,
            bounds=bounds,
            tol=tol,
            options={"maxiter": maxiter},
        )

        # Build parameter dict
        params = {f"param_{i}": val for i, val in enumerate(sol.x)}

        return CalibrationResult(
            parameters=params,
            parameters_array=sol.x,
            success=sol.success,
            residual=np.sqrt(sol.fun),  # Convert back from weighted squared
            iterations=sol.nfev if hasattr(sol, "nfev") else 0,
            message=sol.message,
            method="minimize",
        )

    except Exception as e:
        logger.error("Vector minimization failed: %s", str(e))
        params = {f"param_{i}": val for i, val in enumerate(x0)}
        return CalibrationResult(
            parameters=params,
            parameters_array=x0,
            success=False,
            message=str(e),
            method="minimize",
        )
