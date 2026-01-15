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

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.optimize as opt

from .det_spec import DetSpec
from .linear_spec import LinearSpec
from .results import DeterministicResult, IrfResult, PathResult, SequenceResult

logger = logging.getLogger(__name__)


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


def calibrate(
    model,
    targets: List[Union[PointTarget, FunctionalTarget]],
    param_to_model: Callable[[np.ndarray], tuple],
    initial_params: np.ndarray,
    solver: str = "deterministic",
    spec: Optional[Union[DetSpec, LinearSpec]] = None,
    bounds: Optional[List[tuple]] = None,
    method: Optional[str] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
    **solver_kwargs,
) -> CalibrationResult:
    """
    Unified calibration function for fitting parameters to target outcomes.

    This function provides a flexible interface for calibrating model parameters
    and/or shocks to match specified target outcomes. It automatically dispatches
    to the appropriate solver and uses special-case logic for scalar/vector and
    just-identified/over-identified problems.

    Parameters
    ----------
    model : Model
        Base model instance to calibrate.
    targets : list of PointTarget or FunctionalTarget
        Target specifications to match.
    param_to_model : callable
        Function that takes parameter array and returns (model, spec) tuple.
        The spec should be appropriate for the chosen solver:
        - For deterministic/linear_sequence: DetSpec with Nt set
        - For linear_irf: LinearSpec with shock_name, shock_size, and Nt

        Example for deterministic/linear_sequence::

            def param_to_model(params):
                # params = [tau, shock_size]
                new_model = model.update_copy(params={"tau": params[0]})
                new_model.solve_steady(calibrate=False)
                new_model.linearize()

                spec = DetSpec(Nt=50)
                spec.add_regime(0)
                spec.add_shock(0, "z_tfp", per=0, val=params[1])

                return new_model, spec

        Example for linear_irf::

            def param_to_model(params):
                new_model = model.update_copy(params={"shock_size": params[0]})
                new_model.solve_steady(calibrate=False)
                new_model.linearize()

                spec = LinearSpec(
                    shock_name="Z_til",
                    shock_size=params[0],
                    Nt=50
                )

                return new_model, spec

    initial_params : array-like
        Initial parameter values for optimization.
    solver : str, default "deterministic"
        Solver to use: "deterministic", "linear_irf", or "linear_sequence".
    spec : DetSpec or LinearSpec, optional
        Specification object for solver. If provided, overrides spec from
        param_to_model. Must include Nt:
        - For deterministic/linear_sequence: DetSpec with Nt attribute
        - For linear_irf: LinearSpec with shock_name, shock_size, and Nt
    bounds : list of tuples, optional
        Parameter bounds for optimization. Each tuple is (lower, upper).
    method : str, optional
        Optimization method. If None, automatically selected based on problem
        structure. Options include 'hybr', 'lm' (for root finding), 'L-BFGS-B',
        'Nelder-Mead' (for minimization).
    tol : float, default 1e-6
        Convergence tolerance.
    maxiter : int, default 100
        Maximum number of iterations.
    **solver_kwargs
        Additional keyword arguments passed to the solver.

    Returns
    -------
    CalibrationResult
        Result object with fitted parameters, diagnostics, and solution.

    Raises
    ------
    ValueError
        If the problem is under-identified (more parameters than targets).

    Notes
    -----
    The function automatically selects the appropriate optimization method based
    on the problem structure:

    - **Just-identified (n_params == n_targets)**:
        - Scalar (n_params == 1): Uses `scipy.optimize.root_scalar`
        - Vector (n_params > 1): Uses `scipy.optimize.root`

    - **Over-identified (n_params < n_targets)**:
        - Scalar (n_params == 1): Uses `scipy.optimize.minimize_scalar`
        - Vector (n_params > 1): Uses `scipy.optimize.minimize`

    Examples
    --------
    >>> # Deterministic path calibration
    >>> from equilibrium.solvers.calibration import calibrate, PointTarget
    >>> from equilibrium.solvers.det_spec import DetSpec
    >>>
    >>> # Define parameter mapping
    >>> def param_to_model(params):
    ...     mod = base_model.update_copy(params={"beta": params[0]})
    ...     mod.solve_steady(calibrate=False)
    ...     mod.linearize()
    ...     spec = DetSpec(Nt=50)
    ...     spec.add_regime(0)
    ...     return mod, spec
    >>>
    >>> # Define target
    >>> targets = [PointTarget(variable="consumption", time=10, value=1.05)]
    >>>
    >>> # Calibrate
    >>> result = calibrate(
    ...     model=base_model,
    ...     targets=targets,
    ...     param_to_model=param_to_model,
    ...     initial_params=np.array([0.96]),
    ...     solver="deterministic",
    ... )
    >>> print(f"Fitted beta: {result.parameters['beta']}")
    >>> print(f"Success: {result.success}")
    """
    # Validate inputs
    if len(targets) == 0:
        raise ValueError("At least one target must be specified")

    # Validate solver choice
    valid_solvers = ["deterministic", "linear_irf", "linear_sequence"]
    if solver not in valid_solvers:
        raise ValueError(f"Unknown solver: {solver}. Must be one of {valid_solvers}.")

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

    # Test evaluation to determine actual number of targets
    # First, extract Nt from spec
    try:
        test_mod, test_spec = param_to_model(initial_params)
        spec_to_use_test = spec if spec is not None else test_spec

        # Extract Nt from the spec
        if isinstance(spec_to_use_test, DetSpec):
            Nt = spec_to_use_test.Nt
        elif isinstance(spec_to_use_test, LinearSpec):
            Nt = spec_to_use_test.Nt
        else:
            raise ValueError(
                "spec must be DetSpec or LinearSpec with Nt attribute. "
                f"Got {type(spec_to_use_test)}"
            )

        # Quick solve to get dimensionality
        if solver == "linear_irf":
            test_solution = _compute_irf_from_linear_model(test_mod, spec_to_use_test)
        elif solver == "linear_sequence":
            from .linear import solve_sequence_linear

            seq_result = solve_sequence_linear(
                spec_to_use_test, test_mod, Nt, **solver_kwargs
            )
            test_solution = (
                seq_result.splice(Nt)
                if seq_result.n_regimes > 1
                else seq_result.regimes[0]
            )
        else:  # deterministic
            from .deterministic import solve, solve_sequence

            if isinstance(spec_to_use_test, DetSpec):
                seq_result = solve_sequence(
                    spec_to_use_test,
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

        test_errors, _ = _evaluate_targets(targets, test_solution)
        n_targets = len(test_errors)

    except Exception as e:
        logger.warning(
            "Could not determine target dimensionality from test evaluation: %s. Assuming minimal targets.",
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

    # Build unified objective function
    def _solve_and_evaluate(params, return_weights=False):
        """
        Compute target errors (and optionally weights) for given parameters.

        This is the core evaluation function used by both root-finding and
        minimization methods.

        Parameters
        ----------
        params : array-like
            Parameter values to evaluate.
        return_weights : bool, default False
            If True, return (errors, weights) tuple for weighted minimization.
            If False, return only errors for root-finding.

        Returns
        -------
        errors : array or float
            Target errors. Scalar if is_scalar and single target.
        weights : array, optional
            Target weights. Only returned if return_weights=True.
        """
        params = np.atleast_1d(params)

        try:
            # Get model and spec from parameters
            mod, spec_from_params = param_to_model(params)

            # Use provided spec if available, otherwise use from param_to_model
            spec_to_use = spec if spec is not None else spec_from_params

            # Extract Nt from the spec
            if isinstance(spec_to_use, DetSpec):
                Nt_local = spec_to_use.Nt
            elif isinstance(spec_to_use, LinearSpec):
                Nt_local = spec_to_use.Nt
            else:
                raise ValueError("spec must be DetSpec or LinearSpec with Nt attribute")

            # Solve using specified solver
            if solver == "deterministic":
                from .deterministic import solve, solve_sequence

                if isinstance(spec_to_use, DetSpec):
                    # Use sequence solver if DetSpec provided
                    seq_result = solve_sequence(
                        spec_to_use,
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
                solution = _compute_irf_from_linear_model(mod, spec_to_use)

            elif solver == "linear_sequence":
                from .linear import solve_sequence_linear

                if not isinstance(spec_to_use, DetSpec):
                    raise ValueError(
                        "linear_sequence solver requires DetSpec specification"
                    )

                seq_result = solve_sequence_linear(
                    spec_to_use,
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
            errors, target_weights = _evaluate_targets(targets, solution)

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
                objective_with_weights, initial_params[0], bounds, method, tol, maxiter
            )
        else:
            # Vector minimization
            result = _solve_vector_minimize(
                objective_with_weights, initial_params, bounds, method, tol, maxiter
            )

    # Extract final solution using the same unified evaluation function
    try:
        final_model, final_spec = param_to_model(result.parameters_array)
        spec_to_use = spec if spec is not None else final_spec

        # Extract Nt from the spec
        if isinstance(spec_to_use, DetSpec):
            Nt_final = spec_to_use.Nt
        elif isinstance(spec_to_use, LinearSpec):
            Nt_final = spec_to_use.Nt
        else:
            raise ValueError("spec must be DetSpec or LinearSpec with Nt attribute")

        if solver == "deterministic":
            from .deterministic import solve, solve_sequence

            if isinstance(spec_to_use, DetSpec):
                seq_result = solve_sequence(
                    spec_to_use,
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
            final_solution = _compute_irf_from_linear_model(final_model, spec_to_use)

        elif solver == "linear_sequence":
            from .linear import solve_sequence_linear

            seq_result = solve_sequence_linear(
                spec_to_use,
                final_model,
                Nt_final,
                **solver_kwargs,
            )
            final_solution = (
                seq_result.splice(Nt_final)
                if seq_result.n_regimes > 1
                else seq_result.regimes[0]
            )

        result.solution = final_solution
        result.model = final_model

    except Exception as e:
        logger.error("Error computing final solution: %s", str(e))
        result.solution = None
        result.model = None

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
            residual=abs(sol.function_calls) if hasattr(sol, "function_calls") else 0.0,
            iterations=sol.iterations if hasattr(sol, "iterations") else 0,
            message=sol.flag if hasattr(sol, "flag") else "",
            method="root_scalar",
        )

    except Exception as e:
        logger.error("Scalar root finding failed: %s", str(e))
        return CalibrationResult(
            parameters={"param_0": x0},
            parameters_array=np.array([x0]),
            success=False,
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
