#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear solver functions for deterministic transition paths.

This module provides functions for computing deterministic paths using
the linearized model dynamics.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .deterministic import _build_regime_steady_label, _save_regime_steady_outputs

logger = logging.getLogger(__name__)


def solve_sequence_linear(
    det_spec,
    mod,
    Nt,
    z_init=None,
    ux_init=None,
    y_init=None,
    calibrate_initial=True,
    recalibrate_regimes=False,
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "npz",
    display_steady: bool = False,
    copy_model: bool = False,
    save_regime_steady: bool = False,
    save_regime_steady_tex: bool = False,
    regime_labels: Optional[Sequence[str]] = None,
):
    """
    Solve a sequence of deterministic paths for multiple regimes using the linear model.

    This function loops over regimes defined in det_spec, computing a
    deterministic path for each one using the linearized model dynamics.
    For each regime after the first, initial conditions are taken from
    the previous regime's solution at the transition time specified in
    det_spec.time_list.
    The initial state at t=0 is the baseline steady state; the first
    regime in det_spec is applied starting at t=1.

    In the linear model, the variables follow:
        x_new = H_x @ x_hat + H_z @ z_hat
        u = G_x @ x_hat + G_z @ z_hat

    where x_hat and z_hat are deviations from steady state.

    Parameters
    ----------
    det_spec : DetSpec
        Specification of the deterministic scenario with regime-specific
        parameters and shocks.
    mod : Model
        The baseline model to use. A copy is made up front so the original
        instance is not mutated; copies are also created for regimes where
        preset_par_list differs from the previous regime.
    Nt : int
        Horizon (number of periods) for each regime's deterministic simulation.
    z_init : array-like, optional
        Initial values for exogenous states for the first regime. If None,
        defaults to zeros.
    ux_init : array-like, optional
        Initial values for endogenous states and controls for the first regime.
        If None, defaults to steady state.
    y_init : array-like, optional
        Initial values for intermediate variables for the first regime. If None,
        they are computed from the initial model when using the steady state.
    calibrate_initial : bool, optional
        Whether to calibrate the model when computing the initial steady state.
        Default is True.
    recalibrate_regimes : bool, optional
        If True, apply the `calibrate_initial` flag to all regime steady-state solves.
        If False (default), only the initial steady state uses `calibrate_initial=True`;
        subsequent regimes with parameter changes use `calibrate=False` to avoid
        re-solving for calibrated parameters that may be preset in the regime.
        Default is False.
    save_path : str or Path, optional
        If provided, save results to this path.
    save_format : str, default "npz"
        Format for saving results. Supported: 'npz', 'json'.
    display_steady : bool, optional
        Whether to display steady state results when solving for each regime.
        Default is False to reduce output when solving multiple regimes.
    copy_model : bool, default False
        If True, operate on a copy of ``mod`` so the original model is not
        mutated by steady-state or linearization updates.
    save_regime_steady : bool, default False
        If True, save steady-state snapshots for each regime model using
        deterministic sequence labels.
    save_regime_steady_tex : bool, default False
        If True, also export regime steady states to LaTeX files. This
        implies steady-state JSON snapshots are also written.
    regime_labels : sequence[str], optional
        Optional human-readable names (one per regime) appended to each
        regime steady-state label.

    Returns
    -------
    SequenceResult
        Result container with DeterministicResult for each regime.
    """
    from .results import DeterministicResult, SequenceResult

    if det_spec.n_regimes == 0:
        raise ValueError("DetSpec must have at least one regime")
    if regime_labels is not None and len(regime_labels) != det_spec.n_regimes:
        raise ValueError(
            "regime_labels length must match det_spec.n_regimes: "
            f"{len(regime_labels)} != {det_spec.n_regimes}"
        )

    # Optionally work on a copy so callers can reuse the original model without mutation.
    # update_copy preserves shared JAX bundles to avoid recompilation.
    base_mod = mod.update_copy() if copy_model else mod

    # Ensure the baseline model's steady state is solved and linearized
    # This is needed for consistency and in case ux_init is None
    if not base_mod._steady_solved:
        logger.info("Steady state not found for baseline model, solving...")
        base_mod.solve_steady(calibrate=calibrate_initial, display=display_steady)
    if not base_mod._linearized:
        logger.info("Linear model not found for baseline model, linearizing...")
        base_mod.linearize()

    regime_results = []
    current_mod = base_mod
    # Use z_init from DetSpec if not provided explicitly, otherwise use parameter
    if z_init is None:
        current_z_init = det_spec.z_init
    else:
        current_z_init = z_init
    # Use ux_init from DetSpec if not provided explicitly, otherwise use parameter
    if ux_init is None:
        if det_spec.ux_init is not None:
            current_ux_init = np.asarray(det_spec.ux_init)
        else:
            current_ux_init = np.concatenate(
                [base_mod.steady_components["u"], base_mod.steady_components["x"]]
            )
    else:
        current_ux_init = np.asarray(ux_init)
    prev_preset_par = None
    start_time = 1
    should_save_regime_steady = save_regime_steady or save_regime_steady_tex
    model_label = getattr(mod, "label", "_default")
    experiment_label = getattr(det_spec, "label", "_default")
    regime_steady_labels: list[str] = []

    def _compute_initial_intermediates(mod_for_init, ux_init_for_y, z_init_for_y):
        if not hasattr(mod_for_init, "intermediates"):
            return None
        if not hasattr(mod_for_init, "var_lists") or not mod_for_init.var_lists:
            return None
        y_names = mod_for_init.var_lists.get("intermediate", [])
        if not y_names:
            return None
        u_init, x_init = np.split(ux_init_for_y, np.array((mod_for_init.N["u"],)))
        params = np.array(
            [mod_for_init.params[key] for key in mod_for_init.var_lists["params"]]
        )
        return mod_for_init.intermediates(u_init, x_init, z_init_for_y, params)

    for regime in range(det_spec.n_regimes):
        # Get preset parameters for this regime.
        # preset_par_list[regime] contains parameter overrides for this regime.
        current_preset_par = det_spec.preset_par_list[regime]

        # Check if we need to recreate the Model
        # For first regime: only need new model if there are parameter overrides
        # For subsequent regimes: need new model if params changed from previous
        has_param_overrides = bool(current_preset_par)
        need_new_model = (regime == 0 and has_param_overrides) or (
            regime > 0 and current_preset_par != prev_preset_par
        )

        if need_new_model:
            # Create new model with updated parameters, solve steady state
            # and linearize
            current_mod = base_mod.update_copy(params=current_preset_par)
            # Only calibrate regime-specific steady states if explicitly requested.
            current_mod.solve_steady(
                calibrate=recalibrate_regimes, display=display_steady
            )
            current_mod.linearize()
        elif regime == 0:
            # First regime with no parameter changes - use baseline copy
            # Note: base_mod's steady state and linearization have already been done by the checks before the loop
            current_mod = base_mod

        if should_save_regime_steady:
            regime_name = regime_labels[regime] if regime_labels is not None else None
            regime_steady_label = _build_regime_steady_label(
                model_label=model_label,
                experiment_label=experiment_label,
                regime_idx=regime,
                regime_name=regime_name,
            )
            _save_regime_steady_outputs(
                current_mod,
                regime_steady_label,
                save_tex=save_regime_steady_tex,
            )
            regime_steady_labels.append(regime_steady_label)

        # Build exogenous paths for this regime
        Z_path = det_spec.build_exog_paths(
            current_mod,
            Nt,
            regime=regime,
            z_init=current_z_init,
        )

        # Compute the UX path using the linear model
        UX = solve_linear_path(current_mod, Z_path, current_ux_init)

        # Compute linearized intermediate variables
        Y = compute_linear_intermediates(current_mod, UX, Z_path)
        if regime == 0:
            y_init_regime = None
            if y_init is not None:
                y_init_regime = np.asarray(y_init)
            elif ux_init is None and det_spec.ux_init is None:
                if current_z_init is None:
                    z_init_for_y = np.zeros(current_mod.N["z"])
                else:
                    z_init_for_y = np.asarray(current_z_init)
                y_init_regime = _compute_initial_intermediates(
                    base_mod, current_ux_init, z_init_for_y
                )
            elif hasattr(current_mod, "var_lists"):
                y_names = current_mod.var_lists.get("intermediate", [])
                if y_names:
                    y_init_regime = np.full(len(y_names), np.nan)

            if y_init_regime is not None:
                if Y is None:
                    raise ValueError(
                        "y_init provided but model has no intermediate variables."
                    )
                if y_init_regime.shape[0] != Y.shape[1]:
                    raise ValueError(
                        "y_init length does not match number of intermediate variables."
                    )
                Y[0, :] = y_init_regime

        # Get variable names from model
        var_names = (
            current_mod.var_lists.get("u", []) + current_mod.var_lists.get("x", [])
            if hasattr(current_mod, "var_lists") and current_mod.var_lists
            else []
        )
        exog_names = current_mod.exog_list if hasattr(current_mod, "exog_list") else []
        y_names = (
            current_mod.var_lists.get("intermediate", [])
            if hasattr(current_mod, "var_lists") and current_mod.var_lists
            else []
        )
        model_label = getattr(current_mod, "label", "_default")

        # Create DeterministicResult for this regime
        # Linear solutions are exact (no iteration), so converged=True, residual=0
        result = DeterministicResult(
            UX=UX,
            Z=Z_path,
            Y=Y,
            model_label=model_label,
            var_names=var_names,
            exog_names=exog_names,
            y_names=y_names,
            terminal_condition="linear",  # Linear model uses linear approximation
            converged=True,
            final_residual=0.0,
        )

        regime_results.append(result)

        # Set up initial conditions for next regime from this regime's solution
        if regime < det_spec.n_regimes - 1:
            transition_time = det_spec.time_list[regime] - start_time
            if transition_time < 0:
                raise ValueError(
                    "time_list entries must be >= the start time of the regime. "
                    f"Got time_list[{regime}]={det_spec.time_list[regime]} with "
                    f"start_time={start_time}."
                )
            if transition_time >= Nt:
                raise ValueError(
                    "time_list entries must be within the regime horizon. "
                    f"Got time_list[{regime}]={det_spec.time_list[regime]} with "
                    f"start_time={start_time} and Nt={Nt}."
                )

            current_z_init = Z_path[transition_time, :]
            current_ux_init = UX[transition_time, :]
            start_time = det_spec.time_list[regime] + 1

        # Remember current params for comparison
        prev_preset_par = current_preset_par

    sequence_result = SequenceResult(
        regimes=regime_results,
        time_list=list(det_spec.time_list),
        model_label=model_label,
        experiment_label=experiment_label,
        regime_steady_labels=regime_steady_labels,
    )

    if save_path is not None:
        sequence_result.save(save_path, format=save_format, overwrite=True)

    return sequence_result


def compute_linear_intermediates(mod, UX, Z, deviations=False):
    """
    Compute linearized intermediate variables for all time periods.

    Uses the linear approximation:
        y = y_ss + J_u @ (u - u_ss) + J_x @ (x - x_ss) + J_z @ (z - z_ss)

    where J_u, J_x, J_z are the Jacobians of intermediate variables with
    respect to u, x, and z.

    Parameters
    ----------
    mod : Model
        The model instance with derivatives for intermediate variables.
    UX : np.ndarray, shape (Nt, N_ux)
        Control and state variables at all time points. Can be in levels
        or deviations from steady state (see deviations parameter).
    Z : np.ndarray, shape (Nt, N_z)
        Exogenous variables at all time points. Can be in levels or
        deviations from steady state (see deviations parameter).
    deviations : bool, default False
        If False (default), UX and Z are in levels, and the function returns
        Y in levels. If True, UX and Z are already deviations from steady state,
        and the function returns Y as deviations.

    Returns
    -------
    Y : np.ndarray, shape (Nt, N_y) or None
        Linearized intermediate variables at all time points, either in levels
        or deviations depending on the deviations parameter.
        Returns None if model has no intermediate variables.
    """
    # Check if model has intermediate variables
    if not hasattr(mod, "var_lists") or not mod.var_lists:
        return None

    y_names = mod.var_lists.get("intermediate", [])
    if not y_names:
        return None

    # Check if derivatives are available
    if not hasattr(mod, "derivatives") or "intermediates" not in mod.derivatives:
        logger.warning(
            "Intermediate variable derivatives not found. "
            "Run model.steady_state_derivatives() first."
        )
        return None

    # Get Jacobian matrices for intermediate variables
    J_u = mod.derivatives["intermediates"].get("u")
    J_x = mod.derivatives["intermediates"].get("x")
    J_z = mod.derivatives["intermediates"].get("z")

    if J_u is None or J_x is None or J_z is None:
        logger.warning("Incomplete Jacobian matrices for intermediate variables.")
        return None

    # Get steady state values
    u_ss = mod.steady_components["u"]
    x_ss = mod.steady_components["x"]
    z_ss = mod.steady_components["z"]

    # Compute steady state intermediate variables
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])
    y_ss = mod.intermediates(u_ss, x_ss, z_ss, params)

    Nt = UX.shape[0]
    N_u = mod.N["u"]
    N_y = len(y_names)
    Y = np.zeros((Nt, N_y))

    for tt in range(Nt):
        u_t, x_t = np.split(UX[tt, :], np.array([N_u]))
        z_t = Z[tt, :]

        if deviations:
            # UX and Z are already deviations, use directly
            u_hat = u_t
            x_hat = x_t
            z_hat = z_t
        else:
            # UX and Z are in levels, compute deviations from steady state
            u_hat = u_t - u_ss
            x_hat = x_t - x_ss
            z_hat = z_t - z_ss

        # Apply linear approximation to get y in deviations
        y_hat = J_u @ u_hat + J_x @ x_hat + J_z @ z_hat

        if deviations:
            # Return deviations
            Y[tt, :] = y_hat
        else:
            # Convert back to levels
            Y[tt, :] = y_ss + y_hat

    return Y


def solve_linear_path(mod, Z_path, ux_init=None):
    """
    Compute the UX path using the linearized model dynamics.

    This function applies the linear policy and transition functions:
        u = u_ss + G_x @ x_hat + G_z @ z_hat
        x_new = x_ss + H_x @ x_hat + H_z @ z_hat

    where x_hat and z_hat are deviations from steady state.

    Parameters
    ----------
    mod : Model
        The model to use. If linearization has not been run, it will be
        performed automatically.
    Z_path : array-like, shape (Nt, N_z)
        Exogenous variables path.
    ux_init : array-like, shape (N_ux,), optional
        Initial control and state variables. If None, uses steady state values.

    Returns
    -------
    UX : array-like, shape (Nt, N_ux)
        Computed control and state variables at all time points.
    """
    # Check if linearization has been run; if not, run it
    if (
        mod.linear_mod is None
        or mod.linear_mod.H_x is None
        or mod.linear_mod.H_z is None
        or mod.linear_mod.G_x is None
        or mod.linear_mod.G_z is None
    ):
        logger.info("Linearization not found, running linearize()")
        mod.linearize()

    # Set default ux_init to steady state if not provided
    if ux_init is None:
        ux_init = np.concatenate(
            [mod.steady_components["u"], mod.steady_components["x"]]
        )
    Nt = Z_path.shape[0]
    N_u = mod.N["u"]
    N_x = mod.N["x"]
    N_ux = N_u + N_x

    # Get steady state values
    u_ss = mod.steady_components["u"]
    x_ss = mod.steady_components["x"]
    z_ss = mod.steady_components["z"]

    # Get linear model matrices
    G_x = mod.linear_mod.G_x
    G_z = mod.linear_mod.G_z
    H_x = mod.linear_mod.H_x
    H_z = mod.linear_mod.H_z

    # Initialize output array
    UX = np.zeros((Nt, N_ux))

    # Split initial ux into u and x
    u_init, x_init = np.split(ux_init, np.array([N_u]))

    # Set initial values
    UX[0, :] = ux_init

    # Forward iteration starting from initial conditions
    x_t = x_init.copy()

    for tt in range(1, Nt):
        # Compute deviations from steady state
        x_hat = x_t - x_ss
        z_hat = Z_path[tt, :] - z_ss

        # Apply linear transition and policy functions
        x_new_hat = H_x @ x_hat + H_z @ z_hat
        u_hat = G_x @ x_hat + G_z @ z_hat

        # Convert back to levels
        x_new = x_ss + x_new_hat
        u_new = u_ss + u_hat

        # Store in UX
        UX[tt, :N_u] = u_new
        UX[tt, N_u:] = x_new

        # Update state for next iteration
        x_t = x_new

    return UX
