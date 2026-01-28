#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:13:04 2022

@author: dan
"""

# import jax
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# from py_tools.utilities import tic, toc

logger = logging.getLogger(__name__)

# Valid algorithm options for the solve function
ALGORITHM_LBJ = "LBJ"
ALGORITHM_SPARSE = "sparse"
VALID_ALGORITHMS = (ALGORITHM_LBJ, ALGORITHM_SPARSE)


def compute_time_period(
    tt, Nt, UX, Z, ux_init, mod, params, N_ux, compute_grad, terminal_condition
):
    """
    Compute objective function values and gradients for a single time period.

    Parameters
    ----------
    tt : int
        Current time index.
    Nt : int
        Total number of time periods.
    UX : array-like, shape (Nt, N_ux)
        Control and state variables at all time points.
    Z : array-like, shape (Nt, N_z)
        Exogenous variables at all time points.
    ux_init : array-like, shape (N_ux,)
        Initial control and state variables.
    mod : object
        Instance of a model.
    params : array-like
        Model parameters.
    N_ux : int
        Number of control and state variables.
    compute_grad : bool
        Whether to compute gradient matrices.
    terminal_condition : str
        Terminal condition of objective function, either 'stable' or 'steady'.

    Returns
    -------
    f_t : array-like, shape (N_ux, 1)
        Objective function values at time tt.
    L_t : array-like, shape (N_ux, N_ux) or None
        Gradient matrix L at time tt, or None if compute_grad is False.
    C_t : array-like, shape (N_ux, N_ux) or None
        Gradient matrix C at time tt, or None if compute_grad is False.
    F_t : array-like, shape (N_ux, N_ux) or None
        Gradient matrix F at time tt, or None if compute_grad is False.
    """
    ux_t = UX[tt, :]
    u_t, x_t = np.split(ux_t, np.array((mod.N["u"],)))
    z_t = Z[tt, :]

    if tt > 0:

        u_lag, x_lag = np.split(UX[tt - 1, :], np.array((mod.N["u"],)))
        z_lag = Z[tt - 1, :]

        # Transition equation
        x_star = mod.fcn("transition", u_lag, x_lag, z_lag, params)

    if tt < Nt - 1:

        u_next, x_next = np.split(UX[tt + 1, :], np.array((mod.N["u"],)))
        z_next = Z[tt + 1, :]

    if tt == 0:

        f_t = (ux_t - ux_init)[:, np.newaxis]

        if compute_grad:
            L_t = np.zeros((N_ux, N_ux))
            C_t = np.eye(N_ux)
            F_t = np.zeros((N_ux, N_ux))

    elif tt == Nt - 1:

        if terminal_condition == "stable":

            x_hat = x_t - mod.steady_components["x"]
            z_hat = z_t - mod.steady_components["z"]

            u_star = (
                mod.steady_components["u"]
                + mod.linear_mod.G_x @ x_hat
                + mod.linear_mod.G_z @ z_hat
            )

            f_t = np.vstack(
                (
                    (u_t - u_star)[:, np.newaxis],
                    (x_t - x_star)[:, np.newaxis],
                )
            )

            if compute_grad:

                trans_derivs = mod.d_all("transition", u_lag, x_lag, z_lag, params)

                L_t = np.vstack(
                    (
                        np.zeros((mod.N["u"], N_ux)),
                        -trans_derivs.as_hstack(["u", "x"]),
                    )
                )

                C_t = np.vstack(
                    (
                        np.hstack((np.eye(mod.N["u"]), -mod.linear_mod.G_x)),
                        np.hstack(
                            (np.zeros((mod.N["x"], mod.N["u"])), np.eye(mod.N["x"]))
                        ),
                    )
                )

                F_t = np.zeros((N_ux, N_ux))

        elif terminal_condition == "steady":

            f_t = np.vstack(
                (
                    (u_t - mod.steady_components["u"])[:, np.newaxis],
                    (x_t - mod.steady_components["x"])[:, np.newaxis],
                )
            )

            if compute_grad:
                L_t = np.zeros((N_ux, N_ux))
                C_t = np.eye(N_ux)
                F_t = np.zeros((N_ux, N_ux))

    else:

        E = mod.fcn("expectations", u_t, x_t, z_t, u_next, x_next, z_next, params)

        f_t = np.vstack(
            (
                mod.fcn("optimality", u_t, x_t, z_t, E, params)[:, np.newaxis],
                (x_t - x_star)[:, np.newaxis],
            )
        )

        if compute_grad:

            # d_opt_d_E = mod.jacobians['optimality']['E'](u_t, x_t, z_t, E, params)
            d_opt_d_E = mod.d("optimality", "E", u_t, x_t, z_t, E, params)

            trans_derivs = mod.d_all("transition", u_lag, x_lag, z_lag, params)

            L_t = np.vstack(
                (
                    np.zeros((mod.N["u"], N_ux)),
                    -trans_derivs.as_hstack(["u", "x"]),
                )
            )

            opt_derivs = mod.d_all("optimality", u_t, x_t, z_t, E, params)
            exp_derivs = mod.d_all(
                "expectations", u_t, x_t, z_t, u_next, x_next, z_next, params
            )

            C_t = np.vstack(
                (
                    np.hstack(
                        [
                            opt_derivs["u"] + d_opt_d_E @ exp_derivs["u"],
                            opt_derivs["x"] + d_opt_d_E @ exp_derivs["x"],
                        ]
                    ),
                    np.hstack(
                        (
                            np.zeros((mod.N["x"], mod.N["u"])),
                            np.eye(mod.N["x"]),
                        )
                    ),
                )
            )

            F_t = np.vstack(
                (
                    np.hstack(
                        [
                            d_opt_d_E @ exp_derivs["u_new"],
                            d_opt_d_E @ exp_derivs["x_new"],
                        ]
                    ),
                    np.zeros((mod.N["x"], N_ux)),
                )
            )

    if not compute_grad:
        L_t = None
        C_t = None
        F_t = None

    return f_t, L_t, C_t, F_t


# @jax.jit
def objfcn(mod, UX, Z, ux_init, compute_grad=True, terminal_condition="stable"):
    """
    Compute objective function values and gradients for a model.

    Parameters
    ----------
    mod : object
        Instance of a model.
    UX : array-like, shape (Nt, N_ux)
        Control and state variables at all time points.
    Z : array-like, shape (Nt, N_z)
        Exogenous variables at all time points.
    ux_init : array-like, shape (N_ux,)
        Initial control and state variables.
    compute_grad : bool, optional (default=True)
        Whether to compute gradient of objective function.
    terminal_condition : str, optional (default='stable')
        Terminal condition of objective function, either 'stable' or 'steady'.

    Returns
    -------
    f_all : array-like, shape (Nt, N_ux)
        Objective function values at all time points.
    step : array-like, shape (Nt, N_ux), or None
        Size of step to take for Newton search, if
        `compute_grad` is True.
    """

    # Initialize values
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])
    # steady_vals = mod.initialize_values(mod.steady_dict)

    Nt, N_ux = UX.shape
    f_all = np.zeros((Nt, N_ux))

    if compute_grad:

        M_all = np.zeros((Nt, N_ux, N_ux))
        d_all = np.zeros((Nt, N_ux))

        M_t = np.zeros((N_ux, N_ux))
        d_t = np.zeros((N_ux, 1))

        step = np.zeros((Nt, N_ux))

    else:

        step = None

    for tt in range(Nt):

        f_t, L_t, C_t, F_t = compute_time_period(
            tt, Nt, UX, Z, ux_init, mod, params, N_ux, compute_grad, terminal_condition
        )

        f_all[tt, :] = f_t.ravel()

        if compute_grad:

            CLM_inv = np.linalg.inv(C_t - L_t @ M_t)
            M_t = CLM_inv @ F_t
            d_t = -CLM_inv @ (f_t + L_t @ d_t)

            M_all[tt, :, :] = M_t
            d_all[tt, :] = d_t.ravel()

    # Backwards pass
    if compute_grad:
        dy = np.zeros(N_ux)
        for tt in range(Nt - 1, -1, -1):

            dy = d_all[tt, :] - M_all[tt, :, :] @ dy
            step[tt, :] = dy

    return f_all, step


def compute_jacobian_blocks(
    mod, UX, Z, ux_init, terminal_condition="stable", compute_grad=True
):
    """
    Compute residual vector and Jacobian blocks for all time periods.

    Parameters
    ----------
    mod : object
        Instance of a model.
    UX : array-like, shape (Nt, N_ux)
        Control and state variables at all time points.
    Z : array-like, shape (Nt, N_z)
        Exogenous variables at all time points.
    ux_init : array-like, shape (N_ux,)
        Initial control and state variables.
    terminal_condition : str, optional (default='stable')
        Terminal condition of objective function, either 'stable' or 'steady'.
    compute_grad : bool, optional (default=True)
        Whether to compute gradient matrices. If False, only residuals are computed
        and L_all, C_all, F_all are returned as None.

    Returns
    -------
    f_all : array-like, shape (Nt, N_ux)
        Residual values at all time points.
    L_all : array-like, shape (Nt, N_ux, N_ux) or None
        Sub-diagonal Jacobian blocks (L_t), or None if compute_grad is False.
    C_all : array-like, shape (Nt, N_ux, N_ux) or None
        Diagonal Jacobian blocks (C_t), or None if compute_grad is False.
    F_all : array-like, shape (Nt, N_ux, N_ux) or None
        Super-diagonal Jacobian blocks (F_t), or None if compute_grad is False.
    """
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])
    Nt, N_ux = UX.shape

    f_all = np.zeros((Nt, N_ux))

    if compute_grad:
        L_all = np.zeros((Nt, N_ux, N_ux))
        C_all = np.zeros((Nt, N_ux, N_ux))
        F_all = np.zeros((Nt, N_ux, N_ux))
    else:
        L_all = None
        C_all = None
        F_all = None

    for tt in range(Nt):
        f_t, L_t, C_t, F_t = compute_time_period(
            tt, Nt, UX, Z, ux_init, mod, params, N_ux, compute_grad, terminal_condition
        )
        f_all[tt, :] = f_t.ravel()
        if compute_grad:
            L_all[tt, :, :] = L_t
            C_all[tt, :, :] = C_t
            F_all[tt, :, :] = F_t

    return f_all, L_all, C_all, F_all


def solve_lbj_step(f_all, L_all, C_all, F_all):
    """
    Compute Newton step using L-B-J algorithm.

    Parameters
    ----------
    f_all : array-like, shape (Nt, N_ux)
        Residual values at all time points.
    L_all : array-like, shape (Nt, N_ux, N_ux)
        Sub-diagonal Jacobian blocks.
    C_all : array-like, shape (Nt, N_ux, N_ux)
        Diagonal Jacobian blocks.
    F_all : array-like, shape (Nt, N_ux, N_ux)
        Super-diagonal Jacobian blocks.

    Returns
    -------
    step : array-like, shape (Nt, N_ux)
        Newton step.
    """
    Nt, N_ux = f_all.shape

    M_all = np.zeros((Nt, N_ux, N_ux))
    d_all = np.zeros((Nt, N_ux))

    M_t = np.zeros((N_ux, N_ux))
    d_t = np.zeros((N_ux, 1))

    # Forward pass
    for tt in range(Nt):
        L_t = L_all[tt, :, :]
        C_t = C_all[tt, :, :]
        F_t = F_all[tt, :, :]
        f_t = f_all[tt, :][:, np.newaxis]

        CLM_inv = np.linalg.inv(C_t - L_t @ M_t)
        M_t = CLM_inv @ F_t
        d_t = -CLM_inv @ (f_t + L_t @ d_t)

        M_all[tt, :, :] = M_t
        d_all[tt, :] = d_t.ravel()

    # Backward pass
    step = np.zeros((Nt, N_ux))
    dy = np.zeros(N_ux)
    for tt in range(Nt - 1, -1, -1):
        dy = d_all[tt, :] - M_all[tt, :, :] @ dy
        step[tt, :] = dy

    return step


def solve_sparse_step(f_all, L_all, C_all, F_all):
    """
    Compute Newton step using sparse matrix solver.

    Builds the block tri-diagonal Jacobian matrix from L_all, C_all, F_all
    and solves the linear system J @ step = -f using a sparse solver.

    Parameters
    ----------
    f_all : array-like, shape (Nt, N_ux)
        Residual values at all time points.
    L_all : array-like, shape (Nt, N_ux, N_ux)
        Sub-diagonal Jacobian blocks.
    C_all : array-like, shape (Nt, N_ux, N_ux)
        Diagonal Jacobian blocks.
    F_all : array-like, shape (Nt, N_ux, N_ux)
        Super-diagonal Jacobian blocks.

    Returns
    -------
    step : array-like, shape (Nt, N_ux)
        Newton step.
    """
    Nt, N_ux = f_all.shape

    # Build block tri-diagonal matrix using scipy.sparse.bmat
    # This is more efficient than explicit COO construction
    blocks = [[None] * Nt for _ in range(Nt)]

    for tt in range(Nt):
        # Diagonal block
        blocks[tt][tt] = sp.csc_matrix(C_all[tt, :, :])

        # Sub-diagonal block (L_t affects row tt, column tt-1)
        if tt > 0:
            blocks[tt][tt - 1] = sp.csc_matrix(L_all[tt, :, :])

        # Super-diagonal block (F_t affects row tt, column tt+1)
        if tt < Nt - 1:
            blocks[tt][tt + 1] = sp.csc_matrix(F_all[tt, :, :])

    # Construct sparse block matrix directly in CSC format
    J_sparse = sp.bmat(blocks, format="csc")

    # Flatten residuals and solve
    f_flat = f_all.ravel()
    step_flat = spla.spsolve(J_sparse, -f_flat)

    # Reshape step back to (Nt, N_ux)
    step = step_flat.reshape((Nt, N_ux))

    return step


def solve(
    mod,
    Z,
    ux_init=None,
    y_init=None,
    terminal_condition="stable",
    tol=1e-8,
    guess_method="linear",
    algorithm=ALGORITHM_LBJ,
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "npz",
):
    """
    Solve a nonlinear perfect foresight path using a model and a set of input parameters.

    Parameters
    ----------
    mod : Model
        The model being used.
    Z : array-like, shape (Nt, N_z)
        The exogenous state variables.
    ux_init : array-like, shape (N_ux,), optional
        The initial control and state variables. If None, defaults to steady state.
    y_init : array-like, shape (N_y,), optional
        Initial values for intermediate variables. When provided, overrides
        Y[0] in the returned DeterministicResult.
    terminal_condition : str, optional
        The terminal condition of the model ('stable' by default).
    tol : float, optional
        The tolerance level for the solution (1e-8 by default).
    guess_method : str, optional
        Method for initializing UX_guess. Must be 'linear' or 'constant'.
        Default is 'linear'.
        - 'constant': Tile ux_init to form UX_guess.
        - 'linear': Use solvers.linear.solve_linear_path to create UX_guess
          with a linear approximation.
    algorithm : str, optional
        Algorithm for solving the linear system. Must be 'LBJ' or 'sparse'.
        Default is 'LBJ'.
        - 'LBJ': Use the L-B-J recursive algorithm (memory efficient).
        - 'sparse': Build sparse Jacobian and use sparse direct solver.
    save_path : str or Path, optional
        If provided, save results to this path.
    save_format : str, optional
        Format for saving ('npz' or 'json'). Default is 'npz'.

    Returns
    -------
    result : DeterministicResult
        Result container with UX, Z, and metadata.

    Notes
    -----
    Uses objfcn to calculate the objective function and its gradient for the
    optimization problem.
    """
    from .results import DeterministicResult

    if algorithm == ALGORITHM_LBJ:
        UX, final_residual = _solve_lbj_internal(
            mod, Z, ux_init, terminal_condition, tol, guess_method
        )
    elif algorithm == ALGORITHM_SPARSE:
        UX, final_residual = _solve_sparse_internal(
            mod, Z, ux_init, terminal_condition, tol, guess_method
        )
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Must be one of {VALID_ALGORITHMS}."
        )

    # Get variable names from model
    var_names = (
        mod.var_lists.get("u", []) + mod.var_lists.get("x", [])
        if hasattr(mod, "var_lists") and mod.var_lists
        else []
    )
    exog_names = mod.exog_list if hasattr(mod, "exog_list") else []
    model_label = getattr(mod, "label", "_default")

    # Compute intermediate variables using mod.intermediates()
    y_names = mod.var_lists.get("intermediate", []) if hasattr(mod, "var_lists") else []
    Y = _compute_intermediate_variables(mod, UX, Z)

    if y_init is not None:
        if Y is None:
            raise ValueError("y_init provided but model has no intermediate variables.")
        y_init = np.asarray(y_init)
        if y_init.shape[0] != Y.shape[1]:
            raise ValueError(
                "y_init length does not match number of intermediate variables."
            )
        Y[0, :] = y_init

    result = DeterministicResult(
        UX=UX,
        Z=np.asarray(Z),
        Y=Y,
        model_label=model_label,
        var_names=var_names,
        exog_names=exog_names,
        y_names=y_names,
        terminal_condition=terminal_condition,
        converged=final_residual <= tol,
        final_residual=final_residual,
    )

    if save_path is not None:
        result.save(save_path, format=save_format, overwrite=True)

    return result


def _compute_intermediate_variables(mod, UX, Z):
    """
    Compute intermediate variables for all time periods.

    Parameters
    ----------
    mod : Model
        The model instance with intermediates() method.
    UX : np.ndarray, shape (Nt, N_ux)
        Control and state variables at all time points.
    Z : np.ndarray, shape (Nt, N_z)
        Exogenous variables at all time points.

    Returns
    -------
    Y : np.ndarray, shape (Nt, N_y)
        Intermediate variables at all time points.
    """
    if not hasattr(mod, "intermediates"):
        return None

    if not hasattr(mod, "var_lists") or not mod.var_lists:
        return None

    y_names = mod.var_lists.get("intermediate", [])
    if not y_names:
        return None

    Nt = UX.shape[0]
    N_y = len(y_names)
    Y = np.zeros((Nt, N_y))

    # Get parameters array
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])

    for tt in range(Nt):
        u_t, x_t = np.split(UX[tt, :], np.array((mod.N["u"],)))
        z_t = Z[tt, :]

        # Call mod.intermediates to get intermediate values for this time period
        y_t = mod.intermediates(u_t, x_t, z_t, params)
        Y[tt, :] = np.asarray(y_t)

    return Y


def _compute_initial_intermediates(mod, ux_init, z_init):
    """
    Compute intermediate variables at the initial period.

    Parameters
    ----------
    mod : Model
        The model instance with intermediates() method.
    ux_init : np.ndarray, shape (N_ux,)
        Initial control and state variables.
    z_init : np.ndarray, shape (N_z,)
        Initial exogenous variables.

    Returns
    -------
    y_init : np.ndarray or None
        Intermediate variables at the initial period, or None if unavailable.
    """
    if not hasattr(mod, "intermediates"):
        return None
    if not hasattr(mod, "var_lists") or not mod.var_lists:
        return None
    y_names = mod.var_lists.get("intermediate", [])
    if not y_names:
        return None

    u_init, x_init = np.split(ux_init, np.array((mod.N["u"],)))
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])
    return mod.intermediates(u_init, x_init, z_init, params)


def _solve_lbj_internal(
    mod,
    Z,
    ux_init=None,
    terminal_condition="stable",
    tol=1e-8,
    guess_method="linear",
):
    """
    Internal solve_lbj that returns (UX, final_residual) tuple.
    """
    from equilibrium.solvers.linear import solve_linear_path

    Nt = Z.shape[0]

    # Set default ux_init to steady state if not provided
    if ux_init is None:
        ux_init = np.concatenate(
            [mod.steady_components["u"], mod.steady_components["x"]]
        )
    else:
        ux_init = np.asarray(ux_init)

    # Build UX_guess based on guess_method
    if guess_method == "constant":
        UX_guess = np.tile(ux_init, (Nt, 1))
    elif guess_method == "linear":
        UX_guess = solve_linear_path(mod, Z, ux_init)
    else:
        raise ValueError(
            f"Unknown guess_method: {guess_method}. Must be 'linear' or 'constant'."
        )

    UX = UX_guess.copy()

    f_all, _ = objfcn(
        mod, UX, Z, ux_init, compute_grad=False, terminal_condition=terminal_condition
    )
    dist = np.sqrt(np.mean(f_all**2))
    logger.info("Initial deterministic residual: %g", dist)

    while dist > tol:

        _, step = objfcn(
            mod,
            UX,
            Z,
            ux_init,
            compute_grad=True,
            terminal_condition=terminal_condition,
        )
        dist_new = dist + 1.0

        while dist_new > dist:

            UX_new = UX + step
            f_all, _ = objfcn(
                mod,
                UX_new,
                Z,
                ux_init,
                compute_grad=False,
                terminal_condition=terminal_condition,
            )
            dist_new = np.sqrt(np.mean(f_all**2))
            step *= 0.5

        UX = UX_new.copy()
        dist = dist_new
        logger.info("Deterministic solver residual: %g", dist)

    return UX, dist


def _solve_sparse_internal(
    mod,
    Z,
    ux_init=None,
    terminal_condition="stable",
    tol=1e-8,
    guess_method="linear",
):
    """
    Internal solve_sparse that returns (UX, final_residual) tuple.
    """
    from equilibrium.solvers.linear import solve_linear_path

    Nt = Z.shape[0]

    # Set default ux_init to steady state if not provided
    if ux_init is None:
        ux_init = np.concatenate(
            [mod.steady_components["u"], mod.steady_components["x"]]
        )
    else:
        ux_init = np.asarray(ux_init)

    # Build UX_guess based on guess_method
    if guess_method == "constant":
        UX_guess = np.tile(ux_init, (Nt, 1))
    elif guess_method == "linear":
        UX_guess = solve_linear_path(mod, Z, ux_init)
    else:
        raise ValueError(
            f"Unknown guess_method: {guess_method}. Must be 'linear' or 'constant'."
        )

    UX = UX_guess.copy()

    # Compute initial residual and Jacobian blocks
    f_all, L_all, C_all, F_all = compute_jacobian_blocks(
        mod, UX, Z, ux_init, terminal_condition, compute_grad=True
    )
    dist = np.sqrt(np.mean(f_all**2))
    logger.info("Initial deterministic residual: %g", dist)

    while dist > tol:

        # Compute step using sparse solver
        step = solve_sparse_step(f_all, L_all, C_all, F_all)
        dist_new = dist + 1.0

        while dist_new > dist:

            UX_new = UX + step
            # Only compute residuals (not Jacobian) for backstepping
            f_all_new, _, _, _ = compute_jacobian_blocks(
                mod, UX_new, Z, ux_init, terminal_condition, compute_grad=False
            )
            dist_new = np.sqrt(np.mean(f_all_new**2))
            step *= 0.5

        UX = UX_new.copy()
        # Compute full Jacobian blocks for the next iteration
        f_all, L_all, C_all, F_all = compute_jacobian_blocks(
            mod, UX, Z, ux_init, terminal_condition, compute_grad=True
        )
        dist = dist_new
        logger.info("Deterministic solver residual: %g", dist)

    return UX, dist


def solve_sequence(
    det_spec,
    mod,
    Nt,
    z_init=None,
    ux_init=None,
    y_init=None,
    terminal_condition="stable",
    tol=1e-8,
    guess_method="linear",
    calibrate_initial=True,
    recalibrate_regimes=False,
    algorithm=ALGORITHM_LBJ,
    save_path: Optional[Union[str, Path]] = None,
    save_format: str = "npz",
    display_steady: bool = False,
    save_results: bool = True,
    copy_model: bool = False,
):
    """
    Solve a sequence of deterministic paths for multiple regimes.

    This function loops over regimes defined in det_spec, computing a
    deterministic path for each one. For each regime after the first,
    initial conditions are taken from the previous regime's solution
    at the transition time specified in det_spec.time_list.
    The initial state at t=0 is the baseline steady state; the first
    regime in det_spec is applied starting at t=1.

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
    terminal_condition : str, optional
        Terminal condition for the solver ('stable' or 'steady'). Default is 'stable'.
    tol : float, optional
        Tolerance level for the solver. Default is 1e-8.
    guess_method : str, optional
        Method for initializing UX_guess in solve(). Must be 'linear' or 'constant'.
        Default is 'linear'.
    calibrate_initial : bool, optional
        Whether to calibrate the model when computing the initial steady state.
        Default is True.
    recalibrate_regimes : bool, optional
        If True, apply the `calibrate_initial` flag to all regime steady-state solves.
        If False (default), only the initial steady state uses `calibrate_initial=True`;
        subsequent regimes with parameter changes use `calibrate=False` to avoid
        re-solving for calibrated parameters that may be preset in the regime.
        Default is False.
    algorithm : str, optional
        Algorithm for solving the linear system. Must be 'LBJ' or 'sparse'.
        Default is 'LBJ'.
        - 'LBJ': Use the L-B-J recursive algorithm (memory efficient).
        - 'sparse': Build sparse Jacobian and use sparse direct solver.
    save_path : str or Path, optional
        If provided and ``save_results`` is True, save results to this path.
    save_format : str, optional
        Format for saving ('npz' or 'json'). Default is 'npz'.
    display_steady : bool, optional
        Whether to display steady state results when solving for each regime.
        Default is False to reduce output when solving multiple regimes.
    save_results : bool, default True
        If True, save the resulting SequenceResult. When True and ``save_path``
        is None, uses the default path where label-based loaders expect results.
        Saving overwrites any existing file at the target path.
    copy_model : bool, default False
        If True, operate on a copy of ``mod`` so the original model is not
        mutated by steady-state or linearization updates.

    Returns
    -------
    SequenceResult
        Result container with DeterministicResult for each regime.
    """
    from .results import SequenceResult

    if det_spec.n_regimes == 0:
        raise ValueError("DetSpec must have at least one regime")

    # Optionally work on a copy so callers can reuse the original model without mutation.
    # update_copy preserves shared JAX bundles to avoid recompilation.
    base_mod = mod.update_copy() if copy_model else mod

    # Ensure the baseline model's steady state is solved
    # This is needed because when ux_init is None, we use steady_components
    if not base_mod._steady_solved:
        logger.info("Steady state not found for baseline model, solving...")
        base_mod.solve_steady(calibrate=calibrate_initial, display=display_steady)
    elif base_mod._calibrated != calibrate_initial:
        logger.warning(
            f"Steady state was previously solved with calibrate={base_mod._calibrated}, "
            f"but calibrate_initial={calibrate_initial} requested. Re-solving..."
        )
        base_mod.solve_steady(calibrate=calibrate_initial, display=display_steady)

    # Ensure linearization is done if needed for terminal condition
    if terminal_condition == "stable" and not base_mod._linearized:
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

    for regime in range(det_spec.n_regimes):
        # Get preset parameters for this regime.
        # preset_par_list[regime] contains parameter overrides for this regime.
        current_preset_par = det_spec.preset_par_list[regime]

        # Check if we need to recreate the Model
        # For first regime: only need new model if there are parameter overrides
        # For subsequent regimes: need new model if params changed from previous
        has_param_overrides = bool(current_preset_par) and len(current_preset_par) > 0
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

        # Build exogenous paths for this regime
        Z_path = det_spec.build_exog_paths(
            current_mod,
            Nt,
            regime=regime,
            z_init=current_z_init,
        )

        ux_init_regime = np.asarray(current_ux_init)
        y_init_regime = None
        if regime == 0:
            if y_init is not None:
                y_init_regime = np.asarray(y_init)
            elif ux_init is None and det_spec.ux_init is None:
                if current_z_init is None:
                    z_init_for_y = np.zeros(current_mod.N["z"])
                else:
                    z_init_for_y = np.asarray(current_z_init)
                y_init_regime = _compute_initial_intermediates(
                    base_mod, ux_init_regime, z_init_for_y
                )
            elif hasattr(current_mod, "var_lists"):
                y_names = current_mod.var_lists.get("intermediate", [])
                if y_names:
                    y_init_regime = np.full(len(y_names), np.nan)

        # Solve for this regime - returns DeterministicResult
        result = solve(
            current_mod,
            Z_path,
            ux_init_regime,
            y_init=y_init_regime,
            terminal_condition=terminal_condition,
            tol=tol,
            guess_method=guess_method,
            algorithm=algorithm,
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
            current_ux_init = result.UX[transition_time, :]
            start_time = det_spec.time_list[regime] + 1

        # Remember current params for comparison
        prev_preset_par = current_preset_par

    model_label = getattr(mod, "label", "_default")
    experiment_label = getattr(det_spec, "label", "_default")
    sequence_result = SequenceResult(
        regimes=regime_results,
        time_list=list(det_spec.time_list),
        model_label=model_label,
        experiment_label=experiment_label,
    )

    if save_results:
        if save_path is None:
            sequence_result.save(format=save_format, overwrite=True)
        else:
            sequence_result.save(save_path, format=save_format, overwrite=True)

    return sequence_result
