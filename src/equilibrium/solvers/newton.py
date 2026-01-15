#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 19:02:11 2025

@author: dan
"""

import logging
from types import SimpleNamespace

from jax import numpy as np

logger = logging.getLogger(__name__)


def gradient(f, x, args=None, kwargs=None, step=1e-5, two_sided=True, f_val=None):
    """
    Compute numerical gradient using finite differences.

    Parameters
    ----------
    f : callable
        Function to differentiate.
    x : array_like
        Point at which to evaluate gradient.
    args : tuple, optional
        Additional positional arguments to f.
    kwargs : dict, optional
        Additional keyword arguments to f.
    step : float, optional
        Finite difference step size. Default is 1e-5.
    two_sided : bool, optional
        If True, use centered differences. Default is True.
    f_val : array_like, optional
        Pre-computed function value at x (used when two_sided=False).

    Returns
    -------
    array_like
        Gradient matrix with shape (len(x), len(f(x))).
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    if (not two_sided) and (f_val is None):
        f_val = f(x, *args, **kwargs)

    grad = None
    for ii in range(len(x)):

        x[ii] += step
        f_hi = f(x, *args, **kwargs)

        if two_sided:

            x[ii] -= 2.0 * step
            f_lo = f(x, *args, **kwargs)
            x[ii] += step

            df_i = np.array(f_hi - f_lo) / (2.0 * step)

        else:

            x[ii] -= step

            df_i = np.array(f_hi - f_val) / step

        if grad is None:

            if df_i.shape == ():
                ncols = 1
            else:
                ncols = len(df_i)

            grad = np.zeros((len(x), ncols))

        grad[ii, :] = df_i

    return grad


def root(
    fun,
    x0,
    args=None,
    kwargs=None,
    grad=None,
    tol=1e-8,
    gradient_kwargs=None,
    max_iterations=50,
    max_backstep_iterations=20,
    verbose=True,
    jac=None,
):
    """
    Find root of a function using Newton's method with backstepping.

    Parameters
    ----------
    fun : callable
        Function for which to find root. Should accept x as first argument.
    x0 : array_like
        Initial guess for root.
    args : tuple, optional
        Additional positional arguments to fun.
    kwargs : dict, optional
        Additional keyword arguments to fun.
    grad : callable, optional
        Function to compute Jacobian. If None, uses finite differences.
    tol : float, optional
        Convergence tolerance for norm of function value. Default is 1e-8.
    gradient_kwargs : dict, optional
        Additional keyword arguments for gradient computation.
    max_iterations : int, optional
        Maximum number of Newton iterations. Default is 50.
    max_backstep_iterations : int, optional
        Maximum number of line search iterations per Newton step. Default is 20.
    verbose : bool, optional
        If True, print iteration progress. Default is True.
    jac : callable, optional
        Alternative name for grad parameter (for scipy compatibility).

    Returns
    -------
    SimpleNamespace
        Result object with attributes:
        - success : bool, whether root was found
        - x : array_like, solution if successful
        - f_val : array_like, function value at solution
        - dist : float, norm of f_val at solution
        - failure_cause : str, reason for failure if not successful
    """
    if grad is None:
        if jac is not None:
            grad = jac

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if gradient_kwargs is None:
        gradient_kwargs = {}

    # Initialization
    x = np.array(x0).copy()
    f_val = fun(x, *args, **kwargs)
    dist = np.linalg.norm(f_val)
    res = {}

    iteration = 0

    if verbose:
        logger.info("Iteration %d: |f| = %g", iteration, dist)

    while (dist > tol) and (iteration <= max_iterations):

        iteration += 1

        # Get Jacobian
        if grad is None:
            grad_val = gradient(
                fun, x, args=args, kwargs=kwargs, f_val=f_val, **gradient_kwargs
            )
        else:
            grad_val = grad(x, *args, **kwargs)

        # Use Jacobian to compute step size
        step = -np.linalg.solve(grad_val.T, f_val)

        # Move in step direction with backstepping line search
        backstep_iteration = 0
        dist_new = dist + 1.0
        while ((dist_new > dist) or (np.isnan(dist_new))) and (
            backstep_iteration <= max_backstep_iterations
        ):
            backstep_iteration += 1
            x_new = x + step
            f_val_new = fun(x_new, *args, **kwargs)
            dist_new = np.linalg.norm(f_val_new)
            step *= 0.5

        if dist_new < dist:
            x = x_new
            f_val = f_val_new
            dist = dist_new
        else:
            res["success"] = False
            res["failure_cause"] = "max_backstep_iterations"
            return SimpleNamespace(**res)

        if verbose:
            logger.info("Iteration %d: |f| = %g", iteration, dist)

    if dist < tol:
        res["x"] = x
        res["f_val"] = f_val
        res["dist"] = dist
        res["success"] = True
    else:
        res["success"] = False
        res["failure_cause"] = "max_iterations"

    return SimpleNamespace(**res)
