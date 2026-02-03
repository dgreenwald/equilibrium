"""Derivative computation utilities for model functions.

This module provides utilities for computing and managing derivatives of model
functions using JAX's automatic differentiation.
"""

import logging

import jax
from jax import numpy as jnp

logger = logging.getLogger(__name__)


class DerivativeResult:
    """
    Container for joint derivative computation with convenient access.

    Allows indexing by variable name (str) or argnum (int).
    User is responsible for ensuring derivatives match current arguments.

    Attributes
    ----------
    _jac_tuple : tuple of jax.Array
        Tuple of Jacobian matrices from jax.jacfwd
    _argnum_to_idx : dict
        Maps JAX argument number to tuple index
    _var_to_argnum : dict
        Maps variable name to JAX argument number

    Examples
    --------
    >>> derivs = mod.d_all("transition", u, x, z, params)
    >>> d_u = derivs["u"]  # Access by variable name
    >>> d_x = derivs["x"]
    >>> # Or access by argnum:
    >>> d_0 = derivs[0]
    >>> # Get as hstack:
    >>> d_ux = derivs.as_hstack(["u", "x"])
    """

    def __init__(self, jac_tuple, argnums_list, var_to_argnum):
        """
        Initialize derivative result container.

        Parameters
        ----------
        jac_tuple : tuple of jax.Array
            Tuple of Jacobian matrices
        argnums_list : list of int
            List of argument numbers (e.g., [0, 1, 2, 3])
        var_to_argnum : dict
            Maps variable name to argument number
        """
        self._jac_tuple = jac_tuple
        self._argnum_to_idx = {argnum: idx for idx, argnum in enumerate(argnums_list)}
        self._var_to_argnum = var_to_argnum

    def __getitem__(self, key):
        """
        Access derivative by variable name (str) or argnum (int).

        Parameters
        ----------
        key : str or int
            Variable name (e.g., "u", "x") or JAX argument number

        Returns
        -------
        jax.Array
            Jacobian matrix for the specified variable/argument
        """
        if isinstance(key, str):
            argnum = self._var_to_argnum[key]
        else:
            argnum = key
        idx = self._argnum_to_idx[argnum]
        return self._jac_tuple[idx]

    def as_tuple(self):
        """
        Return raw tuple of Jacobians.

        Returns
        -------
        tuple of jax.Array
            Raw Jacobian tuple
        """
        return self._jac_tuple

    def as_hstack(self, vars=None):
        """
        Return as horizontally stacked array.

        Parameters
        ----------
        vars : list of str, optional
            Variable names to include. If None, includes all in order.

        Returns
        -------
        jax.Array
            Horizontally stacked Jacobian matrix

        Examples
        --------
        >>> derivs.as_hstack(["u", "x"])  # Only u and x
        >>> derivs.as_hstack()  # All variables
        """
        if vars is None:
            return jnp.hstack(self._jac_tuple)
        return jnp.hstack([self[v] for v in vars])


def trace_args(name, *args):
    """
    Log diagnostic information about function arguments for debugging.

    Parameters
    ----------
    name : str
        Name of the function being traced.
    *args : array_like
        Variable number of arguments to inspect.
    """
    logger.debug("%s inputs:", name)
    for i, a in enumerate(args):
        logger.debug(
            "  arg%s: shape=%s, dtype=%s, type=%s",
            i,
            getattr(a, "shape", None),
            getattr(a, "dtype", None),
            type(a),
        )


def standardize_args(*args):
    """
    Convert arguments to JAX arrays with float64 dtype.

    Parameters
    ----------
    *args : array_like
        Variable number of arguments to standardize.

    Returns
    -------
    list[jax.Array]
        List of arguments converted to JAX arrays with float64 dtype.

    Raises
    ------
    AssertionError
        If any converted argument is not a JAX Array.
    """
    std = [jnp.array(arg, dtype=jnp.float64) for arg in args]
    for i, a in enumerate(std):
        assert isinstance(a, jax.Array), f"Arg {i} is not a jax.Array: {type(a)}"
    return std
