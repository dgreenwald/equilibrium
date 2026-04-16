"""Vendored numerical helpers for MCMC estimation.

These four functions are inlined from py_tools to avoid an external dependency.
All are pure NumPy.
"""

import numpy as np
import scipy.linalg as sla


def hessian(f, x_in, eps=1e-4):
    """Estimate the Hessian of *f* at *x_in* by central finite differences.

    Parameters
    ----------
    f : callable
        Scalar-valued function accepting a 1-D array.
    x_in : numpy.ndarray
        Point at which the Hessian is evaluated.  1-D array of length ``n``.
    eps : float, optional
        Step size for finite differences, by default ``1e-4``.

    Returns
    -------
    numpy.ndarray
        Symmetric Hessian matrix of shape ``(n, n)``.
    """
    x = np.array(x_in, dtype=float, copy=True)
    n = len(x)
    H = np.empty((n, n))
    inv_4eps2 = 1.0 / (4.0 * eps * eps)

    for ii in range(n):
        for jj in range(ii, n):
            x[ii] += eps
            x[jj] += eps
            val = f(x)

            x[jj] -= 2.0 * eps
            if ii != jj:
                val -= f(x)
            else:
                val -= 2.0 * f(x)

            x[ii] -= 2.0 * eps
            val += f(x)

            x[jj] += 2.0 * eps
            if ii != jj:
                val -= f(x)

            x[ii] += eps
            x[jj] -= eps

            val *= inv_4eps2
            H[ii, jj] = val
            H[jj, ii] = val

    return H


def _logit(x, lb=0.0, ub=1.0):
    """Logit transform: map ``(lb, ub)`` to the real line."""
    return np.log(x - lb) - np.log(ub - x)


def _logistic(x, lb=0.0, ub=1.0):
    """Logistic transform: map the real line to ``(lb, ub)``."""
    return lb + (ub - lb) / (1.0 + np.exp(-x))


def bound_transform(vals, lb, ub, to_bdd=True):
    """Transform values between unbounded and bounded representations.

    Applies element-wise transformations based on which bounds are finite:

    * Both bounds finite: logistic / logit transform.
    * Lower bound only: exponential / log shift from *lb*.
    * Upper bound only: negative exponential / log shift from *ub*.

    Parameters
    ----------
    vals : numpy.ndarray
        Values to transform.
    lb : numpy.ndarray
        Element-wise lower bounds.  Use ``-numpy.inf`` for no lower bound.
    ub : numpy.ndarray
        Element-wise upper bounds.  Use ``numpy.inf`` for no upper bound.
    to_bdd : bool, optional
        If ``True`` (default), map unbounded values to the bounded domain.
        If ``False``, map bounded values to the unbounded domain.

    Returns
    -------
    numpy.ndarray
        Transformed values with the same shape as *vals*.
    """
    trans_vals = vals.copy()

    ix_lb = lb > -np.inf
    ix_ub = ub < np.inf

    ix_both = ix_lb & ix_ub
    ix_lb_only = ix_lb & (~ix_ub)
    ix_ub_only = (~ix_lb) & ix_ub

    if to_bdd:
        trans_vals[ix_both] = _logistic(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = lb[ix_lb_only] + np.exp(vals[ix_lb_only])
        trans_vals[ix_ub_only] = ub[ix_ub_only] - np.exp(vals[ix_ub_only])
    else:
        trans_vals[ix_both] = _logit(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = np.log(vals[ix_lb_only] - lb[ix_lb_only])
        trans_vals[ix_ub_only] = np.log(ub[ix_ub_only] - vals[ix_ub_only])

    return trans_vals


def robust_cholesky(A, min_eig=1e-12):
    """Compute a Cholesky-like factor of *A*, clipping small eigenvalues.

    Decomposes *A* via eigen-decomposition and clips eigenvalues to
    ``min_eig`` before constructing the square-root factor, so the result
    is well-defined even when *A* is only positive semi-definite.

    Uses ``scipy.linalg.eigh`` which exploits symmetry for ~2× speed
    and guarantees real eigenvalues.

    Parameters
    ----------
    A : numpy.ndarray
        Square symmetric matrix of shape ``(n, n)``.
    min_eig : float, optional
        Minimum eigenvalue threshold, by default ``1e-12``.

    Returns
    -------
    numpy.ndarray
        Matrix ``L`` of shape ``(n, n)`` such that ``L @ L.T`` is a
        positive semi-definite approximation to *A*.
    """
    vals, vecs = sla.eigh(A)
    np.maximum(vals, min_eig, out=vals)
    np.sqrt(vals, out=vals)
    return vecs * vals[np.newaxis, :]


def rsolve(b, A):
    """Solve the right-hand linear system ``b @ inv(A)``.

    Equivalent to solving ``A.T @ x.T = b.T`` and transposing.

    Parameters
    ----------
    b : numpy.ndarray
        Right-hand side array of shape ``(m, n)``.
    A : numpy.ndarray
        Square coefficient matrix of shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Solution array of shape ``(m, n)``.
    """
    return sla.solve(A.T, b.T).T
