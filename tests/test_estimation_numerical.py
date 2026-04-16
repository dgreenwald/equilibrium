"""Tests for estimation/numerical.py vendored helpers."""

import numpy as np
import pytest

from equilibrium.estimation import numerical as nm


def test_hessian_quadratic():
    # f(x, y) = x^2 + 2*y^2 → H = diag(2, 4)
    f = lambda x: x[0] ** 2 + 2.0 * x[1] ** 2
    H = nm.hessian(f, np.array([1.0, 1.0]))
    expected = np.diag([2.0, 4.0])
    np.testing.assert_allclose(H, expected, atol=1e-6)


def test_hessian_cross_term():
    # f(x, y) = x*y → H = [[0, 1], [1, 0]]
    f = lambda x: x[0] * x[1]
    H = nm.hessian(f, np.array([1.0, 2.0]))
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(H, expected, atol=1e-6)


def test_bound_transform_both_bounds_roundtrip():
    vals = np.array([0.3, 0.7])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    unbounded = nm.bound_transform(vals, lb, ub, to_bdd=False)
    back = nm.bound_transform(unbounded, lb, ub, to_bdd=True)
    np.testing.assert_allclose(back, vals, atol=1e-12)


def test_bound_transform_lower_only_roundtrip():
    vals = np.array([0.5, 2.0])
    lb = np.array([0.0, 1.0])
    ub = np.array([np.inf, np.inf])
    unbounded = nm.bound_transform(vals, lb, ub, to_bdd=False)
    back = nm.bound_transform(unbounded, lb, ub, to_bdd=True)
    np.testing.assert_allclose(back, vals, atol=1e-12)


def test_bound_transform_upper_only_roundtrip():
    vals = np.array([-0.5, -2.0])
    lb = np.array([-np.inf, -np.inf])
    ub = np.array([0.0, 0.0])
    unbounded = nm.bound_transform(vals, lb, ub, to_bdd=False)
    back = nm.bound_transform(unbounded, lb, ub, to_bdd=True)
    np.testing.assert_allclose(back, vals, atol=1e-12)


def test_bound_transform_unbounded_passthrough():
    vals = np.array([3.0, -1.0])
    lb = np.array([-np.inf, -np.inf])
    ub = np.array([np.inf, np.inf])
    result = nm.bound_transform(vals, lb, ub, to_bdd=True)
    np.testing.assert_array_equal(result, vals)


def test_robust_cholesky_psd():
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = nm.robust_cholesky(A)
    np.testing.assert_allclose(L @ L.T, A, atol=1e-10)


def test_robust_cholesky_clips_small_eigenvalue():
    # Near-singular matrix; should not raise
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])
    L = nm.robust_cholesky(A, min_eig=1e-8)
    result = L @ L.T
    # Result must be PSD
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals >= 0.0)


def test_rsolve():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    A = A @ A.T + np.eye(4)  # make it positive definite
    b = rng.standard_normal((3, 4))
    x = nm.rsolve(b, A)
    np.testing.assert_allclose(x @ A, b, atol=1e-10)
