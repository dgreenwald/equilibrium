"""Core contract tests for stateless JAX approximation evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equilibrium.approx import (
    evaluate_bases_jax,
    evaluate_jax,
    make_jax_data,
    make_smolyak_chebyshev,
    make_tensor_uniform_hat,
)


@pytest.fixture
def chebyshev_function():
    return make_smolyak_chebyshev(
        dimension=2,
        max_levels=(2, 2),
        max_total_level=2,
        lb=np.array([-2.0, 1.0]),
        ub=np.array([3.0, 5.0]),
    )


def test_basis_evaluation_matches_numpy(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    points = np.array([[0.2, 2.0], [1.5, 4.0]])
    canonical_points = chebyshev_function._transform_to_grid(points)

    expected = chebyshev_function.scheme.evaluate_bases(canonical_points)
    actual = evaluate_bases_jax(data, points)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jitted_vector_evaluation_uses_explicit_coefficients(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    points = jnp.array([[0.2, 2.0], [1.5, 4.0]])
    coefficients = jnp.arange(
        chebyshev_function.get_n_points() * 2, dtype=jnp.float64
    ).reshape(chebyshev_function.get_n_points(), 2)

    compiled = jax.jit(evaluate_jax)
    actual = compiled(data, coefficients, points)
    expected = evaluate_bases_jax(data, points) @ coefficients

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual.shape == (2, 2)
    assert chebyshev_function.coefficients is None


def test_single_point_scalar_shape(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    coefficients = jnp.ones(chebyshev_function.get_n_points())

    result = evaluate_jax(data, coefficients, jnp.array([0.2, 2.0]))

    assert result.shape == ()


def test_evaluation_is_differentiable_in_points_and_coefficients(
    chebyshev_function,
):
    data = make_jax_data(chebyshev_function)
    coefficients = jnp.arange(
        chebyshev_function.get_n_points(), dtype=jnp.float64
    )
    point = jnp.array([0.2, 2.0])

    point_gradient = jax.grad(evaluate_jax, argnums=2)(data, coefficients, point)
    coefficient_gradient = jax.grad(evaluate_jax, argnums=1)(
        data, coefficients, point
    )

    assert point_gradient.shape == (2,)
    assert coefficient_gradient.shape == coefficients.shape
    np.testing.assert_allclose(
        coefficient_gradient,
        evaluate_bases_jax(data, point),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("points", "match"),
    [
        (jnp.ones(3), r"points must have shape \(2,\)"),
        (jnp.ones((3, 1)), r"points must have shape \(n_eval, 2\)"),
        (jnp.ones((1, 1, 2)), "points must be one- or two-dimensional"),
    ],
)
def test_invalid_point_shapes_raise(chebyshev_function, points, match):
    data = make_jax_data(chebyshev_function)

    with pytest.raises(ValueError, match=match):
        evaluate_bases_jax(data, points)


def test_invalid_coefficient_shapes_raise(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    point = jnp.array([0.2, 2.0])
    n_basis = chebyshev_function.get_n_points()

    with pytest.raises(ValueError, match="coefficients must have shape"):
        evaluate_jax(data, jnp.ones((n_basis, 1, 1)), point)
    with pytest.raises(ValueError, match="leading dimension"):
        evaluate_jax(data, jnp.ones(n_basis + 1), point)


def test_hat_basis_is_rejected_by_jax_adapter():
    function = make_tensor_uniform_hat(
        dimension=1,
        n_points=3,
        lb=np.array([-1.0]),
        ub=np.array([1.0]),
    )

    with pytest.raises(NotImplementedError, match="ChebyshevBasis1d only"):
        make_jax_data(function)
