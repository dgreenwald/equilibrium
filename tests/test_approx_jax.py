"""Core contract tests for stateless JAX approximation evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equilibrium.approx import (
    ChebyshevBasis1d,
    UniformHatBasis1d,
    evaluate_bases_jax,
    evaluate_jax,
    make_jax_data,
    make_smolyak_chebyshev,
    make_tensor_chebyshev,
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


def _make_case(case_name):
    if case_name == "tensor_1d":
        function = make_tensor_chebyshev(
            dimension=1,
            n_points=4,
            lb=np.array([-3.0]),
            ub=np.array([2.0]),
        )
        points = np.array([[-2.25], [-0.1], [1.75]])
    elif case_name == "smolyak_2d":
        function = make_smolyak_chebyshev(
            dimension=2,
            max_levels=(2, 2),
            max_total_level=2,
            lb=np.array([-2.0, 1.0]),
            ub=np.array([3.0, 5.0]),
        )
        points = np.array([[-1.5, 1.4], [0.2, 2.0], [2.25, 4.5]])
    elif case_name == "tensor_3d":
        function = make_tensor_chebyshev(
            dimension=3,
            n_points=(3, 2, 3),
            lb=np.array([-2.0, 1.0, 10.0]),
            ub=np.array([1.0, 5.0, 14.0]),
        )
        points = np.array([[-1.7, 1.5, 10.5], [-0.25, 3.0, 12.25], [0.8, 4.75, 13.5]])
    elif case_name == "smolyak_3d":
        function = make_smolyak_chebyshev(
            dimension=3,
            max_levels=(2, 1, 2),
            max_total_level=3,
            lb=np.array([-2.0, 1.0, 10.0]),
            ub=np.array([1.0, 5.0, 14.0]),
        )
        points = np.array([[-1.7, 1.5, 10.5], [-0.25, 3.0, 12.25], [0.8, 4.75, 13.5]])
    else:  # pragma: no cover - guarded by parametrization
        raise ValueError(f"unknown test case: {case_name}")
    return function, points


@pytest.mark.parametrize(
    "case_name", ["tensor_1d", "smolyak_2d", "tensor_3d", "smolyak_3d"]
)
def test_numpy_jax_parity_across_schemes_and_dimensions(case_name):
    function, points = _make_case(case_name)
    data = make_jax_data(function)
    canonical_points = function._transform_to_grid(points)
    expected_basis = function.scheme.evaluate_bases(canonical_points)

    actual_basis = evaluate_bases_jax(data, points)

    np.testing.assert_allclose(actual_basis, expected_basis, rtol=1e-12, atol=1e-12)

    scalar_coefficients = np.linspace(-0.5, 1.5, function.get_n_points())
    vector_coefficients = np.column_stack(
        (scalar_coefficients, scalar_coefficients**2, -scalar_coefficients)
    )
    np.testing.assert_allclose(
        evaluate_jax(data, scalar_coefficients, points),
        expected_basis @ scalar_coefficients,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        evaluate_jax(data, vector_coefficients, points),
        expected_basis @ vector_coefficients,
        rtol=1e-12,
        atol=1e-12,
    )


def test_single_and_batched_basis_shapes(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    point = jnp.array([0.2, 2.0])
    points = jnp.stack((point, jnp.array([1.5, 4.0])))

    single = evaluate_bases_jax(data, point)
    batched = evaluate_bases_jax(data, points)

    assert single.shape == (chebyshev_function.get_n_points(),)
    assert batched.shape == (2, chebyshev_function.get_n_points())
    np.testing.assert_allclose(single, batched[0], rtol=1e-12, atol=1e-12)


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
    coefficients = jnp.arange(chebyshev_function.get_n_points(), dtype=jnp.float64)
    point = jnp.array([0.2, 2.0])

    point_gradient = jax.grad(evaluate_jax, argnums=2)(data, coefficients, point)
    coefficient_gradient = jax.grad(evaluate_jax, argnums=1)(data, coefficients, point)

    assert point_gradient.shape == (2,)
    assert coefficient_gradient.shape == coefficients.shape
    np.testing.assert_allclose(
        coefficient_gradient,
        evaluate_bases_jax(data, point),
        rtol=1e-12,
        atol=1e-12,
    )


def test_point_jacobian_matches_numpy_analytical_gradient(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    coefficients = np.linspace(-0.75, 1.25, chebyshev_function.get_n_points())
    point = np.array([0.2, 2.0])
    chebyshev_function.coefficients = coefficients.copy()

    _, expected_gradient = chebyshev_function.evaluate_with_gradient(point)
    actual_gradient = jax.jacfwd(evaluate_jax, argnums=2)(
        data, jnp.asarray(coefficients), jnp.asarray(point)
    )

    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11
    )


def test_vmap_matches_direct_batch_evaluation(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    coefficients = jnp.arange(
        chebyshev_function.get_n_points() * 2, dtype=jnp.float64
    ).reshape(chebyshev_function.get_n_points(), 2)
    points = jnp.array([[0.2, 2.0], [1.5, 4.0], [-1.25, 1.4]])

    mapped = jax.vmap(evaluate_jax, in_axes=(None, None, 0))(data, coefficients, points)
    direct = evaluate_jax(data, coefficients, points)

    np.testing.assert_allclose(mapped, direct, rtol=1e-12, atol=1e-12)


def test_explicit_coefficients_do_not_mutate_fitted_function(chebyshev_function):
    grid_points = chebyshev_function.get_grid_points()
    chebyshev_function.fit(grid_points[:, 0] + grid_points[:, 1])
    original_coefficients = chebyshev_function.coefficients.copy()
    data = make_jax_data(chebyshev_function)
    replacement = jnp.full(chebyshev_function.get_n_points(), 2.0)

    original_result = evaluate_jax(data, original_coefficients, jnp.array([0.2, 2.0]))
    replacement_result = evaluate_jax(data, replacement, jnp.array([0.2, 2.0]))

    assert not np.isclose(original_result, replacement_result)
    np.testing.assert_array_equal(
        chebyshev_function.coefficients, original_coefficients
    )


def test_jax_arrays_use_float64(chebyshev_function):
    data = make_jax_data(chebyshev_function)
    coefficients = jnp.ones(chebyshev_function.get_n_points(), dtype=jnp.float64)

    result = evaluate_jax(data, coefficients, jnp.array([0.2, 2.0]))

    assert data.lower_bounds.dtype == jnp.float64
    assert data.upper_bounds.dtype == jnp.float64
    assert result.dtype == jnp.float64


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


def test_mixed_basis_scheme_is_rejected_by_jax_adapter(chebyshev_function):
    chebyshev_function.scheme._bases = (
        ChebyshevBasis1d(),
        UniformHatBasis1d(),
    )

    with pytest.raises(NotImplementedError, match="chebyshev, uniform_hat"):
        make_jax_data(chebyshev_function)
