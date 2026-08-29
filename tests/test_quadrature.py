"""Tests for quadrature and exogenous-process data containers."""

import math

import jax
import numpy as np
import pytest

from equilibrium.solvers.quadrature import (
    ExogenousProcess,
    JaxExogenousProcess,
    JaxQuadratureRule,
    QuadratureRule,
    deterministic_quadrature,
    gauss_hermite_normal,
    tensor_gauss_hermite,
)

jax.config.update("jax_enable_x64", True)


def tensor_rule() -> QuadratureRule:
    return QuadratureRule(
        nodes=np.array([[-1.0], [1.0]]),
        weights=np.array([0.25, 0.75]),
        kind="tensor",
        orders=(2,),
    )


def test_rule_properties_and_scalar_integration() -> None:
    rule = tensor_rule()

    assert rule.dimension == 1
    assert rule.n_nodes == 2
    assert rule.integrate(np.array([2.0, 6.0])) == pytest.approx(5.0)


def test_rule_integrates_an_arbitrary_axis() -> None:
    rule = tensor_rule()
    values = np.array([[2.0, 6.0], [10.0, 14.0], [20.0, 28.0]])

    np.testing.assert_allclose(rule.integrate(values, axis=1), [5.0, 13.0, 26.0])
    np.testing.assert_allclose(rule.integrate(values, axis=-1), [5.0, 13.0, 26.0])


@pytest.mark.parametrize(
    ("nodes", "weights", "message"),
    [
        (np.ones(2), np.ones(2) / 2, "two-dimensional"),
        (np.ones((2, 1)), np.ones((2, 1)) / 2, "one-dimensional"),
        (np.ones((0, 1)), np.ones(0), "at least one node"),
        (np.ones((2, 1)), np.ones(3) / 3, "same number"),
        (np.array([[np.nan]]), np.ones(1), "finite"),
        (np.ones((1, 1)), np.array([np.inf]), "finite"),
        (np.ones((2, 1)), np.array([0.2, 0.2]), "sum to one"),
    ],
)
def test_rule_rejects_invalid_array_data(nodes, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        QuadratureRule(nodes, weights, "tensor", (1,))


def test_tensor_rule_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="negative weights"):
        QuadratureRule(
            np.array([[0.0], [1.0]]),
            np.array([1.1, -0.1]),
            "tensor",
            (2,),
        )


def test_smolyak_rule_retains_negative_weights() -> None:
    rule = QuadratureRule(
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([-0.25, 1.5, -0.25]),
        "smolyak",
        None,
        level=1,
    )

    np.testing.assert_array_equal(rule.weights, [-0.25, 1.5, -0.25])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "unknown", "orders": (1,)},
        {"kind": "tensor", "orders": None},
        {"kind": "tensor", "orders": [1]},
        {"kind": "tensor", "orders": (1, 2)},
        {"kind": "tensor", "orders": (0,)},
        {"kind": "tensor", "orders": (1,), "level": 0},
        {"kind": "smolyak", "orders": (1,), "level": 0},
        {"kind": "smolyak", "orders": None, "level": None},
    ],
)
def test_rule_rejects_invalid_metadata(kwargs) -> None:
    with pytest.raises(ValueError):
        QuadratureRule(np.zeros((1, 1)), np.ones(1), **kwargs)


def test_integration_rejects_invalid_values_shape() -> None:
    rule = tensor_rule()

    with pytest.raises(ValueError, match="at least one dimension"):
        rule.integrate(np.array(1.0))
    with pytest.raises(ValueError, match="axis length"):
        rule.integrate(np.ones(3))
    with pytest.raises(ValueError, match="invalid integration axis"):
        rule.integrate(np.ones(2), axis=1)


def test_rule_defensively_copies_and_freezes_arrays() -> None:
    nodes = np.array([[-1.0], [1.0]])
    weights = np.array([0.5, 0.5])
    rule = QuadratureRule(nodes, weights, "tensor", (2,))
    nodes[0, 0] = 100.0
    weights[0] = 1.0

    assert rule.nodes[0, 0] == -1.0
    assert rule.weights[0] == 0.5
    with pytest.raises(ValueError, match="read-only"):
        rule.nodes[0, 0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        rule.weights[0] = 0.0


def test_deterministic_quadrature_contract() -> None:
    rule = deterministic_quadrature()

    assert rule.kind == "deterministic"
    assert rule.orders == ()
    assert rule.level is None
    assert rule.dimension == 0
    assert rule.n_nodes == 1
    assert rule.nodes.shape == (1, 0)
    np.testing.assert_array_equal(rule.weights, [1.0])
    assert rule.integrate(np.array([7.0])) == 7.0


def test_rule_as_jax_preserves_values_and_float64() -> None:
    jax_rule = tensor_rule().as_jax()

    assert isinstance(jax_rule, JaxQuadratureRule)
    assert jax_rule.nodes.dtype == np.float64
    assert jax_rule.weights.dtype == np.float64
    np.testing.assert_array_equal(jax_rule.nodes, [[-1.0], [1.0]])
    np.testing.assert_array_equal(jax_rule.weights, [0.25, 0.75])


def test_exogenous_process_contract_and_jax_conversion() -> None:
    persistence = np.array([[0.9, 0.1], [0.0, 0.8]])
    impact = np.array([[0.2], [0.3]])
    process = ExogenousProcess(("a", "b"), persistence, impact)
    persistence[0, 0] = 0.0
    impact[0, 0] = 0.0

    np.testing.assert_array_equal(process.persistence, [[0.9, 0.1], [0.0, 0.8]])
    np.testing.assert_array_equal(process.innovation_impact, [[0.2], [0.3]])
    assert not process.persistence.flags.writeable
    assert not process.innovation_impact.flags.writeable

    jax_process = process.as_jax()
    assert isinstance(jax_process, JaxExogenousProcess)
    assert jax_process.persistence.dtype == np.float64
    assert jax_process.innovation_impact.dtype == np.float64
    np.testing.assert_array_equal(jax_process.persistence, process.persistence)
    np.testing.assert_array_equal(
        jax_process.innovation_impact, process.innovation_impact
    )


@pytest.mark.parametrize(
    ("names", "persistence", "impact", "message"),
    [
        (["a"], np.eye(1), np.eye(1), "tuple of strings"),
        (("a", "a"), np.eye(2), np.eye(2), "unique"),
        (("a",), np.eye(2), np.ones((1, 1)), "persistence"),
        (("a",), np.eye(1), np.ones(1), "innovation_impact"),
        (("a",), np.eye(1), np.ones((2, 1)), "innovation_impact"),
        (("a",), np.array([[np.nan]]), np.eye(1), "finite"),
        (("a",), np.eye(1), np.array([[np.inf]]), "finite"),
    ],
)
def test_exogenous_process_rejects_invalid_data(
    names, persistence, impact, message
) -> None:
    with pytest.raises(ValueError, match=message):
        ExogenousProcess(names, persistence, impact)


def test_zero_dimensional_exogenous_process() -> None:
    process = ExogenousProcess((), np.empty((0, 0)), np.empty((0, 0)))

    assert process.persistence.shape == (0, 0)
    assert process.innovation_impact.shape == (0, 0)


@pytest.mark.parametrize("degree", [1, 2, 3, 5, 10])
def test_gauss_hermite_standard_normal_contract(degree: int) -> None:
    rule = gauss_hermite_normal(degree)

    assert rule.nodes.shape == (degree, 1)
    assert rule.weights.shape == (degree,)
    assert rule.kind == "tensor"
    assert rule.orders == (degree,)
    assert np.all(rule.weights > 0.0)
    assert rule.weights.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(rule.nodes[:, 0], -rule.nodes[::-1, 0], atol=1e-14)
    np.testing.assert_allclose(rule.weights, rule.weights[::-1], atol=1e-15)


@pytest.mark.parametrize("degree", [1, 2, 3, 5, 10])
def test_gauss_hermite_exact_standard_normal_moments(degree: int) -> None:
    rule = gauss_hermite_normal(degree)
    nodes = rule.nodes[:, 0]

    for power in range(2 * degree):
        expected = 0.0 if power % 2 else math.prod(range(1, power, 2))
        actual = rule.integrate(nodes**power)
        assert actual == pytest.approx(expected, rel=2e-12, abs=1e-9)


def test_gauss_hermite_nonstandard_normal_moments() -> None:
    mu = -1.25
    sigma = 2.5
    rule = gauss_hermite_normal(5, mu=mu, sigma=sigma)
    nodes = rule.nodes[:, 0]

    assert rule.integrate(nodes) == pytest.approx(mu)
    assert rule.integrate((nodes - mu) ** 2) == pytest.approx(sigma**2)
    assert rule.integrate((nodes - mu) ** 3) == pytest.approx(0.0, abs=1e-13)


def test_gauss_hermite_matches_reference_convention() -> None:
    degree = 4
    mu = 0.75
    sigma = 1.6
    reference_nodes, reference_weights = np.polynomial.hermite.hermgauss(degree)
    reference_weights /= reference_weights.sum()
    reference_nodes = mu + np.sqrt(2.0) * sigma * reference_nodes

    rule = gauss_hermite_normal(degree, mu=mu, sigma=sigma)

    np.testing.assert_array_equal(rule.nodes[:, 0], reference_nodes)
    np.testing.assert_array_equal(rule.weights, reference_weights)


@pytest.mark.parametrize("degree", [0, -1, 1.5, True, "3"])
def test_gauss_hermite_rejects_invalid_degree(degree) -> None:
    with pytest.raises(ValueError, match="degree must be a positive integer"):
        gauss_hermite_normal(degree)


@pytest.mark.parametrize("mu", [np.nan, np.inf, -np.inf, True, 1.0j, [0.0]])
def test_gauss_hermite_rejects_invalid_mean(mu) -> None:
    with pytest.raises(ValueError, match="mu must be a finite real scalar"):
        gauss_hermite_normal(3, mu=mu)


@pytest.mark.parametrize(
    "sigma", [0.0, -1.0, np.nan, np.inf, -np.inf, True, 1.0j, [1.0]]
)
def test_gauss_hermite_rejects_invalid_sigma(sigma) -> None:
    with pytest.raises(ValueError, match="sigma"):
        gauss_hermite_normal(3, sigma=sigma)


@pytest.mark.parametrize("dimension", range(5))
def test_tensor_rule_dimensions_and_node_counts(dimension: int) -> None:
    rule = tensor_gauss_hermite(2, dimension=dimension)

    expected_nodes = 1 if dimension == 0 else 2**dimension
    assert rule.dimension == dimension
    assert rule.n_nodes == expected_nodes
    assert rule.nodes.shape == (expected_nodes, dimension)
    assert rule.weights.shape == (expected_nodes,)
    assert rule.weights.sum() == pytest.approx(1.0)
    if dimension == 0:
        assert rule.kind == "deterministic"
        assert rule.orders == ()
    else:
        assert rule.kind == "tensor"
        assert rule.orders == (2,) * dimension


def test_tensor_rule_infers_dimensions() -> None:
    assert tensor_gauss_hermite(3).orders == (3,)
    assert tensor_gauss_hermite([2, 3, 4]).orders == (2, 3, 4)
    assert tensor_gauss_hermite([]).nodes.shape == (1, 0)


def test_tensor_rule_has_documented_cartesian_order() -> None:
    first = gauss_hermite_normal(2).nodes[:, 0]
    second = gauss_hermite_normal(3).nodes[:, 0]
    rule = tensor_gauss_hermite((2, 3))

    expected_nodes = np.array(
        [
            [first[0], second[0]],
            [first[0], second[1]],
            [first[0], second[2]],
            [first[1], second[0]],
            [first[1], second[1]],
            [first[1], second[2]],
        ]
    )
    expected_weights = np.outer(
        gauss_hermite_normal(2).weights,
        gauss_hermite_normal(3).weights,
    ).reshape(-1)
    np.testing.assert_array_equal(rule.nodes, expected_nodes)
    np.testing.assert_array_equal(rule.weights, expected_weights)


def test_tensor_rule_nonstandard_marginal_and_cross_moments() -> None:
    means = np.array([-1.0, 0.5, 2.0])
    sigmas = np.array([0.4, 1.5, 2.5])
    rule = tensor_gauss_hermite((3, 4, 5), mu=means, sigma=sigmas)

    for coordinate in range(3):
        centered = rule.nodes[:, coordinate] - means[coordinate]
        assert rule.integrate(rule.nodes[:, coordinate]) == pytest.approx(
            means[coordinate]
        )
        assert rule.integrate(centered**2) == pytest.approx(sigmas[coordinate] ** 2)

    centered = rule.nodes - means
    covariance = np.einsum("n,ni,nj->ij", rule.weights, centered, centered)
    np.testing.assert_allclose(covariance, np.diag(sigmas**2), atol=2e-14)
    assert rule.integrate(rule.nodes[:, 0] * rule.nodes[:, 2]) == pytest.approx(
        means[0] * means[2]
    )


def test_tensor_rule_anisotropic_polynomial_exactness() -> None:
    rule = tensor_gauss_hermite((2, 3))
    x = rule.nodes[:, 0]
    y = rule.nodes[:, 1]

    # The supported degree is three in x and five in y.
    assert rule.integrate(x**2 * y**4) == pytest.approx(3.0)
    assert rule.integrate(x**3 * y**5) == pytest.approx(0.0, abs=1e-13)


def test_tensor_rule_allocation_guard_runs_before_rule_construction(
    monkeypatch,
) -> None:
    def unexpected_constructor(*args, **kwargs):
        raise AssertionError("one-dimensional rule was constructed")

    monkeypatch.setattr(
        "equilibrium.solvers.quadrature.gauss_hermite_normal",
        unexpected_constructor,
    )
    with pytest.raises(ValueError, match="exceeding max_nodes"):
        tensor_gauss_hermite((100, 100), max_nodes=9_999)


def test_tensor_rule_allocation_guard_can_be_disabled() -> None:
    rule = tensor_gauss_hermite((2, 2), max_nodes=None)

    assert rule.n_nodes == 4


@pytest.mark.parametrize(
    ("degrees", "dimension", "message"),
    [
        (0, None, "positive integers"),
        (-1, None, "positive integers"),
        (True, None, "integer or a sequence"),
        (1.5, None, "integer or a sequence"),
        ((2, 0), None, "positive integers"),
        ((2, True), None, "positive integers"),
        ((2, 3), 3, "one entry per dimension"),
        (2, -1, "nonnegative integer"),
        (2, 1.5, "nonnegative integer"),
        (2, True, "nonnegative integer"),
    ],
)
def test_tensor_rule_rejects_invalid_degrees_and_dimension(
    degrees, dimension, message
) -> None:
    with pytest.raises(ValueError, match=message):
        tensor_gauss_hermite(degrees, dimension=dimension)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("mu", [0.0], "one entry per dimension"),
        ("mu", [0.0, np.nan], "finite real scalar"),
        ("mu", "zero", "scalar or a sequence"),
        ("sigma", [1.0], "one entry per dimension"),
        ("sigma", [1.0, 0.0], "strictly positive"),
        ("sigma", [1.0, np.inf], "finite real scalar"),
    ],
)
def test_tensor_rule_rejects_invalid_distribution_parameters(
    parameter, value, message
) -> None:
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=message):
        tensor_gauss_hermite((2, 2), **kwargs)


@pytest.mark.parametrize("max_nodes", [0, -1, 1.5, True, "100"])
def test_tensor_rule_rejects_invalid_max_nodes(max_nodes) -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        tensor_gauss_hermite(2, max_nodes=max_nodes)
