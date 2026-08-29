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
