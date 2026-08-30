"""Tests for quadrature and exogenous-process data containers."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equilibrium.model import Model
from equilibrium.solvers.quadrature import (
    ExogenousProcess,
    JaxExogenousProcess,
    JaxQuadratureRule,
    QuadratureRule,
    _merge_nodes,
    deterministic_quadrature,
    exogenous_process_from_model,
    gauss_hermite_normal,
    next_exogenous_states_jax,
    smolyak_gauss_hermite,
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


@pytest.mark.parametrize(
    ("dimension", "expected_counts"),
    [
        (1, (1, 2, 3, 4)),
        (2, (1, 5, 13, 29)),
        (3, (1, 7, 25, 69)),
        (4, (1, 9, 41, 137)),
    ],
)
def test_smolyak_node_counts_and_contract(
    dimension: int, expected_counts: tuple[int, ...]
) -> None:
    for level, expected_count in enumerate(expected_counts):
        rule = smolyak_gauss_hermite(dimension, level)

        assert rule.kind == "smolyak"
        assert rule.orders is None
        assert rule.level == level
        assert rule.dimension == dimension
        assert rule.n_nodes == expected_count
        assert np.all(np.isfinite(rule.nodes))
        assert np.all(np.isfinite(rule.weights))
        assert rule.weights.sum() == pytest.approx(1.0, abs=5e-15)

        sort_keys = tuple(
            rule.nodes[:, coordinate]
            for coordinate in range(rule.dimension - 1, -1, -1)
        )
        np.testing.assert_array_equal(np.lexsort(sort_keys), np.arange(rule.n_nodes))


def test_smolyak_one_dimensional_level_two_fixture() -> None:
    rule = smolyak_gauss_hermite(1, 2)

    np.testing.assert_allclose(rule.nodes[:, 0], [-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
    np.testing.assert_allclose(rule.weights, [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def test_smolyak_two_dimensional_level_one_fixture() -> None:
    rule = smolyak_gauss_hermite(2, 1)

    expected_nodes = np.array(
        [[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    )
    expected_weights = np.array([0.5, 0.5, -1.0, 0.5, 0.5])
    np.testing.assert_allclose(rule.nodes, expected_nodes, atol=1e-15)
    np.testing.assert_allclose(rule.weights, expected_weights, atol=1e-15)
    assert np.any(rule.weights < 0.0)


def test_smolyak_two_dimensional_level_two_fixture() -> None:
    root_three = np.sqrt(3.0)
    rule = smolyak_gauss_hermite(2, 2)
    expected_nodes = np.array(
        [
            [-root_three, 0.0],
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -root_three],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, root_three],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [root_three, 0.0],
        ]
    )
    expected_weights = np.array(
        [
            1.0 / 6.0,
            1.0 / 4.0,
            -1.0 / 2.0,
            1.0 / 4.0,
            1.0 / 6.0,
            -1.0 / 2.0,
            4.0 / 3.0,
            -1.0 / 2.0,
            1.0 / 6.0,
            1.0 / 4.0,
            -1.0 / 2.0,
            1.0 / 4.0,
            1.0 / 6.0,
        ]
    )
    np.testing.assert_allclose(rule.nodes, expected_nodes, atol=1e-15)
    np.testing.assert_allclose(rule.weights, expected_weights, atol=1e-15)


def test_smolyak_level_zero_is_node_at_marginal_means() -> None:
    means = (-2.0, 0.5, 4.0)
    rule = smolyak_gauss_hermite(3, 0, mu=means, sigma=(0.2, 1.0, 3.0))

    np.testing.assert_array_equal(rule.nodes, [means])
    np.testing.assert_array_equal(rule.weights, [1.0])


def test_smolyak_nonstandard_normal_moments() -> None:
    means = np.array([-1.0, 0.5, 2.0])
    sigmas = np.array([0.4, 1.5, 2.5])
    level_one = smolyak_gauss_hermite(3, 1, mu=means, sigma=sigmas)
    centered_one = level_one.nodes - means

    np.testing.assert_allclose(level_one.integrate(level_one.nodes), means, atol=2e-14)
    covariance = np.einsum("n,ni,nj->ij", level_one.weights, centered_one, centered_one)
    np.testing.assert_allclose(covariance, np.diag(sigmas**2), atol=3e-14)

    level_two = smolyak_gauss_hermite(3, 2, mu=means, sigma=sigmas)
    centered_two = level_two.nodes - means
    for coordinate in range(3):
        fourth_moment = level_two.integrate(centered_two[:, coordinate] ** 4)
        assert fourth_moment == pytest.approx(3.0 * sigmas[coordinate] ** 4)
    mixed_moment = level_two.integrate(
        centered_two[:, 0] ** 2 * centered_two[:, 2] ** 2
    )
    assert mixed_moment == pytest.approx(sigmas[0] ** 2 * sigmas[2] ** 2)


@pytest.mark.parametrize("dimension", range(1, 5))
@pytest.mark.parametrize("level", range(4))
def test_smolyak_is_symmetric_and_deterministic(dimension: int, level: int) -> None:
    first = smolyak_gauss_hermite(dimension, level)
    second = smolyak_gauss_hermite(dimension, level)

    np.testing.assert_array_equal(first.nodes, second.nodes)
    np.testing.assert_array_equal(first.weights, second.weights)
    for coordinate in range(dimension):
        assert first.integrate(first.nodes[:, coordinate] ** 3) == pytest.approx(
            0.0, abs=2e-14
        )


def test_merge_nodes_uses_coordinatewise_tolerance_and_earliest_node() -> None:
    nodes = np.array([[1.0, 0.0], [5e-15, -5e-15], [2e-14, 0.0], [0.0, 0.0]])
    weights = np.array([1.0, 0.75, 2.0, 0.25])

    merged_nodes, merged_weights = _merge_nodes(nodes, weights, tolerance=1e-14)

    np.testing.assert_array_equal(merged_nodes, [[0.0, 0.0], [2e-14, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(merged_weights, [1.0, 2.0, 1.0])


def test_smolyak_weight_filter_cannot_hide_invalid_sum() -> None:
    with pytest.raises(ValueError, match="do not sum to one"):
        smolyak_gauss_hermite(2, 1, weight_tolerance=0.75)
    with pytest.raises(ValueError, match="removed every"):
        smolyak_gauss_hermite(2, 1, weight_tolerance=1.0)


def test_smolyak_allocation_guard_runs_before_tensor_construction(monkeypatch) -> None:
    def unexpected_constructor(*args, **kwargs):
        raise AssertionError("tensor component was constructed")

    monkeypatch.setattr(
        "equilibrium.solvers.quadrature.tensor_gauss_hermite",
        unexpected_constructor,
    )
    with pytest.raises(ValueError, match="165 candidate nodes"):
        smolyak_gauss_hermite(4, 3, max_nodes=164)


def test_smolyak_allocation_guard_can_be_disabled() -> None:
    assert smolyak_gauss_hermite(2, 2, max_nodes=None).n_nodes == 13


def test_smolyak_sparse_and_tensor_smooth_function_accuracy() -> None:
    sparse = smolyak_gauss_hermite(4, 3)
    tensor = tensor_gauss_hermite(4, dimension=4)
    exact = np.exp(0.08)
    sparse_value = sparse.integrate(np.exp(0.2 * sparse.nodes.sum(axis=1)))
    tensor_value = tensor.integrate(np.exp(0.2 * tensor.nodes.sum(axis=1)))

    assert sparse.n_nodes == 137
    assert tensor.n_nodes == 256
    assert sparse.n_nodes < tensor.n_nodes
    assert abs(sparse_value - exact) < 2e-6
    assert abs(tensor_value - exact) < 1e-8


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True])
def test_smolyak_rejects_invalid_dimension(dimension) -> None:
    with pytest.raises(ValueError, match="dimension must be a positive integer"):
        smolyak_gauss_hermite(dimension, 1)


@pytest.mark.parametrize("level", [-1, 1.5, True])
def test_smolyak_rejects_invalid_level(level) -> None:
    with pytest.raises(ValueError, match="level must be a nonnegative integer"):
        smolyak_gauss_hermite(2, level)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("mu", [0.0], "one entry per dimension"),
        ("mu", [0.0, np.nan], "finite real scalar"),
        ("sigma", [1.0], "one entry per dimension"),
        ("sigma", [1.0, 0.0], "strictly positive"),
        ("sigma", [1.0, np.inf], "finite real scalar"),
    ],
)
def test_smolyak_rejects_invalid_distribution_parameters(
    parameter, value, message
) -> None:
    with pytest.raises(ValueError, match=message):
        smolyak_gauss_hermite(2, 1, **{parameter: value})


@pytest.mark.parametrize("name", ["merge_tolerance", "weight_tolerance"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, True, "small"])
def test_smolyak_rejects_invalid_tolerances(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        smolyak_gauss_hermite(2, 1, **{name: value})


@pytest.mark.parametrize("max_nodes", [0, -1, 1.5, True, "100"])
def test_smolyak_rejects_invalid_max_nodes(max_nodes) -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        smolyak_gauss_hermite(2, 1, max_nodes=max_nodes)


def test_exogenous_process_resolves_model_parameters_in_exog_order() -> None:
    model = Model(
        exog_list=["technology", "preference"],
        params={
            "PERS_preference": 0.7,
            "VOL_preference": 0.04,
            "PERS_technology": 0.95,
            "VOL_technology": 0.01,
        },
    )
    original_params = model.params.copy()
    original_exog = model.exog_list.copy()

    process = exogenous_process_from_model(model)

    assert process.names == ("technology", "preference")
    np.testing.assert_array_equal(process.persistence, np.diag([0.95, 0.7]))
    np.testing.assert_array_equal(process.innovation_impact, np.diag([0.01, 0.04]))
    assert model.params == original_params
    assert model.exog_list == original_exog


def test_exogenous_process_accepts_full_and_rectangular_overrides() -> None:
    persistence = np.array([[0.8, 0.1], [-0.2, 0.6]])
    impact = np.array([[0.3], [0.4]])

    class ExplicitModel:
        exog_list = ["a", "b"]

        @property
        def params(self):
            raise AssertionError("params should not be read with complete overrides")

        @property
        def linear_mod(self):
            raise AssertionError("linear_mod must never be read")

    process = exogenous_process_from_model(
        ExplicitModel(),
        persistence=persistence,
        innovation_impact=impact,
    )

    np.testing.assert_array_equal(process.persistence, persistence)
    np.testing.assert_array_equal(process.innovation_impact, impact)


def test_exogenous_process_partial_override_reads_only_needed_defaults() -> None:
    model = SimpleNamespace(
        exog_list=["a", "b"],
        params={"VOL_a": 0.1, "VOL_b": 0.2},
    )
    persistence = np.array([[0.8, 0.1], [0.0, 0.7]])

    process = exogenous_process_from_model(model, persistence=persistence)

    np.testing.assert_array_equal(process.persistence, persistence)
    np.testing.assert_array_equal(process.innovation_impact, np.diag([0.1, 0.2]))


def test_exogenous_process_does_not_read_linear_model() -> None:
    class ModelWithoutReadableLinearization:
        exog_list = ["a"]
        params = {"PERS_a": 0.9, "VOL_a": 0.2}

        @property
        def linear_mod(self):
            raise AssertionError("linear_mod must never be read")

    process = exogenous_process_from_model(ModelWithoutReadableLinearization())

    np.testing.assert_array_equal(process.innovation_impact, [[0.2]])


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (SimpleNamespace(params={}), "exog_list"),
        (SimpleNamespace(exog_list=["a"]), "provide params"),
        (
            SimpleNamespace(exog_list=["a"], params={"VOL_a": 0.1}),
            "PERS_a",
        ),
        (
            SimpleNamespace(exog_list=["a"], params={"PERS_a": 0.9}),
            "VOL_a",
        ),
        (
            SimpleNamespace(
                exog_list=["a", "a"],
                params={"PERS_a": 0.9, "VOL_a": 0.1},
            ),
            "unique",
        ),
        (
            SimpleNamespace(exog_list=["a"], params={"PERS_a": np.nan, "VOL_a": 0.1}),
            "finite",
        ),
    ],
)
def test_exogenous_process_rejects_invalid_model_data(model, message) -> None:
    with pytest.raises(ValueError, match=message):
        exogenous_process_from_model(model)


@pytest.mark.parametrize(
    ("persistence", "impact", "message"),
    [
        (np.eye(3), np.eye(2), "persistence"),
        (np.eye(2), np.ones(2), "innovation_impact"),
        (np.eye(2), np.ones((3, 1)), "innovation_impact"),
        (np.array([[np.inf, 0.0], [0.0, 1.0]]), np.eye(2), "finite"),
        (np.eye(2), np.array([[0.1], [np.nan]]), "finite"),
    ],
)
def test_exogenous_process_rejects_invalid_overrides(
    persistence, impact, message
) -> None:
    model = SimpleNamespace(exog_list=["a", "b"])
    with pytest.raises(ValueError, match=message):
        exogenous_process_from_model(
            model, persistence=persistence, innovation_impact=impact
        )


def test_exogenous_process_supports_zero_exogenous_variables() -> None:
    process = exogenous_process_from_model(SimpleNamespace(exog_list=[], params={}))

    assert process.names == ()
    assert process.persistence.shape == (0, 0)
    assert process.innovation_impact.shape == (0, 0)


def test_next_exogenous_states_single_and_batched_shapes() -> None:
    process = ExogenousProcess(
        ("a", "b"),
        np.array([[0.8, 0.1], [-0.2, 0.6]]),
        np.array([[0.3], [0.4]]),
    ).as_jax()
    nodes = jnp.array([[-1.0], [0.0], [1.0]])
    current = jnp.array([2.0, -1.0])
    batch = jnp.array([[2.0, -1.0], [0.5, 3.0]])

    single_result = next_exogenous_states_jax(process, current, nodes)
    batch_result = next_exogenous_states_jax(process, batch, nodes)
    expected_single = np.array([[1.2, -1.4], [1.5, -1.0], [1.8, -0.6]])

    assert single_result.shape == (3, 2)
    assert batch_result.shape == (2, 3, 2)
    np.testing.assert_allclose(single_result, expected_single)
    np.testing.assert_allclose(batch_result[0], expected_single)


def test_next_exogenous_states_eager_jit_and_vmap_parity() -> None:
    process = ExogenousProcess(
        ("a", "b"),
        np.array([[0.9, 0.2], [0.0, 0.7]]),
        np.array([[0.1, -0.2], [0.3, 0.4]]),
    ).as_jax()
    nodes = jnp.array([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]])
    batch = jnp.array([[1.0, 2.0], [-0.5, 0.25], [3.0, -2.0]])

    eager = next_exogenous_states_jax(process, batch, nodes)
    jitted = jax.jit(next_exogenous_states_jax)(process, batch, nodes)
    mapped = jax.vmap(lambda state: next_exogenous_states_jax(process, state, nodes))(
        batch
    )

    np.testing.assert_allclose(jitted, eager)
    np.testing.assert_allclose(mapped, eager)
    assert eager.dtype == np.float64


def test_next_exogenous_states_matches_conditional_moments_without_double_scaling() -> (
    None
):
    persistence = np.array([[0.85, 0.1], [-0.05, 0.7]])
    impact = np.array([[0.2, 0.0], [0.1, 0.3]])
    process = ExogenousProcess(("a", "b"), persistence, impact).as_jax()
    rule = tensor_gauss_hermite(3, dimension=2)
    current = jnp.array([1.2, -0.4])

    next_states = next_exogenous_states_jax(process, current, rule.as_jax().nodes)
    conditional_mean = rule.integrate(np.asarray(next_states))
    deviations = np.asarray(next_states) - conditional_mean
    covariance = np.einsum("n,ni,nj->ij", rule.weights, deviations, deviations)

    np.testing.assert_allclose(conditional_mean, persistence @ current, atol=2e-14)
    np.testing.assert_allclose(covariance, impact @ impact.T, atol=2e-14)


def test_model_volatility_is_applied_exactly_once() -> None:
    model = SimpleNamespace(
        exog_list=["a", "b"],
        params={"PERS_a": 0.9, "PERS_b": 0.8, "VOL_a": 0.2, "VOL_b": 0.5},
    )
    process = exogenous_process_from_model(model).as_jax()
    rule = tensor_gauss_hermite(3, dimension=2)
    states = next_exogenous_states_jax(process, jnp.zeros(2), rule.as_jax().nodes)
    covariance = np.einsum("n,ni,nj->ij", rule.weights, states, states)

    np.testing.assert_allclose(covariance, np.diag([0.2**2, 0.5**2]), atol=2e-14)


def test_next_exogenous_states_gradients() -> None:
    persistence = jnp.array([[0.8, 0.1], [-0.2, 0.6]])
    impact = jnp.array([[0.3, -0.1], [0.2, 0.4]])
    current = jnp.array([1.5, -0.5])
    nodes = jnp.array([[-1.0, 0.5], [0.25, -0.75], [1.0, 0.0]])
    cotangent = jnp.array([[1.0, -2.0], [0.5, 0.25], [-1.5, 3.0]])

    def loss(phi, innovation_matrix, state):
        process = JaxExogenousProcess(phi, innovation_matrix)
        next_states = next_exogenous_states_jax(process, state, nodes)
        return jnp.sum(cotangent * next_states)

    phi_gradient, impact_gradient, state_gradient = jax.grad(loss, argnums=(0, 1, 2))(
        persistence, impact, current
    )
    expected_phi_gradient = np.einsum("ne,j->ej", cotangent, current)
    expected_impact_gradient = np.einsum("ne,nk->ek", cotangent, nodes)
    expected_state_gradient = np.einsum("ne,ej->j", cotangent, persistence)

    np.testing.assert_allclose(phi_gradient, expected_phi_gradient)
    np.testing.assert_allclose(impact_gradient, expected_impact_gradient)
    np.testing.assert_allclose(state_gradient, expected_state_gradient)


def test_next_exogenous_states_zero_dimensional_contract() -> None:
    process = ExogenousProcess((), np.empty((0, 0)), np.empty((0, 0))).as_jax()
    nodes = deterministic_quadrature().as_jax().nodes

    single = next_exogenous_states_jax(process, jnp.empty(0), nodes)
    batched = next_exogenous_states_jax(process, jnp.empty((4, 0)), nodes)

    assert single.shape == (1, 0)
    assert batched.shape == (4, 1, 0)


@pytest.mark.parametrize(
    ("process", "current", "nodes", "message"),
    [
        (
            JaxExogenousProcess(jnp.ones(2), jnp.eye(2)),
            jnp.ones(2),
            jnp.ones((1, 2)),
            "square matrix",
        ),
        (
            JaxExogenousProcess(jnp.eye(2), jnp.ones(2)),
            jnp.ones(2),
            jnp.ones((1, 1)),
            "innovation_impact",
        ),
        (
            JaxExogenousProcess(jnp.eye(2), jnp.ones((2, 1))),
            jnp.ones(2),
            jnp.ones(1),
            "two-dimensional",
        ),
        (
            JaxExogenousProcess(jnp.eye(2), jnp.ones((2, 1))),
            jnp.ones(2),
            jnp.ones((1, 2)),
            "matching innovation dimensions",
        ),
        (
            JaxExogenousProcess(jnp.eye(2), jnp.ones((2, 1))),
            jnp.ones((1, 1, 2)),
            jnp.ones((1, 1)),
            "one- or two-dimensional",
        ),
        (
            JaxExogenousProcess(jnp.eye(2), jnp.ones((2, 1))),
            jnp.ones(3),
            jnp.ones((1, 1)),
            "n_exogenous columns",
        ),
    ],
)
def test_next_exogenous_states_rejects_invalid_shapes(
    process, current, nodes, message
) -> None:
    with pytest.raises(ValueError, match=message):
        next_exogenous_states_jax(process, current, nodes)


def test_phase_two_api_is_exported_from_solvers_only() -> None:
    import equilibrium
    import equilibrium.solvers as solvers

    expected = {
        "QuadratureRule",
        "JaxQuadratureRule",
        "ExogenousProcess",
        "JaxExogenousProcess",
        "deterministic_quadrature",
        "gauss_hermite_normal",
        "tensor_gauss_hermite",
        "smolyak_gauss_hermite",
        "exogenous_process_from_model",
        "next_exogenous_states_jax",
    }

    assert expected <= set(solvers.__all__)
    assert all(hasattr(solvers, name) for name in expected)
    assert all(not hasattr(equilibrium, name) for name in expected)
