"""Tests for collocation configuration and domain resolution."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from equilibrium import Model
from equilibrium.solvers import colloc_spec
from equilibrium.solvers.colloc_spec import (
    CollocationSpec,
    _normalize_dimension_integers,
    _resolve_collocation_prerequisites,
    _validate_collocation_model,
)
from equilibrium.solvers.quadrature import ExogenousProcess


class _FakeModel:
    def __init__(
        self,
        *,
        n_controls: int = 1,
        n_x: int = 1,
        z_names: tuple[str, ...] = ("z",),
        x_steady: np.ndarray | None = None,
        h_x: np.ndarray | None = None,
        h_z: np.ndarray | None = None,
    ) -> None:
        n_z = len(z_names)
        x_names = tuple(f"x{index}" for index in range(n_x))
        control_names = tuple(f"u{index}" for index in range(n_controls))
        self.inner_functions = object()
        self.var_lists = {
            "u": list(control_names),
            "x": list(x_names),
            "z": list(z_names),
            "params": [],
        }
        self.N = {"u": n_controls, "x": n_x, "z": n_z}
        self.exog_list = list(z_names)
        self.params = {
            **{f"PERS_{name}": 0.8 for name in z_names},
            **{f"VOL_{name}": 0.1 for name in z_names},
        }
        self.steady_components = {
            "u": np.zeros(n_controls),
            "x": (
                np.zeros(n_x)
                if x_steady is None
                else np.asarray(x_steady, dtype=np.float64)
            ),
            "z": np.zeros(n_z),
            "params": np.empty(0),
        }
        self._steady_solved = True
        self.linear_mod = SimpleNamespace(
            H_x=(
                0.5 * np.eye(n_x) if h_x is None else np.asarray(h_x, dtype=np.float64)
            ),
            H_z=(
                np.zeros((n_x, n_z))
                if h_z is None
                else np.asarray(h_z, dtype=np.float64)
            ),
        )
        self.linearize_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def linearize(self, Phi, impact_matrix):
        self.linearize_calls.append(
            (np.array(Phi, copy=True), np.array(impact_matrix, copy=True))
        )
        return self.linear_mod


class _FakeApproximation:
    def __init__(self, lower, upper):
        self.lb = np.asarray(lower, dtype=np.float64)
        self.ub = np.asarray(upper, dtype=np.float64)


def _explicit_process(names=("z",), persistence=None, impact=None) -> ExogenousProcess:
    dimension = len(names)
    if persistence is None:
        persistence = 0.8 * np.eye(dimension)
    if impact is None:
        impact = 0.1 * np.eye(dimension)
    return ExogenousProcess(
        names=tuple(names),
        persistence=np.asarray(persistence),
        innovation_impact=np.asarray(impact),
    )


def _make_rbc_model() -> Model:
    model = Model(label="colloc_spec_rbc")
    model.params.update(
        {"alp": 0.6, "bet": 0.95, "delta": 0.1, "gam": 2.0, "Z_bar": 0.5}
    )
    model.steady_guess.update({"I": 0.5, "log_K": np.log(6.0)})
    model.rules["intermediate"] += [
        ("K_new", "I + (1.0 - delta) * K"),
        ("K", "np.exp(log_K)"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
    ]
    model.rules["expectations"] += [
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))")
    ]
    model.rules["transition"] += [("log_K", "np.log(K_new)")]
    model.rules["optimality"] += [("I", "E_Om_K - 1.0")]
    model.add_exog("Z_til", pers=0.95, vol=0.1)
    model.finalize()
    result = model.solve_steady(calibrate=False, display=False)
    assert result.success
    return model


def test_collocation_spec_defaults():
    spec = CollocationSpec()

    assert spec.approximation == "smolyak_chebyshev"
    assert spec.algorithm == "hybrid"
    assert spec.initialization == "linear"
    assert spec.domain is None
    assert spec.quadrature_degree == 5
    assert spec.tolerance == 1e-8


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("approximation", "hat", "approximation must be one of"),
        ("quadrature_kind", "auto", "quadrature_kind must be one of"),
        ("algorithm", "fixed_point", "algorithm must be one of"),
        ("initialization", "zero", "initialization must be one of"),
        ("extrapolation", "clip", "extrapolation must be one of"),
        ("max_levels", -1, "max_levels entries"),
        ("max_levels", (), "max_levels cannot be empty"),
        ("max_total_level", -1, "max_total_level"),
        ("tensor_points", 0, "tensor_points entries"),
        ("quadrature_degree", 0, "quadrature_degree entries"),
        ("quadrature_level", -1, "quadrature_level"),
        ("quadrature_max_nodes", 0, "quadrature_max_nodes"),
        ("domain_stddevs", 0.0, "domain_stddevs"),
        ("domain_min_half_width", np.inf, "domain_min_half_width"),
        ("tolerance", 0.0, "tolerance"),
        ("inner_tolerance", np.nan, "inner_tolerance"),
        ("max_time_iterations", 0, "max_time_iterations"),
        ("max_inner_iterations", 0, "max_inner_iterations"),
        ("max_newton_iterations", 0, "max_newton_iterations"),
        ("max_backtracks", -1, "max_backtracks"),
        ("damping", 0.0, "damping"),
        ("damping", 1.1, "damping"),
        ("hybrid_max_time_iterations", 0, "hybrid_max_time_iterations"),
        ("max_newton_unknowns", 0, "max_newton_unknowns"),
        ("verbose", 1, "verbose must be a boolean"),
    ],
)
def test_collocation_spec_rejects_invalid_values(keyword, value, match):
    with pytest.raises(ValueError, match=match):
        CollocationSpec(**{keyword: value})


def test_collocation_spec_requires_hybrid_switch_above_solve_tolerance():
    with pytest.raises(ValueError, match="greater than or equal"):
        CollocationSpec(tolerance=1e-4, hybrid_switch_tolerance=1e-5)


@pytest.mark.parametrize(
    ("domain", "match"),
    [
        ([[-1.0], [1.0]], r"\(lower, upper\) tuple"),
        (([-1.0],), r"\(lower, upper\) tuple"),
        (([[-1.0]], [[1.0]]), "one-dimensional"),
        (([-1.0], [1.0, 2.0]), "matching shapes"),
        (([-1.0], [np.inf]), "finite"),
        (([1.0], [1.0]), "less than"),
    ],
)
def test_collocation_spec_rejects_invalid_domain(domain, match):
    with pytest.raises(ValueError, match=match):
        CollocationSpec(domain=domain)


def test_collocation_spec_defensively_copies_domain():
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    spec = CollocationSpec(domain=(lower, upper))

    lower[0] = -10.0
    upper[0] = 10.0

    np.testing.assert_array_equal(spec.domain[0], [-1.0, -2.0])
    np.testing.assert_array_equal(spec.domain[1], [1.0, 2.0])
    assert not spec.domain[0].flags.writeable
    assert not spec.domain[1].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        spec.domain[0][0] = 0.0


def test_normalize_dimension_integers_broadcasts_and_checks_length():
    assert _normalize_dimension_integers(3, 2, "levels", minimum=0) == (3, 3)
    assert _normalize_dimension_integers((1, 2), 2, "levels", minimum=0) == (
        1,
        2,
    )
    with pytest.raises(ValueError, match="length 2"):
        _normalize_dimension_integers((1,), 2, "levels", minimum=0)


def test_model_validation_requires_finalize_and_steady_state():
    model = _FakeModel()
    model.inner_functions = None
    with pytest.raises(RuntimeError, match="finalized"):
        _validate_collocation_model(model)

    model = _FakeModel()
    model._steady_solved = False
    with pytest.raises(RuntimeError, match="steady state must be solved"):
        _validate_collocation_model(model)


@pytest.mark.parametrize(
    ("n_controls", "n_x", "z_names", "match"),
    [
        (0, 1, ("z",), "at least one control"),
        (1, 0, (), "at least one dynamic state"),
    ],
)
def test_model_validation_rejects_degenerate_systems(n_controls, n_x, z_names, match):
    model = _FakeModel(n_controls=n_controls, n_x=n_x, z_names=z_names)
    with pytest.raises(ValueError, match=match):
        _validate_collocation_model(model)


def test_model_validation_enforces_exogenous_order():
    model = _FakeModel(z_names=("z1", "z2"))
    model.exog_list = ["z2", "z1"]
    with pytest.raises(RuntimeError, match="order"):
        _validate_collocation_model(model)


def test_real_finalized_model_uses_repository_variable_order():
    model = _make_rbc_model()

    dimensions = _validate_collocation_model(model)

    assert dimensions.control_names == ("I",)
    assert dimensions.x_names == ("log_K",)
    assert dimensions.z_names == ("Z_til",)
    assert dimensions.state_names == ("log_K", "Z_til")


def test_real_model_automatic_domain_is_finite_and_uses_volatility():
    model = _make_rbc_model()

    prerequisites = _resolve_collocation_prerequisites(
        model, CollocationSpec(initialization="steady")
    )

    assert prerequisites.domain.source == "automatic"
    assert prerequisites.domain.lower.shape == (2,)
    assert np.all(np.isfinite(prerequisites.domain.lower))
    assert np.all(np.isfinite(prerequisites.domain.upper))
    assert np.all(prerequisites.domain.lower < prerequisites.domain.upper)
    np.testing.assert_allclose(
        model.linear_mod.impact_matrix,
        np.diag([model.params["VOL_Z_til"]]),
    )


def test_explicit_domain_and_steady_initialization_skip_linearization():
    model = _FakeModel()
    spec = CollocationSpec(domain=([-2.0, -1.0], [2.0, 1.0]), initialization="steady")

    prerequisites = _resolve_collocation_prerequisites(model, spec)

    assert prerequisites.domain.source == "explicit"
    assert prerequisites.domain.covariance is None
    assert not prerequisites.linearized
    assert model.linearize_calls == []
    np.testing.assert_array_equal(prerequisites.domain.lower, [-2.0, -1.0])
    np.testing.assert_array_equal(prerequisites.domain.upper, [2.0, 1.0])


def test_explicit_domain_dimension_is_validated_against_model():
    model = _FakeModel()
    spec = CollocationSpec(domain=([-1.0], [1.0]), initialization="steady")
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        _resolve_collocation_prerequisites(model, spec)


def test_custom_approximation_supplies_domain_without_linearization():
    model = _FakeModel()
    spec = CollocationSpec(initialization="steady")
    approximation = _FakeApproximation([-3.0, -2.0], [3.0, 2.0])

    prerequisites = _resolve_collocation_prerequisites(
        model, spec, approximation=approximation
    )

    assert prerequisites.domain.source == "approximation"
    assert not prerequisites.linearized
    assert model.linearize_calls == []


def test_linear_initialization_relinearizes_with_resolved_process():
    model = _FakeModel()
    process = _explicit_process(persistence=[[0.7]], impact=[[0.25, 0.1]])
    spec = CollocationSpec(domain=([-2.0, -1.0], [2.0, 1.0]))

    prerequisites = _resolve_collocation_prerequisites(model, spec, process=process)

    assert prerequisites.linearized
    assert len(model.linearize_calls) == 1
    phi, impact = model.linearize_calls[0]
    np.testing.assert_array_equal(phi, process.persistence)
    np.testing.assert_array_equal(impact, process.innovation_impact)


def test_explicit_coefficients_bypass_linear_initialization():
    model = _FakeModel()
    spec = CollocationSpec(domain=([-2.0, -1.0], [2.0, 1.0]))

    prerequisites = _resolve_collocation_prerequisites(
        model, spec, initial_coefficients_provided=True
    )

    assert not prerequisites.linearized
    assert model.linearize_calls == []


def test_default_process_is_resolved_in_model_order():
    model = _FakeModel(n_x=0, z_names=("z2", "z1"))
    model.params.update({"PERS_z2": 0.4, "PERS_z1": 0.7, "VOL_z2": 0.2, "VOL_z1": 0.3})
    spec = CollocationSpec(domain=([-1.0, -1.0], [1.0, 1.0]), initialization="steady")

    prerequisites = _resolve_collocation_prerequisites(model, spec)

    assert prerequisites.process.names == ("z2", "z1")
    np.testing.assert_array_equal(
        prerequisites.process.persistence, np.diag([0.4, 0.7])
    )
    np.testing.assert_array_equal(
        prerequisites.process.innovation_impact, np.diag([0.2, 0.3])
    )


def test_custom_process_must_match_model_names_and_order():
    model = _FakeModel(z_names=("z1", "z2"))
    process = _explicit_process(names=("z2", "z1"))
    spec = CollocationSpec(domain=([-1.0] * 3, [1.0] * 3), initialization="steady")

    with pytest.raises(ValueError, match="same order"):
        _resolve_collocation_prerequisites(model, spec, process=process)


def test_automatic_domain_solves_joint_stationary_covariance():
    model = _FakeModel(
        x_steady=np.array([2.0]),
        h_x=np.array([[0.5]]),
        h_z=np.array([[0.2]]),
    )
    process = _explicit_process(persistence=[[0.8]], impact=[[0.1]])
    spec = CollocationSpec(
        initialization="steady", domain_stddevs=2.5, domain_min_half_width=1e-8
    )

    prerequisites = _resolve_collocation_prerequisites(model, spec, process=process)

    assert prerequisites.linearized
    assert prerequisites.domain.source == "automatic"
    covariance = prerequisites.domain.covariance
    state_matrix = np.array([[0.5, 0.2], [0.0, 0.8]])
    shock_matrix = np.array([[0.0], [0.1]])
    np.testing.assert_allclose(
        covariance,
        state_matrix @ covariance @ state_matrix.T + shock_matrix @ shock_matrix.T,
        rtol=1e-12,
        atol=1e-14,
    )
    expected_center = np.array([2.0, 0.0])
    expected_width = 2.5 * np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(
        prerequisites.domain.lower, expected_center - expected_width
    )
    np.testing.assert_allclose(
        prerequisites.domain.upper, expected_center + expected_width
    )
    assert not prerequisites.domain.covariance.flags.writeable


def test_automatic_domain_supports_rectangular_correlated_impact():
    model = _FakeModel(n_x=0, z_names=("z1", "z2"))
    process = _explicit_process(
        names=("z1", "z2"),
        persistence=[[0.5, 0.1], [0.0, 0.6]],
        impact=[[0.2], [0.1]],
    )

    prerequisites = _resolve_collocation_prerequisites(
        model, CollocationSpec(initialization="steady"), process=process
    )

    covariance = prerequisites.domain.covariance
    np.testing.assert_allclose(
        covariance,
        process.persistence @ covariance @ process.persistence.T
        + process.innovation_impact @ process.innovation_impact.T,
        rtol=1e-12,
        atol=1e-14,
    )


def test_automatic_domain_uses_scale_aware_floor_for_zero_variance():
    model = _FakeModel(x_steady=np.array([2.0]))
    process = _explicit_process(impact=np.zeros((1, 0)))
    spec = CollocationSpec(initialization="steady", domain_min_half_width=1e-4)

    prerequisites = _resolve_collocation_prerequisites(model, spec, process=process)

    np.testing.assert_allclose(prerequisites.domain.lower, [1.9998, -0.0001])
    np.testing.assert_allclose(prerequisites.domain.upper, [2.0002, 0.0001])


def test_automatic_domain_rejects_unstable_dynamics():
    model = _FakeModel(h_x=np.array([[1.0]]))
    process = _explicit_process()

    with pytest.raises(ValueError, match="stable.*explicit domain"):
        _resolve_collocation_prerequisites(
            model, CollocationSpec(initialization="steady"), process=process
        )


def test_automatic_domain_rejects_nonfinite_linear_matrices():
    model = _FakeModel(h_z=np.array([[np.nan]]))
    process = _explicit_process()

    with pytest.raises(ValueError, match="finite linear state dynamics"):
        _resolve_collocation_prerequisites(
            model, CollocationSpec(initialization="steady"), process=process
        )


def test_automatic_domain_rejects_wrong_linear_matrix_shapes():
    model = _FakeModel()
    model.linear_mod.H_z = np.zeros((2, 1))

    with pytest.raises(RuntimeError, match="incompatible shapes"):
        _resolve_collocation_prerequisites(
            model,
            CollocationSpec(initialization="steady"),
            process=_explicit_process(),
        )


def test_automatic_domain_rejects_materially_negative_variance(monkeypatch):
    model = _FakeModel()

    monkeypatch.setattr(
        colloc_spec,
        "solve_discrete_lyapunov",
        lambda state_matrix, innovation_covariance: np.diag([-0.1, 0.1]),
    )

    with pytest.raises(ValueError, match="negative variances"):
        _resolve_collocation_prerequisites(
            model,
            CollocationSpec(initialization="steady"),
            process=_explicit_process(),
        )


def test_automatic_domain_rejects_invalid_covariance_shape(monkeypatch):
    model = _FakeModel()

    monkeypatch.setattr(
        colloc_spec,
        "solve_discrete_lyapunov",
        lambda state_matrix, innovation_covariance: np.eye(1),
    )

    with pytest.raises(ValueError, match="invalid shape"):
        _resolve_collocation_prerequisites(
            model,
            CollocationSpec(initialization="steady"),
            process=_explicit_process(),
        )


def test_automatic_domain_clips_roundoff_negative_variance(monkeypatch):
    model = _FakeModel()

    monkeypatch.setattr(
        colloc_spec,
        "solve_discrete_lyapunov",
        lambda state_matrix, innovation_covariance: np.diag([-1e-14, 0.1]),
    )

    prerequisites = _resolve_collocation_prerequisites(
        model,
        CollocationSpec(initialization="steady"),
        process=_explicit_process(),
    )

    assert prerequisites.domain.covariance[0, 0] == 0.0
