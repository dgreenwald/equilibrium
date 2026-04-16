"""Tests for measurement selection and state-space likelihood bridge."""

import numpy as np

from equilibrium import Model
from equilibrium.estimation import build_state_space, log_likelihood, log_likelihood_ssm


def _make_observable_rbc_model():
    mod = Model()
    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
        }
    )
    mod.steady_guess.update(
        {
            "I": 0.5,
            "log_K": np.log(6.0),
        }
    )

    mod.rules["intermediate"] += [
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("R_ann", "400.0 * np.log(fk + (1.0 - delta))"),
        ("uc", "c ** (-gam)"),
        ("K", "np.exp(log_K)"),
    ]
    mod.rules["expectations"] += [
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]
    mod.rules["read_expectations"] += [
        ("exp_return_gap", "E_Om_K - 1.0"),
    ]
    mod.rules["transition"] += [
        ("log_K", "np.log(K_new)"),
    ]
    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)
    mod.finalize()
    mod.solve_steady(calibrate=False, display=False)
    linear_model = mod.linearize()
    return mod, linear_model


def test_build_state_space_selects_existing_model_variables():
    mod, linear_model = _make_observable_rbc_model()
    observables = ["y", "I", "Z_til", "R_ann", "exp_return_gap"]

    ssm = build_state_space(mod, observables=observables)
    assert ssm.A.shape == linear_model.A_s.shape
    assert ssm.R.shape == linear_model.B_s.shape
    assert ssm.Z.shape == (5, mod.N["u"] + mod.N["x"] + mod.N["z"])
    np.testing.assert_allclose(ssm.b, [mod.steady_dict[name] for name in observables])

    np.testing.assert_allclose(ssm.Z[1], np.array([1.0, 0.0, 0.0]))

    z_col = mod.N["u"] + mod.N["x"] + mod.var_lists["z"].index("Z_til")
    expected_z_row = np.zeros(ssm.Z.shape[1])
    expected_z_row[z_col] = 1.0
    np.testing.assert_allclose(ssm.Z[2], expected_z_row)
    np.testing.assert_allclose(
        ssm.Z[4],
        np.asarray(linear_model.L[mod.var_lists["read_expectations"].index("exp_return_gap")]),
    )


def test_build_state_space_and_log_likelihood():
    mod, _ = _make_observable_rbc_model()
    observables = ["y", "c"]

    ssm = build_state_space(mod, observables=observables, meas_err={"y": 0.1})
    assert ssm.H.shape == (2, 2)
    np.testing.assert_allclose(np.diag(ssm.H), [0.01, 0.0])

    np.random.seed(0)
    data, _ = ssm.simulate(x_1=np.zeros(ssm.Nx), Nt=20)
    ll = log_likelihood(mod, data, observables=observables, meas_err={"y": 0.1})
    assert np.isfinite(ll)
    np.testing.assert_allclose(
        ll,
        log_likelihood_ssm(ssm, data),
    )


def test_log_likelihood_bad_shape_returns_floor():
    mod, _ = _make_observable_rbc_model()
    data = np.zeros((10, 1))
    assert log_likelihood(mod, data, observables=["y", "c"]) == -1e10


def test_log_likelihood_ssm_bad_shape_returns_floor():
    mod, _ = _make_observable_rbc_model()
    ssm = build_state_space(mod, observables=["y", "c"])
    data = np.zeros((10, 1))
    assert log_likelihood_ssm(ssm, data) == -1e10
