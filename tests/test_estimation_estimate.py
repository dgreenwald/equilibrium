"""Tests for high-level estimation entry point."""

import numpy as np
import pytest

from equilibrium import Model
from equilibrium.estimation import EstimParam, build_state_space, estimate
from equilibrium.settings import get_settings


def _make_estimation_model():
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
    mod.linearize()
    return mod


def test_estimate_validates_data_shape():
    mod = _make_estimation_model()
    params = [EstimParam(name="bet", prior="norm", mean=0.95, sd=0.02)]
    data = np.zeros((10, 1))

    with pytest.raises(ValueError, match="data must have 2 columns"):
        estimate(
            mod,
            params,
            data,
            observables=["y", "c"],
            estimation_label="shape_check",
            find_mode=False,
            compute_hessian=False,
            CH_inv=np.eye(1),
            Nsim=5,
        )


def test_estimate_validates_observables():
    mod = _make_estimation_model()
    params = [EstimParam(name="bet", prior="norm", mean=0.95, sd=0.02)]
    data = np.zeros((10, 1))

    with pytest.raises(ValueError, match="Unknown observables"):
        estimate(
            mod,
            params,
            data,
            observables=["not_a_var"],
            estimation_label="obs_check",
            find_mode=False,
            compute_hessian=False,
            CH_inv=np.eye(1),
            Nsim=5,
        )


def test_estimate_smoke_single_chain(tmp_path):
    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path
    try:
        mod = _make_estimation_model()
        observables = ["y", "c"]
        ssm = build_state_space(mod, observables=observables)

        np.random.seed(0)
        data, _ = ssm.simulate(x_1=np.zeros(ssm.Nx), Nt=15)

        result = estimate(
            mod,
            [
                EstimParam(
                    name="bet",
                    prior="norm",
                    mean=0.95,
                    sd=0.02,
                    lb=0.8,
                    ub=0.999,
                )
            ],
            data,
            observables=observables,
            estimation_label="smoke_run",
            Nsim=8,
            n_chains=1,
            find_mode=False,
            compute_hessian=False,
            CH_inv=np.array([[0.01]]),
            sample_kwargs={"log": False},
        )

        assert result.mode.shape == (1,)
        assert len(result.chains) == 1
        assert result.chains[0].draws.shape == (8, 1)
        assert np.all(np.isfinite(result.chains[0].post_sim))
        assert (tmp_path / "estimation" / mod.label / "smoke_run" / "chain0.npz").exists()
    finally:
        settings.paths.save_dir = old_save_dir
