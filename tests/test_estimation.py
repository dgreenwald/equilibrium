"""Top-level public estimation API smoke test."""

import numpy as np

from equilibrium import (
    EstimParam,
    Model,
    estimate,
    load_estimation,
)
from equilibrium.estimation import build_state_space
from equilibrium.settings import get_settings


def _make_public_estimation_model():
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


def test_top_level_estimation_api_smoke(tmp_path):
    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path
    try:
        mod = _make_public_estimation_model()
        observables = ["y", "c"]
        ssm = build_state_space(mod, observables=observables)

        np.random.seed(0)
        data, _ = ssm.simulate(x_1=np.zeros(ssm.Nx), Nt=12)

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
            estimation_label="top_level_smoke",
            Nsim=6,
            n_chains=1,
            find_mode=False,
            compute_hessian=False,
            CH_inv=np.array([[0.01]]),
            sample_kwargs={"log": False},
        )

        assert result.estimation_label == "top_level_smoke"
        loaded = load_estimation(mod.label, "top_level_smoke")
        assert loaded.param_names == ["bet"]
        assert loaded.observables == observables
        assert len(loaded.chains) == 1
        np.testing.assert_allclose(loaded.chains[0].draws, result.chains[0].draws)
    finally:
        settings.paths.save_dir = old_save_dir
