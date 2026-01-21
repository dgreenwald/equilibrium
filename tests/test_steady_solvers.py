#!/usr/bin/env python3
"""Tests for steady-state solver selection."""

import numpy as np
import pytest

from equilibrium import Model


def create_simple_model():
    """Create a simple RBC model for testing."""
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
        ("K", "np.exp(log_K)"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
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

    return mod


@pytest.mark.parametrize("solver", ["scipy", "newton"])
def test_solve_steady_works_with_solver(solver):
    mod = create_simple_model()
    result = mod.solve_steady(calibrate=False, display=False, solver=solver)

    assert result.success
    assert np.isfinite(np.asarray(result.x)).all()


def test_solve_steady_solvers_are_consistent():
    mod_scipy = create_simple_model()
    res_scipy = mod_scipy.solve_steady(calibrate=False, display=False, solver="scipy")

    mod_newton = create_simple_model()
    res_newton = mod_newton.solve_steady(
        calibrate=False, display=False, solver="newton"
    )

    assert res_scipy.success
    assert res_newton.success
    assert np.allclose(res_scipy.x, res_newton.x, atol=1e-6, rtol=1e-6)
