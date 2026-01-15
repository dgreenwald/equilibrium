#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for handling empty u or x arrays in Model.solve_steady.

This module tests that the Model class properly handles cases where
all state variables (x) or all policy variables (u), or both, have
analytical solutions provided in rules['analytical_steady'].
When this happens, the lists can be empty and array operations must
handle this gracefully.
"""

import jax
import numpy as np
import pytest

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def test_empty_x_with_analytical_steady():
    """Test that solve_steady works when all x variables have analytical solutions."""
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
        ("K", "np.exp(log_K)"),
        ("K_new", "I + (1.0 - delta) * K"),
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

    # Provide analytical solution for log_K, making x empty
    # In steady state: K = I / delta (from K_new = K)
    mod.rules["analytical_steady"] += [
        ("log_K", "np.log(I / delta)"),
    ]

    mod.rules["optimality"] += [
        # Optimality: residual = I - (E_Om_K - 1.0) should be zero
        ("I", "E_Om_K - 1.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    # This should work without crashing even though x is empty
    result = mod.solve_steady(
        calibrate=False,
        save=False,
        load_initial_guess=False,
        display=False,
    )

    # Should successfully solve
    assert result.success
    assert hasattr(mod, "steady_dict")
    assert mod.steady_dict != {}

    # Verify the steady state is consistent
    K_val = float(mod.steady_dict["K"])
    I_val = float(mod.steady_dict["I"])
    # In steady state: K = I / delta
    assert np.isclose(K_val, I_val / mod.params["delta"], rtol=1e-6)


def test_empty_u_with_analytical_steady():
    """Test that solve_steady works when all u variables have analytical solutions."""
    mod = Model()

    # Use correct analytical steady-state values (pre-computed from solving the model)
    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
            "I_ss": 0.5416168218211491,  # Correct analytical steady-state investment
        }
    )

    mod.steady_guess.update(
        {
            "log_K": np.log(6.0),
            "I": 0.6,
        }
    )

    mod.rules["intermediate"] += [
        ("K", "np.exp(log_K)"),
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("log_K", "np.log(K_new)"),
    ]

    # Optimality condition: residual = I - (E_Om_K - 1.0) should be zero
    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    # Provide correct analytical solution for I
    mod.rules["analytical_steady"] += [
        ("I", "I_ss"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    # This should work without crashing even though u is empty
    result = mod.solve_steady(
        calibrate=False,
        save=False,
        load_initial_guess=False,
        display=False,
    )

    # Should successfully solve
    assert result.success
    assert hasattr(mod, "steady_dict")
    assert mod.steady_dict != {}

    # Verify the analytical solution is used
    I_val = float(mod.steady_dict["I"])
    assert np.isclose(I_val, mod.params["I_ss"], rtol=1e-8)


def test_both_empty_with_analytical_steady():
    """Test that solve_steady works when both u and x have analytical solutions."""
    mod = Model()

    # Use correct analytical steady-state values (pre-computed)
    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
            "log_K_ss": 1.689388594620209,  # Correct analytical steady-state log_K
            "I_ss": 0.5416168218211491,  # Correct analytical steady-state I
        }
    )

    mod.steady_guess.update(
        {
            "log_K": np.log(6.0),
            "I": 0.6,
        }
    )

    # Model with proper economic structure
    mod.rules["intermediate"] += [
        ("K", "np.exp(log_K)"),
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("log_K", "np.log(K_new)"),
    ]

    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    # Both optimality and transition have correct analytical solutions
    mod.rules["analytical_steady"] += [
        ("log_K", "log_K_ss"),
        ("I", "I_ss"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    # This should work without crashing even though both u and x are empty
    result = mod.solve_steady(
        calibrate=False,
        save=False,
        load_initial_guess=False,
        display=False,
    )

    # Should successfully solve
    assert result.success
    assert hasattr(mod, "steady_dict")
    assert mod.steady_dict != {}

    # Verify the analytical solutions are correct
    K_val = float(mod.steady_dict["K"])
    I_val = float(mod.steady_dict["I"])
    log_K_val = float(mod.steady_dict["log_K"])

    assert np.isclose(log_K_val, mod.params["log_K_ss"], rtol=1e-8)
    assert np.isclose(I_val, mod.params["I_ss"], rtol=1e-8)

    # Verify steady state condition: K_new should equal K
    K_new_val = float(mod.steady_dict["K_new"])
    assert np.isclose(K_new_val, K_val, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
