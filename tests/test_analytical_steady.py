#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the analytical_steady validation in Model.solve_steady.

This module tests that incorrect analytical_steady rules are detected
and an appropriate error is raised during steady state computation.

The analytical_steady rules allow users to provide analytical formulas for
steady state values instead of letting the solver find them numerically.
If these formulas are incorrect, the resulting steady state won't satisfy
the original transition equations. These tests verify that such errors
are caught and reported clearly.
"""

import jax
import numpy as np
import pytest

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def create_model_with_analytical_steady(analytical_steady_formula):
    """
    Create a model with a specific analytical_steady formula for log_K.

    The model has two state variables (log_K and log_H) so that even when
    we provide an analytical formula for log_K, the solver still needs to
    find the steady state for the remaining equations.

    Parameters
    ----------
    analytical_steady_formula : str
        The analytical formula for log_K in steady state.
        Correct formula is: "np.log(I / delta)"

    Returns
    -------
    Model
        A model with the specified analytical_steady rule.
    """
    mod = Model()

    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
            "rho_h": 0.95,  # Persistence for habit
        }
    )

    mod.steady_guess.update(
        {
            "I": 0.5,
            "log_K": np.log(6.0),
            "log_H": np.log(0.5),  # Habit stock
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
        ("H", "np.exp(log_H)"),
        ("H_new", "rho_h * H + (1 - rho_h) * c"),  # Habit accumulates from consumption
    ]

    mod.rules["expectations"] += [
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("log_K", "np.log(K_new)"),
        ("log_H", "np.log(H_new)"),  # Second state variable
    ]

    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    # Analytical steady state for log_K
    mod.rules["analytical_steady"] += [
        ("log_K", analytical_steady_formula),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_correct_analytical_steady_succeeds():
    """Test that a model with correct analytical_steady rules solves successfully."""
    # Correct formula: in steady state, K_new = K
    # => I + (1 - delta) * K = K
    # => I = delta * K
    # => K = I / delta
    # => log_K = np.log(I / delta)
    mod = create_model_with_analytical_steady("np.log(I / delta)")

    # Should solve without errors
    mod.solve_steady(calibrate=True, display=False)

    assert mod.res_steady.success

    # Verify the steady state values make sense
    K = float(mod.steady_dict["K"])
    I_val = float(mod.steady_dict["I"])
    delta = mod.params["delta"]

    # K_new should equal K
    K_new = I_val + (1 - delta) * K
    assert np.isclose(K_new, K, rtol=1e-8)


def test_incorrect_analytical_steady_raises_error():
    """Test that a model with incorrect analytical_steady rules raises an error."""
    # Incorrect formula: adding a constant offset
    mod = create_model_with_analytical_steady("np.log(I / delta) + 0.3")

    # Should raise ValueError due to failed validation
    with pytest.raises(ValueError, match="Steady state verification failed"):
        mod.solve_steady(calibrate=True, display=False)


def test_incorrect_analytical_steady_error_mentions_variable():
    """Test that the error message mentions the problematic variable."""
    # Incorrect formula
    mod = create_model_with_analytical_steady("np.log(I / delta) + 0.3")

    # Should raise ValueError with information about which equation failed
    with pytest.raises(ValueError, match="log_K"):
        mod.solve_steady(calibrate=True, display=False)


def test_incorrect_analytical_steady_mentions_analytical_rules():
    """Test that the error message hints about analytical_steady rules."""
    # Incorrect formula
    mod = create_model_with_analytical_steady("np.log(I / delta) + 0.3")

    # Should raise ValueError with a hint about analytical_steady rules
    with pytest.raises(ValueError, match="analytical_steady"):
        mod.solve_steady(calibrate=True, display=False)


def test_incorrect_analytical_steady_reports_residual():
    """Test that the error message includes the residual value."""
    # Incorrect formula
    mod = create_model_with_analytical_steady("np.log(I / delta) + 0.3")

    # Should raise ValueError with a residual value
    with pytest.raises(ValueError, match=r"log_K=.*e"):
        mod.solve_steady(calibrate=True, display=False)


def test_model_without_analytical_steady_unaffected():
    """Test that models without analytical_steady rules work normally."""
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

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    # No analytical_steady rules

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    # Should solve without errors
    mod.solve_steady(calibrate=True, display=False)

    assert mod.res_steady.success


def test_analytical_steady_with_different_error_magnitudes():
    """Test that both small and large errors in analytical_steady are caught."""
    # Small error (offset of 0.01)
    mod_small = create_model_with_analytical_steady("np.log(I / delta) + 0.01")

    with pytest.raises(ValueError, match="Steady state verification failed"):
        mod_small.solve_steady(calibrate=True, display=False)

    # Large error (offset of 1.0)
    mod_large = create_model_with_analytical_steady("np.log(I / delta) + 1.0")

    with pytest.raises(ValueError, match="Steady state verification failed"):
        mod_large.solve_steady(calibrate=True, display=False)


def test_analytical_steady_with_scale_error():
    """Test that scale errors in analytical_steady formulas are caught."""
    # Multiply by wrong factor
    mod = create_model_with_analytical_steady("np.log(I / (0.5 * delta))")

    with pytest.raises(ValueError, match="Steady state verification failed"):
        mod.solve_steady(calibrate=True, display=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
