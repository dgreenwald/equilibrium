#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for handling steady-state solve failures gracefully.

This module tests that the Model class properly handles cases where
the steady-state solver fails, ensuring no AttributeError is raised
when accessing steady_dict after a failed solve.
"""

import jax
import numpy as np
import pytest

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def create_problematic_model():
    """
    Create a model that may fail to solve from certain initial guesses.

    This model is intentionally set up with a difficult initial guess
    that can cause the solver to fail, allowing us to test the error
    handling path.
    """
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

    # Use a bad initial guess that might cause solver to fail
    mod.steady_guess.update(
        {
            "I": 0.0001,  # Very small value that may cause issues
            "log_K": -10.0,  # Extreme value
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

    return mod


def test_steady_dict_initialized():
    """Test that steady_dict is initialized to empty dict in __init__."""
    mod = Model()

    # Before finalize, steady_dict should exist as empty dict
    assert hasattr(mod, "steady_dict")
    assert mod.steady_dict == {}


def test_steady_dict_on_failed_solve():
    """Test that steady_dict remains defined (as empty dict) after failed solve."""
    mod = create_problematic_model()

    # Try to solve with load_initial_guess=False to avoid loading from cache
    # This may fail due to bad initial guess
    result = mod.solve_steady(
        calibrate=False,
        save=False,
        load_initial_guess=False,
        display=False,
    )

    # Whether solve succeeded or failed, steady_dict should be defined
    assert hasattr(mod, "steady_dict")

    # If solve failed, steady_dict should be empty dict
    if not result.success:
        assert mod.steady_dict == {}


def test_no_attribute_error_on_failed_solve():
    """Test that no AttributeError is raised when solver fails."""
    mod = create_problematic_model()

    # This should not raise AttributeError even if solve fails
    # The solve_steady method should handle failures gracefully
    try:
        mod.solve_steady(
            calibrate=False,
            save=False,
            load_initial_guess=False,
            display=False,
        )
        # Test passes if we get here without AttributeError
    except AttributeError as e:
        if "steady_dict" in str(e):
            pytest.fail(f"AttributeError related to steady_dict was raised: {e}")
        else:
            # Re-raise if it's a different AttributeError
            raise


def test_steady_dict_preserved_on_successful_solve():
    """Test that steady_dict is properly set when solve succeeds."""
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

    # Use reasonable initial guess
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

    result = mod.solve_steady(
        calibrate=False,
        save=False,
        load_initial_guess=False,
        display=False,
    )

    # On successful solve, steady_dict should be populated
    if result.success:
        assert hasattr(mod, "steady_dict")
        assert mod.steady_dict != {}
        # steady_dict can be either a dict or NamedTuple depending on implementation
        # Check for the presence of expected variables
        if hasattr(mod.steady_dict, "_fields"):
            # NamedTuple - check if K is in fields
            assert "K" in mod.steady_dict._fields
        else:
            # dict - check if K is a key
            assert "K" in mod.steady_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
