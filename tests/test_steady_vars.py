#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the _STEADY suffix variable functionality.

This module tests that variables with _STEADY suffix are properly handled:
- In steady state models, x_STEADY is replaced with x
- In dynamic models, x_STEADY remains a parameter with the steady state value
"""

import jax
import numpy as np
import pytest

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def set_model_with_steady_vars(flags=None, params=None, steady_guess=None, **kwargs):
    """
    Create a model that uses _STEADY suffix variables.

    This model uses c_STEADY in the optimality condition to represent the
    steady state value of consumption.
    """
    mod = Model(flags=flags, params=params, steady_guess=steady_guess, **kwargs)

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

    # Use c_STEADY in the optimality condition
    # This tests the steady state variable feature
    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0 + 0.0 * c_STEADY"),
    ]

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_steady_vars_in_rules():
    """Test that _STEADY variables are detected in rules."""
    from equilibrium.core.rules import RuleProcessor

    rp = RuleProcessor()

    rules = {
        "intermediate": {"x": "a + b_STEADY"},
        "optimality": {"y": "c_STEADY + d"},
    }

    steady_vars = rp.find_steady_vars(rules)
    assert "b" in steady_vars
    assert "c" in steady_vars
    assert len(steady_vars) == 2


def test_replace_steady_vars_steady_flag_true():
    """Test that _STEADY is replaced with the base variable when steady_flag is True."""
    from equilibrium.core.rules import RuleProcessor
    from equilibrium.utils.containers import MyOrderedDict

    rp = RuleProcessor()

    rules = {
        "intermediate": MyOrderedDict([("x", "a + b_STEADY")]),
        "optimality": MyOrderedDict([("y", "c_STEADY + d")]),
    }

    new_rules = rp.replace_steady_vars(rules, steady_flag=True)

    assert new_rules["intermediate"]["x"] == "a + b"
    assert new_rules["optimality"]["y"] == "c + d"


def test_replace_steady_vars_steady_flag_false():
    """Test that _STEADY is kept when steady_flag is False."""
    from equilibrium.core.rules import RuleProcessor
    from equilibrium.utils.containers import MyOrderedDict

    rp = RuleProcessor()

    rules = {
        "intermediate": MyOrderedDict([("x", "a + b_STEADY")]),
        "optimality": MyOrderedDict([("y", "c_STEADY + d")]),
    }

    new_rules = rp.replace_steady_vars(rules, steady_flag=False)

    # Rules should remain unchanged
    assert new_rules["intermediate"]["x"] == "a + b_STEADY"
    assert new_rules["optimality"]["y"] == "c_STEADY + d"


def test_model_with_steady_vars_parameter_created():
    """Test that _STEADY parameters are created in dynamic model."""
    mod = set_model_with_steady_vars()

    # c_STEADY should be added as a parameter
    assert "c_STEADY" in mod.params
    # Before solving steady state, the value should be 0.0 (initial)
    assert mod.params["c_STEADY"] == 0.0


def test_model_with_steady_vars_after_solve():
    """Test that _STEADY parameters are set to steady state values after solving."""
    mod = set_model_with_steady_vars()

    # Solve the steady state
    mod.solve_steady(calibrate=True, display=False)

    # c_STEADY should now have the steady state value of c
    assert "c_STEADY" in mod.params

    # Get the steady state value of c
    c_steady_value = float(mod.steady_dict["c"])

    # c_STEADY parameter should equal the steady state value of c
    assert np.isclose(mod.params["c_STEADY"], c_steady_value, rtol=1e-6)


def test_steady_model_replaces_steady_vars():
    """Test that the steady state model has _STEADY replaced with base variable."""
    mod = set_model_with_steady_vars()

    # The steady model should not have c_STEADY in its params (it's replaced with c)
    # Instead, it should use c directly
    # We can check this by verifying the steady model solves correctly

    mod.solve_steady(calibrate=True, display=False)

    # The model should solve successfully
    assert mod.res_steady.success

    # Check that steady state values are reasonable (c > 0)
    c_steady = float(mod.steady_dict["c"])
    assert c_steady > 0


def test_steady_vars_in_mod_steady():
    """Test that mod_steady correctly handles _STEADY replacement."""
    mod = set_model_with_steady_vars()

    # mod_steady should have the rules transformed
    # In mod_steady (steady_flag=True), c_STEADY should become c
    assert mod.mod_steady is not None
    assert mod.mod_steady.steady_flag is True


def test_multiple_steady_vars():
    """Test model with multiple _STEADY variables."""
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

    # Use multiple _STEADY variables
    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0 + 0.0 * c_STEADY + 0.0 * K_STEADY"),
    ]

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    # Both c_STEADY and K_STEADY should be parameters
    assert "c_STEADY" in mod.params
    assert "K_STEADY" in mod.params

    # Solve and verify values are set
    mod.solve_steady(calibrate=True, display=False)

    c_steady_value = float(mod.steady_dict["c"])
    K_steady_value = float(mod.steady_dict["K"])

    assert np.isclose(mod.params["c_STEADY"], c_steady_value, rtol=1e-6)
    assert np.isclose(mod.params["K_STEADY"], K_steady_value, rtol=1e-6)


def test_update_copy_preserves_steady_vars():
    """Test that update_copy correctly handles _STEADY variables."""
    mod = set_model_with_steady_vars()
    mod.solve_steady(calibrate=True, display=False)

    # Create an updated copy
    mod_new = mod.update_copy(params={"bet": mod.params["bet"] + 0.01})
    mod_new.solve_steady(calibrate=False, display=False)

    # c_STEADY should exist in the new model
    assert "c_STEADY" in mod_new.params

    # And should be updated to the new steady state
    c_steady_new = float(mod_new.steady_dict["c"])
    assert np.isclose(mod_new.params["c_STEADY"], c_steady_new, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
