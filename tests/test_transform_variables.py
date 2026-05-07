#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for variable transformation feature in Model class.

This module tests the transform_variables() and log_transform() methods
that allow users to transform variables (e.g., to log-space) before
model compilation.
"""

import jax
import numpy as np
import pytest

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def test_log_transform_basic():
    """Test basic log transformation of a single variable."""
    mod = Model()

    mod.params.update(
        {
            "alpha": 0.3,
            "beta": 0.95,
            "delta": 0.1,
        }
    )

    mod.steady_guess.update(
        {
            "C": 1.0,
            "K": 10.0,
        }
    )

    # Define rules using K (not log_K)
    mod.rules["intermediate"] += [
        ("Y", "K ** alpha"),
        ("I", "Y - C"),
    ]

    mod.rules["transition"] += [
        ("K", "I + (1 - delta) * K"),
    ]

    mod.rules["optimality"] += [
        ("C", "beta * C_NEXT - C"),
    ]

    # Apply log transformation to K
    mod.log_transform(["K"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that K was renamed to log_K in transition rules
    assert "log_K" in mod.rules["transition"]
    assert "K" not in mod.rules["transition"]

    # Check that the transition rule contains np.log
    trans_rule = mod.rules["transition"]["log_K"]
    assert "np.log" in trans_rule

    # Check that intermediate rules reference np.exp(log_K)
    y_rule = mod.rules["intermediate"]["Y"]
    assert "np.exp(log_K)" in y_rule

    # Check that steady_guess was transformed
    assert "log_K" in mod.steady_guess
    assert "K" not in mod.steady_guess
    # Check that the value is approximately log(10)
    assert abs(mod.steady_guess["log_K"] - np.log(10.0)) < 1e-10


def test_log_transform_multiple_variables():
    """Test log transformation of multiple variables."""
    mod = Model()

    mod.params.update(
        {
            "alpha": 0.3,
            "beta": 0.95,
            "delta": 0.1,
        }
    )

    mod.steady_guess.update(
        {
            "C": 1.0,
            "K": 10.0,
            "I": 1.0,
        }
    )

    mod.rules["intermediate"] += [
        ("Y", "K ** alpha"),
    ]

    mod.rules["transition"] += [
        ("K", "I + (1 - delta) * K"),
        ("C", "0.5 * C + 0.5 * Y"),
    ]

    mod.rules["optimality"] += [
        ("I", "beta * I_NEXT - I"),
    ]

    # Apply log transformation to both K and C
    mod.log_transform(["K", "C"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that both variables were transformed
    assert "log_K" in mod.rules["transition"]
    assert "log_C" in mod.rules["transition"]
    assert "K" not in mod.rules["transition"]
    assert "C" not in mod.rules["transition"]

    # Check that steady_guess was transformed for both
    assert "log_K" in mod.steady_guess
    assert "log_C" in mod.steady_guess
    assert "K" not in mod.steady_guess
    assert "C" not in mod.steady_guess


def test_transform_variables_custom_prefix():
    """Test custom transformation with custom prefix."""
    mod = Model()

    mod.params.update({"alpha": 0.3})
    mod.steady_guess.update({"K": 10.0, "C": 1.0})

    mod.rules["intermediate"] += [
        ("Y", "K ** alpha"),
    ]

    mod.rules["transition"] += [
        ("K", "0.9 * K"),
    ]

    mod.rules["optimality"] += [
        ("C", "Y - C"),
    ]

    # Apply custom transformation with custom prefix
    mod.transform_variables(["K"], "np.log", "np.exp", prefix="mylog")

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that the custom prefix was used
    assert "mylog_K" in mod.rules["transition"]
    assert "K" not in mod.rules["transition"]

    # Check intermediate rule
    y_rule = mod.rules["intermediate"]["Y"]
    assert "np.exp(mylog_K)" in y_rule


def test_transform_variables_auto_prefix():
    """Test automatic prefix generation from function name."""
    mod = Model()

    mod.params.update({"alpha": 0.3})
    mod.steady_guess.update({"K": 10.0, "C": 1.0})

    mod.rules["transition"] += [
        ("K", "0.9 * K"),
    ]

    mod.rules["optimality"] += [
        ("C", "K - C"),
    ]

    # Apply transformation without specifying prefix
    # Should auto-generate "log" from "np.log"
    mod.transform_variables(["K"], "np.log", "np.exp")

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that auto-generated prefix was used
    assert "log_K" in mod.rules["transition"]


def test_transform_with_next_suffix():
    """Test that transformations handle _NEXT suffix correctly."""
    mod = Model()

    mod.params.update(
        {
            "alpha": 0.3,
            "beta": 0.95,
        }
    )

    mod.steady_guess.update(
        {
            "C": 1.0,
            "K": 10.0,
        }
    )

    mod.rules["intermediate"] += [
        ("Y", "K ** alpha"),
    ]

    mod.rules["expectations"] += [
        ("E_val", "beta * K_NEXT / K"),
    ]

    mod.rules["transition"] += [
        ("K", "0.9 * K"),
    ]

    mod.rules["optimality"] += [
        ("C", "C - Y"),
    ]

    # Apply log transformation
    mod.log_transform(["K"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that K_NEXT was transformed correctly in expectations
    e_val_rule = mod.rules["expectations"]["E_val"]
    assert "np.exp(log_K_NEXT)" in e_val_rule
    assert "np.exp(log_K)" in e_val_rule
    # Make sure the old variable name is completely replaced
    assert "K_NEXT" not in e_val_rule or "log_K_NEXT" in e_val_rule
    # Verify no untransformed K appears (except as part of log_K)
    assert e_val_rule.count("K") == e_val_rule.count("log_K")


def test_transform_in_optimality_rules():
    """Test transformations in optimality rules."""
    mod = Model()

    mod.params.update(
        {
            "alpha": 0.3,
            "beta": 0.95,
        }
    )

    mod.steady_guess.update(
        {
            "x": 1.0,
            "y": 2.0,
        }
    )

    mod.rules["transition"] += [
        ("y", "0.5 * y + 0.5 * x"),
    ]

    # x appears on LHS in optimality rule
    mod.rules["optimality"] += [
        ("x", "x - y"),
    ]

    # Apply transformation to x
    mod.log_transform(["x"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that x was transformed in optimality rule
    assert "log_x" in mod.rules["optimality"]
    opt_rule = mod.rules["optimality"]["log_x"]
    # References to x on the RHS should use the inverse transform
    assert "np.exp(log_x)" in opt_rule
    # Optimality rules are residuals (driven to zero), so the transform
    # function (np.log) is NOT applied as a wrapper — that would change
    # the zero-crossing incorrectly.
    assert "np.log" not in opt_rule


def test_transform_with_analytical_steady():
    """Test that transformations work with analytical_steady rules."""
    mod = Model()

    mod.params.update(
        {
            "alpha": 0.3,
            "delta": 0.1,
        }
    )

    mod.steady_guess.update(
        {
            "K": 10.0,
            "I": 1.0,
        }
    )

    mod.rules["intermediate"] += [
        ("Y", "K ** alpha"),
    ]

    mod.rules["transition"] += [
        ("K", "I + (1 - delta) * K"),
    ]

    mod.rules["optimality"] += [
        ("I", "Y - I"),
    ]

    # Provide analytical steady state formula for K
    mod.rules["analytical_steady"] += [
        ("K", "I / delta"),
    ]

    # Apply log transformation
    mod.log_transform(["K"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that analytical_steady was transformed
    assert "log_K" in mod.rules["analytical_steady"]
    analytical_rule = mod.rules["analytical_steady"]["log_K"]
    assert "np.log" in analytical_rule


def test_no_transformation_if_not_registered():
    """Test that variables are not transformed if no transformation is registered."""
    mod = Model()

    mod.params.update({"alpha": 0.3})
    mod.steady_guess.update({"K": 10.0, "C": 1.0})

    mod.rules["transition"] += [
        ("K", "0.9 * K"),
    ]

    mod.rules["optimality"] += [
        ("C", "K - C"),
    ]

    # Don't register any transformations
    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that K was not transformed
    assert "K" in mod.rules["transition"]
    assert "log_K" not in mod.rules["transition"]


def test_transform_with_intermediate_rules():
    """Test transformations when variable is defined in intermediate rules."""
    mod = Model()

    mod.params.update(
        {
            "alpha": 0.3,
            "delta": 0.1,
        }
    )

    mod.steady_guess.update(
        {
            "K": 10.0,
            "I": 1.0,
        }
    )

    # K is used in intermediate rule
    mod.rules["intermediate"] += [
        ("Y", "K ** alpha"),
        ("C", "Y - I"),
    ]

    mod.rules["transition"] += [
        ("K", "I + (1 - delta) * K"),
    ]

    mod.rules["optimality"] += [
        ("I", "C - I"),
    ]

    # Apply transformation to K
    mod.log_transform(["K"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Check that K was transformed in transition
    assert "log_K" in mod.rules["transition"]

    # Check that intermediate rules use inverse transform
    y_rule = mod.rules["intermediate"]["Y"]
    assert "np.exp(log_K)" in y_rule


def test_multiple_transformations_sequentially():
    """Test applying multiple transformations sequentially."""
    mod = Model()

    mod.params.update({"alpha": 0.3})
    mod.steady_guess.update(
        {
            "K": 10.0,
            "C": 1.0,
            "I": 1.0,
        }
    )

    mod.rules["transition"] += [
        ("K", "0.9 * K"),
        ("C", "0.5 * C + 0.5 * K"),
    ]

    mod.rules["optimality"] += [
        ("I", "K - C"),
    ]

    # Apply transformations separately
    mod.log_transform(["K"])
    mod.log_transform(["C"])

    mod.add_exog("z", pers=0.9, vol=0.01)
    mod.finalize()

    # Both should be transformed
    assert "log_K" in mod.rules["transition"]
    assert "log_C" in mod.rules["transition"]


def test_prefix_generation_from_different_functions():
    """Test automatic prefix generation for various function names."""
    mod = Model()

    # Test the prefix generation method directly
    assert mod._generate_prefix_from_function("np.log") == "log"
    assert mod._generate_prefix_from_function("jnp.sqrt") == "sqrt"
    assert mod._generate_prefix_from_function("np.square") == "square"
    # Test with special characters
    assert mod._generate_prefix_from_function("lambda x: x**2") == "lambda_x_x_2"


def test_log_transform_with_analytical_steady_solves():
    """Test that log_transform works end-to-end when the variable has an analytical_steady rule.

    This is a regression test for a bug where Model.log_transform([var]) would
    fail with a KeyError or incorrect steady-state values when var had an
    analytical_steady solution, because init_dict was populated before
    _apply_transformations() ran and thus retained old variable names.
    """
    mod = Model()

    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 1.0,
        }
    )

    mod.steady_guess.update(
        {
            "I": 0.5,
            "K": 6.0,
        }
    )

    mod.rules["intermediate"] += [
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * K ** (alp - 1.0)"),
        ("y", "Z * K ** alp"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
    ]

    mod.rules["expectations"] += [
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("K", "K_new"),
    ]

    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    # K = I/delta in steady state (capital accumulation K_new=K iff I = delta*K)
    mod.rules["analytical_steady"] += [
        ("K", "I / delta"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    # Apply log transform to the variable that has an analytical_steady rule
    mod.log_transform(["K"])

    mod.finalize()

    # Verify the rules were correctly transformed
    assert "log_K" in mod.rules["analytical_steady"]
    assert "np.log" in mod.rules["analytical_steady"]["log_K"]
    assert "log_K" in mod.rules["transition"]

    # This should solve without errors (was previously broken due to init_dict bug)
    mod.solve_steady(calibrate=False, display=False)

    assert mod.res_steady.success

    # Verify steady-state consistency: K_ss = I_ss / delta
    K_ss = float(np.exp(mod.steady_dict["log_K"]))
    I_ss = float(mod.steady_dict["I"])
    delta = mod.params["delta"]

    assert np.isclose(K_ss, I_ss / delta, rtol=1e-6), (
        f"K_ss={K_ss:.6f} should equal I_ss/delta={I_ss/delta:.6f}"
    )


def test_log_transform_state_var_without_analytical_steady_solves():
    """Test that log_transform works for a state variable without analytical_steady.

    This ensures the init_dict fix doesn't break the non-analytical case,
    and that the initial guess is correctly transformed for the solver.
    """
    mod = Model()

    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 1.0,
        }
    )

    mod.steady_guess.update(
        {
            "I": 0.5,
            "K": 6.0,
        }
    )

    mod.rules["intermediate"] += [
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * K ** (alp - 1.0)"),
        ("y", "Z * K ** alp"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
    ]

    mod.rules["expectations"] += [
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("K", "K_new"),
    ]

    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    # No analytical_steady - solver must find log_K numerically
    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.log_transform(["K"])
    mod.finalize()

    # init_dict should have log_K (transformed), not K (old name)
    assert "log_K" in mod.init_dict
    assert "K" not in mod.init_dict

    mod.solve_steady(calibrate=False, display=False)

    assert mod.res_steady.success

    K_ss = float(np.exp(mod.steady_dict["log_K"]))
    I_ss = float(mod.steady_dict["I"])
    delta = mod.params["delta"]

    assert np.isclose(K_ss, I_ss / delta, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
