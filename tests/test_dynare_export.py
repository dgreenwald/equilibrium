#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Model.to_dynare method, particularly the initval block generation.

This module tests that the to_dynare method correctly generates:
1. Exogenous variables set to zero
2. Endogenous states and policy controls from steady_dict (if solved) or analytical_steady/steady_guess
3. Intermediate variables using their rules
4. Expectations variables with _NEXT suffixes removed for steady state
"""

import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def create_simple_rbc_model():
    """
    Create a simple RBC model for testing Dynare export.

    Returns
    -------
    Model
        A basic RBC model with capital accumulation.
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

    return mod


def create_model_with_analytical_steady():
    """
    Create a model with analytical_steady rules.

    Returns
    -------
    Model
        A model with analytical steady state formula.
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

    # Analytical steady state for log_K
    mod.rules["analytical_steady"] += [
        ("log_K", "np.log(I / delta)"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_to_dynare_creates_file():
    """Test that to_dynare creates a .mod file."""
    mod = create_simple_rbc_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_model.mod"
        content = mod.to_dynare(output_path=output_path)

        assert output_path.exists()
        assert content == output_path.read_text()


def test_to_dynare_has_initval_block_unsolved():
    """Test that initval block is generated for unsolved model."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that initval block is present
    assert "initval;" in content
    assert "end;" in content

    # Check that exogenous variable is set to zero
    assert "Z_til = 0;" in content

    # Check that endogenous variables are initialized
    # Should use steady_guess values
    assert "I =" in content
    assert "log_K =" in content


def test_to_dynare_has_initval_block_solved():
    """Test that initval block uses steady_dict when model is solved."""
    mod = create_simple_rbc_model()
    mod.solve_steady(calibrate=False, display=False)

    content = mod.to_dynare()

    # Check that initval block is present
    assert "initval;" in content

    # Check that exogenous variable is set to zero
    assert "Z_til = 0;" in content

    # Check that endogenous variables use steady state values
    I_steady = float(mod.steady_dict["I"])
    log_K_steady = float(mod.steady_dict["log_K"])

    # These should appear in the initval block with high precision
    assert f"I = {I_steady:.16f};" in content
    assert f"log_K = {log_K_steady:.16f};" in content


def test_to_dynare_initval_has_intermediate_vars():
    """Test that initval block includes intermediate variables."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that intermediate variables are in initval block
    assert "K = exp(log_K);" in content
    assert "Z = Z_bar + Z_til;" in content
    assert "y = Z * (K ^ alp);" in content
    assert "c = y - I;" in content


def test_to_dynare_initval_has_expectations_flattened():
    """Test that initval block has expectations with _NEXT removed."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # The expectation rule has _NEXT suffixes
    # In initval block, these should be removed for steady state
    assert "E_Om_K = bet * (uc / uc) * (fk + (1.0 - delta));" in content


def test_to_dynare_initval_with_analytical_steady():
    """Test that initval block uses analytical_steady rules when present."""
    mod = create_model_with_analytical_steady()

    content = mod.to_dynare()

    # Check that analytical_steady formula is used in initval
    # Should see log_K = log(I / delta)
    assert "log_K = log(I / delta);" in content


def test_to_dynare_initval_with_solved_analytical_steady():
    """Test that initval block uses steady_dict even when analytical_steady exists."""
    mod = create_model_with_analytical_steady()
    mod.solve_steady(calibrate=False, display=False)

    content = mod.to_dynare()

    # When solved, should use numerical value from steady_dict
    log_K_steady = float(mod.steady_dict["log_K"])
    assert f"log_K = {log_K_steady:.16f};" in content

    # Should not use the analytical formula when steady state is solved
    assert "log_K = log(I / delta);" not in content


def test_to_dynare_structure():
    """Test the overall structure of the Dynare .mod file."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that blocks appear in the expected order
    param_idx = content.find("parameters")
    var_idx = content.find("var ")
    varexo_idx = content.find("varexo")
    model_idx = content.find("model;")
    initval_idx = content.find("initval;")

    # All should be present (>= 0, not necessarily > 0)
    assert param_idx >= 0
    assert var_idx >= 0
    assert varexo_idx >= 0
    assert model_idx >= 0
    assert initval_idx >= 0

    # And in the correct order
    assert param_idx < var_idx < varexo_idx < model_idx < initval_idx


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
