#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for symbolic preference block generator.

This module tests the preference_block function which uses SymPy
to automatically compute marginal utilities through symbolic differentiation.
"""

import pytest


def test_preference_block_basic_crra():
    """Test basic CRRA preferences without housing or labor."""
    from equilibrium.blocks import preference_block

    block = preference_block(util_type="crra")

    # Check that uc_AGENT exists
    assert "uc_AGENT" in block.rules["intermediate"]

    # Check Lagrange multipliers exist
    assert "Lam_1_AGENT" in block.rules["intermediate"]
    assert "Lam_0_AGENT" in block.rules["intermediate"]
    assert block.rules["intermediate"]["Lam_1_AGENT"] == "uc_AGENT"
    assert block.rules["intermediate"]["Lam_0_AGENT"] == "uc_AGENT / bet_ATYPE"

    # Check nominal marginal utility exists by default
    assert "Lam_1_nom_AGENT" in block.rules["intermediate"]
    assert block.rules["intermediate"]["Lam_1_nom_AGENT"] == "Lam_1_AGENT / pi"

    # Should not have housing or labor related variables
    assert "uh_AGENT" not in block.rules["intermediate"]
    assert "n_un_AGENT" not in block.rules["intermediate"]
    assert "x_AGENT" not in block.rules["intermediate"]
    assert "v_AGENT" not in block.rules["intermediate"]


def test_preference_block_unit_eis():
    """Test unit EIS (log) preferences."""
    from equilibrium.blocks import preference_block

    block = preference_block(util_type="unit_eis")

    # Check that uc_AGENT exists
    assert "uc_AGENT" in block.rules["intermediate"]

    # For log utility u = log(c), we have uc = 1/c
    # Check that the expression contains "np.log" or represents derivative correctly
    uc_expr = block.rules["intermediate"]["uc_AGENT"]
    # The derivative of log(c) is 1/c
    # In symbolic form, this should be something like "1/c_AGENT" or "c_AGENT**(-1)"
    assert "c_AGENT" in uc_expr


def test_preference_block_risk_neutral():
    """Test risk neutral preferences."""
    from equilibrium.blocks import preference_block

    block = preference_block(util_type="risk_neutral")

    # Check that uc_AGENT exists
    assert "uc_AGENT" in block.rules["intermediate"]

    # For risk neutral u = c, we have uc = 1
    uc_expr = block.rules["intermediate"]["uc_AGENT"]
    # Should be constant 1
    assert uc_expr == "1"


def test_preference_block_ghh_with_labor():
    """Test GHH preferences with labor."""
    from equilibrium.blocks import preference_block

    block = preference_block(util_type="ghh", labor=True)

    # Check that uc_AGENT exists
    assert "uc_AGENT" in block.rules["intermediate"]

    # Check that labor disutility exists
    assert "n_un_AGENT" in block.rules["intermediate"]
    assert "v_AGENT" in block.rules["intermediate"]

    # Check that composite consumption x exists
    assert "x_AGENT" in block.rules["intermediate"]

    # Verify v contains labor disutility formula
    v_expr = block.rules["intermediate"]["v_AGENT"]
    assert "eta_ATYPE" in v_expr
    assert "n_AGENT" in v_expr
    assert "varphi_ATYPE" in v_expr


def test_preference_block_cobb_douglas_housing():
    """Test Cobb-Douglas housing aggregation."""
    from equilibrium.blocks import preference_block

    block = preference_block(housing=True, housing_spec="cobb_douglas")

    # Check that housing marginal utility exists
    assert "uh_AGENT" in block.rules["intermediate"]

    # Check that composite consumption exists
    assert "x_AGENT" in block.rules["intermediate"]

    # Check that consumption marginal utility exists
    assert "uc_AGENT" in block.rules["intermediate"]

    # Verify x contains Cobb-Douglas formula with h and c
    x_expr = block.rules["intermediate"]["x_AGENT"]
    assert "h_AGENT" in x_expr
    assert "c_AGENT" in x_expr
    assert "xi_ATYPE" in x_expr


def test_preference_block_ces_housing():
    """Test CES housing aggregation."""
    from equilibrium.blocks import preference_block

    block = preference_block(housing=True, housing_spec="ces")

    # Check that housing marginal utility exists
    assert "uh_AGENT" in block.rules["intermediate"]

    # Check that composite consumption exists
    assert "x_AGENT" in block.rules["intermediate"]

    # Verify x contains CES formula
    x_expr = block.rules["intermediate"]["x_AGENT"]
    assert "h_AGENT" in x_expr
    assert "c_AGENT" in x_expr
    assert "xi_ATYPE" in x_expr
    assert "eps_h_ATYPE" in x_expr


def test_preference_block_h_exponent_housing():
    """Test h_exponent housing aggregation."""
    from equilibrium.blocks import preference_block

    block = preference_block(housing=True, housing_spec="h_exponent")

    # Check that housing marginal utility exists
    assert "uh_AGENT" in block.rules["intermediate"]

    # Check that composite consumption exists
    assert "x_AGENT" in block.rules["intermediate"]

    # Verify x contains h^xi * c formula
    x_expr = block.rules["intermediate"]["x_AGENT"]
    assert "h_AGENT" in x_expr
    assert "c_AGENT" in x_expr
    assert "xi_ATYPE" in x_expr


def test_preference_block_substitutes_housing():
    """Test substitutes housing aggregation."""
    from equilibrium.blocks import preference_block

    block = preference_block(housing=True, housing_spec="substitutes")

    # Check that housing marginal utility exists
    assert "uh_AGENT" in block.rules["intermediate"]

    # Check that composite consumption exists
    assert "x_AGENT" in block.rules["intermediate"]

    # Verify x contains linear combination
    x_expr = block.rules["intermediate"]["x_AGENT"]
    assert "h_AGENT" in x_expr
    assert "c_AGENT" in x_expr
    assert "xi_ATYPE" in x_expr


def test_preference_block_ql_housing():
    """Test ql_housing specification."""
    from equilibrium.blocks import preference_block

    block = preference_block(housing=True, housing_spec="ql_housing", labor=True)

    # Check that housing marginal utility exists
    assert "uh_AGENT" in block.rules["intermediate"]

    # Check that composite consumption exists
    assert "x_AGENT" in block.rules["intermediate"]

    # Check that labor disutility exists
    assert "v_AGENT" in block.rules["intermediate"]

    # Verify x contains the ql_housing formula
    x_expr = block.rules["intermediate"]["x_AGENT"]
    assert "h_AGENT" in x_expr
    assert "c_AGENT" in x_expr
    assert "v_AGENT" in x_expr


def test_preference_block_no_nominal():
    """Test symbolic preferences without nominal marginal utility."""
    from equilibrium.blocks import preference_block

    block = preference_block(nominal=False)

    # Check that nominal marginal utility does not exist
    assert "Lam_1_nom_AGENT" not in block.rules["intermediate"]

    # But other rules should still exist
    assert "uc_AGENT" in block.rules["intermediate"]
    assert "Lam_1_AGENT" in block.rules["intermediate"]
    assert "Lam_0_AGENT" in block.rules["intermediate"]


def test_preference_block_custom_agent_atype():
    """Test symbolic preferences with custom agent and atype."""
    from equilibrium.blocks import preference_block

    block = preference_block(agent="borrower", atype="b")

    # Check that variables use the custom agent suffix
    assert "uc_borrower" in block.rules["intermediate"]
    assert "Lam_1_borrower" in block.rules["intermediate"]
    assert "Lam_0_borrower" in block.rules["intermediate"]

    # Check that the expression uses custom atype
    uc_expr = block.rules["intermediate"]["uc_borrower"]
    assert "psi_b" in uc_expr
    assert "c_borrower" in uc_expr


def test_preference_block_returns_model_block():
    """Test that preference_block returns a ModelBlock instance."""
    from equilibrium.blocks import preference_block
    from equilibrium.model import ModelBlock

    block = preference_block()

    assert isinstance(block, ModelBlock)


def test_preference_block_has_intermediate_rules():
    """Test that preference_block adds rules to the intermediate category."""
    from equilibrium.blocks import preference_block

    block = preference_block()

    # All rules should be in the 'intermediate' category
    assert len(block.rules["intermediate"]) > 0

    # Other categories should be empty
    for key in ["transition", "optimality", "expectations", "calibration"]:
        assert len(block.rules[key]) == 0


def test_preference_block_invalid_util_type():
    """Test that invalid util_type raises ValueError."""
    from equilibrium.blocks import preference_block

    with pytest.raises(ValueError, match="Invalid util_type"):
        preference_block(util_type="invalid")


def test_preference_block_invalid_housing_spec():
    """Test that invalid housing_spec raises ValueError."""
    from equilibrium.blocks import preference_block

    with pytest.raises(ValueError, match="Invalid housing_spec"):
        preference_block(housing=True, housing_spec="invalid")


def test_preference_block_integration_with_model():
    """Test that preference_block can be added to a Model."""
    from equilibrium.blocks import preference_block
    from equilibrium.model import Model

    model = Model()
    block = preference_block()

    # Should be able to add the block to a model
    model.add_block(block)

    # Check that rules were added
    assert "uc_AGENT" in model.rules["intermediate"]
    assert "Lam_1_AGENT" in model.rules["intermediate"]


def test_preference_block_can_be_renamed():
    """Test that preference_block can be used with replacements for multiple agents."""
    from equilibrium.blocks import preference_block
    from equilibrium.model import Model

    model = Model()

    # Add blocks for multiple agents using replacement
    borrower_block = preference_block()
    lender_block = preference_block()

    model.add_block(borrower_block, rename={"AGENT": "borrower", "ATYPE": "b"})
    model.add_block(lender_block, rename={"AGENT": "lender", "ATYPE": "l"})

    # Check that both agents have their own rules
    assert "uc_borrower" in model.rules["intermediate"]
    assert "uc_lender" in model.rules["intermediate"]
    assert "Lam_1_borrower" in model.rules["intermediate"]
    assert "Lam_1_lender" in model.rules["intermediate"]


def test_preference_block_crra_derivative():
    """Test that CRRA marginal utility matches expected analytical form."""
    from equilibrium.blocks import preference_block

    block = preference_block(util_type="crra")

    # For CRRA u = c^(1-psi)/(1-psi), we have uc = c^(-psi)
    uc_expr = block.rules["intermediate"]["uc_AGENT"]
    # The expression should contain c_AGENT**(-psi_ATYPE) or equivalent
    assert "c_AGENT" in uc_expr
    assert "psi_ATYPE" in uc_expr


def test_preference_block_labor_only():
    """Test labor disutility without housing."""
    from equilibrium.blocks import preference_block

    block = preference_block(labor=True, util_type="crra")

    # Check labor disutility exists
    assert "n_un_AGENT" in block.rules["intermediate"]
    assert "v_AGENT" in block.rules["intermediate"]

    # Should not have housing
    assert "uh_AGENT" not in block.rules["intermediate"]

    # Verify v contains labor disutility formula
    v_expr = block.rules["intermediate"]["v_AGENT"]
    assert "eta_ATYPE" in v_expr
    assert "n_AGENT" in v_expr


def test_preference_block_ghh_housing_labor():
    """Test GHH with both housing and labor."""
    from equilibrium.blocks import preference_block

    block = preference_block(
        util_type="ghh", housing=True, labor=True, housing_spec="cobb_douglas"
    )

    # Check all components exist
    assert "uc_AGENT" in block.rules["intermediate"]
    assert "uh_AGENT" in block.rules["intermediate"]
    assert "n_un_AGENT" in block.rules["intermediate"]
    assert "v_AGENT" in block.rules["intermediate"]
    assert "x_AGENT" in block.rules["intermediate"]


def test_sympy_to_numpy_conversion():
    """Test that SymPy expressions are correctly converted to NumPy format."""
    import sympy as sp

    from equilibrium.blocks.symbolic import _sympy_to_numpy

    # Test simple expression
    c = sp.symbols("c")
    expr = c**2
    result = _sympy_to_numpy(expr, "AGENT", "ATYPE")
    assert "c_AGENT" in result

    # Test log expression
    expr = sp.log(c)
    result = _sympy_to_numpy(expr, "AGENT", "ATYPE")
    assert "np.log" in result
    assert "c_AGENT" in result

    # Test exp expression
    expr = sp.exp(c)
    result = _sympy_to_numpy(expr, "AGENT", "ATYPE")
    assert "np.exp" in result
    assert "c_AGENT" in result


def test_blocks_module_exports_preference_block():
    """Test that blocks module exports preference_block."""
    import equilibrium.blocks

    assert hasattr(equilibrium.blocks, "preference_block")


def test_basic_import():
    """Test basic import of preference_block from equilibrium.blocks."""
    from equilibrium.blocks import preference_block

    # Function should be available
    block = preference_block()
    assert "uc_AGENT" in block.rules["intermediate"]
