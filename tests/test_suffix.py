#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the suffix parameter in Model.add_block.
"""

from equilibrium.model import Model, ModelBlock


def test_suffix_basic():
    """Test basic suffix application."""
    model = Model()

    block = ModelBlock(
        rules={
            "intermediate": [("K", "10.0"), ("Y", "K * 0.5")],
        }
    )

    model.add_block(block, suffix="_firm")

    # Check LHS variables are suffixed
    assert "K_firm" in model.rules["intermediate"]
    assert "Y_firm" in model.rules["intermediate"]

    # Check RHS references are suffixed
    assert model.rules["intermediate"]["Y_firm"] == "K_firm * 0.5"


def test_suffix_with_next():
    """Test that _NEXT is handled specially."""
    model = Model()

    block = ModelBlock(
        rules={
            "intermediate": [("K", "10.0")],
            "expectations": [("E_K", "K_NEXT * 0.95")],
        }
    )

    model.add_block(block, suffix="_firm")

    # K_NEXT should become K_firm_NEXT (preserve _NEXT)
    assert "E_K_firm" in model.rules["expectations"]
    assert model.rules["expectations"]["E_K_firm"] == "K_firm_NEXT * 0.95"


def test_suffix_does_not_match_partial():
    """Test that suffix doesn't match partial variable names."""
    model = Model()

    block = ModelBlock(
        rules={
            "intermediate": [
                ("K", "5.0"),
                ("log_K", "np.log(K)"),
            ]
        }
    )

    model.add_block(block, suffix="_firm")

    # Both K and log_K should be suffixed independently
    assert "K_firm" in model.rules["intermediate"]
    assert "log_K_firm" in model.rules["intermediate"]

    # log_K expression should reference K_firm (K is suffixed)
    assert model.rules["intermediate"]["log_K_firm"] == "np.log(K_firm)"


def test_suffix_with_params():
    """Test that LHS variables in steady_guess are suffixed, but params are not."""
    model = Model()

    block = ModelBlock(
        rules={"intermediate": [("K", "alpha * 10")]},
        params={"alpha": 0.33},
        steady_guess={"K": 5.0},
    )

    model.add_block(block, suffix="_firm")

    # Variables are suffixed
    assert "K_firm" in model.rules["intermediate"]
    assert "K_firm" in model.steady_guess

    # Params are NOT suffixed (they're not LHS variables)
    assert "alpha" in model.params
    assert "alpha_firm" not in model.params

    # Expression references the original param name
    assert model.rules["intermediate"]["K_firm"] == "alpha * 10"


def test_suffix_with_exog():
    """Test that exogenous variables are suffixed."""
    model = Model()

    block = ModelBlock(
        rules={"intermediate": [("A", "Z * 1.0")]},
        exog_list=["Z"],
    )

    model.add_block(block, suffix="_tech")

    # Exog variable should be suffixed
    assert "Z_tech" in model.exog_list

    # Rule should reference suffixed exog
    assert "A_tech" in model.rules["intermediate"]
    assert model.rules["intermediate"]["A_tech"] == "Z_tech * 1.0"


def test_suffix_and_rename():
    """Test that both suffix and rename can be used together."""
    model = Model()

    block = ModelBlock(
        rules={
            "intermediate": [
                ("K_AGENT", "10.0"),
                ("Y", "K_AGENT * 0.5"),
            ]
        }
    )

    # Apply suffix first, then rename
    model.add_block(block, suffix="_firm", rename={"AGENT": "corp"})

    # K_AGENT -> K_AGENT_firm (suffix) -> K_corp_firm (rename)
    assert "K_corp_firm" in model.rules["intermediate"]
    # Y -> Y_firm (suffix), no rename
    assert "Y_firm" in model.rules["intermediate"]

    # Expression should use renamed variable
    assert model.rules["intermediate"]["Y_firm"] == "K_corp_firm * 0.5"


def test_suffix_multiple_blocks():
    """Test adding multiple blocks with different suffixes."""
    model = Model()

    block1 = ModelBlock(rules={"intermediate": [("K", "10.0"), ("Y", "K * 0.5")]})

    block2 = ModelBlock(rules={"intermediate": [("K", "20.0"), ("C", "K * 0.3")]})

    model.add_block(block1, suffix="_firm")
    model.add_block(block2, suffix="_household")

    # Both blocks should have their own suffixed variables
    assert "K_firm" in model.rules["intermediate"]
    assert "Y_firm" in model.rules["intermediate"]
    assert "K_household" in model.rules["intermediate"]
    assert "C_household" in model.rules["intermediate"]


def test_suffix_empty_block():
    """Test that suffix works with empty block."""
    model = Model()

    block = ModelBlock()
    model.add_block(block, suffix="_test")

    # Should not raise an error
    assert True


def test_suffix_preserves_external_variables():
    """Test that suffix doesn't affect variables not defined in the block."""
    model = Model()

    # Add an external variable first
    model.rules["intermediate"] += [("external_var", "1.0")]

    # Add a block that references the external variable
    block = ModelBlock(rules={"intermediate": [("internal_var", "external_var * 2.0")]})

    model.add_block(block, suffix="_block")

    # internal_var should be suffixed
    assert "internal_var_block" in model.rules["intermediate"]

    # external_var should NOT be suffixed (not defined in block)
    assert model.rules["intermediate"]["internal_var_block"] == "external_var * 2.0"
    # Original external_var should still exist
    assert "external_var" in model.rules["intermediate"]


def test_suffix_with_complex_expressions():
    """Test suffix with complex mathematical expressions."""
    model = Model()

    block = ModelBlock(
        rules={
            "intermediate": [
                ("K", "10.0"),
                ("Y", "np.exp(K) + np.log(K) * K**2"),
            ]
        }
    )

    model.add_block(block, suffix="_firm")

    # All K references should be suffixed
    expected = "np.exp(K_firm) + np.log(K_firm) * K_firm**2"
    assert model.rules["intermediate"]["Y_firm"] == expected
