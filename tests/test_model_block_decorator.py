#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the @model_block decorator.
"""


def test_model_block_decorator_basic():
    """Test that the @model_block decorator works for basic usage."""
    from equilibrium.model import ModelBlock, model_block

    @model_block
    def test_block(block, *, param=True):
        block.rules["intermediate"] += [("x", "param * 2")]

    result = test_block(param=True)

    assert isinstance(result, ModelBlock)
    assert "x" in result.rules["intermediate"]
    assert result.rules["intermediate"]["x"] == "param * 2"


def test_model_block_decorator_with_multiple_rules():
    """Test that the decorator handles multiple rules correctly."""
    from equilibrium.model import model_block

    @model_block
    def test_block(block):
        block.rules["intermediate"] += [
            ("a", "1.0"),
            ("b", "a * 2"),
        ]
        block.rules["transition"] += [("c", "b")]

    result = test_block()

    assert "a" in result.rules["intermediate"]
    assert "b" in result.rules["intermediate"]
    assert "c" in result.rules["transition"]


def test_model_block_decorator_with_flags_params():
    """Test that the decorator works with flags and params."""
    from equilibrium.model import model_block

    @model_block
    def test_block(block, *, use_flag=True):
        if use_flag:
            block.flags["enabled"] = True
            block.params["alpha"] = 0.5

        block.rules["intermediate"] += [("x", "alpha * 2")]

    result = test_block(use_flag=True)

    assert result.flags["enabled"] is True
    assert result.params["alpha"] == 0.5
    assert "x" in result.rules["intermediate"]


def test_model_block_decorator_preserves_function_name():
    """Test that the decorator preserves the original function name."""
    from equilibrium.model import model_block

    @model_block
    def my_custom_block(block):
        block.rules["intermediate"] += [("x", "1.0")]

    assert my_custom_block.__name__ == "my_custom_block"


def test_model_block_decorator_preserves_docstring():
    """Test that the decorator preserves the original function docstring."""
    from equilibrium.model import model_block

    @model_block
    def documented_block(block):
        """This is a test docstring."""
        block.rules["intermediate"] += [("x", "1.0")]

    assert documented_block.__doc__ == "This is a test docstring."


def test_investment_block_works_with_decorator():
    """Test that investment_block works correctly with the decorator."""
    from equilibrium.blocks import investment_block
    from equilibrium.model import ModelBlock

    block = investment_block()

    assert isinstance(block, ModelBlock)
    assert "Phi_inv_AGENT" in block.rules["intermediate"]
    assert "Q_AGENT" in block.rules["intermediate"]


def test_st_bond_block_works_with_decorator():
    """Test that st_bond_block works correctly with the decorator."""
    from equilibrium.blocks import st_bond_block
    from equilibrium.model import ModelBlock

    block = st_bond_block()

    assert isinstance(block, ModelBlock)
    assert "E_Lam_AGENT" in block.rules["expectations"]
    assert "R_new" in block.rules["optimality"]


def test_st_bond_block_with_lag_parameter():
    """Test that st_bond_block works with include_lag parameter."""
    from equilibrium.blocks import st_bond_block

    block_no_lag = st_bond_block(include_lag=False)
    block_with_lag = st_bond_block(include_lag=True)

    assert "R_lag" not in block_no_lag.rules["transition"]
    assert "R_lag" in block_with_lag.rules["transition"]


def test_preference_block_works_with_decorator():
    """Test that preference_block works correctly with the decorator."""
    from equilibrium.blocks import preference_block
    from equilibrium.model import ModelBlock

    block = preference_block()

    assert isinstance(block, ModelBlock)
    assert "uc_AGENT" in block.rules["intermediate"]
    assert "Lam_1_AGENT" in block.rules["intermediate"]


def test_model_block_decorator_exported_from_model():
    """Test that model_block is exported from equilibrium.model."""
    from equilibrium.model import model_block

    assert callable(model_block)


def test_model_block_decorator_works_in_chain():
    """Test that decorated blocks can be combined with add_block."""
    from equilibrium.model import Model, model_block

    @model_block
    def block1(block):
        block.rules["intermediate"] += [("a", "1.0")]

    @model_block
    def block2(block):
        block.rules["intermediate"] += [("b", "2.0")]

    model = Model()
    model.add_block(block1())
    model.add_block(block2())

    assert "a" in model.rules["intermediate"]
    assert "b" in model.rules["intermediate"]


def test_decorated_block_supports_renaming():
    """Test that blocks created with decorator support renaming."""
    from equilibrium.model import Model, model_block

    @model_block
    def agent_block(block):
        block.params["alpha_AGENT"] = 0.5
        block.rules["intermediate"] += [("x_AGENT", "alpha_AGENT * 2")]

    model = Model()
    model.add_block(agent_block(), rename={"AGENT": "borrower"})

    assert "alpha_borrower" in model.params
    assert "x_borrower" in model.rules["intermediate"]


def test_decorator_usage_example_from_problem_statement():
    """Test the example usage pattern from the problem statement."""
    from equilibrium.model import model_block

    @model_block
    def cdf_block(block):
        block.rules["intermediate"] += [
            ("z_e_bar_VAR", "(np.log(e_bar_VAR) - MU) / SIG"),
            (
                "cdf_VAR",
                "0.5 * (1.0 + jax.scipy.special.erf(z_e_bar_VAR / np.sqrt(2.0)))",
            ),
        ]

    result = cdf_block()

    assert "z_e_bar_VAR" in result.rules["intermediate"]
    assert "cdf_VAR" in result.rules["intermediate"]
