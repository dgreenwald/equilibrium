#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


def test_validate_unique_rule_keys_no_duplicates():
    """Test validation passes when no duplicate keys exist across rule categories."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["transition"]["x"] = "x_lag"
    mod.rules["optimality"]["c"] = "c_rule"
    mod.rules["intermediate"]["y"] = "c + x"

    # Should not raise
    mod._validate_unique_rule_keys()


def test_validate_unique_rule_keys_with_duplicates():
    """Test validation fails when duplicate keys exist across rule categories."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["transition"]["x"] = "x_lag"
    mod.rules["optimality"]["x"] = "x_opt_rule"  # Duplicate

    with pytest.raises(ValueError, match="Duplicate keys found"):
        mod._validate_unique_rule_keys()


def test_validate_unique_rule_keys_multiple_duplicates():
    """Test validation reports all duplicates."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["transition"]["x"] = "x_lag"
    mod.rules["optimality"]["x"] = "x_opt_rule"  # Duplicate
    mod.rules["intermediate"]["y"] = "y_inter"
    mod.rules["expectations"]["y"] = "y_exp"  # Another duplicate

    with pytest.raises(ValueError) as excinfo:
        mod._validate_unique_rule_keys()

    error_msg = str(excinfo.value)
    assert "'x'" in error_msg
    assert "'y'" in error_msg
    assert "transition" in error_msg
    assert "optimality" in error_msg
    assert "intermediate" in error_msg
    assert "expectations" in error_msg


def test_validate_unique_rule_keys_analytical_steady_excluded():
    """Test that analytical_steady can duplicate keys from other categories."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["transition"]["x"] = "x_lag"
    mod.rules["optimality"]["c"] = "c_rule"
    mod.rules["analytical_steady"]["x"] = "x_steady_formula"  # Same as transition
    mod.rules["analytical_steady"]["c"] = "c_steady_formula"  # Same as optimality

    # Should not raise - analytical_steady is allowed to have duplicates
    mod._validate_unique_rule_keys()


def test_validate_unique_rule_keys_calibration_excluded():
    """Test that calibration can duplicate keys from other categories."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["optimality"]["bet"] = "bet_rule"
    mod.rules["calibration"]["bet"] = "K - 6.0"  # Same as optimality - typical pattern

    # Should not raise - calibration is allowed to have duplicates
    mod._validate_unique_rule_keys()


def test_validate_unique_rule_keys_same_category():
    """Test that duplicate keys within the same MyOrderedDict are handled correctly."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["transition"]["x"] = "x_lag"
    mod.rules["transition"]["x"] = "x_lag_new"  # Overwrites in same dict, no error

    # Should not raise - same key in same category just overwrites
    mod._validate_unique_rule_keys()


def test_unique_rule_keys_class_constant():
    """Test that UNIQUE_RULE_KEYS is defined correctly."""
    from equilibrium.model.model import Model

    assert hasattr(Model, "UNIQUE_RULE_KEYS")
    assert isinstance(Model.UNIQUE_RULE_KEYS, tuple)
    # analytical_steady and calibration should NOT be in UNIQUE_RULE_KEYS
    assert "analytical_steady" not in Model.UNIQUE_RULE_KEYS
    assert "calibration" not in Model.UNIQUE_RULE_KEYS
    # These should be in UNIQUE_RULE_KEYS
    expected = {
        "transition",
        "expectations",
        "optimality",
        "intermediate",
        "derived_param",
        "read_expectations",
    }
    assert set(Model.UNIQUE_RULE_KEYS) == expected


def test_add_block_overwrite():
    from equilibrium.model.model import Model, ModelBlock

    mod = Model()
    mod.flags["mode"] = "base"
    mod.params.update({"alpha": 0.1})
    mod.steady_guess.update({"K": 1.0})
    mod.exog_list.append("Z")
    mod.rules["intermediate"]["a"] = "old_expr"

    block = ModelBlock(
        flags={"mode": "new", "extra": True},
        params={"alpha": 0.2, "beta": 0.9},
        steady_guess={"K": 2.0, "C": 3.0},
        rules={
            "intermediate": [("a", "new_expr"), ("b", "expr_b")],
            "transition": [("k", "np.log(K_new)")],
        },
        exog_list=["Z", "Y"],
    )

    result = mod.add_block(block, overwrite=True)

    assert result is mod
    assert mod.flags["mode"] == "new"
    assert mod.flags["extra"] is True
    assert mod.params["alpha"] == 0.2
    assert mod.params["beta"] == 0.9
    assert mod.steady_guess["K"] == 2.0
    assert mod.steady_guess["C"] == 3.0
    assert mod.exog_list == ["Z", "Y"]
    assert mod.rules["intermediate"]["a"] == "new_expr"
    assert mod.rules["intermediate"]["b"] == "expr_b"
    assert mod.rules["transition"]["k"] == "np.log(K_new)"


def test_add_block_no_overwrite():
    from equilibrium.model.model import Model

    mod = Model()
    mod.flags["mode"] = "base"
    mod.params.update({"alpha": 0.1})
    mod.steady_guess.update({"K": 1.0})
    mod.exog_list.append("Z")
    mod.rules["intermediate"]["a"] = "old_expr"

    mod.add_block(
        flags={"mode": "new", "level": "debug"},
        params={"alpha": 0.5, "gamma": 2.0},
        steady_guess={"K": 3.0, "H": 4.0},
        rules={
            "intermediate": [("a", "new_expr"), ("b", "expr_b")],
            "transition": [("k", "np.log(K_new)")],
        },
        exog_list=["Z", "Y"],
        overwrite=False,
    )

    assert mod.flags["mode"] == "base"
    assert mod.flags["level"] == "debug"
    assert mod.params["alpha"] == 0.1
    assert mod.params["gamma"] == 2.0
    assert mod.steady_guess["K"] == 1.0
    assert mod.steady_guess["H"] == 4.0
    assert mod.exog_list == ["Z", "Y"]
    assert mod.rules["intermediate"]["a"] == "old_expr"
    assert mod.rules["intermediate"]["b"] == "expr_b"
    assert mod.rules["transition"]["k"] == "np.log(K_new)"


def test_add_block_with_replacements():
    from equilibrium.model.model import Model, ModelBlock

    mod = Model()
    block = ModelBlock(
        flags={"mode_AGENT": "enabled"},
        params={"alpha_AGENT": 0.1},
        steady_guess={"K_AGENT": 1.0},
        exog_list=["Z_AGENT"],
        rules={
            "intermediate": [("x_AGENT", "alpha_AGENT * K_AGENT")],
            "transition": [("K_AGENT", "np.log(K_new_AGENT)")],
        },
    )

    mod.add_block(block, rename={"AGENT": "borrower"})

    assert "mode_borrower" in mod.flags
    assert mod.flags["mode_borrower"] == "enabled"
    assert "alpha_borrower" in mod.params
    assert "K_borrower" in mod.steady_guess
    assert mod.exog_list == ["Z_borrower"]
    assert "x_borrower" in mod.rules["intermediate"]
    assert mod.rules["intermediate"]["x_borrower"] == "alpha_borrower * K_borrower"
    assert "K_borrower" in mod.rules["transition"]
    assert mod.rules["transition"]["K_borrower"] == "np.log(K_new_borrower)"


def test_add_block_replacement_conflict():
    from equilibrium.model.model import Model, ModelBlock

    mod = Model()
    block = ModelBlock(
        params={"x_AGENT": 1.0},
    )

    with pytest.raises(ValueError):
        mod.add_block(block, rename={"AGE": "short", "AGENT": "long"})


def test_model_block_inherits_from_base_model_block():
    """Verify ModelBlock is a subclass of BaseModelBlock."""
    from equilibrium.model.model import BaseModelBlock, ModelBlock

    assert issubclass(ModelBlock, BaseModelBlock)


def test_model_block_uses_model_rule_keys():
    """Verify ModelBlock uses Model.RULE_KEYS automatically."""
    from equilibrium.model.model import Model, ModelBlock

    block = ModelBlock()
    assert block.rule_keys == Model.RULE_KEYS


def test_model_block_has_all_standard_rule_categories():
    """Verify ModelBlock initializes all standard rule categories."""
    from equilibrium.model.model import Model, ModelBlock

    block = ModelBlock()
    for key in Model.RULE_KEYS:
        assert key in block.rules


def test_base_model_block_accepts_custom_rule_keys():
    """Verify BaseModelBlock can use custom rule keys."""
    from equilibrium.model.model import BaseModelBlock

    custom_keys = ("foo", "bar", "baz")
    block = BaseModelBlock(rule_keys=custom_keys)
    assert block.rule_keys == custom_keys
    for key in custom_keys:
        assert key in block.rules


def test_model_rule_keys_class_constant():
    """Verify Model.RULE_KEYS is defined and is a tuple with expected structure."""
    from equilibrium.model.model import Model

    assert hasattr(Model, "RULE_KEYS")
    assert isinstance(Model.RULE_KEYS, tuple)
    # Verify expected minimum length and that all values are non-empty strings
    assert len(Model.RULE_KEYS) >= 7
    for key in Model.RULE_KEYS:
        assert isinstance(key, str)
        assert key
    # Verify key rule categories are present (core functionality depends on these)
    assert "intermediate" in Model.RULE_KEYS
    assert "transition" in Model.RULE_KEYS
    assert "expectations" in Model.RULE_KEYS
    assert "optimality" in Model.RULE_KEYS


def test_model_uses_rule_keys_constant():
    """Verify Model instance rule_keys matches Model.RULE_KEYS."""
    from equilibrium.model.model import Model

    mod = Model()
    assert mod.rule_keys == Model.RULE_KEYS


def test_add_block_with_base_model_block():
    """Verify add_block accepts BaseModelBlock instances."""
    from equilibrium.model.model import BaseModelBlock, Model

    mod = Model()
    block = BaseModelBlock(
        params={"custom_param": 1.0},
        rule_keys=Model.RULE_KEYS,
    )
    mod.add_block(block)
    assert "custom_param" in mod.params


def test_model_block_without_rule_keys_arg():
    """Verify ModelBlock can be instantiated without rule_keys argument."""
    from equilibrium.model.model import ModelBlock

    # Should not raise TypeError
    block = ModelBlock(
        flags={"flag1": True},
        params={"param1": 1.0},
    )
    assert block.flags["flag1"] is True
    assert block.params["param1"] == 1.0


# Tests for BaseModelBlock.add_block


def test_base_model_block_add_block_basic():
    """Verify BaseModelBlock.add_block merges two blocks correctly."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate", "transition")

    block1 = BaseModelBlock(
        flags={"flag1": True},
        params={"alpha": 0.1},
        steady_guess={"K": 1.0},
        exog_list=["Z"],
        rules={"intermediate": [("a", "expr_a")]},
        rule_keys=rule_keys,
    )

    block2 = BaseModelBlock(
        flags={"flag2": False},
        params={"beta": 0.9},
        steady_guess={"C": 2.0},
        exog_list=["Y"],
        rules={"transition": [("k", "np.log(K)")]},
        rule_keys=rule_keys,
    )

    result = block1.add_block(block2)

    # Should return self for chaining
    assert result is block1

    # Check flags merged
    assert block1.flags["flag1"] is True
    assert block1.flags["flag2"] is False

    # Check params merged
    assert block1.params["alpha"] == 0.1
    assert block1.params["beta"] == 0.9

    # Check steady_guess merged
    assert block1.steady_guess["K"] == 1.0
    assert block1.steady_guess["C"] == 2.0

    # Check exog_list merged
    assert block1.exog_list == ["Z", "Y"]

    # Check rules merged
    assert "a" in block1.rules["intermediate"]
    assert "k" in block1.rules["transition"]


def test_base_model_block_add_block_with_overwrite():
    """Verify BaseModelBlock.add_block overwrites when requested."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate",)

    block1 = BaseModelBlock(
        flags={"mode": "original"},
        params={"alpha": 0.1},
        rules={"intermediate": [("a", "old_expr")]},
        rule_keys=rule_keys,
    )

    block2 = BaseModelBlock(
        flags={"mode": "updated"},
        params={"alpha": 0.2},
        rules={"intermediate": [("a", "new_expr")]},
        rule_keys=rule_keys,
    )

    block1.add_block(block2, overwrite=True)

    assert block1.flags["mode"] == "updated"
    assert block1.params["alpha"] == 0.2
    assert block1.rules["intermediate"]["a"] == "new_expr"


def test_base_model_block_add_block_no_overwrite():
    """Verify BaseModelBlock.add_block preserves existing values by default."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate",)

    block1 = BaseModelBlock(
        flags={"mode": "original"},
        params={"alpha": 0.1},
        rules={"intermediate": [("a", "old_expr")]},
        rule_keys=rule_keys,
    )

    block2 = BaseModelBlock(
        flags={"mode": "updated"},
        params={"alpha": 0.2},
        rules={"intermediate": [("a", "new_expr")]},
        rule_keys=rule_keys,
    )

    block1.add_block(block2, overwrite=False)

    # Original values should be preserved
    assert block1.flags["mode"] == "original"
    assert block1.params["alpha"] == 0.1
    assert block1.rules["intermediate"]["a"] == "old_expr"


def test_base_model_block_add_block_different_rule_keys_raises():
    """Verify add_block raises ValueError when rule_keys differ."""
    from equilibrium.model.model import BaseModelBlock

    block1 = BaseModelBlock(rule_keys=("foo", "bar"))
    block2 = BaseModelBlock(rule_keys=("baz", "qux"))

    with pytest.raises(ValueError, match="different rule_keys"):
        block1.add_block(block2)


def test_base_model_block_add_block_with_rename():
    """Verify add_block applies rename replacements."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate",)

    block1 = BaseModelBlock(rule_keys=rule_keys)
    block2 = BaseModelBlock(
        params={"alpha_AGENT": 0.1},
        rules={"intermediate": [("x_AGENT", "alpha_AGENT * 2")]},
        rule_keys=rule_keys,
    )

    block1.add_block(block2, rename={"AGENT": "lender"})

    assert "alpha_lender" in block1.params
    assert "x_lender" in block1.rules["intermediate"]
    assert block1.rules["intermediate"]["x_lender"] == "alpha_lender * 2"


def test_base_model_block_add_block_with_keywords():
    """Verify add_block accepts keyword components instead of block."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate",)

    block1 = BaseModelBlock(rule_keys=rule_keys)

    block1.add_block(
        params={"alpha": 0.1},
        rules={"intermediate": [("a", "alpha * 2")]},
    )

    assert block1.params["alpha"] == 0.1
    assert block1.rules["intermediate"]["a"] == "alpha * 2"


def test_base_model_block_add_block_rejects_both_block_and_keywords():
    """Verify add_block raises ValueError when both block and keywords provided."""
    from equilibrium.model.model import BaseModelBlock

    block1 = BaseModelBlock(rule_keys=())
    block2 = BaseModelBlock(rule_keys=())

    with pytest.raises(ValueError, match="Provide either"):
        block1.add_block(block2, params={"alpha": 0.1})


# Tests for + operator


def test_base_model_block_add_operator_basic():
    """Verify BaseModelBlock + operator creates a new merged block."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate", "transition")

    block1 = BaseModelBlock(
        flags={"flag1": True},
        params={"alpha": 0.1},
        steady_guess={"K": 1.0},
        exog_list=["Z"],
        rules={"intermediate": [("a", "expr_a")]},
        rule_keys=rule_keys,
    )

    block2 = BaseModelBlock(
        flags={"flag2": False},
        params={"beta": 0.9},
        steady_guess={"C": 2.0},
        exog_list=["Y"],
        rules={"transition": [("k", "np.log(K)")]},
        rule_keys=rule_keys,
    )

    result = block1 + block2

    # Should return a new block, not modify originals
    assert result is not block1
    assert result is not block2

    # Original blocks should be unchanged
    assert "flag2" not in block1.flags
    assert "flag1" not in block2.flags

    # Check result has merged contents
    assert result.flags["flag1"] is True
    assert result.flags["flag2"] is False
    assert result.params["alpha"] == 0.1
    assert result.params["beta"] == 0.9
    assert result.steady_guess["K"] == 1.0
    assert result.steady_guess["C"] == 2.0
    assert result.exog_list == ["Z", "Y"]
    assert "a" in result.rules["intermediate"]
    assert "k" in result.rules["transition"]


def test_base_model_block_add_operator_preserves_left_values():
    """Verify + operator preserves left operand values on conflict."""
    from equilibrium.model.model import BaseModelBlock

    rule_keys = ("intermediate",)

    block1 = BaseModelBlock(
        params={"alpha": 0.1},
        rules={"intermediate": [("a", "original")]},
        rule_keys=rule_keys,
    )

    block2 = BaseModelBlock(
        params={"alpha": 0.9},
        rules={"intermediate": [("a", "new")]},
        rule_keys=rule_keys,
    )

    result = block1 + block2

    # Left operand values should be preserved
    assert result.params["alpha"] == 0.1
    assert result.rules["intermediate"]["a"] == "original"


def test_base_model_block_add_operator_different_rule_keys_raises():
    """Verify + operator raises ValueError when rule_keys differ."""
    from equilibrium.model.model import BaseModelBlock

    block1 = BaseModelBlock(rule_keys=("foo",))
    block2 = BaseModelBlock(rule_keys=("bar",))

    with pytest.raises(ValueError, match="different rule_keys"):
        _ = block1 + block2


def test_base_model_block_add_operator_with_non_block_returns_not_implemented():
    """Verify + operator returns NotImplemented for non-block types."""
    from equilibrium.model.model import BaseModelBlock

    block = BaseModelBlock(rule_keys=())

    result = block.__add__("not a block")
    assert result is NotImplemented


def test_model_block_add_operator():
    """Verify ModelBlock + operator works (inherits from BaseModelBlock)."""
    from equilibrium.model.model import BaseModelBlock, ModelBlock

    block1 = ModelBlock(
        flags={"flag1": True},
        params={"alpha": 0.1},
    )

    block2 = ModelBlock(
        flags={"flag2": False},
        params={"beta": 0.9},
    )

    result = block1 + block2

    assert isinstance(result, BaseModelBlock)
    assert result.flags["flag1"] is True
    assert result.flags["flag2"] is False
    assert result.params["alpha"] == 0.1
    assert result.params["beta"] == 0.9


def test_model_block_add_block_method():
    """Verify ModelBlock.add_block works (inherits from BaseModelBlock)."""
    from equilibrium.model.model import ModelBlock

    block1 = ModelBlock(
        flags={"flag1": True},
        params={"alpha": 0.1},
    )

    block2 = ModelBlock(
        flags={"flag2": False},
        params={"beta": 0.9},
    )

    result = block1.add_block(block2)

    assert result is block1
    assert block1.flags["flag1"] is True
    assert block1.flags["flag2"] is False
    assert block1.params["alpha"] == 0.1
    assert block1.params["beta"] == 0.9
