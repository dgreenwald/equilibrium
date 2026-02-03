"""Test suite for exclude_vars parameter in add_block methods."""

import pytest

from equilibrium import Model
from equilibrium.model.model import BaseModelBlock, ModelBlock


class TestExcludeVarsBasic:
    """Test basic exclusion functionality."""

    def test_exclude_vars_basic(self):
        """Basic exclusion of single variable."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0", "y": "2.0"},
            "transition": {"z": "x + y"},
        }

        model.add_block(rules=block_rules, exclude_vars={"y"})

        # x and z should be present, y should be excluded
        assert "x" in model.core.rules["intermediate"]
        assert "y" not in model.core.rules["intermediate"]
        assert "z" in model.core.rules["transition"]

    def test_exclude_vars_multiple(self):
        """Exclude multiple variables."""
        model = Model()

        block_rules = {
            "intermediate": {"a": "1.0", "b": "2.0", "c": "3.0"},
            "optimality": {"d": "a + b"},
        }

        model.add_block(rules=block_rules, exclude_vars={"b", "d"})

        # a and c should be present, b and d excluded
        assert "a" in model.core.rules["intermediate"]
        assert "b" not in model.core.rules["intermediate"]
        assert "c" in model.core.rules["intermediate"]
        assert "d" not in model.core.rules["optimality"]

    def test_exclude_vars_empty_set(self):
        """Empty set behaves like None (no exclusion)."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0", "y": "2.0"},
        }

        model.add_block(rules=block_rules, exclude_vars=set())

        # All variables should be present
        assert "x" in model.core.rules["intermediate"]
        assert "y" in model.core.rules["intermediate"]

    def test_exclude_vars_as_list(self):
        """Accept list type for exclude_vars."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0", "y": "2.0", "z": "3.0"},
        }

        model.add_block(rules=block_rules, exclude_vars=["y", "z"])

        # Only x should be present
        assert "x" in model.core.rules["intermediate"]
        assert "y" not in model.core.rules["intermediate"]
        assert "z" not in model.core.rules["intermediate"]


class TestExcludeVarsWithTransforms:
    """Test exclusion with suffix and rename transformations."""

    def test_exclude_vars_with_suffix(self):
        """Exclusion after suffix transformation."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0", "y": "2.0"},
        }

        # Suffix adds "_firm" to variables, so exclude x_firm
        model.add_block(rules=block_rules, suffix="_firm", exclude_vars={"x_firm"})

        # y_firm should be present, x_firm excluded
        assert "x_firm" not in model.core.rules["intermediate"]
        assert "y_firm" in model.core.rules["intermediate"]

    def test_exclude_vars_with_rename(self):
        """Exclusion after rename transformation."""
        model = Model()

        block_rules = {
            "intermediate": {"x_AGENT": "1.0", "y_AGENT": "2.0"},
        }

        # Rename replaces AGENT with household
        model.add_block(
            rules=block_rules,
            rename={"AGENT": "household"},
            exclude_vars={"x_household"},
        )

        # y_household should be present, x_household excluded
        assert "x_household" not in model.core.rules["intermediate"]
        assert "y_household" in model.core.rules["intermediate"]

    def test_exclude_vars_combined_transforms(self):
        """With both suffix and rename."""
        model = Model()

        block_rules = {
            "intermediate": {"var_AGENT": "1.0", "other_AGENT": "2.0"},
        }

        # Suffix first (_firm), then rename (AGENT -> h)
        # Result: var_AGENT -> var_AGENT_firm -> var_h_firm
        model.add_block(
            rules=block_rules,
            suffix="_firm",
            rename={"AGENT": "h"},
            exclude_vars={"var_h_firm"},  # Final name after both transforms
        )

        # other_h_firm should be present, var_h_firm excluded
        assert "var_h_firm" not in model.core.rules["intermediate"]
        assert "other_h_firm" in model.core.rules["intermediate"]

    def test_exclude_vars_with_next_suffix(self):
        """Variables with _NEXT excluded properly."""
        model = Model()

        block_rules = {
            "expectations": {"x_NEXT": "x + 1.0"},
            "transition": {"x": "x_NEXT"},
        }

        model.add_block(rules=block_rules, suffix="_firm", exclude_vars={"x_firm_NEXT"})

        # x_firm should be present, x_firm_NEXT excluded
        assert "x_firm" in model.core.rules["transition"]
        assert "x_firm_NEXT" not in model.core.rules["expectations"]


class TestExcludeVarsPreservesOthers:
    """Test that exclusion doesn't affect non-excluded items."""

    def test_exclude_vars_preserves_other_rules(self):
        """Non-excluded rules still added."""
        model = Model()

        block_rules = {
            "intermediate": {"a": "1.0", "b": "2.0", "c": "3.0"},
            "optimality": {"d": "a + b", "e": "c * 2"},
        }

        model.add_block(rules=block_rules, exclude_vars={"b"})

        # All except b should be present
        assert "a" in model.core.rules["intermediate"]
        assert "b" not in model.core.rules["intermediate"]
        assert "c" in model.core.rules["intermediate"]
        assert "d" in model.core.rules["optimality"]
        assert "e" in model.core.rules["optimality"]

    def test_exclude_vars_from_exog_list(self):
        """Exogenous variables excluded."""
        model = Model()

        model.add_block(
            exog_list=["z_tfp", "z_demand", "z_other"], exclude_vars={"z_demand"}
        )

        # z_tfp and z_other should be present, z_demand excluded
        assert "z_tfp" in model.core.exog_list
        assert "z_demand" not in model.core.exog_list
        assert "z_other" in model.core.exog_list

    def test_exclude_vars_does_not_affect_params(self):
        """Parameters not excluded."""
        model = Model()

        model.add_block(
            params={"alpha": 0.5, "beta": 0.95},
            rules={"intermediate": {"x": "alpha + beta"}},
            exclude_vars={"x"},
        )

        # Parameters should remain
        assert "alpha" in model.core.params
        assert "beta" in model.core.params
        # But rule should be excluded
        assert "x" not in model.core.rules["intermediate"]

    def test_exclude_vars_does_not_affect_steady_guess(self):
        """Steady guess not excluded."""
        model = Model()

        model.add_block(
            steady_guess={"x": 1.0, "y": 2.0},
            rules={"intermediate": {"x": "1.0", "y": "2.0"}},
            exclude_vars={"y"},
        )

        # Steady guess should remain for both
        assert "x" in model.core.steady_guess
        assert "y" in model.core.steady_guess
        # But y rule should be excluded
        assert "x" in model.core.rules["intermediate"]
        assert "y" not in model.core.rules["intermediate"]

    def test_exclude_vars_does_not_affect_flags(self):
        """Flags not excluded."""
        model = Model()

        model.add_block(
            flags={"enable_feature": True},
            rules={"intermediate": {"x": "1.0"}},
            exclude_vars={"x"},
        )

        # Flag should remain
        assert "enable_feature" in model.core.flags
        assert model.core.flags["enable_feature"] is True
        # But rule should be excluded
        assert "x" not in model.core.rules["intermediate"]


class TestExcludeVarsInteractions:
    """Test interactions with other parameters."""

    def test_exclude_vars_with_overwrite_true(self):
        """Exclusion overrides overwrite."""
        model = Model()

        # Add initial rules
        model.add_block(rules={"intermediate": {"x": "1.0", "y": "2.0"}})

        # Try to overwrite with exclusion
        model.add_block(
            rules={"intermediate": {"x": "10.0", "y": "20.0"}},
            overwrite=True,
            exclude_vars={"y"},
        )

        # x should be overwritten, y should remain original (excluded from merge)
        assert model.core.rules["intermediate"]["x"] == "10.0"
        assert model.core.rules["intermediate"]["y"] == "2.0"

    def test_exclude_vars_nonexistent_var(self):
        """No error for non-existent variable."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0", "y": "2.0"},
        }

        # Should not raise error even though 'z' doesn't exist
        model.add_block(rules=block_rules, exclude_vars={"z"})

        # x and y should be present
        assert "x" in model.core.rules["intermediate"]
        assert "y" in model.core.rules["intermediate"]


class TestExcludeVarsAcrossCategories:
    """Test exclusion across different rule categories."""

    def test_exclude_vars_across_rule_categories(self):
        """Works for all rule types."""
        model = Model()

        block_rules = {
            "intermediate": {"int_var": "1.0"},
            "transition": {"trans_var": "2.0"},
            "optimality": {"opt_var": "3.0"},
            "expectations": {"exp_var_NEXT": "4.0"},
        }

        model.add_block(
            rules=block_rules, exclude_vars={"int_var", "opt_var", "exp_var_NEXT"}
        )

        # Only trans_var should be present
        assert "int_var" not in model.core.rules["intermediate"]
        assert "trans_var" in model.core.rules["transition"]
        assert "opt_var" not in model.core.rules["optimality"]
        assert "exp_var_NEXT" not in model.core.rules["expectations"]


class TestExcludeVarsMethodVariants:
    """Test with different add_block invocation methods."""

    def test_exclude_vars_model_add_block(self):
        """Test via Model.add_block()."""
        model = Model()

        block = ModelBlock()
        block.rules["intermediate"] = {"x": "1.0", "y": "2.0"}

        model.add_block(block, exclude_vars={"y"})

        assert "x" in model.core.rules["intermediate"]
        assert "y" not in model.core.rules["intermediate"]

    def test_exclude_vars_base_model_block(self):
        """Test BaseModelBlock directly."""
        # BaseModelBlock needs rule_keys initialized
        rule_keys = [
            "intermediate",
            "transition",
            "optimality",
            "expectations",
            "calibration",
            "analytical_steady",
        ]
        base = BaseModelBlock(rule_keys=rule_keys)

        block = BaseModelBlock(rule_keys=rule_keys)
        block.rules["intermediate"] = {"x": "1.0", "y": "2.0"}

        base.add_block(block, exclude_vars={"y"})

        assert "x" in base.rules["intermediate"]
        assert "y" not in base.rules["intermediate"]


class TestExcludeVarsErrors:
    """Test error handling."""

    def test_exclude_vars_invalid_type(self):
        """TypeError for invalid type."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0"},
        }

        # Dict is not a valid type
        with pytest.raises(
            TypeError, match="exclude_vars must be a set, list, or None"
        ):
            model.add_block(rules=block_rules, exclude_vars={"y": True})

    def test_exclude_vars_invalid_type_string(self):
        """TypeError for string instead of set/list."""
        model = Model()

        block_rules = {
            "intermediate": {"x": "1.0"},
        }

        # String is not a valid type
        with pytest.raises(
            TypeError, match="exclude_vars must be a set, list, or None"
        ):
            model.add_block(rules=block_rules, exclude_vars="y")


class TestExcludeVarsUseCases:
    """Test realistic use cases."""

    def test_exclude_vars_custom_implementation(self):
        """Use case: exclude variable to provide custom implementation."""
        model = Model()

        # Standard block with K_new formula
        block_rules = {
            "intermediate": {"I": "0.5", "delta": "0.1"},
            "transition": {"K_new": "(1 - delta) * K + I"},
        }

        # Add block but exclude K_new to define it ourselves
        model.add_block(rules=block_rules, exclude_vars={"K_new"})

        # Now add custom K_new
        model.rules["transition"]["K_new"] = "custom_formula(K, I)"

        # Standard variables present
        assert "I" in model.core.rules["intermediate"]
        assert "delta" in model.core.rules["intermediate"]
        # Custom K_new present
        assert "K_new" in model.core.rules["transition"]
        assert model.core.rules["transition"]["K_new"] == "custom_formula(K, I)"

    def test_exclude_vars_partial_block_usage(self):
        """Use case: use only part of a block."""
        model = Model()

        # Large block with many variables
        block_rules = {
            "intermediate": {
                "var1": "expr1",
                "var2": "expr2",
                "var3": "expr3",
                "var4": "expr4",
                "var5": "expr5",
            }
        }

        # Only want var1, var2, var3
        model.add_block(rules=block_rules, exclude_vars={"var4", "var5"})

        # First three present, last two excluded
        assert "var1" in model.core.rules["intermediate"]
        assert "var2" in model.core.rules["intermediate"]
        assert "var3" in model.core.rules["intermediate"]
        assert "var4" not in model.core.rules["intermediate"]
        assert "var5" not in model.core.rules["intermediate"]
