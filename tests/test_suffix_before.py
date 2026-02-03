"""Test suite for suffix_before parameter in add_block methods."""

import pytest

from equilibrium import Model
from equilibrium.model.model import BaseModelBlock


class TestSuffixBeforeBasic:
    """Test basic suffix_before functionality."""

    def test_suffix_before_single_term(self):
        """Basic case with single special term."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0", "I": "2.0"},
        }

        model.add_block(rules=block_rules, suffix="_firm", suffix_before=["_AGENT"])

        # C_AGENT should become C_firm_AGENT, I should become I_firm
        assert "C_firm_AGENT" in model.core.rules["intermediate"]
        assert "I_firm" in model.core.rules["intermediate"]
        assert "C_AGENT_firm" not in model.core.rules["intermediate"]

    def test_suffix_before_multiple_terms(self):
        """Multiple special terms in list."""
        model = Model()

        block_rules = {
            "intermediate": {
                "C_AGENT": "1.0",
                "I_ATYPE": "2.0",
                "K": "3.0",
            },
        }

        model.add_block(
            rules=block_rules, suffix="_firm", suffix_before=["_AGENT", "_ATYPE"]
        )

        # Each should have suffix inserted before its special term
        assert "C_firm_AGENT" in model.core.rules["intermediate"]
        assert "I_firm_ATYPE" in model.core.rules["intermediate"]
        assert "K_firm" in model.core.rules["intermediate"]

    def test_suffix_before_as_string(self):
        """Accept single string instead of list."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0"},
        }

        model.add_block(rules=block_rules, suffix="_firm", suffix_before="_AGENT")

        assert "C_firm_AGENT" in model.core.rules["intermediate"]

    def test_suffix_before_multiple_consecutive_terms(self):
        """Variable ending with multiple consecutive special terms."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT_ATYPE": "1.0"},
        }

        model.add_block(
            rules=block_rules, suffix="_firm", suffix_before=["_AGENT", "_ATYPE"]
        )

        # Suffix goes before the entire block of special terms
        assert "C_firm_AGENT_ATYPE" in model.core.rules["intermediate"]

    def test_suffix_before_different_order(self):
        """Special terms in different order."""
        model = Model()

        block_rules = {
            "intermediate": {"C_ATYPE_AGENT": "1.0"},
        }

        model.add_block(
            rules=block_rules, suffix="_firm", suffix_before=["_AGENT", "_ATYPE"]
        )

        # Suffix goes before the block regardless of term order
        assert "C_firm_ATYPE_AGENT" in model.core.rules["intermediate"]


class TestSuffixBeforeWithNext:
    """Test interaction with temporal _NEXT."""

    def test_suffix_before_with_next_in_expression(self):
        """Variable with _NEXT in expression."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0", "x": "2.0", "y": "0.0"},
            "expectations": {"y": "C_AGENT_NEXT + x_NEXT + 1"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        # C_AGENT becomes C_firm_AGENT, x becomes x_firm, y becomes y_firm
        assert "C_firm_AGENT" in model.core.rules["intermediate"]
        assert "x_firm" in model.core.rules["intermediate"]
        assert "y_firm" in model.core.rules["expectations"]
        # In expression: C_AGENT_NEXT becomes C_firm_AGENT_NEXT, x_NEXT becomes x_firm_NEXT
        assert "C_firm_AGENT_NEXT" in model.core.rules["expectations"]["y_firm"]
        assert "x_firm_NEXT" in model.core.rules["expectations"]["y_firm"]

    def test_suffix_before_plain_variable_with_next(self):
        """Plain variable (no special terms) with _NEXT."""
        model = Model()

        block_rules = {
            "intermediate": {"K": "1.0", "y": "0.0"},
            "expectations": {"y": "K_NEXT + 1"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        # K becomes K_firm, y becomes y_firm
        assert "K_firm" in model.core.rules["intermediate"]
        assert "y_firm" in model.core.rules["expectations"]
        # In expression: K_NEXT becomes K_firm_NEXT
        assert "K_firm_NEXT" in model.core.rules["expectations"]["y_firm"]

    def test_suffix_before_variable_ending_with_next(self):
        """Variable that literally ends with _NEXT in LHS."""
        model = Model()

        # Unusual but valid: variable named x_NEXT
        block_rules = {
            "intermediate": {"x_NEXT": "1.0"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        # x_NEXT should become x_firm_NEXT (_NEXT is always special)
        assert "x_firm_NEXT" in model.core.rules["intermediate"]


class TestSuffixBeforeEdgeCases:
    """Test edge cases."""

    def test_suffix_before_middle_of_name(self):
        """Special term in middle of variable name."""
        model = Model()

        block_rules = {
            "intermediate": {"x_AGENT_y": "1.0"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        # _AGENT is not at end, so suffix goes at end
        assert "x_AGENT_y_firm" in model.core.rules["intermediate"]
        assert "x_firm_AGENT_y" not in model.core.rules["intermediate"]

    def test_suffix_before_empty_list(self):
        """Empty list behaves like None."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0"},
        }

        model.add_block(rules=block_rules, suffix="_firm", suffix_before=[])

        # No special handling, suffix at end
        assert "C_AGENT_firm" in model.core.rules["intermediate"]

    def test_suffix_before_none(self):
        """None (default) behavior."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0"},
        }

        model.add_block(rules=block_rules, suffix="_firm", suffix_before=None)

        # No special handling, suffix at end
        assert "C_AGENT_firm" in model.core.rules["intermediate"]

    def test_suffix_before_no_suffix(self):
        """suffix_before without suffix parameter."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0"},
        }

        # No suffix, so suffix_before should be ignored
        model.add_block(rules=block_rules, suffix_before=["_AGENT"])

        # Original variable name unchanged
        assert "C_AGENT" in model.core.rules["intermediate"]

    def test_suffix_before_variable_without_special_term(self):
        """Variable doesn't end with any special term."""
        model = Model()

        block_rules = {
            "intermediate": {"consumption": "1.0"},
        }

        model.add_block(
            rules=block_rules, suffix="_firm", suffix_before=["_AGENT", "_ATYPE"]
        )

        # No special term, suffix at end
        assert "consumption_firm" in model.core.rules["intermediate"]


class TestSuffixBeforeWithRename:
    """Test interaction with rename parameter."""

    def test_suffix_before_then_rename(self):
        """Suffix applied before rename."""
        model = Model()

        block_rules = {
            "intermediate": {
                "C_AGENT": "wage_AGENT * hours",
                "wage_AGENT": "base_wage",
            },
        }

        model.add_block(
            rules=block_rules,
            suffix="_worker",
            suffix_before=["_AGENT"],
            rename={"AGENT": "h"},
        )

        # Transformation: C_AGENT -> C_worker_AGENT -> C_worker_h
        assert "C_worker_h" in model.core.rules["intermediate"]
        assert "wage_worker_h" in model.core.rules["intermediate"]
        assert "C_worker_AGENT" not in model.core.rules["intermediate"]

        # Expression should also be transformed
        assert "wage_worker_h" in model.core.rules["intermediate"]["C_worker_h"]

    def test_suffix_before_with_rename_multiple_terms(self):
        """Multiple special terms with rename."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT_ATYPE": "1.0"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT", "_ATYPE"],
            rename={"AGENT": "household", "ATYPE": "saver"},
        )

        # C_AGENT_ATYPE -> C_firm_AGENT_ATYPE -> C_firm_household_saver
        assert "C_firm_household_saver" in model.core.rules["intermediate"]


class TestSuffixBeforeWithExcludeVars:
    """Test interaction with exclude_vars parameter."""

    def test_suffix_before_with_exclude_vars(self):
        """exclude_vars applied after suffix_before."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0", "I_AGENT": "2.0"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT"],
            exclude_vars={"C_firm_AGENT"},  # Use transformed name
        )

        # I_AGENT should be present, C_AGENT excluded
        assert "I_firm_AGENT" in model.core.rules["intermediate"]
        assert "C_firm_AGENT" not in model.core.rules["intermediate"]

    def test_suffix_before_with_rename_and_exclude(self):
        """All three transformations together."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0", "I_AGENT": "2.0"},
        }

        model.add_block(
            rules=block_rules,
            suffix="_worker",
            suffix_before=["_AGENT"],
            rename={"AGENT": "h"},
            exclude_vars={"I_worker_h"},  # Use final name after all transforms
        )

        # C should be present, I excluded
        assert "C_worker_h" in model.core.rules["intermediate"]
        assert "I_worker_h" not in model.core.rules["intermediate"]


class TestSuffixBeforeInRHS:
    """Test that suffix_before works in RHS expressions."""

    def test_suffix_before_in_expression(self):
        """Variables in RHS transformed correctly."""
        model = Model()

        block_rules = {
            "intermediate": {
                "wage_AGENT": "base_wage",
                "C_AGENT": "wage_AGENT * hours",
                "S_AGENT": "income - C_AGENT",
            },
        }

        model.add_block(
            rules=block_rules,
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        # Check RHS transformations - all LHS variables transformed in RHS too
        assert (
            model.core.rules["intermediate"]["C_firm_AGENT"]
            == "wage_firm_AGENT * hours"
        )
        assert (
            model.core.rules["intermediate"]["S_firm_AGENT"] == "income - C_firm_AGENT"
        )


class TestSuffixBeforeErrors:
    """Test error handling."""

    def test_suffix_before_invalid_type(self):
        """TypeError for invalid type."""
        model = Model()

        block_rules = {
            "intermediate": {"C_AGENT": "1.0"},
        }

        with pytest.raises(
            TypeError, match="suffix_before must be a list, str, or None"
        ):
            model.add_block(
                rules=block_rules,
                suffix="_firm",
                suffix_before={"_AGENT": True},  # Dict not valid
            )


class TestSuffixBeforeParams:
    """Test that suffix_before works with params, exog, etc."""

    def test_suffix_before_with_params(self):
        """Params get transformed if they're also LHS variables."""
        model = Model()

        block_rules = {
            "intermediate": {
                "C_AGENT": "alpha_AGENT",
                "alpha_AGENT": "0.5",  # Also define as LHS variable
            },
        }

        model.add_block(
            rules=block_rules,
            params={"alpha_AGENT": 0.5},  # Param with same name
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        # Both variable and param should be transformed
        assert "C_firm_AGENT" in model.core.rules["intermediate"]
        assert "alpha_firm_AGENT" in model.core.rules["intermediate"]
        assert "alpha_firm_AGENT" in model.core.params

    def test_suffix_before_with_exog_list(self):
        """Exogenous variables transformed."""
        model = Model()

        model.add_block(
            exog_list=["z_AGENT", "eps"],
            suffix="_firm",
            suffix_before=["_AGENT"],
        )

        assert "z_firm_AGENT" in model.core.exog_list
        assert "eps_firm" in model.core.exog_list


class TestSuffixBeforeMethodVariants:
    """Test with different invocation methods."""

    def test_suffix_before_base_model_block(self):
        """Test BaseModelBlock directly."""
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
        block.rules["intermediate"] = {"C_AGENT": "1.0"}

        base.add_block(block, suffix="_firm", suffix_before=["_AGENT"])

        assert "C_firm_AGENT" in base.rules["intermediate"]
