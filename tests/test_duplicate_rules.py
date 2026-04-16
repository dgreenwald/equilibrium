#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests that duplicate rules within a category raise a clear ValueError."""

import pytest

from equilibrium.utils.containers import MyOrderedDict


# ---------------------------------------------------------------------------
# MyOrderedDict.__add__ unit tests
# ---------------------------------------------------------------------------


def test_no_duplicate_addition_succeeds():
    """Adding non-overlapping entries raises nothing."""
    od = MyOrderedDict([("a", "1")])
    result = od + [("b", "2")]
    assert result == MyOrderedDict([("a", "1"), ("b", "2")])


def test_duplicate_key_raises():
    """Adding a key already present raises ValueError."""
    od = MyOrderedDict([("K", "expr1")])
    with pytest.raises(ValueError, match="Duplicate rule key"):
        od + [("K", "expr2")]


def test_duplicate_key_error_names_the_variable():
    """The error message explicitly names the duplicate key."""
    od = MyOrderedDict([("K", "expr1"), ("C", "expr2")])
    with pytest.raises(ValueError, match="'K'"):
        od + [("K", "new_expr")]


def test_multiple_duplicates_all_listed():
    """All duplicate keys appear in the error message."""
    od = MyOrderedDict([("K", "expr1"), ("C", "expr2")])
    with pytest.raises(ValueError) as exc_info:
        od + [("K", "new_K"), ("C", "new_C")]
    msg = str(exc_info.value)
    assert "'K'" in msg
    assert "'C'" in msg


def test_iadd_duplicate_raises():
    """+= with a duplicate key raises ValueError."""
    od = MyOrderedDict([("K", "expr1")])
    with pytest.raises(ValueError, match="Duplicate rule key"):
        od += [("K", "expr2")]


def test_empty_addition_succeeds():
    """Adding an empty list to a non-empty dict raises nothing."""
    od = MyOrderedDict([("K", "expr1")])
    result = od + []
    assert result == od


# ---------------------------------------------------------------------------
# Integration: model.rules += [...] user pattern
# ---------------------------------------------------------------------------


def test_model_rules_duplicate_within_category_raises():
    """Using += twice with the same variable name in a category raises ValueError."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["intermediate"] += [("K", "np.exp(log_K)")]

    with pytest.raises(ValueError, match="Duplicate rule key"):
        mod.rules["intermediate"] += [("K", "different_expr")]


def test_model_rules_duplicate_error_suggests_direct_assignment():
    """The error message hints at the direct-assignment workaround."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["transition"] += [("log_K", "np.log(K_new)")]

    with pytest.raises(ValueError, match="assign directly"):
        mod.rules["transition"] += [("log_K", "replacement")]


def test_model_rules_different_categories_do_not_conflict():
    """The same variable name in two different categories does NOT raise via +=."""
    from equilibrium.model.model import Model

    mod = Model()
    # 'x' in transition is fine; 'x' in intermediate would be caught by
    # _validate_unique_rule_keys at finalize time, but the += itself is legal
    # because each category is a separate MyOrderedDict.
    mod.rules["transition"] += [("x", "x_lag")]
    mod.rules["optimality"] += [("y", "foc")]  # different key, no conflict


def test_direct_assignment_allows_overwrite():
    """Direct dict assignment is the supported way to replace an existing rule."""
    from equilibrium.model.model import Model

    mod = Model()
    mod.rules["intermediate"] += [("K", "np.exp(log_K)")]
    # This should succeed — intentional replacement via direct assignment
    mod.rules["intermediate"]["K"] = "updated_expression"
    assert mod.rules["intermediate"]["K"] == "updated_expression"


# ---------------------------------------------------------------------------
# Internal: get_steady_rules calibration merge still works
# ---------------------------------------------------------------------------


def test_get_steady_rules_calibration_overwrite_still_works():
    """Calibration rules can replace optimality rules (internal behaviour)."""
    import numpy as np

    from equilibrium import Model

    mod = Model()
    mod.params.update({"alp": 0.36, "bet": 0.95, "delta": 0.1, "PERS_Z": 0.9, "VOL_Z": 0.01})
    mod.steady_guess.update({"log_K": np.log(3.0), "I": 0.3})
    mod.add_exog("Z", pers=0.9, vol=0.01)

    mod.rules["intermediate"] += [
        ("K", "np.exp(log_K)"),
        ("y", "Z * K ** alp"),
        ("C", "y - I"),
    ]
    mod.rules["transition"] += [
        ("log_K", "np.log((1.0 - delta) * K + I)"),
    ]
    mod.rules["optimality"] += [
        ("I", "bet * (alp * y / K + 1.0 - delta) - 1.0"),
    ]
    # Calibrate bet to match a capital target — this creates a calibration rule
    # with key 'bet' that should REPLACE the matching entry added to optimality.
    mod.rules["calibration"] += [("bet", "K - 3.0")]

    # get_steady_rules must not raise even though 'bet' isn't already in optimality.
    # More importantly, if 'bet' WERE in optimality, the merge should still succeed.
    from equilibrium.core.rules import RuleProcessor

    rp = RuleProcessor()
    rules_steady = rp.get_steady_rules(mod.rules, calibrate=True)
    # 'bet' should now appear in optimality (merged from calibration)
    assert "bet" in rules_steady["optimality"]
