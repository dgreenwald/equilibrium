#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for initial_guess_from_label parameter in solve_steady.

Tests that when initial_guess_from_label is set:
1. Loads initial guess from the alternative model's saved steady state
2. Raises FileNotFoundError when alternative file doesn't exist
3. Works with both calibrated and non-calibrated steady states
4. Existing behavior is unchanged when parameter is None
"""

import numpy as np
import pytest

from equilibrium import Model
from equilibrium.settings import get_settings


def create_test_model(label="test_model", **kwargs):
    """Create a simple test model with calibration."""
    mod = Model(label=label, **kwargs)

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

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_load_initial_guess_from_alternative_label():
    """Test that initial_guess_from_label loads from alternative model."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    baseline_label = "test_baseline_model"
    alt_label = "test_alt_model"

    # Clean up any existing files
    for label in [baseline_label, alt_label]:
        for f in save_dir.glob(f"{label}*"):
            f.unlink()

    try:
        # Step 1: Create and solve baseline model with calibration
        baseline_model = create_test_model(label=baseline_label)
        baseline_model.solve_steady(calibrate=True, save=True, display=False)
        baseline_bet = baseline_model.params["bet"]
        baseline_I = baseline_model.steady_dict["I"]

        # Step 2: Create alternative model with different calibration target
        alt_model = create_test_model(label=alt_label)
        alt_model.rules["calibration"]["bet"] = "K - 8.0"  # Replace existing target
        alt_model.finalize()
        alt_model.solve_steady(calibrate=True, save=True, display=False)
        alt_bet = alt_model.params["bet"]
        alt_I = alt_model.steady_dict["I"]

        # Verify they are different
        assert not np.isclose(
            baseline_bet, alt_bet
        ), "Models should have different bet values"
        assert not np.isclose(
            baseline_I, alt_I
        ), "Models should have different I values"

        # Step 3: Create new baseline model and load from alternative model's initial guess
        new_baseline = create_test_model(label=baseline_label)
        # Use wrong initial values to ensure we're actually loading
        new_baseline.params.overwrite_item("bet", 0.80)
        new_baseline.steady_guess.update({"I": 0.001, "log_K": np.log(0.5)})

        # Solve with initial guess from alternative model
        new_baseline.solve_steady(
            calibrate=True,
            initial_guess_from_label=alt_label,
            display=False,
        )

        # The initial guess should be close to alt_model's values
        # (Since we use alt model as initial guess, and baseline recalibrates,
        # the final result should converge to baseline's calibration target,
        # but we can verify the initial guess was loaded by checking init_dict)
        assert "I" in new_baseline.init_dict, "I should be in init_dict after loading"

        # The loaded initial guess should be from alt_model
        assert np.isclose(
            new_baseline.init_dict["I"], alt_I, rtol=0.01
        ), f"Should load initial guess I from {alt_label}, expected {alt_I}, got {new_baseline.init_dict['I']}"

    finally:
        # Clean up
        for label in [baseline_label, alt_label]:
            for f in save_dir.glob(f"{label}*"):
                f.unlink()


def test_initial_guess_from_label_raises_file_not_found():
    """Test that initial_guess_from_label raises FileNotFoundError when file doesn't exist."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label = "test FileNotFoundError"
    fake_label = "nonexistent_model"

    # Clean up any existing files
    for f in save_dir.glob(f"{label}*"):
        f.unlink()
    for f in save_dir.glob(f"{fake_label}*"):
        f.unlink()

    try:
        mod = create_test_model(label=label)

        # Should raise FileNotFoundError because nonexistent_model_steady_state.json doesn't exist
        with pytest.raises(FileNotFoundError, match="nonexistent_model"):
            mod.solve_steady(
                calibrate=True,
                initial_guess_from_label=fake_label,
                display=False,
            )

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


def test_initial_guess_from_label_with_non_calibrated_steady_state():
    """Test that initial_guess_from_label works with non-calibrated steady states."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label1 = "test_noncalib_1"
    label2 = "test_noncalib_2"

    # Clean up any existing files
    for label in [label1, label2]:
        for f in save_dir.glob(f"{label}*"):
            f.unlink()

    try:
        # Step 1: Create and solve model without calibration
        model1 = create_test_model(label=label1)
        model1.rules["calibration"] = {}  # No calibration
        model1.finalize()
        model1.solve_steady(calibrate=False, save=True, display=False)
        I_1 = model1.steady_dict["I"]

        # Step 2: Create second model and load from first
        model2 = create_test_model(label=label2)
        model2.rules["calibration"] = {}
        model2.finalize()

        # Use bad initial guess
        model2.steady_guess.update({"I": 0.001, "log_K": np.log(0.5)})

        # Solve with initial guess from model1
        model2.solve_steady(
            calibrate=False,
            initial_guess_from_label=label1,
            display=False,
        )

        # Should load the initial guess from model1
        assert np.isclose(
            model2.init_dict["I"], I_1, rtol=0.01
        ), f"Should load initial guess from {label1}"

        # And should converge to similar steady state
        assert np.isclose(
            model2.steady_dict["I"], I_1, rtol=0.01
        ), "Should converge to similar steady state"

    finally:
        # Clean up
        for label in [label1, label2]:
            for f in save_dir.glob(f"{label}*"):
                f.unlink()


def test_initial_guess_from_label_none_preserves_default_behavior():
    """Test that initial_guess_from_label=None preserves existing behavior."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label = "test_default_behavior"

    # Clean up any existing files
    for f in save_dir.glob(f"{label}*"):
        f.unlink()

    try:
        # Step 1: Solve and save
        model1 = create_test_model(label=label)
        model1.solve_steady(calibrate=True, save=True, display=False)
        calibrated_bet = model1.params["bet"]
        I_ss = model1.steady_dict["I"]

        # Step 2: Load with initial_guess_from_label=None (default)
        model2 = create_test_model(label=label)
        model2.params.overwrite_item("bet", 0.85)  # Wrong value

        # Explicitly pass None to verify default behavior
        model2.solve_steady(
            calibrate=True,
            load_initial_guess=True,
            initial_guess_from_label=None,  # Explicit None
            display=False,
        )

        # Should load from its own label (model2.label == model1.label == label)
        assert np.isclose(
            model2.params["bet"], calibrated_bet, rtol=0.01
        ), "Should load from own label when initial_guess_from_label=None"

        assert np.isclose(
            model2.steady_dict["I"], I_ss, rtol=0.01
        ), "Should converge to same steady state"

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


def test_initial_guess_from_label_parameter_filtering():
    """Test that parameter filtering still works with initial_guess_from_label."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label1 = "test_filtering_1"
    label2 = "test_filtering_2"

    # Clean up any existing files
    for label in [label1, label2]:
        for f in save_dir.glob(f"{label}*"):
            f.unlink()

    try:
        # Step 1: Create and save model with calibration
        model1 = create_test_model(label=label1)
        model1.solve_steady(calibrate=True, save=True, display=False)
        calibrated_bet = model1.params["bet"]

        # Step 2: Create model2 where bet is no longer calibrated
        model2 = create_test_model(label=label2)
        model2.rules["calibration"] = {}  # Remove calibration
        model2.finalize()

        preset_bet = 0.88
        model2.params.overwrite_item("bet", preset_bet)

        # Solve with initial guess from model1
        model2.solve_steady(
            calibrate=True,
            initial_guess_from_label=label1,
            display=False,
        )

        # Should keep preset bet because it's not in model2's calibration rules
        # even though model1 had it calibrated
        assert np.isclose(
            model2.params["bet"], preset_bet
        ), f"Should keep preset bet={preset_bet} when not in calibration rules"

        assert not np.isclose(
            model2.params["bet"], calibrated_bet
        ), "Should NOT load calibrated bet when it's not in calibration rules"

    finally:
        # Clean up
        for label in [label1, label2]:
            for f in save_dir.glob(f"{label}*"):
                f.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
