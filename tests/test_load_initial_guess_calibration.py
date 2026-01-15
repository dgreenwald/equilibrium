#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for load_initial_guess behavior with calibration parameters.

Tests that when load_initial_guess=True:
1. Always tries to load from calibrated version first
2. Loads steady state values correctly
3. Only loads calibrated parameters that are still in rules['calibration']
4. Ignores calibrated parameters no longer in rules['calibration']
"""

import numpy as np
import pytest

from equilibrium import Model
from equilibrium.settings import get_settings


def set_test_model(label="test_model", **kwargs):
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


def test_load_initial_guess_with_calibration():
    """Test that load_initial_guess loads calibrated parameters correctly."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label = "test_load_calib"

    # Clean up any existing files
    for f in save_dir.glob(f"{label}*"):
        f.unlink()

    try:
        # Step 1: Solve and save with calibration
        mod1 = set_test_model(label=label)
        initial_bet = mod1.params["bet"]
        mod1.solve_steady(calibrate=True, save=True, display=False)
        calibrated_bet = mod1.params["bet"]

        # Verify calibration actually changed bet
        assert not np.isclose(
            initial_bet, calibrated_bet
        ), "Calibration should have changed bet"

        # Step 2: Load with calibrate=True and load_initial_guess=True
        mod2 = set_test_model(label=label)
        mod2.params.overwrite_item("bet", 0.90)  # Set wrong initial value
        mod2.solve_steady(calibrate=True, load_initial_guess=True, display=False)

        # Should load calibrated bet
        assert np.isclose(
            mod2.params["bet"], calibrated_bet
        ), "Should load calibrated bet when calibrate=True and load_initial_guess=True"

        # Step 3: Load with calibrate=False and load_initial_guess=True
        # Should STILL load calibrated bet because it tries calibrated version first
        mod3 = set_test_model(label=label)
        mod3.params.overwrite_item("bet", 0.85)
        mod3.solve_steady(calibrate=False, load_initial_guess=True, display=False)

        # Should load calibrated bet (from calibrated file)
        assert np.isclose(
            mod3.params["bet"], calibrated_bet
        ), "Should load calibrated bet even with calibrate=False when load_initial_guess=True"

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


def test_load_initial_guess_filters_non_calibration_params():
    """Test that parameters no longer in calibration rules are not loaded."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label = "test_filter_params"

    # Clean up any existing files
    for f in save_dir.glob(f"{label}*"):
        f.unlink()

    try:
        # Step 1: Solve with calibration and save
        mod1 = set_test_model(label=label)
        mod1.solve_steady(calibrate=True, save=True, display=False)
        calibrated_bet = mod1.params["bet"]

        # Step 2: Create new model where bet is no longer being calibrated
        mod2 = set_test_model(label=label)
        mod2.rules["calibration"] = {}  # Remove all calibration rules
        preset_bet = 0.88
        mod2.params.overwrite_item("bet", preset_bet)
        mod2.finalize()

        # Solve with load_initial_guess=True
        mod2.solve_steady(calibrate=True, load_initial_guess=True, display=False)

        # Should keep preset bet because it's not in calibration rules
        assert np.isclose(
            mod2.params["bet"], preset_bet
        ), f"Should keep preset bet={preset_bet} when it's no longer in calibration rules"

        assert not np.isclose(
            mod2.params["bet"], calibrated_bet
        ), "Should NOT load calibrated bet when it's not in calibration rules"

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


def test_load_initial_guess_loads_steady_state_values():
    """Test that load_initial_guess loads steady state variable values into init_dict."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label = "test_load_vars"

    # Clean up any existing files
    for f in save_dir.glob(f"{label}*"):
        f.unlink()

    try:
        # Step 1: Solve and save
        mod1 = set_test_model(label=label)
        mod1.solve_steady(calibrate=True, save=True, display=False)

        # Get steady state values
        I_ss = mod1.steady_dict["I"]
        bet_cal = mod1.params["bet"]

        # Step 2: Load in new model with load_initial_guess=True
        mod2 = set_test_model(label=label)

        # Solve with load_initial_guess - this should load values into init_dict
        mod2.solve_steady(calibrate=True, load_initial_guess=True, display=False)

        # Verify that init_dict was updated with loaded values
        # (This is what load_initial_guess is supposed to do)
        assert "I" in mod2.init_dict, "I should be in init_dict"
        assert "log_K" in mod2.init_dict, "log_K should be in init_dict"

        # After successful solve with calibrate=True, steady state should match
        assert np.isclose(
            mod2.steady_dict["I"], I_ss, rtol=0.01
        ), f"Should have similar steady state value for I: {mod2.steady_dict['I']} vs {I_ss}"

        # And calibrated parameter should be loaded
        assert np.isclose(
            mod2.params["bet"], bet_cal, rtol=0.01
        ), "Should have loaded calibrated bet parameter"

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


def test_load_initial_guess_fallback_to_non_calibrated():
    """Test that load_initial_guess falls back to non-calibrated version if needed."""
    settings = get_settings()
    save_dir = settings.paths.save_dir

    label = "test_fallback"

    # Clean up any existing files
    for f in save_dir.glob(f"{label}*"):
        f.unlink()

    try:
        # Step 1: Solve WITHOUT calibration and save
        mod1 = set_test_model(label=label)
        mod1.rules["calibration"] = {}  # No calibration
        mod1.finalize()
        mod1.solve_steady(calibrate=False, save=True, display=False)

        I_ss = mod1.steady_dict["I"]

        # Verify that files were saved
        files = list(save_dir.glob(f"{label}*"))
        assert len(files) > 0, "Should have saved at least one file"

        # Step 2: Load with load_initial_guess=True
        # Since no calibrated file exists, it should fall back to non-calibrated
        mod2 = set_test_model(label=label)
        mod2.rules["calibration"] = {}
        mod2.finalize()
        mod2.solve_steady(calibrate=False, load_initial_guess=True, display=False)

        # Should load the steady state value
        assert np.isclose(
            mod2.steady_dict["I"], I_ss
        ), "Should load steady state from fallback file"

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


def test_initial_guess_propagates_to_child_model():
    """Test that loaded initial guess is properly passed to child steady model.

    This test verifies the fix for the issue where the loaded initial guess
    was not being propagated to the child model (mod_steady for non-calibrating
    solves or mod_steady_cal for calibrating solves), causing solves to fail or
    use incorrect initial guesses.
    """
    settings = get_settings()
    save_dir = settings.paths.save_dir
    label = "test_init_propagation"

    try:
        # Create and solve first model
        mod1 = Model(label=label)
        mod1.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
            }
        )
        mod1.steady_guess.update({"I": 0.5, "log_K": np.log(6.0)})
        mod1.rules["intermediate"] += [
            ("K_new", "I + (1.0 - delta) * K"),
            ("Z", "Z_bar + Z_til"),
            ("fk", "alp * Z * (K ** (alp - 1.0))"),
            ("y", "Z * (K ** alp)"),
            ("c", "y - I"),
            ("uc", "c ** (-gam)"),
            ("K", "np.exp(log_K)"),
        ]
        mod1.rules["expectations"] += [
            ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
        ]
        mod1.rules["transition"] += [("log_K", "np.log(K_new)")]
        mod1.rules["optimality"] += [("I", "E_Om_K - 1.0")]
        mod1.rules["calibration"] += [("bet", "K - 6.0")]
        mod1.add_exog("Z_til", pers=0.95, vol=0.1)
        mod1.finalize()

        # Solve and save
        res1 = mod1.solve_steady(
            calibrate=True, save=True, load_initial_guess=False, display=False
        )

        assert res1.success, "First solve should succeed"
        I_ss = mod1.steady_dict["I"]
        log_K_ss = mod1.steady_dict["log_K"]

        # Create second model with bad initial guess
        mod2 = Model(label=label)
        mod2.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
            }
        )
        # Use bad initial guess
        mod2.steady_guess.update({"I": 0.001, "log_K": np.log(0.5)})
        mod2.rules["intermediate"] += [
            ("K_new", "I + (1.0 - delta) * K"),
            ("Z", "Z_bar + Z_til"),
            ("fk", "alp * Z * (K ** (alp - 1.0))"),
            ("y", "Z * (K ** alp)"),
            ("c", "y - I"),
            ("uc", "c ** (-gam)"),
            ("K", "np.exp(log_K)"),
        ]
        mod2.rules["expectations"] += [
            ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
        ]
        mod2.rules["transition"] += [("log_K", "np.log(K_new)")]
        mod2.rules["optimality"] += [("I", "E_Om_K - 1.0")]
        mod2.rules["calibration"] += [("bet", "K - 6.0")]
        mod2.add_exog("Z_til", pers=0.95, vol=0.1)
        mod2.finalize()

        # The initial guess should be bad before loading
        assert mod2.init_dict["I"] < 0.01, "Initial guess should be bad before loading"

        # Now solve with load_initial_guess=True
        # This should load the good initial guess from mod1 and use it
        res2 = mod2.solve_steady(
            calibrate=True, save=False, load_initial_guess=True, display=False
        )

        # The solve should succeed because the loaded initial guess is good
        assert res2.success, (
            "Second solve should succeed with loaded initial guess. "
            "If it fails, the initial guess was not properly propagated to child model."
        )

        # The solution should be the same
        assert np.isclose(mod2.steady_dict["I"], I_ss, rtol=1e-6)
        assert np.isclose(mod2.steady_dict["log_K"], log_K_ss, rtol=1e-6)

    finally:
        # Clean up
        for f in save_dir.glob(f"{label}*"):
            f.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
