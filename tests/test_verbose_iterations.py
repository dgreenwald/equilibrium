#!/usr/bin/env python3
"""
Test verbose_iterations feature for steady state solving.
"""

from pathlib import Path

import jax
import numpy as np

from equilibrium import Model
from equilibrium.settings import get_settings

jax.config.update("jax_enable_x64", True)


def create_simple_model():
    """Create a simple RBC model for testing."""
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
        ("K", "np.exp(log_K)"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
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


def test_verbose_iterations_creates_log_file():
    """Test that verbose_iterations creates a log file with iteration details."""
    settings = get_settings()
    debug_dir = Path(settings.paths.debug_dir)

    # Get list of existing log files before running
    existing_logs = (
        set(debug_dir.glob("*_steady_iterations_*.txt"))
        if debug_dir.exists()
        else set()
    )

    # Create model and solve with verbose_iterations
    mod = create_simple_model()
    result = mod.solve_steady(calibrate=False, verbose_iterations=True, display=False)

    # Check that solve succeeded
    assert result.success, "Steady state solve should succeed"

    # Get list of log files after running
    new_logs = set(debug_dir.glob("*_steady_iterations_*.txt"))

    # Verify a new log file was created
    created_logs = new_logs - existing_logs
    assert len(created_logs) == 1, "Should create exactly one new log file"

    # Read the log file and verify it contains expected content
    log_file = list(created_logs)[0]
    with open(log_file, "r") as f:
        content = f.read()

    # Verify header
    assert "Steady State Solve Attempt - Detailed Iteration Log" in content
    assert "Model Label: _default_steady" in content

    # Verify it contains iteration information
    assert "Iteration 0:" in content
    assert "Intermediate Variables:" in content
    assert "State Variables (x):" in content
    assert "Control Variables (u):" in content
    assert "Residuals:" in content

    # Verify equations have values in brackets
    assert "K [" in content and "] = np.exp(log_K [" in content

    # Verify residuals show equilibrium conditions with values
    assert "(transition) [" in content  # New format with residual value in brackets
    assert "(optimality) [" in content
    assert "_new [" in content  # Should show x_new values in transition residuals


def test_verbose_iterations_disabled_by_default():
    """Test that verbose_iterations is disabled by default (no log file created)."""
    settings = get_settings()
    debug_dir = Path(settings.paths.debug_dir)

    # Get list of existing log files before running
    existing_logs = (
        set(debug_dir.glob("*_steady_iterations_*.txt"))
        if debug_dir.exists()
        else set()
    )

    # Create model and solve without verbose_iterations
    mod = create_simple_model()
    result = mod.solve_steady(calibrate=False, display=False)

    # Check that solve succeeded
    assert result.success, "Steady state solve should succeed"

    # Get list of log files after running
    new_logs = (
        set(debug_dir.glob("*_steady_iterations_*.txt"))
        if debug_dir.exists()
        else set()
    )

    # Verify no new log file was created
    created_logs = new_logs - existing_logs
    assert (
        len(created_logs) == 0
    ), "Should not create log file when verbose_iterations=False"
