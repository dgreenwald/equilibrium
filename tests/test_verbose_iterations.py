#!/usr/bin/env python3
"""
Test verbose_iterations feature for steady state solving.
"""

from pathlib import Path

import jax
import numpy as np

from equilibrium import Model
from equilibrium.settings import get_settings
from equilibrium.utils import io

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


def test_verbose_iterations_prunes_old_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIUM_PATHS__DEBUG_DIR", str(tmp_path))
    monkeypatch.setenv("EQUILIBRIUM_DEBUG__KEEP_ITERATION_LOGS", "1")
    get_settings.cache_clear()
    settings = get_settings()

    mod = create_simple_model()
    res1 = mod.solve_steady(calibrate=False, verbose_iterations=True, display=False)
    assert res1.success

    mod2 = create_simple_model()
    res2 = mod2.solve_steady(calibrate=False, verbose_iterations=True, display=False)
    assert res2.success

    steady_label = mod.mod_steady.label if mod.mod_steady is not None else mod.label
    logs = sorted(
        settings.paths.debug_dir.glob(f"{steady_label}_steady_iterations_*.txt")
    )
    assert len(logs) == 1, "Should keep only the most recent iteration log"


def test_prune_files_by_stem(tmp_path):
    stem = "model_steady_iterations"
    filenames = [
        "model_steady_iterations_20250101_000000.txt",
        "model_steady_iterations_20250102_000000.txt",
        "model_steady_iterations_20250103_000000.txt",
    ]
    for name in filenames:
        (tmp_path / name).write_text("test", encoding="utf-8")

    io.prune_files_by_stem(tmp_path, stem, keep=2, suffix=".txt")

    remaining = sorted(tmp_path.glob(f"{stem}_*.txt"))
    assert len(remaining) == 2
    assert remaining[-1].name.endswith("20250103_000000.txt")
