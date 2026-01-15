#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for label-based loading of IRFs and deterministic results.

This module tests the new functionality for loading saved IRFs and deterministic
results by their labels, enabling memory-efficient workflows.
"""


import numpy as np
import pytest

from equilibrium import (
    Model,
    load_deterministic_result,
    load_model_irfs,
    load_sequence_result,
)
from equilibrium.solvers import deterministic, linear
from equilibrium.solvers.det_spec import DetSpec


def set_model(label="_default"):
    """Create a simple test model."""
    mod = Model(label=label)

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

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_det_spec_label_attribute():
    """Test that DetSpec now has a label attribute."""
    # Test default label
    spec1 = DetSpec()
    assert hasattr(spec1, "label")
    assert spec1.label == "_default"

    # Test custom label
    spec2 = DetSpec(label="my_experiment")
    assert spec2.label == "my_experiment"

    # Test label can be set after creation
    spec3 = DetSpec()
    spec3.label = "another_experiment"
    assert spec3.label == "another_experiment"


def test_sequence_result_experiment_label():
    """Test that SequenceResult stores and preserves experiment_label."""
    mod = set_model()
    mod.solve_steady(calibrate=True)

    # Create DetSpec with label
    spec = DetSpec(label="test_experiment")
    spec.add_regime(0, preset_par_regime={})
    spec.add_shock(0, "Z_til", 0, 0.01)

    # Solve sequence
    result = deterministic.solve_sequence(spec, mod, Nt=10, save_results=False)

    # Check that experiment_label was set
    assert result.experiment_label == "test_experiment"


def test_save_and_load_sequence_result_with_label(tmp_path):
    """Test saving and loading SequenceResult with experiment label."""
    from equilibrium.settings import get_settings

    # Temporarily override save_dir
    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="test_model")
        mod.solve_steady(calibrate=True)

        # Create DetSpec with label
        spec = DetSpec(label="experiment_1")
        spec.add_regime(0, preset_par_regime={})
        spec.add_shock(0, "Z_til", 0, 0.01)

        # Solve and save
        result = deterministic.solve_sequence(spec, mod, Nt=10, save_results=False)
        save_path = result.save()

        # Check filename includes both labels
        assert "test_model_experiment_1" in str(save_path)
        assert save_path.exists()

        # Load using loader function
        loaded = load_sequence_result("test_model", "experiment_1", save_dir=tmp_path)

        # Verify data matches
        assert loaded.model_label == "test_model"
        assert loaded.experiment_label == "experiment_1"
        assert loaded.n_regimes == result.n_regimes
        assert np.allclose(loaded.regimes[0].UX, result.regimes[0].UX)

    finally:
        settings.paths.save_dir = old_save_dir


def test_load_sequence_result_without_experiment_label(tmp_path):
    """Test loading SequenceResult when no experiment label is used."""
    from equilibrium.settings import get_settings

    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="test_model")
        mod.solve_steady(calibrate=True)

        # Create DetSpec without custom label (uses _default)
        spec = DetSpec()
        spec.add_regime(0, preset_par_regime={})
        spec.add_shock(0, "Z_til", 0, 0.01)

        # Solve and save
        result = deterministic.solve_sequence(spec, mod, Nt=10, save_results=False)
        save_path = result.save()

        # Filename should include the default experiment label
        assert save_path.name == "test_model__default.npz"

        # Load without experiment label
        loaded = load_sequence_result("test_model", save_dir=tmp_path)

        assert loaded.model_label == "test_model"
        assert np.allclose(loaded.regimes[0].UX, result.regimes[0].UX)

    finally:
        settings.paths.save_dir = old_save_dir


def test_save_and_load_deterministic_result_with_label(tmp_path):
    """Test saving and loading DeterministicResult with experiment label."""
    from equilibrium.settings import get_settings

    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="test_model")
        mod.solve_steady(calibrate=True)

        spec = DetSpec(label="experiment_2")
        spec.add_regime(0, preset_par_regime={})
        spec.add_shock(0, "Z_til", 0, 0.01)

        result = deterministic.solve_sequence(spec, mod, Nt=10, save_results=False)

        # Get the first regime's DeterministicResult and save it
        det_result = result.regimes[0]
        save_path = det_result.save(experiment_label="experiment_2")

        # Check filename
        assert "test_model_experiment_2" in str(save_path)
        assert save_path.exists()

        # Load using loader function
        loaded = load_deterministic_result(
            "test_model", "experiment_2", save_dir=tmp_path
        )

        assert loaded.model_label == det_result.model_label
        assert np.allclose(loaded.UX, det_result.UX)
        assert np.allclose(loaded.Z, det_result.Z)

    finally:
        settings.paths.save_dir = old_save_dir


def test_load_model_irfs(tmp_path):
    """Test loading IRFs by model label."""
    from equilibrium.settings import get_settings

    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="irf_test_model")
        mod.solve_steady(calibrate=True)
        mod.linearize()

        # Compute and save IRFs
        Nt_irf = 20
        mod.compute_linear_irfs(Nt_irf=Nt_irf)
        from equilibrium.io import resolve_output_path

        save_path = resolve_output_path(
            None,
            result_type="irfs",
            model_label="irf_test_model",
            save_dir=tmp_path,
            suffix=".npz",
        )

        assert save_path.exists()

        # Load all IRFs
        irfs = load_model_irfs("irf_test_model", save_dir=tmp_path)

        # Should return a dict
        assert isinstance(irfs, dict)
        assert "Z_til" in irfs

        # Check IrfResult properties
        irf_result = irfs["Z_til"]
        assert irf_result.model_label == "irf_test_model"
        assert irf_result.shock_name == "Z_til"
        assert irf_result.UX.shape[0] == Nt_irf

        # Load specific shock
        irf_single = load_model_irfs("irf_test_model", shock="Z_til", save_dir=tmp_path)
        assert irf_single.shock_name == "Z_til"
        assert np.allclose(irf_single.UX, irf_result.UX)

    finally:
        settings.paths.save_dir = old_save_dir


def test_load_model_irfs_nonexistent_shock(tmp_path):
    """Test that loading a non-existent shock raises KeyError."""
    from equilibrium.settings import get_settings

    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="irf_test_model2")
        mod.solve_steady(calibrate=True)
        mod.linearize()

        mod.compute_linear_irfs(Nt_irf=20)

        # Try to load non-existent shock
        with pytest.raises(KeyError, match="Shock 'nonexistent' not found"):
            load_model_irfs("irf_test_model2", shock="nonexistent", save_dir=tmp_path)

    finally:
        settings.paths.save_dir = old_save_dir


def test_load_nonexistent_file_errors():
    """Test that loading non-existent files raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_irfs("nonexistent_model")

    with pytest.raises(FileNotFoundError):
        load_deterministic_result("nonexistent_model", "nonexistent_experiment")

    with pytest.raises(FileNotFoundError):
        load_sequence_result("nonexistent_model", "nonexistent_experiment")


def test_multiple_experiments_same_model(tmp_path):
    """Test saving multiple experiments for the same model."""
    from equilibrium.settings import get_settings

    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="baseline")
        mod.solve_steady(calibrate=True)

        # Run two different experiments
        experiments = ["small_shock", "large_shock"]
        shock_sizes = [0.01, 0.05]

        for exp_label, shock_size in zip(experiments, shock_sizes):
            spec = DetSpec(label=exp_label)
            spec.add_regime(0, preset_par_regime={})
            spec.add_shock(0, "Z_til", 0, shock_size)

            result = deterministic.solve_sequence(spec, mod, Nt=10, save_results=False)
            result.save()

        # Load both experiments
        result1 = load_sequence_result("baseline", "small_shock", save_dir=tmp_path)
        result2 = load_sequence_result("baseline", "large_shock", save_dir=tmp_path)

        # Verify they're different
        assert result1.experiment_label == "small_shock"
        assert result2.experiment_label == "large_shock"

        # The larger shock should have larger deviations
        assert np.abs(result2.regimes[0].UX).max() > np.abs(result1.regimes[0].UX).max()

    finally:
        settings.paths.save_dir = old_save_dir


def test_linear_solver_with_labeled_detspec(tmp_path):
    """Test that linear solver also respects DetSpec label."""
    from equilibrium.settings import get_settings

    settings = get_settings()
    old_save_dir = settings.paths.save_dir
    settings.paths.save_dir = tmp_path

    try:
        mod = set_model(label="linear_test")
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(label="linear_experiment")
        spec.add_regime(0, preset_par_regime={})
        spec.add_shock(0, "Z_til", 0, 0.01)

        result = linear.solve_sequence_linear(spec, mod, Nt=10)

        # Check label was passed through
        assert result.experiment_label == "linear_experiment"

        # Save and load
        result.save()
        loaded = load_sequence_result(
            "linear_test", "linear_experiment", save_dir=tmp_path
        )

        assert loaded.experiment_label == "linear_experiment"
        assert np.allclose(loaded.regimes[0].UX, result.regimes[0].UX)

    finally:
        settings.paths.save_dir = old_save_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
