#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for calibration I/O and automatic saving.
"""


import numpy as np
import pytest

from equilibrium import Model
from equilibrium.solvers.calibration import (
    CalibrationResult,
    PointTarget,
    ShockParam,
    calibrate,
)
from equilibrium.solvers.linear_spec import LinearSpec
from equilibrium.utils.io import (
    read_calibrated_param,
    read_calibrated_params,
    save_calibrated_params,
)


def create_simple_model(label="test_calib_io"):
    """Create a simple RBC model for testing."""
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
    mod.solve_steady(calibrate=False)
    mod.linearize()

    return mod


class TestCalibrationIO:
    """Tests for calibration I/O functions."""

    def test_save_and_read_params(self, tmp_path):
        """Test basic saving and reading of parameters."""
        params = {
            "beta": 0.95,
            "regime_tau_r1": 0.3,
            "regime_tau_r2": 0.4,
            "shock_Z_r1_t0": 0.01,
            "shock_Z_r2_t0": 0.02,
        }
        label = "test_io_basic"

        # Save
        path = save_calibrated_params(params, label, save_dir=tmp_path)
        assert path.exists()
        assert path.name == f"{label}_calibrated_params.json"

        # Read full
        read = read_calibrated_params(label, save_dir=tmp_path)
        assert read == params

    def test_read_params_with_regime_filtering(self, tmp_path):
        """Test reading parameters with regime filtering and renaming."""
        params = {
            "beta": 0.95,  # Global
            "regime_tau_r1": 0.3,  # Regime 1 specific
            "regime_tau_r2": 0.4,  # Regime 2 specific
            "regime_phi_r1_2": 0.5,  # Shared 1 & 2
            "shock_Z_r1_t0": 0.01,  # Shock regime 1
            "shock_Z_r2_t0": 0.02,  # Shock regime 2
        }
        label = "test_io_regime"
        save_calibrated_params(params, label, save_dir=tmp_path)

        # 1. Read regime 1
        r1_params = read_calibrated_params(label, save_dir=tmp_path, regime=1)

        # Check globals kept
        assert r1_params["beta"] == 0.95

        # Check renaming
        assert r1_params["tau"] == 0.3
        assert "regime_tau_r1" not in r1_params

        # Check shared regime
        assert r1_params["phi"] == 0.5

        # Check filtering
        assert "regime_tau_r2" not in r1_params
        assert "tau" in r1_params  # Confirmed above, but to be clear

        # Check shock filtering
        assert "shock_Z_r1_t0" in r1_params
        assert "shock_Z_r2_t0" not in r1_params

        # 2. Read regime 2
        r2_params = read_calibrated_params(label, save_dir=tmp_path, regime=2)

        assert r2_params["beta"] == 0.95
        assert r2_params["tau"] == 0.4
        assert r2_params["phi"] == 0.5
        assert "shock_Z_r1_t0" not in r2_params
        assert "shock_Z_r2_t0" in r2_params

    def test_read_calibrated_param_helper(self, tmp_path):
        """Test the helper function for reading single parameters."""
        params = {
            "beta": 0.95,
            "regime_tau_r1": 0.3,
        }
        label = "test_io_single"
        save_calibrated_params(params, label, save_dir=tmp_path)

        # Read global
        val = read_calibrated_param(label, "beta", save_dir=tmp_path)
        assert val == 0.95

        # Read specific with regime
        val = read_calibrated_param(label, "tau", save_dir=tmp_path, regime=1)
        assert val == 0.3

        # Read specific without regime (should fail or return full name if used?)
        # read_calibrated_param("tau") would fail because key is "regime_tau_r1"
        with pytest.raises(KeyError):
            read_calibrated_param(label, "tau", save_dir=tmp_path)

        # Can read full name
        val = read_calibrated_param(label, "regime_tau_r1", save_dir=tmp_path)
        assert val == 0.3


class TestCalibrateAutoSave:
    """Tests for automatic saving in calibrate()."""

    def test_auto_save_on_success(self, tmp_path):
        """Test that calibrate saves parameters on success when label provided."""
        model = create_simple_model()

        # Simple calibration
        calib_params = [ShockParam("Z_til", initial=0.005, bounds=(0.0001, 0.1))]

        # Fake target matching the initial guess roughly to ensure success
        # But here we want to solve it.
        # Let's just run a real quick one.
        ref_spec = LinearSpec(shock_name="Z_til", shock_size=0.01, Nt=10)
        # We know 0.01 works for shock_size=0.01 if we target something generated by it

        # Generate target
        irf = model.linear_mod.compute_irfs(10)["Z_til"]
        target_val = irf.UX[5, irf.var_names.index("I")] * 0.01

        targets = [PointTarget(variable="I", time=5, value=target_val)]

        label = "auto_save_success"

        # Run calibrate with label
        result = calibrate(
            model=model,
            targets=targets,
            calib_params=calib_params,
            solver="linear_irf",
            spec=ref_spec,
            label=label,
            save_dir=tmp_path,
        )

        assert result.success

        # Check file
        expected_file = tmp_path / f"{label}_calibrated_params.json"
        assert expected_file.exists()

        # Check content
        params = read_calibrated_params(label, save_dir=tmp_path)
        assert len(params) == 1
        assert "shock_Z_til_r0_t0" in params

    def test_error_on_failure(self, tmp_path):
        """Test that calibrate raises RuntimeError on failure when label provided."""
        model = create_simple_model()

        # Impossible calibration
        calib_params = [ShockParam("Z_til", initial=0.01, bounds=(0.0, 0.1))]

        # Target that is impossible (e.g. negative investment when it should be positive or something extreme)
        # Or just use a bad target with very tight bounds/tolerance/maxiter
        # Set maxiter=1 to force failure

        targets = [PointTarget(variable="I", time=5, value=100.0)]

        label = "auto_save_failure"

        with pytest.raises(RuntimeError, match="Calibration failed"):
            calibrate(
                model=model,
                targets=targets,
                calib_params=calib_params,
                solver="linear_irf",
                spec=LinearSpec(shock_name="Z_til", shock_size=0.01, Nt=10),
                label=label,
                save_dir=tmp_path,
                maxiter=1,  # Force failure
            )

        # Check file does NOT exist
        expected_file = tmp_path / f"{label}_calibrated_params.json"
        assert not expected_file.exists()

    def test_manual_save(self, tmp_path):
        """Test manually saving from result."""
        result = CalibrationResult(parameters={"p": 1.0}, success=True)

        label = "manual_save"
        result.save(label, save_dir=tmp_path)

        assert (tmp_path / f"{label}_calibrated_params.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
