#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for IrfResult, PathResult, and related functionality.
"""

import tempfile

import jax
import numpy as np
import pytest

from equilibrium import IrfResult, Model, PathResult
from equilibrium.plot import plot_irf_results

jax.config.update("jax_enable_x64", True)


def set_model(flags=None, params=None, steady_guess=None, **kwargs):
    """Create a simple model for testing."""
    mod = Model(flags=flags, params=params, steady_guess=steady_guess, **kwargs)

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


class TestPathResult:
    """Tests for the PathResult base class."""

    def test_create_path_result(self):
        """Test creating a PathResult with minimal data."""
        UX = np.random.randn(10, 2)
        Z = np.random.randn(10, 1)

        result = PathResult(
            UX=UX,
            Z=Z,
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        assert result.UX.shape == (10, 2)
        assert result.Z.shape == (10, 1)
        assert result.Y is None
        assert result.var_names == ["I", "log_K"]
        assert result.exog_names == ["Z_til"]

    def test_path_result_with_intermediates(self):
        """Test creating a PathResult with intermediate variables."""
        UX = np.random.randn(10, 2)
        Z = np.random.randn(10, 1)
        Y = np.random.randn(10, 5)

        result = PathResult(
            UX=UX,
            Z=Z,
            Y=Y,
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            y_names=["K_new", "fk", "y", "c", "uc"],
        )

        assert result.Y is not None
        assert result.Y.shape == (10, 5)
        assert len(result.y_names) == 5

    def test_path_result_save_load(self):
        """Test saving and loading PathResult."""
        UX = np.random.randn(10, 2)
        Z = np.random.randn(10, 1)
        Y = np.random.randn(10, 3)

        result = PathResult(
            UX=UX,
            Z=Z,
            Y=Y,
            model_label="test_model",
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            y_names=["y", "c", "K"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/path_result.npz"
            result.save(filepath, overwrite=True)

            loaded = PathResult.load(filepath)

            assert np.allclose(loaded.UX, result.UX)
            assert np.allclose(loaded.Z, result.Z)
            assert np.allclose(loaded.Y, result.Y)
            assert loaded.model_label == result.model_label
            assert loaded.var_names == result.var_names
            assert loaded.exog_names == result.exog_names
            assert loaded.y_names == result.y_names


class TestIrfResult:
    """Tests for the IrfResult class."""

    def test_create_irf_result(self):
        """Test creating an IrfResult."""
        UX = np.random.randn(20, 2)
        Z = np.random.randn(20, 1)
        Y = np.random.randn(20, 5)

        result = IrfResult(
            UX=UX,
            Z=Z,
            Y=Y,
            model_label="test_model",
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            y_names=["K_new", "fk", "y", "c", "uc"],
            shock_name="Z_til",
            shock_size=1.0,
        )

        assert result.shock_name == "Z_til"
        assert result.shock_size == 1.0
        assert result.UX.shape == (20, 2)
        assert result.Z.shape == (20, 1)
        assert result.Y.shape == (20, 5)

    def test_irf_result_save_load(self):
        """Test saving and loading IrfResult."""
        UX = np.random.randn(20, 2)
        Z = np.random.randn(20, 1)
        Y = np.random.randn(20, 3)

        result = IrfResult(
            UX=UX,
            Z=Z,
            Y=Y,
            model_label="test_model",
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            y_names=["y", "c", "K"],
            shock_name="Z_til",
            shock_size=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/irf_result.npz"
            result.save(filepath, overwrite=True)

            loaded = IrfResult.load(filepath)

            assert np.allclose(loaded.UX, result.UX)
            assert np.allclose(loaded.Z, result.Z)
            assert np.allclose(loaded.Y, result.Y)
            assert loaded.model_label == result.model_label
            assert loaded.shock_name == result.shock_name
            assert loaded.shock_size == result.shock_size
            assert loaded.var_names == result.var_names
            assert loaded.exog_names == result.exog_names
            assert loaded.y_names == result.y_names


class TestComputeIrfsWithIntermediates:
    """Tests for computing IRFs with intermediate variables."""

    def test_compute_irfs_returns_dict(self):
        """Test that compute_linear_irfs returns a dict of IrfResults."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        irf_dict = mod.compute_linear_irfs(20)

        # Should return a dictionary
        assert isinstance(irf_dict, dict)

        # Should have one entry per shock
        assert len(irf_dict) == len(mod.exog_list)
        assert "Z_til" in irf_dict

        # Each entry should be an IrfResult
        for shock_name, irf_result in irf_dict.items():
            assert isinstance(irf_result, IrfResult)
            assert irf_result.shock_name == shock_name

    def test_irf_result_contains_intermediate_variables(self):
        """Test that IrfResults contain intermediate variables."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        irf_dict = mod.compute_linear_irfs(20)
        irf_result = irf_dict["Z_til"]

        # Should have Y (intermediate variables)
        assert irf_result.Y is not None
        assert irf_result.Y.shape[0] == 20  # 20 periods

        # Should have intermediate variable names
        assert len(irf_result.y_names) > 0
        # Expected intermediate variables from the model
        expected_intermediates = ["K_new", "Z", "fk", "y", "c", "uc", "K"]
        for var in expected_intermediates:
            assert var in irf_result.y_names

    def test_irf_intermediate_vars_nonzero(self):
        """Test that intermediate variables respond to shocks."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        irf_dict = mod.compute_linear_irfs(20)
        irf_result = irf_dict["Z_til"]

        # Intermediate variables should not all be zero
        # (they should respond to the shock)
        assert not np.allclose(irf_result.Y, 0.0)

        # At least some intermediate variables should have non-negligible responses
        max_abs_response = np.max(np.abs(irf_result.Y))
        assert max_abs_response > 1e-6

    def test_irf_dimensions_match(self):
        """Test that all arrays in IrfResult have consistent dimensions."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        Nt = 25
        irf_dict = mod.compute_linear_irfs(Nt)
        irf_result = irf_dict["Z_til"]

        # All arrays should have Nt periods
        assert irf_result.UX.shape[0] == Nt
        assert irf_result.Z.shape[0] == Nt
        assert irf_result.Y.shape[0] == Nt

        # Column dimensions should match variable names
        assert irf_result.UX.shape[1] == len(irf_result.var_names)
        assert irf_result.Z.shape[1] == len(irf_result.exog_names)
        assert irf_result.Y.shape[1] == len(irf_result.y_names)

    def test_backward_compat_irfs_tensor_still_exists(self):
        """Test that the old .irfs tensor is still available."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        _ = mod.compute_linear_irfs(20)

        # The old irfs tensor should still be stored
        assert mod.linear_mod.irfs is not None
        assert mod.linear_mod.irfs.shape == (1, 20, mod.linear_mod.A.shape[0])


class TestPlotIrfResults:
    """Tests for plotting IrfResult dictionaries."""

    def test_plot_single_irf_dict(self):
        """Test plotting a single dict of IrfResults."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        irf_dict = mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_irf_results(
                irf_dict,
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
            )

            # Should create at least one plot file
            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_multiple_irf_dicts(self):
        """Test plotting multiple dicts of IrfResults for comparison."""
        mod1 = set_model(label="model1")
        mod1.solve_steady(calibrate=True)
        mod1.linearize()
        irf_dict1 = mod1.compute_linear_irfs(20)

        mod2 = set_model(label="model2")
        mod2.params["bet"] = 0.96  # Different parameter
        mod2.solve_steady(calibrate=True)
        mod2.linearize()
        irf_dict2 = mod2.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_irf_results(
                [irf_dict1, irf_dict2],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                result_names=["Baseline", "High Beta"],
            )

            # Should create at least one plot file
            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_irf_with_intermediate_vars(self):
        """Test plotting IRFs including intermediate variables."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        irf_dict = mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Include both state variables and intermediate variables
            paths = plot_irf_results(
                irf_dict,
                include_list=["I", "log_K", "y", "c"],  # Mix of UX and Y vars
                plot_dir=tmpdir,
            )

            # Should create plots successfully
            assert len(paths) > 0
            for path in paths:
                assert path.exists()


class TestMultiShockModel:
    """Tests with a model that has multiple shocks."""

    def test_multi_shock_irf_dict(self):
        """Test IRFs with a model that has multiple exogenous shocks."""
        mod = Model(label="two_shock_model")
        mod.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
                "G_bar": 0.2,
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
            ("G", "G_bar + G_til"),
            ("fk", "alp * Z * (K ** (alp - 1.0))"),
            ("y", "Z * (K ** alp)"),
            ("c", "y - I - G"),
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

        # Add two shocks
        mod.add_exog("Z_til", pers=0.95, vol=0.1)
        mod.add_exog("G_til", pers=0.9, vol=0.05)

        mod.finalize()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        irf_dict = mod.compute_linear_irfs(20)

        # Should have two entries
        assert len(irf_dict) == 2
        assert "Z_til" in irf_dict
        assert "G_til" in irf_dict

        # Each should be an IrfResult with the correct shock name
        assert irf_dict["Z_til"].shock_name == "Z_til"
        assert irf_dict["G_til"].shock_name == "G_til"

        # Both should have intermediate variables
        assert irf_dict["Z_til"].Y is not None
        assert irf_dict["G_til"].Y is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
