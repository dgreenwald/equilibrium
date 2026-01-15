#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for new plot_model_irfs functionality with multiple shocks.
"""

import tempfile

import jax
import numpy as np
import pytest

from equilibrium import Model
from equilibrium.plot import plot_model_irfs

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


class TestPlotModelIrfsMultipleShocks:
    """Tests for new plot_model_irfs functionality with multiple shocks."""

    def test_plot_all_shocks_default(self):
        """Test plotting IRFs for all shocks when shocks parameter is None."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without shocks - should plot all shocks in model
            paths = plot_model_irfs(
                [mod],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
            )

            # Should have created plots for Z_til shock
            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_multiple_shocks_explicit(self):
        """Test plotting IRFs for multiple explicitly specified shocks."""
        # Create a model with two shocks
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
        mod.add_exog("Z_til", pers=0.95, vol=0.1)
        mod.add_exog("G_til", pers=0.9, vol=0.05)
        mod.finalize()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Plot IRFs for both shocks
            paths = plot_model_irfs(
                [mod],
                shocks=["Z_til", "G_til"],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
            )

            # Should have created plots for both shocks
            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_missing_shock_in_some_models(self):
        """Test that missing shock in some models uses NaN values."""
        # Create two models, one with Z_til, one with both Z_til and G_til
        mod1 = set_model(label="model1")
        mod1.solve_steady(calibrate=True)
        mod1.linearize()
        mod1.compute_linear_irfs(20)

        mod2 = Model(label="model2")
        mod2.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
                "G_bar": 0.2,
            }
        )
        mod2.steady_guess.update(
            {
                "I": 0.5,
                "log_K": np.log(6.0),
            }
        )
        mod2.rules["intermediate"] += [
            ("K_new", "I + (1.0 - delta) * K"),
            ("Z", "Z_bar + Z_til"),
            ("G", "G_bar + G_til"),
            ("fk", "alp * Z * (K ** (alp - 1.0))"),
            ("y", "Z * (K ** alp)"),
            ("c", "y - I - G"),
            ("uc", "c ** (-gam)"),
            ("K", "np.exp(log_K)"),
        ]
        mod2.rules["expectations"] += [
            ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
        ]
        mod2.rules["transition"] += [
            ("log_K", "np.log(K_new)"),
        ]
        mod2.rules["optimality"] += [
            ("I", "E_Om_K - 1.0"),
        ]
        mod2.rules["calibration"] += [
            ("bet", "K - 6.0"),
        ]
        mod2.add_exog("Z_til", pers=0.95, vol=0.1)
        mod2.add_exog("G_til", pers=0.9, vol=0.05)
        mod2.finalize()
        mod2.solve_steady(calibrate=True)
        mod2.linearize()
        mod2.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Plot IRFs for G_til shock which is missing from mod1
            paths = plot_model_irfs(
                [mod1, mod2],
                shock="G_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                model_names=["Model1", "Model2"],
            )

            # Should succeed without crashing
            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_all_shocks_from_multiple_models(self):
        """Test that default shocks includes union from all models."""
        # Create two models with different shocks
        mod1 = set_model(label="model1")  # Has Z_til
        mod1.solve_steady(calibrate=True)
        mod1.linearize()
        mod1.compute_linear_irfs(20)

        mod2 = Model(label="model2")
        mod2.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
                "G_bar": 0.2,
            }
        )
        mod2.steady_guess.update(
            {
                "I": 0.5,
                "log_K": np.log(6.0),
            }
        )
        mod2.rules["intermediate"] += [
            ("K_new", "I + (1.0 - delta) * K"),
            ("Z", "Z_bar + Z_til"),
            ("G", "G_bar + G_til"),
            ("fk", "alp * Z * (K ** (alp - 1.0))"),
            ("y", "Z * (K ** alp)"),
            ("c", "y - I - G"),
            ("uc", "c ** (-gam)"),
            ("K", "np.exp(log_K)"),
        ]
        mod2.rules["expectations"] += [
            ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
        ]
        mod2.rules["transition"] += [
            ("log_K", "np.log(K_new)"),
        ]
        mod2.rules["optimality"] += [
            ("I", "E_Om_K - 1.0"),
        ]
        mod2.rules["calibration"] += [
            ("bet", "K - 6.0"),
        ]
        mod2.add_exog("Z_til", pers=0.95, vol=0.1)
        mod2.add_exog("G_til", pers=0.9, vol=0.05)
        mod2.finalize()
        mod2.solve_steady(calibrate=True)
        mod2.linearize()
        mod2.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without shocks - should plot all shocks from both models
            paths = plot_model_irfs(
                [mod1, mod2],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                model_names=["Model1", "Model2"],
            )

            # Should have created plots for both Z_til and G_til
            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_backward_compatibility_single_shock(self):
        """Test that shock parameter (singular) still works for backward compatibility."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use old API with shock (singular)
            paths = plot_model_irfs(
                [mod],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_both_shock_and_shocks_raises(self):
        """Test that providing both shock and shocks raises ValueError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Cannot specify both"):
                plot_model_irfs(
                    [mod],
                    shock="Z_til",
                    shocks=["Z_til"],
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_unknown_shock_in_all_models_raises(self):
        """Test that a shock not in any model raises ValueError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="not found in any model"):
                plot_model_irfs(
                    [mod],
                    shock="nonexistent_shock",
                    include_list=["I"],
                    plot_dir=tmpdir,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
