#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for plotting functionality for DeterministicResult and SequenceResult.
"""

import os
import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from equilibrium import Model
from equilibrium.plot import PlotSpec, plot_deterministic_results, plot_model_irfs
from equilibrium.settings import get_settings
from equilibrium.solvers import deterministic
from equilibrium.solvers.det_spec import DetSpec
from equilibrium.solvers.results import (
    DeterministicResult,
    SequenceResult,
    SeriesTransform,
)

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


def test_plot_linear_irfs(tmp_path):
    """Ensure plot_linear_irfs writes IRF figures to the configured plot directory."""

    old_dir = os.environ.get("EQUILIBRIUM_PATHS__DATA_DIR")
    try:
        os.environ["EQUILIBRIUM_PATHS__DATA_DIR"] = str(tmp_path)
        get_settings.cache_clear()

        mod = set_model(label="plot_irf_test")
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(Nt_irf=6)

        shock_scales = {shock: 0.5 for shock in mod.exog_list}
        subset = mod.all_vars[: min(4, len(mod.all_vars))]
        results = mod.plot_linear_irfs(shock_sizes=shock_scales, include_list=subset)

        assert set(results) == set(mod.exog_list)
        expected_dir = Path(tmp_path) / "plots" / "irfs" / mod.label
        for paths in results.values():
            assert paths
            for path in paths:
                assert path.exists()
                assert path.parent == expected_dir

        results_truncated = mod.plot_linear_irfs(
            shock_sizes=shock_scales, include_list=subset, n_periods=4
        )
        assert set(results_truncated) == set(mod.exog_list)

        with pytest.raises(ValueError):
            mod.plot_linear_irfs(
                shock_sizes=shock_scales, include_list=subset, n_periods=10
            )
    finally:
        if old_dir is None:
            os.environ.pop("EQUILIBRIUM_PATHS__DATA_DIR", None)
        else:
            os.environ["EQUILIBRIUM_PATHS__DATA_DIR"] = old_dir
        get_settings.cache_clear()


def test_plot_spec_smoke(tmp_path):
    """Ensure PlotSpec integrates with deterministic and IRF plotting."""
    ux = np.column_stack(
        [
            np.log([1.0, 1.05, 1.1, 1.15, 1.2]),
            [0.0, 0.1, 0.2, 0.3, 0.4],
        ]
    )
    det_result = DeterministicResult(
        UX=ux,
        Z=np.zeros((5, 1)),
        var_names=["log_A", "B"],
        exog_names=["Z_til"],
        model_label="plot_spec_model",
    )

    plot_spec = PlotSpec(
        var_titles={"log_A": "A", "B": "B"},
        group_colors={"Baseline": "#1f77b4"},
        series_transforms={"log_A": SeriesTransform(log_to_level=True, diff=True)},
        plot_kwargs={"legend_loc": "upper right"},
    )

    det_paths = plot_deterministic_results(
        [det_result],
        include_list=["log_A", "B"],
        plot_dir=tmp_path,
        result_names=["Baseline"],
        plot_spec=plot_spec,
    )
    assert det_paths
    for path in det_paths:
        assert path.exists()

    mod = set_model(label="plot_spec_irf")
    mod.solve_steady(calibrate=True)
    mod.linearize()
    mod.compute_linear_irfs(6)

    irf_paths = plot_model_irfs(
        [mod],
        shock="Z_til",
        include_list=["I", "log_K"],
        plot_dir=tmp_path,
        plot_spec=PlotSpec(var_titles={"Z_til": "Technology"}),
    )
    assert irf_paths
    for path in irf_paths:
        assert path.exists()


class TestSequenceResultSplice:
    """Tests for SequenceResult.splice method."""

    def test_splice_single_regime(self):
        """Test splicing a single-regime sequence."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 20
        result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        # Splice with T_max equal to the regime length
        spliced = result.splice(T_max=Nt)

        assert spliced.UX.shape[0] == Nt
        assert spliced.Z.shape[0] == Nt
        assert np.allclose(spliced.UX, result.regimes[0].UX)
        assert np.allclose(spliced.Z, result.regimes[0].Z)
        assert spliced.var_names == result.regimes[0].var_names
        assert spliced.exog_names == result.regimes[0].exog_names
        assert spliced.terminal_condition == result.regimes[0].terminal_condition

    def test_splice_single_regime_truncated(self):
        """Test splicing a single-regime sequence with truncation."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 20
        result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        # Splice with T_max less than regime length
        T_max = 10
        spliced = result.splice(T_max=T_max)

        assert spliced.UX.shape[0] == T_max
        assert spliced.Z.shape[0] == T_max
        assert np.allclose(spliced.UX, result.regimes[0].UX[:T_max])
        assert np.allclose(spliced.Z, result.regimes[0].Z[:T_max])

    def test_transform_series(self):
        """Test applying SeriesTransform specs to a DeterministicResult."""
        ux = np.column_stack(
            [
                np.log(np.array([1.0, 1.1, 1.2])),
                np.array([1.0, 2.0, 3.0]),
            ]
        )
        z = np.array([[0.0], [0.0], [0.0]])

        result = DeterministicResult(
            UX=ux,
            Z=z,
            var_names=["log_Y", "C"],
            exog_names=["Z_til"],
        )

        transforms = {
            "log_Y": SeriesTransform(log_to_level=True, diff=True, to_percent=True),
            "C": {"diff": True, "scale": 2.0},
        }

        transformed = result.transform(series_transforms=transforms)

        assert np.allclose(transformed.UX[:, 0], np.array([0.0, 10.0, 20.0]))
        assert np.allclose(transformed.UX[:, 1], np.array([0.0, 2.0, 4.0]))
        assert np.allclose(transformed.Z[:, 0], np.array([0.0, 0.0, 0.0]))

    def test_splice_two_regimes(self):
        """Test splicing two regimes together."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        # Create two-regime specification with transition at time 5
        spec = DetSpec(n_regimes=2, time_list=[5])
        spec.add_shock(0, "Z_til", 0, 0.1)
        spec.add_shock(1, "Z_til", 0, 0.05)

        Nt = 15
        result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        # Splice together
        T_max = 20
        spliced = result.splice(T_max=T_max)

        # Should have:
        # - Regime 0: periods 0-5 (6 periods)
        # - Regime 1: periods 1-14 (14 periods, skipping period 0 which is duplicate)
        # Total: min(20, 6 + 14) = 20
        expected_len = min(T_max, 6 + 14)
        assert spliced.UX.shape[0] == expected_len

        # First 6 periods should match regime 0
        assert np.allclose(spliced.UX[:6, :], result.regimes[0].UX[:6, :])
        assert np.allclose(spliced.Z[:6, :], result.regimes[0].Z[:6, :])

        # Periods 6 onwards should come from regime 1 (starting at index 1)
        assert np.allclose(
            spliced.UX[6:, :], result.regimes[1].UX[1 : 1 + (expected_len - 6), :]
        )

    def test_splice_preserves_metadata(self):
        """Test that splicing preserves metadata correctly."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=2, time_list=[5])
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 15
        result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        spliced = result.splice(T_max=20)

        # Metadata should match first regime
        assert spliced.model_label == result.model_label
        assert spliced.var_names == result.regimes[0].var_names
        assert spliced.exog_names == result.regimes[0].exog_names
        assert spliced.terminal_condition == result.regimes[0].terminal_condition

    def test_splice_empty_sequence_raises(self):
        """Test that splicing an empty sequence raises ValueError."""
        seq = SequenceResult(regimes=[], time_list=[], model_label="test")

        with pytest.raises(ValueError, match="Cannot splice an empty"):
            seq.splice(T_max=10)

    def test_splice_convergence_status(self):
        """Test that splice correctly aggregates convergence status."""
        # Create synthetic results
        result1 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            converged=True,
            final_residual=1e-10,
        )
        result2 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            converged=False,
            final_residual=1e-5,
        )

        seq = SequenceResult(
            regimes=[result1, result2],
            time_list=[5],
            model_label="test",
        )

        spliced = seq.splice(T_max=15)

        # Should be False since one regime didn't converge
        assert spliced.converged is False
        # Should have max residual
        assert spliced.final_residual == 1e-5

    def test_splice_mismatched_var_names_raises(self):
        """Test that splice raises when var_names differ across regimes."""
        result1 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )
        result2 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_C"],  # Different!
            exog_names=["Z_til"],
        )

        seq = SequenceResult(
            regimes=[result1, result2],
            time_list=[5],
            model_label="test",
        )

        with pytest.raises(ValueError, match="var_names.*differs"):
            seq.splice(T_max=15)

    def test_splice_mismatched_exog_names_raises(self):
        """Test that splice raises when exog_names differ across regimes."""
        result1 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )
        result2 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_different"],  # Different!
        )

        seq = SequenceResult(
            regimes=[result1, result2],
            time_list=[5],
            model_label="test",
        )

        with pytest.raises(ValueError, match="exog_names.*differs"):
            seq.splice(T_max=15)

    def test_splice_mismatched_terminal_condition_raises(self):
        """Test that splice raises when terminal_condition differs."""
        result1 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            terminal_condition="stable",
        )
        result2 = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            terminal_condition="steady",  # Different!
        )

        seq = SequenceResult(
            regimes=[result1, result2],
            time_list=[5],
            model_label="test",
        )

        with pytest.raises(ValueError, match="terminal_condition.*differs"):
            seq.splice(T_max=15)


class TestPlotDeterministicResults:
    """Tests for plot_deterministic_results function."""

    def test_plot_single_deterministic_result(self):
        """Test plotting a single DeterministicResult."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        Nt = 10
        z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]
        result = deterministic.solve(mod, z_trans)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_deterministic_results(
                [result],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                prefix="single_det",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_multiple_deterministic_results(self):
        """Test plotting multiple DeterministicResults."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        Nt = 10
        z_trans1 = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]
        result1 = deterministic.solve(mod, z_trans1)

        # Create second result with slight shock - use numpy array copy
        z_trans2 = np.array(z_trans1, copy=True)
        z_trans2[0, :] = z_trans2[0, :] + 0.1
        result2 = deterministic.solve(mod, z_trans2)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_deterministic_results(
                [result1, result2],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                result_names=["Baseline", "Shocked"],
                prefix="multi_det",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_sequence_result(self):
        """Test plotting a SequenceResult (auto-splice)."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 15
        result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_deterministic_results(
                [result],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                prefix="seq",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_sequence_result_with_T_max(self):
        """Test plotting a SequenceResult with explicit T_max."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=2, time_list=[5])
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 15
        result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_deterministic_results(
                [result],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                T_max=10,
                prefix="seq_tmax",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_mixed_results(self):
        """Test plotting a mix of DeterministicResult and SequenceResult."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        Nt = 10
        z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]
        det_result = deterministic.solve(mod, z_trans)

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.1)
        seq_result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_deterministic_results(
                [det_result, seq_result],
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                result_names=["Deterministic", "Sequence"],
                prefix="mixed",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_empty_results_raises(self):
        """Test that empty results list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="non-empty"):
                plot_deterministic_results([], include_list=["I"], plot_dir=tmpdir)

    def test_plot_empty_include_list_raises(self):
        """Test that empty include_list raises ValueError."""
        result = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="None of the variables"):
                plot_deterministic_results([result], include_list=[], plot_dir=tmpdir)

    def test_plot_nonexistent_variable_raises(self):
        """Test that requesting nonexistent variable raises ValueError."""
        result = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="None of the variables"):
                plot_deterministic_results(
                    [result],
                    include_list=["nonexistent_var"],
                    plot_dir=tmpdir,
                )

    def test_plot_mismatched_result_names_raises(self):
        """Test that mismatched result_names length raises ValueError."""
        result = DeterministicResult(
            UX=np.zeros((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="result_names length"):
                plot_deterministic_results(
                    [result],
                    include_list=["I"],
                    plot_dir=tmpdir,
                    result_names=["Name1", "Name2"],  # Two names but one result
                )

    def test_plot_invalid_result_type_raises(self):
        """Test that invalid result type raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(TypeError, match="unsupported type"):
                plot_deterministic_results(
                    ["not a result"],  # Invalid type
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_plot_with_missing_variable_fills_nan(self):
        """Test that missing variables are filled with NaN."""
        # Create two results with different variables
        result1 = DeterministicResult(
            UX=np.ones((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )
        result2 = DeterministicResult(
            UX=np.ones((10, 1)) * 2,
            Z=np.zeros((10, 1)),
            var_names=["I"],  # Missing log_K
            exog_names=["Z_til"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # This should succeed - plot_deterministic_results handles missing vars
            paths = plot_deterministic_results(
                [result1, result2],
                include_list=["I"],  # Only request I which both have
                plot_dir=tmpdir,
                result_names=["Full", "Partial"],
            )

            assert len(paths) > 0

    def test_plot_default_include_list(self):
        """Test that include_list defaults to all variables when None."""
        result = DeterministicResult(
            UX=np.ones((10, 3)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K", "c"],
            exog_names=["Z_til"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without include_list - should use all variables
            paths = plot_deterministic_results(
                [result],
                plot_dir=tmpdir,
                prefix="default_vars",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_default_plot_dir(self):
        """Test that plot_dir defaults to Settings.paths.plot_dir / 'deterministic'."""
        from equilibrium.settings import get_settings

        result = DeterministicResult(
            UX=np.ones((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        # Call without plot_dir - should use default from Settings
        paths = plot_deterministic_results(
            [result],
            include_list=["I"],
            prefix="default_dir",
        )

        settings = get_settings()
        expected_dir = settings.paths.plot_dir / "deterministic" / "_default"

        assert len(paths) > 0
        for path in paths:
            assert path.exists()
            assert path.parent == expected_dir

        # Clean up the created files
        for path in paths:
            path.unlink()

    def test_plot_all_defaults(self):
        """Test plotting with all default values (no include_list, no plot_dir)."""
        from equilibrium.settings import get_settings

        result = DeterministicResult(
            UX=np.ones((10, 2)),
            Z=np.zeros((10, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        # Call with minimal arguments - both defaults should apply
        paths = plot_deterministic_results(
            [result],
            prefix="all_defaults",
        )

        settings = get_settings()
        expected_dir = settings.paths.plot_dir / "deterministic" / "_default"

        assert len(paths) > 0
        for path in paths:
            assert path.exists()
            assert path.parent == expected_dir

        # Clean up the created files
        for path in paths:
            path.unlink()

    def test_plot_intermediate_variables(self):
        """Test that intermediate variables (Y) can be plotted."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        Nt = 10
        z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]
        result = deterministic.solve(mod, z_trans)

        # Verify result has Y and y_names
        assert result.Y is not None
        assert len(result.y_names) > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            # Plot a mix of UX and Y variables
            include_list = [
                "I",
                "log_K",
                "c",
                "y",
                "K",
            ]  # I, log_K are in UX (var_names); c, y, K are in Y (y_names)
            paths = plot_deterministic_results(
                [result],
                include_list=include_list,
                plot_dir=tmpdir,
                prefix="intermediate",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_default_include_list_with_intermediate(self):
        """Test that include_list defaults to all variables including intermediates."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        Nt = 10
        z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]
        result = deterministic.solve(mod, z_trans)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without include_list - should use all variables including Y
            paths = plot_deterministic_results(
                [result],
                plot_dir=tmpdir,
                prefix="all_with_intermediate",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()


class TestPlotModelIrfs:
    """Tests for plot_model_irfs function."""

    def test_plot_single_model(self):
        """Test plotting IRFs from a single model."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_model_irfs(
                [mod],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                prefix="single_irf",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_multiple_models(self):
        """Test plotting IRFs from multiple models."""
        mod1 = set_model(label="model1")
        mod1.solve_steady(calibrate=True)
        mod1.linearize()
        mod1.compute_linear_irfs(20)

        # Create second model with different parameter
        mod2 = set_model(label="model2")
        mod2.params["alp"] = 0.5
        mod2.solve_steady(calibrate=True)
        mod2.linearize()
        mod2.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_model_irfs(
                [mod1, mod2],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                model_names=["Baseline", "Alternative"],
                prefix="multi_irf",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_with_shock_size(self):
        """Test plotting IRFs with shock size scaling."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_model_irfs(
                [mod],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                shock_size=2.0,
                prefix="scaled_irf",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_with_n_periods(self):
        """Test plotting IRFs with explicit n_periods."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(50)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_model_irfs(
                [mod],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                n_periods=20,
                prefix="truncated_irf",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_default_model_names(self):
        """Test that model_names defaults to model labels."""
        mod1 = set_model(label="test_label_1")
        mod1.solve_steady(calibrate=True)
        mod1.linearize()
        mod1.compute_linear_irfs(20)

        mod2 = set_model(label="test_label_2")
        mod2.solve_steady(calibrate=True)
        mod2.linearize()
        mod2.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without model_names - should use labels
            paths = plot_model_irfs(
                [mod1, mod2],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
                prefix="default_names",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_default_include_list(self):
        """Test that include_list defaults to first model's all_vars."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without include_list
            paths = plot_model_irfs(
                [mod],
                shock="Z_til",
                plot_dir=tmpdir,
                prefix="default_vars",
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()

    def test_plot_default_plot_dir(self):
        """Test that plot_dir defaults to Settings.paths.plot_dir / 'irfs'."""
        from equilibrium.settings import get_settings

        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        # Call without plot_dir - should use default from Settings
        paths = plot_model_irfs(
            [mod],
            shock="Z_til",
            include_list=["I", "log_K"],
            prefix="default_dir",
        )

        settings = get_settings()
        expected_dir = settings.paths.plot_dir / "irfs" / mod.label

        assert len(paths) > 0
        for path in paths:
            assert path.exists()
            assert path.parent == expected_dir

        # Clean up the created files
        for path in paths:
            path.unlink()

    def test_plot_default_title_and_prefix(self):
        """Test that title_str and prefix have correct defaults based on shock."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without title_str and prefix
            paths = plot_model_irfs(
                [mod],
                shock="Z_til",
                include_list=["I", "log_K"],
                plot_dir=tmpdir,
            )

            assert len(paths) > 0
            for path in paths:
                assert path.exists()
                # Check filename starts with default prefix
                assert "irf_to_Z_til" in path.name

    def test_plot_with_model_labels(self, tmp_path):
        """Test plotting IRFs by loading from model labels."""
        from equilibrium.plot import plot_model_irfs

        old_dir = os.environ.get("EQUILIBRIUM_PATHS__DATA_DIR")
        try:
            os.environ["EQUILIBRIUM_PATHS__DATA_DIR"] = str(tmp_path)
            get_settings.cache_clear()

            mod = set_model(label="label_irf")
            mod.solve_steady(calibrate=True)
            mod.linearize()
            mod.compute_linear_irfs(10)

            paths = plot_model_irfs(
                model_labels=["label_irf"],
                shock="Z_til",
                include_list=["I", "log_K"],
                prefix="label_irf_plot",
            )

            expected_dir = Path(tmp_path) / "plots" / "irfs" / "label_irf"
            assert len(paths) > 0
            for path in paths:
                assert path.exists()
                assert path.parent == expected_dir
        finally:
            if old_dir is None:
                os.environ.pop("EQUILIBRIUM_PATHS__DATA_DIR", None)
            else:
                os.environ["EQUILIBRIUM_PATHS__DATA_DIR"] = old_dir
            get_settings.cache_clear()

    def test_plot_empty_models_raises(self):
        """Test that empty models list raises ValueError."""
        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="non-empty"):
                plot_model_irfs(
                    [],
                    shock="Z_til",
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_plot_invalid_model_type_raises(self):
        """Test that invalid model type raises TypeError."""
        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(TypeError, match="unsupported type"):
                plot_model_irfs(
                    ["not a model"],
                    shock="Z_til",
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_plot_unlinearized_model_raises(self):
        """Test that unlinearized model raises RuntimeError."""
        # Create a model that hasn't been finalized, so linear_mod is None.
        # This tests the case where linearize() was never called since the model
        # was not finalized.
        mod = Model(label="test_unlinearized")
        mod.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
            }
        )
        # Don't call finalize, linearize or solve_steady

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="has not been linearized"):
                plot_model_irfs(
                    [mod],
                    shock="Z_til",
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_plot_model_without_irfs_raises(self):
        """Test that model without computed IRFs raises RuntimeError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        # Don't compute IRFs

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="has no computed IRFs"):
                plot_model_irfs(
                    [mod],
                    shock="Z_til",
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_plot_unknown_shock_raises(self):
        """Test that unknown shock raises ValueError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="not found in any model"):
                plot_model_irfs(
                    [mod],
                    shock="nonexistent_shock",
                    include_list=["I"],
                    plot_dir=tmpdir,
                )

    def test_plot_nonexistent_variable_raises(self):
        """Test that requesting nonexistent variable raises ValueError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="None of the variables"):
                plot_model_irfs(
                    [mod],
                    shock="Z_til",
                    include_list=["nonexistent_var"],
                    plot_dir=tmpdir,
                )

    def test_plot_mismatched_model_names_raises(self):
        """Test that mismatched model_names length raises ValueError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="model_names length"):
                plot_model_irfs(
                    [mod],
                    shock="Z_til",
                    include_list=["I"],
                    plot_dir=tmpdir,
                    model_names=["Name1", "Name2"],  # Two names but one model
                )

    def test_plot_n_periods_too_large_raises(self):
        """Test that n_periods exceeding available horizon raises ValueError."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()
        mod.compute_linear_irfs(20)

        from equilibrium.plot import plot_model_irfs

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="exceeds minimum available"):
                plot_model_irfs(
                    [mod],
                    shock="Z_til",
                    include_list=["I"],
                    plot_dir=tmpdir,
                    n_periods=100,  # Much larger than computed horizon
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
