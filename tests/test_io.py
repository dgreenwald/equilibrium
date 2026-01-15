#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the I/O module functions.
"""

import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from equilibrium import Model, read_steady_value, read_steady_values
from equilibrium.io import load_results, resolve_output_path, save_results

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


class TestResolveOutputPath:
    """Tests for resolve_output_path function."""

    def test_explicit_path(self):
        """Test that explicit path is used directly."""
        explicit = Path("/tmp/my/explicit/path.npz")
        result = resolve_output_path(explicit)
        assert result == explicit

    def test_explicit_path_string(self):
        """Test that explicit string path is converted to Path."""
        explicit = "/tmp/my/explicit/path.npz"
        result = resolve_output_path(explicit)
        assert result == Path(explicit)

    def test_default_path_construction(self):
        """Test default path construction with model_label and result_type."""
        result = resolve_output_path(
            result_type="irfs",
            model_label="test_model",
            suffix=".npz",
        )
        assert result.name == "test_model.npz"
        assert "irfs" in str(result)

    def test_timestamp_in_filename(self):
        """Test that timestamp is appended when requested."""
        result = resolve_output_path(
            result_type="irfs",
            model_label="test_model",
            timestamp=True,
            suffix=".npz",
        )
        # Filename should contain model_label and a timestamp pattern
        assert result.stem.startswith("test_model_")
        assert len(result.stem) > len("test_model_")  # Has timestamp appended

    def test_suffix_normalization(self):
        """Test that suffix is properly normalized."""
        result1 = resolve_output_path(model_label="test", suffix=".npz")
        result2 = resolve_output_path(model_label="test", suffix="npz")
        assert result1.suffix == result2.suffix == ".npz"


class TestSaveLoadResults:
    """Tests for save_results and load_results functions."""

    def test_save_load_npz_roundtrip(self):
        """Test that npz format round-trips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            data = {
                "arr1": np.array([1.0, 2.0, 3.0]),
                "arr2": np.array([[1, 2], [3, 4]]),
                "scalar": np.array(3.14),
            }
            metadata = {"model_label": "test", "version": 1}

            save_results(data, path, format="npz", metadata=metadata)
            loaded = load_results(path)

            assert np.allclose(loaded["arr1"], data["arr1"])
            assert np.allclose(loaded["arr2"], data["arr2"])
            assert np.isclose(loaded["scalar"], data["scalar"])
            assert loaded["__metadata__"]["model_label"] == "test"

    def test_save_load_json_roundtrip(self):
        """Test that json format round-trips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {
                "arr1": np.array([1.0, 2.0, 3.0]),
                "arr2": np.array([[1, 2], [3, 4]]),
            }
            metadata = {"model_label": "test"}

            save_results(data, path, format="json", metadata=metadata)
            loaded = load_results(path)

            assert np.allclose(loaded["arr1"], data["arr1"])
            assert np.allclose(loaded["arr2"], data["arr2"])
            assert loaded["__metadata__"]["model_label"] == "test"

    def test_overwrite_protection(self):
        """Test that overwrite=False prevents file replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            data = {"arr": np.array([1, 2, 3])}

            save_results(data, path, format="npz")

            with pytest.raises(FileExistsError):
                save_results(data, path, format="npz", overwrite=False)

    def test_overwrite_allowed(self):
        """Test that overwrite=True allows file replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            data1 = {"arr": np.array([1, 2, 3])}
            data2 = {"arr": np.array([4, 5, 6])}

            save_results(data1, path, format="npz")
            save_results(data2, path, format="npz", overwrite=True)

            loaded = load_results(path)
            assert np.allclose(loaded["arr"], data2["arr"])

    def test_parent_directory_creation(self):
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "test.npz"
            data = {"arr": np.array([1, 2, 3])}

            save_results(data, path, format="npz")
            assert path.exists()

    def test_unsupported_save_format(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"
            data = {"arr": np.array([1, 2, 3])}

            with pytest.raises(ValueError, match="Unsupported format"):
                save_results(data, path, format="xyz")

    def test_unsupported_load_format(self):
        """Test that unsupported file format raises ValueError on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"
            path.write_text("dummy")

            with pytest.raises(ValueError, match="Unsupported file format"):
                load_results(path)

    def test_file_not_found(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path/file.npz")


class TestLinearModelSaveIrfs:
    """Tests for LinearModel.save_irfs method."""

    def test_save_irfs_raises_if_not_computed(self):
        """Test that save_irfs raises RuntimeError if IRFs haven't been computed."""
        mod = set_model(label="test_irfs")
        mod.solve_steady(calibrate=True)
        mod.linearize()

        # IRFs have not been computed yet
        with pytest.raises(RuntimeError, match="No IRFs have been computed"):
            mod.linear_mod.save_irfs()

    def test_save_irfs_basic(self):
        """Test basic save_irfs functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model(label="test_irfs")
            mod.solve_steady(calibrate=True)
            mod.linearize()
            mod.compute_linear_irfs(20)

            path = Path(tmpdir) / "irfs.npz"
            result_path = mod.linear_mod.save_irfs(path, overwrite=True)

            assert result_path.exists()
            loaded = load_results(result_path)
            assert "irfs" in loaded
            assert loaded["irfs"].shape == mod.linear_mod.irfs.shape

    def test_save_irfs_with_matrices(self):
        """Test save_irfs with include_matrices=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model(label="test_irfs")
            mod.solve_steady(calibrate=True)
            mod.linearize()
            mod.compute_linear_irfs(20)

            path = Path(tmpdir) / "irfs_with_matrices.npz"
            mod.linear_mod.save_irfs(path, include_matrices=True, overwrite=True)

            loaded = load_results(path)
            assert "irfs" in loaded
            assert "A" in loaded
            assert "B" in loaded
            assert "G_x" in loaded
            assert "H_x" in loaded

    def test_save_irfs_json_format(self):
        """Test save_irfs with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model(label="test_irfs")
            mod.solve_steady(calibrate=True)
            mod.linearize()
            mod.compute_linear_irfs(10)

            path = Path(tmpdir) / "irfs.json"
            mod.linear_mod.save_irfs(path, format="json", overwrite=True)

            loaded = load_results(path)
            assert "irfs" in loaded
            assert "__metadata__" in loaded

    def test_save_irfs_metadata(self):
        """Test that save_irfs includes proper metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model(label="test_model_label")
            mod.solve_steady(calibrate=True)
            mod.linearize()
            mod.compute_linear_irfs(10)

            path = Path(tmpdir) / "irfs.npz"
            mod.linear_mod.save_irfs(path, overwrite=True)

            loaded = load_results(path)
            metadata = loaded["__metadata__"]
            assert metadata["model_label"] == "test_model_label"
            assert "shock_names" in metadata
            assert "var_names" in metadata


class TestModelSaveLinearIrfs:
    """Tests for Model.save_linear_irfs convenience method."""

    def test_save_linear_irfs_delegates_to_linear_mod(self):
        """Test that save_linear_irfs delegates to linear_mod.save_irfs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model(label="delegate_test")
            mod.solve_steady(calibrate=True)
            mod.linearize()
            mod.compute_linear_irfs(15)

            path = Path(tmpdir) / "irfs.npz"
            result = mod.save_linear_irfs(path, overwrite=True)

            assert result.exists()
            loaded = load_results(result)
            assert "irfs" in loaded

    def test_save_linear_irfs_raises_without_linearization(self):
        """Test that save_linear_irfs raises if model not linearized."""
        mod = set_model(label="not_linearized")
        mod.solve_steady(calibrate=True)
        # Don't linearize

        # The model is linearized automatically during finalize, but IRFs are not computed
        mod.linear_mod.irfs = None  # Clear any computed IRFs

        with pytest.raises(RuntimeError, match="No IRFs have been computed"):
            mod.save_linear_irfs()


class TestSolveSequenceLinearSave:
    """Tests for solve_sequence_linear with save functionality."""

    def test_solve_sequence_linear_with_save(self):
        """Test that solve_sequence_linear can save results."""
        from equilibrium.solvers import linear
        from equilibrium.solvers.det_spec import DetSpec

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            spec = DetSpec(n_regimes=1)
            spec.add_shock(0, "Z_til", 0, 0.1)

            save_path = Path(tmpdir) / "linear_results.npz"

            result = linear.solve_sequence_linear(
                spec, mod, Nt=20, save_path=save_path, save_format="npz"
            )

            assert result.n_regimes == 1
            assert save_path.exists()

            loaded = load_results(save_path)
            assert "UX_regime_0" in loaded
            assert "Z_regime_0" in loaded
            assert np.allclose(loaded["UX_regime_0"], result.regimes[0].UX)


class TestDeterministicResult:
    """Tests for DeterministicResult save/load functionality."""

    def test_save_load_roundtrip_npz(self):
        """Test that DeterministicResult round-trips correctly with npz format."""
        from equilibrium.solvers import deterministic
        from equilibrium.solvers.results import DeterministicResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            Nt = 10
            z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]

            result = deterministic.solve(mod, z_trans)

            # Save
            save_path = Path(tmpdir) / "result.npz"
            result.save(save_path, overwrite=True)

            # Load
            loaded = DeterministicResult.load(save_path)

            # Verify
            assert np.allclose(loaded.UX, result.UX)
            assert np.allclose(loaded.Z, result.Z)
            assert loaded.model_label == result.model_label
            assert loaded.terminal_condition == result.terminal_condition
            assert loaded.converged == result.converged
            assert np.isclose(loaded.final_residual, result.final_residual)

    def test_save_load_roundtrip_json(self):
        """Test that DeterministicResult round-trips correctly with json format."""
        from equilibrium.solvers import deterministic
        from equilibrium.solvers.results import DeterministicResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            Nt = 5
            z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]

            result = deterministic.solve(mod, z_trans)

            # Save
            save_path = Path(tmpdir) / "result.json"
            result.save(save_path, format="json", overwrite=True)

            # Load
            loaded = DeterministicResult.load(save_path)

            # Verify
            assert np.allclose(loaded.UX, result.UX)
            assert np.allclose(loaded.Z, result.Z)

    def test_solve_with_save_path(self):
        """Test that solve() with save_path saves correctly."""
        from equilibrium.solvers import deterministic
        from equilibrium.solvers.results import DeterministicResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            Nt = 5
            z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]

            save_path = Path(tmpdir) / "saved_result.npz"
            result = deterministic.solve(mod, z_trans, save_path=save_path)

            assert save_path.exists()

            # Load and verify
            loaded = DeterministicResult.load(save_path)
            assert np.allclose(loaded.UX, result.UX)


class TestSequenceResult:
    """Tests for SequenceResult save/load functionality."""

    def test_save_load_roundtrip_npz(self):
        """Test that SequenceResult round-trips correctly with npz format."""
        from equilibrium.solvers import deterministic
        from equilibrium.solvers.det_spec import DetSpec
        from equilibrium.solvers.results import SequenceResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            spec = DetSpec(n_regimes=2, time_list=[5])
            spec.add_shock(0, "Z_til", 0, 0.1)
            spec.add_shock(1, "Z_til", 0, 0.05)

            result = deterministic.solve_sequence(spec, mod, Nt=10, save_results=False)

            # Save
            save_path = Path(tmpdir) / "sequence.npz"
            result.save(save_path, overwrite=True)

            # Load
            loaded = SequenceResult.load(save_path)

            # Verify
            assert loaded.n_regimes == result.n_regimes
            assert loaded.time_list == result.time_list
            assert loaded.model_label == result.model_label

            for i in range(result.n_regimes):
                assert np.allclose(loaded.regimes[i].UX, result.regimes[i].UX)
                assert np.allclose(loaded.regimes[i].Z, result.regimes[i].Z)

    def test_save_load_roundtrip_json(self):
        """Test that SequenceResult round-trips correctly with json format."""
        from equilibrium.solvers import linear
        from equilibrium.solvers.det_spec import DetSpec
        from equilibrium.solvers.results import SequenceResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            spec = DetSpec(n_regimes=1)
            spec.add_shock(0, "Z_til", 0, 0.1)

            result = linear.solve_sequence_linear(spec, mod, Nt=5)

            # Save
            save_path = Path(tmpdir) / "sequence.json"
            result.save(save_path, format="json", overwrite=True)

            # Load
            loaded = SequenceResult.load(save_path)

            # Verify
            assert loaded.n_regimes == result.n_regimes
            assert np.allclose(loaded.regimes[0].UX, result.regimes[0].UX)

    def test_solve_sequence_with_save_path(self):
        """Test that solve_sequence() with save_path saves correctly."""
        from equilibrium.solvers import deterministic
        from equilibrium.solvers.det_spec import DetSpec
        from equilibrium.solvers.results import SequenceResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            spec = DetSpec(n_regimes=1)
            spec.add_shock(0, "Z_til", 0, 0.1)

            save_path = Path(tmpdir) / "sequence_result.npz"
            result = deterministic.solve_sequence(spec, mod, Nt=10, save_path=save_path)

            assert save_path.exists()

            # Load and verify
            loaded = SequenceResult.load(save_path)
            assert np.allclose(loaded.regimes[0].UX, result.regimes[0].UX)

    def test_fluent_api_save(self):
        """Test fluent API pattern: solve(...).save()"""
        from equilibrium.solvers import deterministic

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model()
            mod.solve_steady(calibrate=True)
            mod.linearize()

            Nt = 5
            z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]

            save_path = Path(tmpdir) / "fluent.npz"
            path = deterministic.solve(mod, z_trans).save(save_path, overwrite=True)

            assert path.exists()
            assert path == save_path


class TestReadSteadyValue:
    """Tests for read_steady_value function."""

    def test_read_existing_variable(self):
        """Test reading a variable from an existing steady state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and solve a model with save=True
            mod = set_model(label="test_read")
            mod.solve_steady(calibrate=True, save=True)

            # Override save_dir to use tmpdir by moving the file
            from equilibrium.settings import get_settings

            settings = get_settings()
            source = settings.paths.save_dir / "test_read_steady_state.json"
            dest = Path(tmpdir) / "test_read_steady_state.json"
            import shutil

            shutil.copy(source, dest)

            # Read a variable
            value = read_steady_value("test_read", "K", save_dir=tmpdir)

            # Verify it matches the model's steady state
            assert isinstance(value, float)
            assert np.isclose(value, mod.steady_dict["K"], rtol=1e-6)

    def test_read_with_default_file_not_found(self):
        """Test that default value is returned when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            value = read_steady_value(
                "nonexistent_model",
                "K",
                default=42.0,
                save_dir=tmpdir,
            )
            assert value == 42.0

    def test_read_with_default_variable_not_found(self):
        """Test that default value is returned when variable doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and solve a model
            mod = set_model(label="test_default_var")
            mod.solve_steady(calibrate=True, save=True)

            # Copy to tmpdir
            from equilibrium.settings import get_settings

            settings = get_settings()
            source = settings.paths.save_dir / "test_default_var_steady_state.json"
            dest = Path(tmpdir) / "test_default_var_steady_state.json"
            import shutil

            shutil.copy(source, dest)

            # Try to read a nonexistent variable with default
            value = read_steady_value(
                "test_default_var",
                "nonexistent_variable",
                default=99.9,
                save_dir=tmpdir,
            )
            assert value == 99.9

    def test_read_error_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist and no default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Steady state file not found"):
                read_steady_value("nonexistent_model", "K", save_dir=tmpdir)

    def test_read_error_variable_not_found(self):
        """Test that KeyError is raised when variable doesn't exist and no default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and solve a model
            mod = set_model(label="test_error_var")
            mod.solve_steady(calibrate=True, save=True)

            # Copy to tmpdir
            from equilibrium.settings import get_settings

            settings = get_settings()
            source = settings.paths.save_dir / "test_error_var_steady_state.json"
            dest = Path(tmpdir) / "test_error_var_steady_state.json"
            import shutil

            shutil.copy(source, dest)

            # Try to read a nonexistent variable without default
            with pytest.raises(KeyError, match="Variable 'xyz' not found"):
                read_steady_value("test_error_var", "xyz", save_dir=tmpdir)

    def test_read_uses_default_save_dir(self):
        """Test that default save_dir from settings is used when not specified."""
        # Create and solve a model (uses default save_dir)
        mod = set_model(label="test_default_dir")
        mod.solve_steady(calibrate=True, save=True)

        # Read without specifying save_dir
        value = read_steady_value("test_default_dir", "K")

        # Verify it matches
        assert isinstance(value, float)
        assert np.isclose(value, mod.steady_dict["K"], rtol=1e-6)

    def test_read_returns_float(self):
        """Test that the function always returns a float."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and solve a model
            mod = set_model(label="test_float_return")
            mod.solve_steady(calibrate=True, save=True)

            # Copy to tmpdir
            from equilibrium.settings import get_settings

            settings = get_settings()
            source = settings.paths.save_dir / "test_float_return_steady_state.json"
            dest = Path(tmpdir) / "test_float_return_steady_state.json"
            import shutil

            shutil.copy(source, dest)

            # Read a variable
            value = read_steady_value("test_float_return", "K", save_dir=tmpdir)

            # Verify it's a float (not np.float64 or other numeric type)
            assert isinstance(value, float)

    def test_cross_model_parameter_sharing(self):
        """Integration test: use one model's output as another model's parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and solve baseline model
            baseline = set_model(label="baseline")
            baseline.solve_steady(calibrate=True, save=True)

            # Copy to tmpdir
            from equilibrium.settings import get_settings

            settings = get_settings()
            source = settings.paths.save_dir / "baseline_steady_state.json"
            dest = Path(tmpdir) / "baseline_steady_state.json"
            import shutil

            shutil.copy(source, dest)

            # Read baseline's capital stock
            baseline_K = read_steady_value("baseline", "K", save_dir=tmpdir)

            # Create variant model using baseline's K as a target parameter
            # (In practice, you'd use this in calibration equations)
            variant = set_model(label="variant")
            variant.params.overwrite_item("K_target", baseline_K)

            # Verify the parameter was set correctly
            assert np.isclose(variant.params["K_target"], baseline.steady_dict["K"])

    def test_read_all_values(self):
        """Test reading all steady state values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = set_model(label="test_read_all")
            mod.solve_steady(calibrate=True, save=True)

            from equilibrium.settings import get_settings

            settings = get_settings()
            source = settings.paths.save_dir / "test_read_all_steady_state.json"
            dest = Path(tmpdir) / "test_read_all_steady_state.json"
            import shutil

            shutil.copy(source, dest)

            values = read_steady_values("test_read_all", save_dir=tmpdir)

            assert isinstance(values, dict)
            assert np.isclose(values["K"], mod.steady_dict["K"], rtol=1e-6)

    def test_read_all_with_default_file_not_found(self):
        """Test that default dict is returned when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default = {"K": 1.0}
            values = read_steady_values(
                "nonexistent_model",
                default=default,
                save_dir=tmpdir,
            )
            assert values == default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
