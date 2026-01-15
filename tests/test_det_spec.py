#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for DetSpec class with tuple-based shocks format.
"""

import numpy as np
import pytest

from equilibrium import Model
from equilibrium.solvers.det_spec import DetSpec


class TestDetSpecBasic:
    """Test basic DetSpec creation and manipulation."""

    def test_empty_initialization(self):
        """Test creating an empty DetSpec."""
        spec = DetSpec()
        assert spec.n_regimes == 0
        assert len(spec.preset_par_list) == 0  # no regimes
        assert len(spec.shocks) == 0
        assert len(spec.time_list) == 0
        assert spec.preset_par_init == {}
        spec.validate()

    def test_initialization_with_n_regimes(self):
        """Test creating DetSpec with n_regimes."""
        spec = DetSpec(n_regimes=3)
        assert spec.n_regimes == 3
        assert len(spec.preset_par_list) == 3  # 3 regimes
        assert len(spec.shocks) == 3
        assert len(spec.time_list) == 2  # n_regimes - 1
        spec.validate()

    def test_initialization_with_shocks(self):
        """Test creating DetSpec with shocks format (tuples)."""
        shocks = [
            [("Z", 0.9, 0.1)],  # regime 0
            [("Z", 0.8, 0.2), ("Y", 0.7, 0.15)],  # regime 1
        ]
        spec = DetSpec(shocks=shocks)
        assert spec.n_regimes == 2
        assert len(spec.shocks) == 2
        assert spec.shocks[0] == [("Z", 0.9, 0.1)]
        assert spec.shocks[1] == [("Z", 0.8, 0.2), ("Y", 0.7, 0.15)]
        spec.validate()


class TestDetSpecAddShock:
    """Test add_shock method."""

    def test_add_shock_to_regime(self):
        """Test adding a single shock to a regime."""
        spec = DetSpec(n_regimes=2)
        spec.add_shock(0, "Z", 0.9, 0.1)
        assert len(spec.shocks[0]) == 1
        assert spec.shocks[0][0] == ("Z", 0.9, 0.1)
        spec.validate()

    def test_add_multiple_shocks_sequentially(self):
        """Test adding multiple shocks to the same regime."""
        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z", 0.9, 0.1)
        spec.add_shock(0, "Y", 0.8, 0.2)
        assert len(spec.shocks[0]) == 2
        assert spec.shocks[0][0] == ("Z", 0.9, 0.1)
        assert spec.shocks[0][1] == ("Y", 0.8, 0.2)
        spec.validate()

    def test_add_shock_creates_regime(self):
        """Test that add_shock creates regime if it doesn't exist."""
        spec = DetSpec()
        spec.add_shock(2, "Z", 0.9, 0.1)
        assert spec.n_regimes == 3
        assert len(spec.shocks) == 3
        assert spec.shocks[2][0] == ("Z", 0.9, 0.1)
        spec.validate()


class TestDetSpecAddShocks:
    """Test add_shocks method."""

    def test_add_shocks_tuple_format(self):
        """Test adding multiple shocks using tuple format."""
        spec = DetSpec(n_regimes=1)
        shocks = [("Z", 0.9, 0.1), ("Y", 0.8, 0.2)]
        spec.add_shocks(0, shocks=shocks)
        assert len(spec.shocks[0]) == 2
        assert spec.shocks[0][0] == ("Z", 0.9, 0.1)
        assert spec.shocks[0][1] == ("Y", 0.8, 0.2)
        spec.validate()


class TestDetSpecAddRegime:
    """Test add_regime method."""

    def test_add_regime_with_shocks(self):
        """Test adding a regime with shocks."""
        spec = DetSpec()
        spec.add_regime(0, shocks_regime=[("Z", 0.9, 0.1)])
        assert spec.n_regimes == 1
        assert spec.shocks[0] == [("Z", 0.9, 0.1)]
        spec.validate()

    def test_add_regime_with_parameters(self):
        """Test adding a regime with parameters."""
        spec = DetSpec()
        spec.add_regime(0, preset_par_regime={"alpha": 0.5})
        assert spec.preset_par_list[0] == {"alpha": 0.5}
        spec.validate()

    def test_add_regime_with_time(self):
        """Test adding a regime with transition time."""
        spec = DetSpec()
        spec.add_regime(0)
        spec.add_regime(1, time_regime=10)
        assert spec.time_list[0] == 10
        spec.validate()

    def test_add_regime_replaces_shocks(self):
        """Test that add_regime replaces existing shocks."""
        spec = DetSpec(shocks=[[("Z", 0.9, 0.1)]])
        spec.add_regime(0, shocks_regime=[("Y", 0.8, 0.2)])
        assert spec.shocks[0] == [("Y", 0.8, 0.2)]
        spec.validate()


class TestDetSpecValidation:
    """Test validation method."""

    def test_validate_correct_spec(self):
        """Test validation passes for correct spec."""
        spec = DetSpec(
            n_regimes=2,
            shocks=[[("Z", 0.9, 0.1)], [("Y", 0.8, 0.2)]],
            time_list=[5],
        )
        spec.validate()  # Should not raise

    def test_validate_shock_tuple_length(self):
        """Test validation catches incorrect tuple length."""
        spec = DetSpec(n_regimes=1)
        spec.shocks[0].append(("Z", 0.9))  # Missing third element
        with pytest.raises(ValueError, match="must be a tuple of length 3"):
            spec.validate()

    def test_validate_shock_var_type(self):
        """Test validation catches incorrect variable type."""
        spec = DetSpec(n_regimes=1)
        spec.shocks[0].append((123, 0.9, 0.1))  # var should be string
        with pytest.raises(ValueError, match="must be a string"):
            spec.validate()

    def test_validate_shock_per_type(self):
        """Test validation catches incorrect persistence type."""
        spec = DetSpec(n_regimes=1)
        spec.shocks[0].append(("Z", "invalid", 0.1))  # per should be numeric
        with pytest.raises(ValueError, match="must be numeric"):
            spec.validate()

    def test_validate_shock_val_type(self):
        """Test validation catches incorrect value type."""
        spec = DetSpec(n_regimes=1)
        spec.shocks[0].append(("Z", 0.9, "invalid"))  # val should be numeric
        with pytest.raises(ValueError, match="must be numeric"):
            spec.validate()


class TestDetSpecUpdateNRegimes:
    """Test update_n_regimes method."""

    def test_update_n_regimes_increases(self):
        """Test increasing number of regimes."""
        spec = DetSpec(n_regimes=1)
        spec.update_n_regimes(3)
        assert spec.n_regimes == 3
        assert len(spec.preset_par_list) == 3
        assert len(spec.shocks) == 3
        assert len(spec.time_list) == 2
        spec.validate()

    def test_update_n_regimes_preserves_data(self):
        """Test that increasing regimes preserves existing data."""
        spec = DetSpec(n_regimes=1, shocks=[[("Z", 0.9, 0.1)]])
        spec.update_n_regimes(2)
        assert spec.shocks[0] == [("Z", 0.9, 0.1)]
        assert spec.shocks[1] == []  # New regime has empty shocks
        spec.validate()


class TestDetSpecComplexScenarios:
    """Test complex usage scenarios."""

    def test_multi_regime_scenario(self):
        """Test a complex multi-regime scenario."""
        spec = DetSpec(preset_par_start={"beta": 0.95})

        # Regime 0: baseline with one shock
        spec.add_regime(0, shocks_regime=[("Z", 0.9, 0.1)])

        # Regime 1: policy change with different shock
        spec.add_regime(
            1,
            preset_par_regime={"beta": 0.96},
            shocks_regime=[("Z", 0.85, 0.15)],
            time_regime=10,
        )

        # Regime 2: return to baseline
        spec.add_regime(
            2,
            preset_par_regime={"beta": 0.95},
            shocks_regime=[("Z", 0.9, 0.05)],
            time_regime=20,
        )

        assert spec.n_regimes == 3
        assert spec.time_list == [10, 20]
        assert spec.preset_par_list[1]["beta"] == 0.96
        assert spec.shocks[1] == [("Z", 0.85, 0.15)]
        spec.validate()

    def test_incremental_shock_addition(self):
        """Test incrementally adding shocks to a regime."""
        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z", 0.9, 0.1)
        spec.add_shocks(0, shocks=[("Y", 0.8, 0.2), ("X", 0.7, 0.15)])

        assert len(spec.shocks[0]) == 3
        assert spec.shocks[0][0] == ("Z", 0.9, 0.1)
        assert spec.shocks[0][1] == ("Y", 0.8, 0.2)
        assert spec.shocks[0][2] == ("X", 0.7, 0.15)
        spec.validate()


class TestDetSpecBuildExogPaths:
    """Test build_exog_paths method."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        mod = Model()

        # Set up basic parameters
        mod.params.update(
            {
                "alp": 0.6,
                "bet": 0.95,
                "delta": 0.1,
                "gam": 2.0,
                "Z_bar": 0.5,
            }
        )

        # Set up steady state guess
        mod.steady_guess.update(
            {
                "I": 0.5,
                "log_K": np.log(6.0),
            }
        )

        # Define model rules
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

        # Add exogenous variable
        mod.add_exog("Z_til", pers=0.95, vol=0.1)

        mod.finalize()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        return mod

    def test_build_exog_paths_default_arguments(self, simple_model):
        """Test build_exog_paths with default arguments."""
        spec = DetSpec(n_regimes=1)
        mod = simple_model
        Nt = 10

        Z_path = spec.build_exog_paths(mod, Nt)

        # Check shapes
        n_exog = len(mod.exog_list)

        assert Z_path.shape == (Nt, n_exog)

        # Check that z_init defaults to zeros
        assert np.allclose(Z_path[0, :], 0.0)

    def test_build_exog_paths_with_custom_z_init(self, simple_model):
        """Test build_exog_paths with custom z_init."""
        spec = DetSpec(n_regimes=1)
        mod = simple_model
        Nt = 10

        n_exog = len(mod.exog_list)
        custom_z = np.ones(n_exog) * 0.3

        Z_path = spec.build_exog_paths(mod, Nt, z_init=custom_z)

        # Check that Z_path[0] matches custom z_init
        assert np.allclose(Z_path[0, :], custom_z)

    def test_build_exog_paths_with_shocks(self, simple_model):
        """Test build_exog_paths with shocks applied."""
        # Create spec with shocks
        spec = DetSpec(n_regimes=1)
        # Shock at period 0 -> innovations array index 1 (due to per+1 offset)
        spec.add_shock(0, "Z_til", 0, 0.1)
        # Shock at period 2 -> innovations array index 3 (due to per+1 offset)
        spec.add_shock(0, "Z_til", 2, 0.05)

        mod = simple_model
        Nt = 10

        Z_path = spec.build_exog_paths(mod, Nt)

        # Check that Z_path[0] is zero (initial state)
        assert np.allclose(Z_path[0, :], 0.0)

        # Manually compute expected Z_path
        # Z[1] should be Phi @ Z[0] + impact @ [0.1]
        # Z[3] should incorporate the second shock
        z = np.zeros(len(mod.exog_list))
        expected_Z = np.zeros((Nt, len(mod.exog_list)))
        expected_Z[0, :] = z

        innovations = np.zeros((Nt, len(mod.exog_list)))
        innovations[1, 0] = 0.1  # First shock at index 1
        innovations[3, 0] = 0.05  # Second shock at index 3

        for tt in range(1, Nt):
            z = (
                mod.linear_mod.Phi @ z
                + mod.linear_mod.impact_matrix @ innovations[tt, :]
            )
            expected_Z[tt, :] = z

        assert np.allclose(Z_path, expected_Z)

    def test_build_exog_paths_regime_selection(self, simple_model):
        """Test build_exog_paths with different regimes."""
        # Create spec with multiple regimes
        spec = DetSpec(n_regimes=2)
        spec.add_shock(0, "Z_til", 0, 0.1)
        spec.add_shock(1, "Z_til", 0, 0.2)

        mod = simple_model
        Nt = 10

        # Build paths for regime 0
        Z_path_0 = spec.build_exog_paths(mod, Nt, regime=0)

        # Build paths for regime 1
        Z_path_1 = spec.build_exog_paths(mod, Nt, regime=1)

        # The paths should be different due to different shocks
        assert not np.allclose(Z_path_0, Z_path_1)

    def test_build_exog_paths_z_evolution(self, simple_model):
        """Test that Z_path evolves correctly according to linear dynamics."""
        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 1, 0.1)  # Single shock at period 1

        mod = simple_model
        Nt = 10

        z_init_custom = np.array([0.05])
        Z_path = spec.build_exog_paths(mod, Nt, z_init=z_init_custom)

        # Verify initial condition
        assert np.allclose(Z_path[0, :], z_init_custom)

        # Verify evolution for period 1 (no shock yet)
        # Note: shock at "per=1" is applied at innovations[2, :]
        # So Z[1] = Phi @ z_init + impact @ 0
        expected_z1 = mod.linear_mod.Phi @ z_init_custom
        assert np.allclose(Z_path[1, :], expected_z1)

        # Verify evolution for period 2 (shock applied)
        expected_z2 = mod.linear_mod.Phi @ Z_path[
            1, :
        ] + mod.linear_mod.impact_matrix @ np.array([0.1])
        assert np.allclose(Z_path[2, :], expected_z2)

    def test_build_exog_paths_no_shocks(self, simple_model):
        """Test build_exog_paths with no shocks (zero innovations)."""
        spec = DetSpec(n_regimes=1)
        mod = simple_model
        Nt = 10

        z_init = np.array([0.2])
        Z_path = spec.build_exog_paths(mod, Nt, z_init=z_init)

        # With no shocks, Z should decay according to Phi
        z = z_init.copy()
        for tt in range(1, Nt):
            z = mod.linear_mod.Phi @ z
            assert np.allclose(Z_path[tt, :], z)

    def test_build_exog_paths_negative_regime(self, simple_model):
        """Test that negative regime raises error."""
        spec = DetSpec(n_regimes=1)
        mod = simple_model
        Nt = 10

        with pytest.raises(ValueError, match="regime must be non-negative"):
            spec.build_exog_paths(mod, Nt, regime=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
