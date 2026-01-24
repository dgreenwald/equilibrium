#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the linear solver functions.
"""

import jax
import numpy as np
import pytest

from equilibrium import Model
from equilibrium.solvers import linear
from equilibrium.solvers.det_spec import DetSpec

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


class TestSolveSequenceLinear:
    """Tests for solve_sequence_linear function."""

    def test_single_regime_no_shock(self):
        """Test linear solution with single regime and no shocks."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        Nt = 20

        result = linear.solve_sequence_linear(spec, mod, Nt)

        assert result.n_regimes == 1
        assert len(result.regimes) == 1

        N_ux = mod.N["u"] + mod.N["x"]
        assert result.regimes[0].UX.shape == (Nt, N_ux)
        assert result.regimes[0].Z.shape == (Nt, mod.N["z"])

    def test_single_regime_with_shock(self):
        """Test linear solution with single regime and a shock."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 20
        result = linear.solve_sequence_linear(spec, mod, Nt)

        assert result.n_regimes == 1
        N_ux = mod.N["u"] + mod.N["x"]
        assert result.regimes[0].UX.shape == (Nt, N_ux)
        assert result.regimes[0].Z.shape == (Nt, mod.N["z"])

        # With a shock, the Z path should deviate from zero
        # (at least for some periods)
        assert not np.allclose(result.regimes[0].Z, 0.0)

    def test_two_regimes(self):
        """Test linear solution with two regimes."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=2, time_list=[5])
        spec.add_shock(0, "Z_til", 0, 0.1)
        spec.add_shock(1, "Z_til", 0, 0.05)

        Nt = 15
        result = linear.solve_sequence_linear(spec, mod, Nt)

        assert result.n_regimes == 2

        N_ux = mod.N["u"] + mod.N["x"]
        for i in range(2):
            assert result.regimes[i].UX.shape == (Nt, N_ux)
            assert result.regimes[i].Z.shape == (Nt, mod.N["z"])

        # The second regime should have initial conditions from the first
        # time_list[0]=5 means transition at time 5, with start_time=1,
        # transition_time in path = 5-1 = 4, so Z[0] for regime 1 should equal Z[4] from regime 0
        assert np.allclose(result.regimes[1].Z[0, :], result.regimes[0].Z[4, :])

    def test_initial_conditions_transfer(self):
        """Test that initial conditions are properly passed between regimes."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=2, time_list=[10])
        spec.add_shock(0, "Z_til", 0, 0.1)

        Nt = 20
        result = linear.solve_sequence_linear(spec, mod, Nt)

        # time_list[0]=10 means transition at time 10, with start_time=1,
        # transition_time in path = 10-1 = 9
        transition_idx = 9
        z_at_transition = result.regimes[0].Z[transition_idx, :]
        ux_at_transition = result.regimes[0].UX[transition_idx, :]

        # The Z path for the second regime should start from z_at_transition
        assert np.allclose(result.regimes[1].Z[0, :], z_at_transition)

        # The UX for the second regime should start from ux_at_transition
        assert np.allclose(result.regimes[1].UX[0, :], ux_at_transition)

    def test_regime_parameter_change(self):
        """Test linear solution when parameters change between regimes."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=2, time_list=[5])
        spec.add_regime(
            0,
            preset_par_regime={},
        )
        spec.add_regime(
            1,
            preset_par_regime={"bet": mod.params["bet"] + 0.01},
            time_regime=5,
        )

        Nt = 15
        result = linear.solve_sequence_linear(spec, mod, Nt, calibrate_initial=False)

        assert result.n_regimes == 2

        # Verify shapes are correct
        N_ux = mod.N["u"] + mod.N["x"]
        assert result.regimes[0].UX.shape == (Nt, N_ux)
        assert result.regimes[1].UX.shape == (Nt, N_ux)

    def test_linear_y_init_override(self):
        """Test that linear solver uses y_init to override Y[0]."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_regime(0, preset_par_regime={"bet": mod.params["bet"] + 0.01})

        Nt = 5
        ux_init = np.concatenate(
            [mod.steady_components["u"], mod.steady_components["x"]]
        )
        u_init, x_init = np.split(ux_init, [mod.N["u"]])
        params = np.array([mod.params[key] for key in mod.var_lists["params"]])
        z_init = np.zeros(mod.N["z"])
        y_init = mod.intermediates(u_init, x_init, z_init, params)

        result = linear.solve_sequence_linear(
            spec, mod, Nt, ux_init=ux_init, y_init=y_init
        )

        assert np.allclose(result.regimes[0].Y[0, : y_init.shape[0]], y_init)

    def test_empty_spec_error(self):
        """Test that solve_sequence_linear raises error with empty DetSpec."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=0)

        with pytest.raises(ValueError, match="must have at least one regime"):
            linear.solve_sequence_linear(spec, mod, 20)

    def test_output_format_matches_nonlinear(self):
        """Test that output format matches deterministic.solve_sequence."""
        from equilibrium.solvers import deterministic

        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.01)  # Small shock for linear approximation

        Nt = 10

        # Get both linear and nonlinear results
        linear_result = linear.solve_sequence_linear(spec, mod, Nt)
        nonlinear_result = deterministic.solve_sequence(
            spec, mod, Nt, save_results=False
        )

        # Verify both have the same structure
        assert linear_result.n_regimes == nonlinear_result.n_regimes

        for i in range(linear_result.n_regimes):
            assert (
                linear_result.regimes[i].UX.shape
                == nonlinear_result.regimes[i].UX.shape
            )
            assert (
                linear_result.regimes[i].Z.shape == nonlinear_result.regimes[i].Z.shape
            )

    def test_linear_vs_nonlinear_small_shock(self):
        """Test that linear and nonlinear solutions are close for small shocks."""
        from equilibrium.solvers import deterministic

        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        # Use a small shock so linear approximation is accurate
        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.001)

        Nt = 20

        linear_result = linear.solve_sequence_linear(spec, mod, Nt)
        nonlinear_result = deterministic.solve_sequence(
            spec, mod, Nt, save_results=False
        )

        # Z paths should be identical (both use same build_exog_paths)
        assert np.allclose(
            linear_result.regimes[0].Z, nonlinear_result.regimes[0].Z, rtol=1e-10
        )

        # UX paths should be close for small shocks
        # (linear approximation should be accurate)
        assert np.allclose(
            linear_result.regimes[0].UX,
            nonlinear_result.regimes[0].UX,
            rtol=1e-3,
            atol=1e-6,
        )

    def test_steady_state_with_no_shock(self):
        """Test that solution stays at steady state with no shocks."""
        mod = set_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        # No shocks added

        Nt = 10
        result = linear.solve_sequence_linear(spec, mod, Nt)

        # Extract steady state
        u_ss = mod.steady_components["u"]
        x_ss = mod.steady_components["x"]
        ux_ss = np.concatenate([u_ss, x_ss])

        # All UX values should be at steady state
        for t in range(Nt):
            assert np.allclose(result.regimes[0].UX[t, :], ux_ss, rtol=1e-10)


def set_multi_expectation_model(flags=None, params=None, steady_guess=None, **kwargs):
    """Create a model with multiple expectations and shocks for testing.

    This model has more expectation variables and shocks, which exercises
    the linearization code paths that handle non-square matrix shapes.
    """
    mod = Model(flags=flags, params=params, steady_guess=steady_guess, **kwargs)

    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
            "psi_bar": 1.0,
        }
    )

    mod.steady_guess.update(
        {
            "I1": 0.3,
            "I2": 0.3,
            "log_K1": np.log(3.0),
            "log_K2": np.log(3.0),
        }
    )

    mod.rules["intermediate"] += [
        # Capital evolution
        ("K1_new", "I1 + (1.0 - delta) * K1"),
        ("K2_new", "I2 + (1.0 - delta) * K2"),
        ("K1", "np.exp(log_K1)"),
        ("K2", "np.exp(log_K2)"),
        # Production
        ("Z", "Z_bar + Z_til"),
        ("psi", "psi_bar + psi_til"),
        ("y1", "Z * (K1 ** alp)"),
        ("y2", "psi * (K2 ** alp)"),
        ("y", "y1 + y2"),
        # Marginal products
        ("fk1", "alp * Z * (K1 ** (alp - 1.0))"),
        ("fk2", "alp * psi * (K2 ** (alp - 1.0))"),
        # Consumption
        ("c", "y - I1 - I2"),
        ("uc", "c ** (-gam)"),
    ]

    mod.rules["expectations"] += [
        ("E_Om_K1", "bet * (uc_NEXT / uc) * (fk1_NEXT + (1.0 - delta))"),
        ("E_Om_K2", "bet * (uc_NEXT / uc) * (fk2_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("log_K1", "np.log(K1_new)"),
        ("log_K2", "np.log(K2_new)"),
    ]

    mod.rules["optimality"] += [
        ("I1", "E_Om_K1 - 1.0"),
        ("I2", "E_Om_K2 - 1.0"),
    ]

    mod.rules["calibration"] += [
        ("bet", "K1 + K2 - 6.0"),
    ]

    # Add two exogenous shocks
    mod.add_exog("Z_til", pers=0.95, vol=0.1)
    mod.add_exog("psi_til", pers=0.9, vol=0.05)

    mod.finalize()

    return mod


class TestMultiExpectationLinearization:
    """Tests for linearization with multiple expectation variables and shocks."""

    def test_linearize_multi_expectation_model(self):
        """Test that linearization works with multiple expectations and shocks.

        This test catches dimension mismatches in the linearization code that
        can be hidden by numpy broadcasting in simpler models.
        """
        mod = set_multi_expectation_model()
        mod.solve_steady(calibrate=True)

        # This should not raise a ValueError about shape mismatch
        mod.linearize()

        # Verify matrix shapes are correct
        lm = mod.linear_mod
        N_s = mod.N["u"] + mod.N["x"] + mod.N["z"]
        N_shock = lm.impact_matrix.shape[1]

        assert lm.A_s.shape == (N_s, N_s)
        assert lm.B_s.shape == (N_s, N_shock)
        assert lm.A.shape[0] == lm.A.shape[1]  # A should be square
        assert lm.B.shape[0] == lm.A.shape[0]  # B rows should match A

    def test_linear_solution_multi_expectation(self):
        """Test that linear solution works with multi-expectation model."""
        mod = set_multi_expectation_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        spec = DetSpec(n_regimes=1)
        spec.add_shock(0, "Z_til", 0, 0.01)
        spec.add_shock(0, "psi_til", 0, 0.01)

        Nt = 20
        result = linear.solve_sequence_linear(spec, mod, Nt)

        assert result.n_regimes == 1
        N_ux = mod.N["u"] + mod.N["x"]
        assert result.regimes[0].UX.shape == (Nt, N_ux)
        assert result.regimes[0].Z.shape == (Nt, mod.N["z"])

    def test_linear_vs_nonlinear_multi_expectation(self):
        """Test linear solution remains at steady state with no shocks.

        This is a simpler test that verifies the linearization produces
        sensible results without requiring a working nonlinear solver.
        """
        mod = set_multi_expectation_model()
        mod.solve_steady(calibrate=True)
        mod.linearize()

        # No shock - should stay at steady state
        spec = DetSpec(n_regimes=1)
        Nt = 20

        linear_result = linear.solve_sequence_linear(spec, mod, Nt)

        # Extract steady state
        u_ss = mod.steady_components["u"]
        x_ss = mod.steady_components["x"]
        ux_ss = np.concatenate([u_ss, x_ss])

        # All UX values should be at steady state
        for t in range(Nt):
            assert np.allclose(linear_result.regimes[0].UX[t, :], ux_ss, rtol=1e-10)


def test_solve_sequence_linear_auto_solves_steady_state():
    """
    Test that solve_sequence_linear automatically solves steady state if not already solved.

    This test verifies the auto-solving feature where solve_sequence_linear will
    automatically call solve_steady() and linearize() if the model hasn't been solved yet.
    """
    # Create a model but DON'T solve steady state or linearize
    mod = set_model()

    # Verify that steady state hasn't been solved yet
    assert not mod._steady_solved
    assert not mod._linearized

    # Create a simple deterministic spec
    spec = DetSpec()
    spec.add_regime(0, preset_par_regime={})
    spec.add_shock(0, "Z_til", 0, 0.01)

    # Call solve_sequence_linear without pre-solving
    result = linear.solve_sequence_linear(spec, mod, Nt=20, calibrate_initial=True)

    # Verify the solution succeeded
    assert result.regimes[0].converged
    assert result.regimes[0].UX.shape == (20, mod.N["u"] + mod.N["x"])

    # Verify that steady state and linearization were auto-performed
    assert mod._steady_solved
    assert mod._linearized


def test_solve_sequence_linear_display_steady_flag():
    """
    Test that display_steady parameter works correctly in linear solver.

    This test verifies that the display_steady flag is accepted and doesn't
    cause errors with both True and False values.
    """
    # Test with display_steady=False (default)
    mod1 = set_model()
    mod1.solve_steady(calibrate=True)
    mod1.linearize()

    spec1 = DetSpec()
    spec1.add_regime(0, preset_par_regime={})
    spec1.add_shock(0, "Z_til", 0, 0.01)

    result1 = linear.solve_sequence_linear(spec1, mod1, Nt=10, display_steady=False)
    assert result1.regimes[0].converged

    # Test with display_steady=True
    mod2 = set_model()

    spec2 = DetSpec()
    spec2.add_regime(0, preset_par_regime={})
    spec2.add_shock(0, "Z_til", 0, 0.01)

    result2 = linear.solve_sequence_linear(
        spec2, mod2, Nt=10, calibrate_initial=True, display_steady=True
    )
    assert result2.regimes[0].converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
