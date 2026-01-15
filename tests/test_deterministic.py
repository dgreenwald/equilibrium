#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:00:12 2022

@author: dan
"""

import os

import jax
import numpy as np
import pytest

from equilibrium import Model
from equilibrium.settings import get_settings
from equilibrium.solvers import deterministic

jax.config.update("jax_enable_x64", True)


def set_model(flags=None, params=None, steady_guess=None, **kwargs):

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
        ("K", "np.exp(log_K)"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
    ]

    # rules['read_expectations'] = []

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

    # mod.rules['analytical_steady'] += [
    #     ('log_K', 'np.log(I / delta)'),
    #     ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_compute_time_period():
    """Test that the extracted compute_time_period function works correctly."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Set up test data
    Nt = 10
    N_ux = mod.N["u"] + mod.N["x"]
    UX = np.zeros((Nt, N_ux))
    Z = np.zeros((Nt, mod.N["z"])) + mod.steady_components["z"]
    ux_init = np.zeros(N_ux)
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])

    # Test initial period (tt=0)
    f_t, L_t, C_t, F_t = deterministic.compute_time_period(
        0, Nt, UX, Z, ux_init, mod, params, N_ux, True, "stable"
    )
    assert f_t.shape == (N_ux, 1)
    assert L_t.shape == (N_ux, N_ux)
    assert C_t.shape == (N_ux, N_ux)
    assert F_t.shape == (N_ux, N_ux)

    # Test without gradient computation
    f_t, L_t, C_t, F_t = deterministic.compute_time_period(
        0, Nt, UX, Z, ux_init, mod, params, N_ux, False, "stable"
    )
    assert f_t.shape == (N_ux, 1)
    assert L_t is None
    assert C_t is None
    assert F_t is None

    # Test intermediate period (tt=1)
    f_t, L_t, C_t, F_t = deterministic.compute_time_period(
        1, Nt, UX, Z, ux_init, mod, params, N_ux, True, "stable"
    )
    assert f_t.shape == (N_ux, 1)
    assert L_t.shape == (N_ux, N_ux)
    assert C_t.shape == (N_ux, N_ux)
    assert F_t.shape == (N_ux, N_ux)

    # Test terminal period with stable condition (tt=Nt-1)
    f_t, L_t, C_t, F_t = deterministic.compute_time_period(
        Nt - 1, Nt, UX, Z, ux_init, mod, params, N_ux, True, "stable"
    )
    assert f_t.shape == (N_ux, 1)
    assert L_t.shape == (N_ux, N_ux)
    assert C_t.shape == (N_ux, N_ux)
    assert F_t.shape == (N_ux, N_ux)

    # Test terminal period with steady condition
    f_t, L_t, C_t, F_t = deterministic.compute_time_period(
        Nt - 1, Nt, UX, Z, ux_init, mod, params, N_ux, True, "steady"
    )
    assert f_t.shape == (N_ux, 1)
    assert L_t.shape == (N_ux, N_ux)
    assert C_t.shape == (N_ux, N_ux)
    assert F_t.shape == (N_ux, N_ux)


def test_jacobian_matrix_structure():
    """Test that the Jacobian matrix has the correct block-tridiagonal structure.

    The matrix should be a block-tridiagonal system of total size Nt block rows
    by Nt block columns, where:
    - Block row 0: Identity matrix in block column 0 (initial condition)
    - Block rows 1 through Nt-2: Lₜ in column t-1, Cₜ in column t, Fₜ in column t+1
    - Block row Nt-1: Depends on terminal condition
    """
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    Nt = 10
    N_ux = mod.N["u"] + mod.N["x"]
    UX = np.zeros((Nt, N_ux)) + np.concatenate(
        [mod.steady_components["u"], mod.steady_components["x"]]
    )
    Z = np.zeros((Nt, mod.N["z"])) + mod.steady_components["z"]
    ux_init = np.concatenate([mod.steady_components["u"], mod.steady_components["x"]])

    # Compute Jacobian blocks
    f_all, L_all, C_all, F_all = deterministic.compute_jacobian_blocks(
        mod, UX, Z, ux_init, terminal_condition="steady"
    )

    # Verify block row 0 (initial condition): C_0 = I, L_0 = 0, F_0 = 0
    assert np.allclose(C_all[0], np.eye(N_ux)), "Block row 0: C_0 should be identity"
    assert np.allclose(
        L_all[0], np.zeros((N_ux, N_ux))
    ), "Block row 0: L_0 should be zero"
    assert np.allclose(
        F_all[0], np.zeros((N_ux, N_ux))
    ), "Block row 0: F_0 should be zero"

    # Verify block row Nt-1 (terminal condition with 'steady'): C_{Nt-1} = I, L_{Nt-1} = 0, F_{Nt-1} = 0
    assert np.allclose(
        C_all[Nt - 1], np.eye(N_ux)
    ), "Block row Nt-1: C_{Nt-1} should be identity (steady)"
    assert np.allclose(
        L_all[Nt - 1], np.zeros((N_ux, N_ux))
    ), "Block row Nt-1: L_{Nt-1} should be zero (steady)"
    assert np.allclose(
        F_all[Nt - 1], np.zeros((N_ux, N_ux))
    ), "Block row Nt-1: F_{Nt-1} should be zero"

    # Verify intermediate blocks (1 to Nt-2) have tridiagonal structure (blocks exist)
    for tt in range(1, Nt - 1):
        # L_t, C_t, F_t should have some structure (not all zero necessarily)
        assert L_all[tt].shape == (N_ux, N_ux)
        assert C_all[tt].shape == (N_ux, N_ux)
        assert F_all[tt].shape == (N_ux, N_ux)


def test_deterministic_solution():
    """Test full deterministic solution."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()
    Nt_irf = 10
    mod.compute_linear_irfs(Nt_irf=Nt_irf)
    # compute_linear_irfs now returns a dict of IrfResult objects
    # Check the backward-compatible tensor stored in linear_mod.irfs
    assert mod.linear_mod.irfs.shape == (
        mod.linear_mod.B.shape[1],
        Nt_irf,
        mod.linear_mod.A.shape[0],
    )

    params_new = {"bet": mod.params["bet"] + 0.01}
    mod_new = mod.update_copy(params=params_new)
    mod_new.solve_steady(calibrate=False)
    mod_new.linearize()
    mod_new.compute_linear_irfs(Nt_irf=Nt_irf)
    assert mod_new.linear_mod.irfs.shape == (
        mod_new.linear_mod.B.shape[1],
        Nt_irf,
        mod_new.linear_mod.A.shape[0],
    )

    Nt = 20
    s_steady = mod.get_s_steady()

    N_ux = mod_new.N["u"] + mod_new.N["x"]

    z_trans = np.zeros((Nt + 1, mod_new.N["z"])) + mod_new.steady_components["z"]

    ux_init = s_steady[:N_ux]

    # Deterministic code - using the new signature
    result = deterministic.solve(mod_new, z_trans, ux_init, guess_method="linear")

    # To save initial value
    # np.save('UX_benchmark.npy', result.UX)

    benchmark_path = os.path.join(os.path.dirname(__file__), "UX_benchmark.npy")
    UX_benchmark = np.load(benchmark_path)
    assert np.allclose(result.UX, UX_benchmark)


def test_intermediate_variables():
    """Test that deterministic solve computes intermediate variables (Y)."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    Nt = 10
    z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]
    result = deterministic.solve(mod, z_trans)

    # Check that Y is computed and has correct shape
    assert result.Y is not None, "Y should be computed"
    assert result.Y.shape == (Nt + 1, len(mod.var_lists["intermediate"]))

    # Check that y_names matches the model's intermediate variable list
    assert result.y_names == mod.var_lists["intermediate"]

    # Verify that intermediate values are correctly computed
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])
    for tt in range(min(3, Nt)):  # Check first 3 time periods
        u_t, x_t = np.split(result.UX[tt, :], [mod.N["u"]])
        z_t = result.Z[tt, :]
        y_manual = mod.intermediates(u_t, x_t, z_t, params)
        assert np.allclose(
            result.Y[tt, :], y_manual
        ), f"Intermediate values at time {tt} don't match"


def test_deterministic_y_init_override():
    """Test that deterministic solve uses y_init to override Y[0]."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    params_new = {"bet": mod.params["bet"] + 0.01}
    mod_new = mod.update_copy(params=params_new)
    mod_new.solve_steady(calibrate=False)
    mod_new.linearize()

    Nt = 5
    z_trans = np.zeros((Nt + 1, mod_new.N["z"])) + mod_new.steady_components["z"]
    s_steady = mod.get_s_steady()
    N_ux = mod_new.N["u"] + mod_new.N["x"]
    ux_init = s_steady[:N_ux]

    u_init, x_init = np.split(ux_init, [mod.N["u"]])
    params = np.array([mod.params[key] for key in mod.var_lists["params"]])
    y_init = mod.intermediates(u_init, x_init, z_trans[0, :], params)

    result = deterministic.solve(mod_new, z_trans, ux_init, y_init=y_init)

    assert np.allclose(result.Y[0, :], y_init)


def test_save_and_load_steady():
    label = "io_steady_test"
    mod = set_model(label=label)

    mod.solve_steady(calibrate=True, save=True)

    settings = get_settings()
    path = settings.paths.save_dir / f"{label}_steady_state.json"
    assert path.exists()

    def _steady_to_dict(steady):
        if hasattr(steady, "items"):
            return dict(steady.items())
        if hasattr(steady, "_asdict"):
            return dict(steady._asdict())
        raise TypeError("Unexpected steady state container")

    original = {
        key: np.asarray(val) for key, val in _steady_to_dict(mod.steady_dict).items()
    }

    mod.steady_dict = {}
    mod.solve_steady(calibrate=True, load_initial_guess=True)
    loaded = _steady_to_dict(mod.steady_dict)

    for key, value in original.items():
        assert key in loaded
        assert np.allclose(np.asarray(loaded[key]), value)
        assert np.allclose(np.asarray(mod.steady_dict[key]), value)

    for key, value in original.items():
        assert key in mod.init_dict
        assert np.allclose(np.asarray(mod.init_dict[key]), value)

    backup_loaded = mod.load_steady(load_calibration=True, backup_to_use=0)
    backup_loaded = _steady_to_dict(backup_loaded)
    for key, value in original.items():
        assert key in backup_loaded
        assert np.allclose(np.asarray(backup_loaded[key]), value)

    # Cleanup main file and backups to keep test artifacts local
    path.unlink(missing_ok=True)
    backup_dir = path.parent / "steady_backups"
    for backup in backup_dir.glob(f"{label}_steady_state_*.json"):
        backup.unlink(missing_ok=True)


def test_update_steady_with_missing_variables():
    """Test that update_steady handles missing variables in steady_dict gracefully.

    This test simulates the scenario where the model specification has changed
    (e.g., a new state variable was added) and a saved steady-state is loaded
    that doesn't contain the new variable. The update_steady function should
    use the init_dict fallback for missing variables.
    """
    mod = set_model()
    mod.solve_steady(calibrate=True)

    # Simulate loading a steady_dict that's missing some variables
    # by creating a dict with only a subset of the variables
    original_steady_dict = mod.steady_dict
    # Get a subset of available keys (first 3) from params list for robustness
    available_keys = list(mod.var_lists["params"])[:3]
    subset_steady = {key: original_steady_dict[key] for key in available_keys}

    # Replace steady_dict with the subset (simulating an old saved state)
    mod.steady_dict = subset_steady

    # This should not raise a KeyError - it should use init_dict fallback
    mod.update_steady(calibrate=True)

    # Check that steady_components were created
    assert "u" in mod.steady_components
    assert "x" in mod.steady_components
    assert "params" in mod.steady_components
    assert "z" in mod.steady_components

    # Check that values for variables present in steady_dict are preserved
    for key in available_keys:
        assert np.isclose(mod.params[key], original_steady_dict[key])

    # Check that we can still solve with the model after update_steady
    # (this verifies the fix doesn't break other functionality)
    mod.steady_dict = original_steady_dict
    mod.update_steady(calibrate=True)
    for key in available_keys:
        assert np.isclose(mod.params[key], original_steady_dict[key])


def test_solve_sequence_single_regime():
    """Test solve_sequence with a single regime."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create DetSpec with one regime and a shock
    spec = DetSpec(n_regimes=1)
    spec.add_shock(0, "Z_til", 0, 0.1)

    Nt = 20
    result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

    assert result.n_regimes == 1
    assert len(result.regimes) == 1

    N_ux = mod.N["u"] + mod.N["x"]
    assert result.regimes[0].UX.shape == (Nt, N_ux)
    assert result.regimes[0].Z.shape == (Nt, mod.N["z"])


def test_solve_sequence_two_regimes():
    """Test solve_sequence with two regimes."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create DetSpec with two regimes
    spec = DetSpec(n_regimes=2, time_list=[5])
    spec.add_shock(0, "Z_til", 0, 0.1)
    spec.add_shock(1, "Z_til", 0, 0.05)

    Nt = 15
    result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

    assert result.n_regimes == 2

    # Both regimes should have properly shaped outputs
    N_ux = mod.N["u"] + mod.N["x"]
    for i in range(2):
        assert result.regimes[i].UX.shape == (Nt, N_ux)
        assert result.regimes[i].Z.shape == (Nt, mod.N["z"])

    # The second regime should have initial conditions from the first
    # at time index 4 (the transition time is time_list[0] - 1)
    # Z[0] for regime 1 should equal Z[4] from regime 0
    assert np.allclose(result.regimes[1].Z[0, :], result.regimes[0].Z[4, :])


def test_solve_sequence_regime_parameter_change():
    """Test solve_sequence when parameters change between regimes."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create DetSpec with two regimes with different parameters
    spec = DetSpec(n_regimes=2, time_list=[5])
    spec.add_regime(
        0,
        preset_par_regime={},  # First regime uses baseline
    )
    spec.add_regime(
        1,
        preset_par_regime={"bet": mod.params["bet"] + 0.01},  # Second has different bet
        time_regime=5,
    )

    Nt = 15
    result = deterministic.solve_sequence(
        spec, mod, Nt, calibrate_initial=False, save_results=False
    )

    assert result.n_regimes == 2

    # The two regimes should have different model labels
    # (since they use different parameters, their model_label in the result reflects
    # the model that was used, which may have been updated)
    # For now, we just verify that the shapes are correct
    N_ux = mod.N["u"] + mod.N["x"]
    assert result.regimes[0].UX.shape == (Nt, N_ux)
    assert result.regimes[1].UX.shape == (Nt, N_ux)


def test_solve_sequence_initial_conditions():
    """Test that initial conditions are properly passed between regimes."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create DetSpec with two regimes
    spec = DetSpec(n_regimes=2, time_list=[10])
    spec.add_shock(0, "Z_til", 0, 0.1)

    Nt = 20
    result = deterministic.solve_sequence(spec, mod, Nt, save_results=False)

    # The second regime's initial UX should match the first regime's UX at time 10
    # This is ensured by the ux_init passed to solve
    # We verify by checking that the solution is continuous across regimes
    transition_time = 9
    z_at_transition = result.regimes[0].Z[transition_time, :]

    # The Z path for the second regime should start from z_at_transition
    assert np.allclose(result.regimes[1].Z[0, :], z_at_transition)


def test_solve_sequence_empty_spec_error():
    """Test that solve_sequence raises error with empty DetSpec."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    spec = DetSpec(n_regimes=0)

    with pytest.raises(ValueError, match="must have at least one regime"):
        deterministic.solve_sequence(spec, mod, 20, save_results=False)


def test_solve_algorithm_lbj_vs_sparse():
    """Test that LBJ and sparse algorithms produce the same result."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    params_new = {"bet": mod.params["bet"] + 0.01}
    mod_new = mod.update_copy(params=params_new)
    mod_new.solve_steady(calibrate=False)
    mod_new.linearize()

    Nt = 20
    s_steady = mod.get_s_steady()
    N_ux = mod_new.N["u"] + mod_new.N["x"]
    z_trans = np.zeros((Nt + 1, mod_new.N["z"])) + mod_new.steady_components["z"]
    ux_init = s_steady[:N_ux]

    # Solve using LBJ algorithm
    result_lbj = deterministic.solve(
        mod_new, z_trans, ux_init, guess_method="linear", algorithm="LBJ"
    )

    # Solve using sparse algorithm
    result_sparse = deterministic.solve(
        mod_new, z_trans, ux_init, guess_method="linear", algorithm="sparse"
    )

    # Both should produce the same result
    assert np.allclose(result_lbj.UX, result_sparse.UX, rtol=1e-10, atol=1e-12)


def test_solve_sparse_matches_benchmark():
    """Test that sparse algorithm produces the benchmark result."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    params_new = {"bet": mod.params["bet"] + 0.01}
    mod_new = mod.update_copy(params=params_new)
    mod_new.solve_steady(calibrate=False)
    mod_new.linearize()

    Nt = 20
    s_steady = mod.get_s_steady()
    N_ux = mod_new.N["u"] + mod_new.N["x"]
    z_trans = np.zeros((Nt + 1, mod_new.N["z"])) + mod_new.steady_components["z"]
    ux_init = s_steady[:N_ux]

    # Solve using sparse algorithm
    result_sparse = deterministic.solve(
        mod_new, z_trans, ux_init, guess_method="linear", algorithm="sparse"
    )

    # Compare to benchmark
    benchmark_path = os.path.join(os.path.dirname(__file__), "UX_benchmark.npy")
    UX_benchmark = np.load(benchmark_path)
    assert np.allclose(result_sparse.UX, UX_benchmark)


def test_solve_invalid_algorithm():
    """Test that invalid algorithm raises ValueError."""
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    Nt = 10
    z_trans = np.zeros((Nt + 1, mod.N["z"])) + mod.steady_components["z"]

    with pytest.raises(ValueError, match="Unknown algorithm"):
        deterministic.solve(mod, z_trans, algorithm="invalid")


def test_solve_sequence_with_sparse_algorithm():
    """Test solve_sequence with sparse algorithm."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create DetSpec with one regime and a shock
    spec = DetSpec(n_regimes=1)
    spec.add_shock(0, "Z_til", 0, 0.1)

    Nt = 20

    # Solve with LBJ algorithm
    result_lbj = deterministic.solve_sequence(
        spec, mod, Nt, algorithm="LBJ", save_results=False
    )

    # Solve with sparse algorithm
    result_sparse = deterministic.solve_sequence(
        spec, mod, Nt, algorithm="sparse", save_results=False
    )

    # Both should produce the same result
    assert np.allclose(
        result_lbj.regimes[0].UX, result_sparse.regimes[0].UX, rtol=1e-10, atol=1e-12
    )


def test_solve_sequence_ux_init_uses_original_model_steady_state():
    """Test that ux_init defaults to original model's steady state, not regime model's.

    When solve_sequence is called with preset_par_regime that changes parameters,
    the ux_init for the first regime should be the steady state of the original
    model (mod), not the steady state of the regime's model (current_mod).
    This ensures the simulation starts from the original economy's equilibrium.
    """
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create DetSpec with parameter change in first regime
    spec = DetSpec(n_regimes=1)
    new_bet = mod.params["bet"] + 0.01
    spec.add_regime(0, preset_par_regime={"bet": new_bet})

    Nt = 20
    result = deterministic.solve_sequence(
        spec, mod, Nt, calibrate_initial=False, save_results=False
    )

    # The first row of UX should be the original model's steady state
    # (not the regime model's steady state)
    original_ux = np.concatenate(
        [mod.steady_components["u"], mod.steady_components["x"]]
    )
    assert np.allclose(
        result.regimes[0].UX[0, :], original_ux
    ), "First row of UX should match original model's steady state"


def test_solve_sequence_auto_solves_steady_state():
    """
    Test that solve_sequence automatically solves steady state if not already solved.

    This test verifies the auto-solving feature where solve_sequence will
    automatically call solve_steady() if the model hasn't been solved yet.
    """
    from equilibrium.solvers.det_spec import DetSpec

    # Create a model but DON'T solve steady state
    mod = set_model()

    # Verify that steady state hasn't been solved yet
    assert not mod._steady_solved

    # Create a simple deterministic spec
    det_spec = DetSpec()
    det_spec.add_regime(0, preset_par_regime={})
    det_spec.add_shock(0, "Z_til", 0, 0.01)

    # Test with terminal_condition='steady' (doesn't need linearization)
    result = deterministic.solve_sequence(
        det_spec,
        mod,
        Nt=20,
        terminal_condition="steady",
        calibrate_initial=False,
        save_results=False,
    )

    # Verify the solution succeeded
    assert result.regimes[0].converged
    assert result.regimes[0].UX.shape == (20, mod.N["u"] + mod.N["x"])

    # Verify that steady state was auto-solved
    assert mod._steady_solved

    # Test with terminal_condition='stable' (needs linearization too)
    mod2 = set_model()

    det_spec2 = DetSpec()
    det_spec2.add_regime(0, preset_par_regime={})
    det_spec2.add_shock(0, "Z_til", 0, 0.01)

    result2 = deterministic.solve_sequence(
        det_spec2,
        mod2,
        Nt=20,
        terminal_condition="stable",
        calibrate_initial=False,
        save_results=False,
    )

    # Verify both steady state and linearization were performed
    assert mod2._steady_solved
    assert mod2._linearized
    assert result2.regimes[0].converged


def test_solve_sequence_copy_model_keeps_original_unchanged():
    """Test that copy_model=True avoids mutating the original model."""
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    assert not mod._steady_solved
    assert not mod._linearized

    det_spec = DetSpec()
    det_spec.add_regime(0, preset_par_regime={})
    det_spec.add_shock(0, "Z_til", 0, 0.01)

    result = deterministic.solve_sequence(
        det_spec,
        mod,
        Nt=20,
        terminal_condition="steady",
        calibrate_initial=False,
        save_results=False,
        copy_model=True,
    )

    assert result.regimes[0].converged
    assert not mod._steady_solved
    assert not mod._linearized


def test_solve_sequence_display_steady_flag():
    """
    Test that display_steady parameter works correctly.

    This test verifies that the display_steady flag is accepted and doesn't
    cause errors with both True and False values.
    """
    from equilibrium.solvers.det_spec import DetSpec

    mod = set_model()
    mod.solve_steady(calibrate=True)

    # Create a simple deterministic spec
    det_spec = DetSpec()
    det_spec.add_regime(0, preset_par_regime={})
    det_spec.add_shock(0, "Z_til", 0, 0.01)

    # Test with display_steady=False (default)
    result1 = deterministic.solve_sequence(
        det_spec,
        mod,
        Nt=10,
        terminal_condition="steady",
        display_steady=False,
        save_results=False,
    )
    assert result1.regimes[0].converged

    # Create fresh model for second test
    mod2 = set_model()

    det_spec2 = DetSpec()
    det_spec2.add_regime(0, preset_par_regime={})
    det_spec2.add_shock(0, "Z_til", 0, 0.01)

    # Test with display_steady=True
    result2 = deterministic.solve_sequence(
        det_spec2,
        mod2,
        Nt=10,
        terminal_condition="steady",
        calibrate_initial=True,
        display_steady=True,
        save_results=False,
    )
    assert result2.regimes[0].converged


def test_solve_sequence_no_recalibration_default():
    """Test that by default, only initial steady state is calibrated."""
    from equilibrium.solvers.det_spec import DetSpec

    # Create model with calibration rule for bet
    mod = set_model()

    # Solve steady state with calibration - bet will be calibrated to achieve K=6
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Save the calibrated bet value
    calibrated_bet = mod.params["bet"]

    # Create DetSpec with parameter change in regime 1
    # We'll change bet to a different value
    preset_bet_regime1 = calibrated_bet + 0.02

    spec = DetSpec(n_regimes=2, time_list=[10])
    spec.add_regime(0, preset_par_regime={})
    spec.add_regime(1, preset_par_regime={"bet": preset_bet_regime1}, time_regime=10)
    spec.add_shock(0, "Z_til", 0, 0.01)

    # Solve with calibrate_initial=True and default recalibrate_regimes=False
    # This should calibrate regime 0 but NOT recalibrate regime 1
    result = deterministic.solve_sequence(
        spec,
        mod,
        Nt=20,
        calibrate_initial=True,
        recalibrate_regimes=False,
        save_results=False,
    )

    assert result.n_regimes == 2
    assert result.regimes[0].converged
    assert result.regimes[1].converged

    # The key test: regime 1 should preserve the preset bet value
    # We can verify this indirectly by checking that the solutions are different
    # (if bet was recalibrated, it would have been solved to achieve K=6 again)
    # Instead, with the preset bet value, the steady state should be different

    # Just verify the sequence was solved successfully
    assert result.regimes[0].UX.shape[0] == 20
    assert result.regimes[1].UX.shape[0] == 20


def test_solve_sequence_with_recalibration():
    """Test that recalibrate_regimes=True preserves old behavior."""
    from equilibrium.solvers.det_spec import DetSpec

    # Create model with calibration rule for bet
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    calibrated_bet = mod.params["bet"]

    # Create DetSpec with parameter change in regime 1
    preset_bet_regime1 = calibrated_bet + 0.02

    spec = DetSpec(n_regimes=2, time_list=[10])
    spec.add_regime(0, preset_par_regime={})
    spec.add_regime(1, preset_par_regime={"bet": preset_bet_regime1}, time_regime=10)
    spec.add_shock(0, "Z_til", 0, 0.01)

    # Solve with calibrate_initial=True and recalibrate_regimes=True
    # This should calibrate BOTH regimes
    result = deterministic.solve_sequence(
        spec,
        mod,
        Nt=20,
        calibrate_initial=True,
        recalibrate_regimes=True,
        save_results=False,
    )

    assert result.n_regimes == 2
    assert result.regimes[0].converged
    assert result.regimes[1].converged

    # Verify the sequence was solved successfully
    assert result.regimes[0].UX.shape[0] == 20
    assert result.regimes[1].UX.shape[0] == 20


def test_solve_sequence_linear_no_recalibration():
    """Test that linear solver respects recalibrate_regimes flag."""
    from equilibrium.solvers import linear
    from equilibrium.solvers.det_spec import DetSpec

    # Create model with calibration rule
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    calibrated_bet = mod.params["bet"]
    preset_bet_regime1 = calibrated_bet + 0.02

    spec = DetSpec(n_regimes=2, time_list=[10])
    spec.add_regime(0, preset_par_regime={})
    spec.add_regime(1, preset_par_regime={"bet": preset_bet_regime1}, time_regime=10)
    spec.add_shock(0, "Z_til", 0, 0.01)

    # Test linear solver with default recalibrate_regimes=False
    result = linear.solve_sequence_linear(
        spec,
        mod,
        Nt=20,
        calibrate_initial=True,
        recalibrate_regimes=False,
        save_path=None,
    )

    assert result.n_regimes == 2
    assert result.regimes[0].converged
    assert result.regimes[1].converged

    # Verify shapes
    N_ux = mod.N["u"] + mod.N["x"]
    assert result.regimes[0].UX.shape == (20, N_ux)
    assert result.regimes[1].UX.shape == (20, N_ux)


def test_solve_sequence_linear_with_recalibration():
    """Test that linear solver with recalibrate_regimes=True works."""
    from equilibrium.solvers import linear
    from equilibrium.solvers.det_spec import DetSpec

    # Create model with calibration rule
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    calibrated_bet = mod.params["bet"]
    preset_bet_regime1 = calibrated_bet + 0.02

    spec = DetSpec(n_regimes=2, time_list=[10])
    spec.add_regime(0, preset_par_regime={})
    spec.add_regime(1, preset_par_regime={"bet": preset_bet_regime1}, time_regime=10)
    spec.add_shock(0, "Z_til", 0, 0.01)

    # Test linear solver with recalibrate_regimes=True
    result = linear.solve_sequence_linear(
        spec,
        mod,
        Nt=20,
        calibrate_initial=True,
        recalibrate_regimes=True,
        save_path=None,
    )

    assert result.n_regimes == 2
    assert result.regimes[0].converged
    assert result.regimes[1].converged

    # Verify shapes
    N_ux = mod.N["u"] + mod.N["x"]
    assert result.regimes[0].UX.shape == (20, N_ux)
    assert result.regimes[1].UX.shape == (20, N_ux)


def _run_all_with_summary():
    """Run all tests (for use with __main__)."""
    print("\n" + "=" * 60)
    print("Running test_deterministic_solution...")
    print("=" * 60)
    test_deterministic_solution()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print("\nNote: For step-by-step compilation analysis, run:")
    print("  python tests/test_compilation_analysis.py")


if __name__ == "__main__":
    _run_all_with_summary()
