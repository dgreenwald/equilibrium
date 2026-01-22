#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Model.to_dynare method, particularly the initval block generation.

This module tests that the to_dynare method correctly generates:
1. Exogenous variables set to zero
2. Endogenous states and policy controls from steady_dict (if solved) or analytical_steady/steady_guess
3. Intermediate variables using their rules
4. Expectations variables with _NEXT suffixes removed for steady state
"""

import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from equilibrium import Model
from equilibrium.solvers.det_spec import DetSpec

jax.config.update("jax_enable_x64", True)


def create_simple_rbc_model():
    """
    Create a simple RBC model for testing Dynare export.

    Returns
    -------
    Model
        A basic RBC model with capital accumulation.
    """
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

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def create_model_with_analytical_steady():
    """
    Create a model with analytical_steady rules.

    Returns
    -------
    Model
        A model with analytical steady state formula.
    """
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

    # Analytical steady state for log_K
    mod.rules["analytical_steady"] += [
        ("log_K", "np.log(I / delta)"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_to_dynare_creates_file():
    """Test that to_dynare creates a .mod file."""
    mod = create_simple_rbc_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_model.mod"
        content = mod.to_dynare(output_path=output_path)

        assert output_path.exists()
        assert content == output_path.read_text()


def test_to_dynare_has_initval_block_unsolved():
    """Test that initval block is generated for unsolved model."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that initval block is present
    assert "initval;" in content
    assert "end;" in content

    # Check that exogenous variable is set to zero
    assert "Z_til = 0;" in content

    # Check that endogenous variables are initialized
    # Should use steady_guess values
    assert "I =" in content
    assert "log_K =" in content


def test_to_dynare_has_initval_block_solved():
    """Test that initval block uses steady_dict when model is solved."""
    mod = create_simple_rbc_model()
    mod.solve_steady(calibrate=False, display=False)

    content = mod.to_dynare()

    # Check that initval block is present
    assert "initval;" in content

    # Check that exogenous variable is set to zero
    assert "Z_til = 0;" in content

    # Check that endogenous variables use steady state values
    I_steady = float(mod.steady_dict["I"])
    log_K_steady = float(mod.steady_dict["log_K"])

    # These should appear in the initval block with high precision
    assert f"I = {I_steady:.16f};" in content
    assert f"log_K = {log_K_steady:.16f};" in content


def test_to_dynare_initval_has_intermediate_vars():
    """Test that initval block includes intermediate variables."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that intermediate variables are in initval block
    assert "K = exp(log_K);" in content
    assert "Z = Z_bar + Z_til;" in content
    assert "y = Z * (K ^ alp);" in content
    assert "c = y - I;" in content


def test_to_dynare_initval_has_expectations_flattened():
    """Test that initval block has expectations with _NEXT removed."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # The expectation rule has _NEXT suffixes
    # In initval block, these should be removed for steady state
    assert "E_Om_K = bet * (uc / uc) * (fk + (1.0 - delta));" in content


def test_to_dynare_initval_with_analytical_steady():
    """Test that initval block uses analytical_steady rules when present."""
    mod = create_model_with_analytical_steady()

    content = mod.to_dynare()

    # Check that analytical_steady formula is used in initval
    # Should see log_K = log(I / delta)
    assert "log_K = log(I / delta);" in content


def test_to_dynare_initval_with_solved_analytical_steady():
    """Test that initval block uses steady_dict even when analytical_steady exists."""
    mod = create_model_with_analytical_steady()
    mod.solve_steady(calibrate=False, display=False)

    content = mod.to_dynare()

    # When solved, should use numerical value from steady_dict
    log_K_steady = float(mod.steady_dict["log_K"])
    assert f"log_K = {log_K_steady:.16f};" in content

    # Should not use the analytical formula when steady state is solved
    assert "log_K = log(I / delta);" not in content


def test_to_dynare_has_shocks_block():
    """Test that shocks block is generated with stderr = 1.0."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that shocks block is present
    assert "shocks;" in content

    # Check that the shock for Z_til is defined with stderr = 1.0
    assert "var e_Z_til; stderr 1.0;" in content


def test_to_dynare_structure():
    """Test the overall structure of the Dynare .mod file."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()

    # Check that blocks appear in the expected order
    param_idx = content.find("parameters")
    var_idx = content.find("var ")
    varexo_idx = content.find("varexo")
    model_idx = content.find("model;")
    initval_idx = content.find("initval;")
    shocks_idx = content.find("shocks;")

    # All should be present (>= 0, not necessarily > 0)
    assert param_idx >= 0
    assert var_idx >= 0
    assert varexo_idx >= 0
    assert model_idx >= 0
    assert initval_idx >= 0
    assert shocks_idx >= 0

    # And in the correct order
    assert param_idx < var_idx < varexo_idx < model_idx < initval_idx < shocks_idx


def test_to_dynare_steady_command():
    """Test that steady=True adds the steady command."""
    mod = create_simple_rbc_model()

    # Without steady parameter (default False)
    content_no_steady = mod.to_dynare()
    assert "steady;" not in content_no_steady

    # With steady=True
    content_with_steady = mod.to_dynare(steady=True)
    assert "steady;" in content_with_steady

    # Verify it appears after shocks block
    shocks_end = content_with_steady.find("end;", content_with_steady.find("shocks;"))
    steady_idx = content_with_steady.find("steady;")
    assert steady_idx > shocks_end, "steady; should appear after shocks block"


def test_to_dynare_stoch_simul_default_off():
    """Test that stoch_simul is not included by default."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare()
    assert "stoch_simul(" not in content


def test_to_dynare_stoch_simul_all_vars():
    """Test that compute_irfs=True adds stoch_simul without variable list."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare(compute_irfs=True)
    assert "stoch_simul(order=1);" in content


def test_to_dynare_stoch_simul_subset_vars():
    """Test that compute_irfs=True with irf_var_list includes variables."""
    mod = create_simple_rbc_model()

    content = mod.to_dynare(compute_irfs=True, irf_var_list=["y", "c"])
    assert "stoch_simul(order=1) y c;" in content


def test_to_dynare_with_det_spec_basic():
    """Test DetSpec with a simple 2-regime parameter change."""
    mod = create_simple_rbc_model()

    # Create DetSpec with parameter change in delta
    # Regime 0: delta=0.15, starts at t=1
    # Regime 1: delta=0.1 (baseline), starts at t=11
    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15})
    spec.add_regime(1, preset_par_regime={"delta": 0.1}, time_regime=11)

    content = mod.to_dynare(det_spec=spec, det_spec_periods=100)

    # Check that TV_delta infrastructure is present
    assert "TV_delta" in content
    assert "e_TV_delta" in content
    assert "mu_TV_delta" in content
    assert "rho_TV_delta" in content
    assert "sig_TV_delta" in content

    # Check that TV_delta AR(1) equation is present
    assert (
        "TV_delta = (1.0 - rho_TV_delta) * mu_TV_delta + rho_TV_delta * TV_delta(-1) + sig_TV_delta * e_TV_delta;"
        in content
    )

    # Check that delta is replaced with (delta + TV_delta) in equations
    # The intermediate equation has: K_new = I + (1.0 - delta) * K
    # Should become: K_new = I + (1.0 - (delta + TV_delta)) * K
    assert "(delta + TV_delta)" in content

    # Check that parameter assignment is NOT modified
    assert "delta = 0.1" in content  # Original parameter value

    # Check that endval blocks are present
    assert "endval(learnt_in=1);" in content  # Regime 0 starts at t=1
    assert "endval(learnt_in=11);" in content  # Regime 1 starts at t=11

    # Check that regime 0 sets delta change
    # Regime 0: delta = 0.15, baseline = 0.1, diff = 0.05
    assert "e_TV_delta = 0.0500000000000000;" in content

    # Check that regime 1 resets delta to baseline
    # Regime 1: delta = 0.1, baseline = 0.1, diff = 0.0
    assert "e_TV_delta = 0.0000000000000000;" in content

    # Check that perfect foresight commands are present
    assert "perfect_foresight_with_expectation_errors_setup(periods=100);" in content
    assert "perfect_foresight_with_expectation_errors_solver;" in content


def test_to_dynare_with_det_spec_shocks():
    """Test DetSpec with exogenous shocks only (no parameter changes)."""
    mod = create_simple_rbc_model()

    # Create DetSpec with only shocks (no parameter changes)
    spec = DetSpec()
    spec.add_regime(0)
    spec.add_shock(0, "Z_til", 5, 0.01)  # Shock at period 5 relative to regime start

    content = mod.to_dynare(det_spec=spec)

    # Should NOT have TV_ infrastructure (no changing parameters)
    assert "TV_" not in content
    assert "e_TV_" not in content

    # Should have shock block for Z_til
    assert "shocks(learnt_in=1);" in content
    assert "var e_Z_til;" in content
    # Shock at period 5 relative to regime start (learnt_in = 1)
    # Absolute period = 5 + 1 - 1 = 5
    assert "periods 5;" in content
    assert "values 0.0100000000000000;" in content


def test_to_dynare_with_det_spec_mixed():
    """Test DetSpec with both parameter changes and shocks."""
    mod = create_simple_rbc_model()

    # Create DetSpec with both parameter changes and shocks
    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15})
    spec.add_shock(0, "Z_til", 5, 0.01)  # Shock at period 5 relative to regime start

    content = mod.to_dynare(det_spec=spec)

    # Should have TV_delta infrastructure
    assert "TV_delta" in content
    assert "(delta + TV_delta)" in content

    # Should have endval block
    assert "endval(learnt_in=1);" in content

    # Should have shock block
    assert "shocks(learnt_in=1);" in content
    assert "var e_Z_til;" in content
    assert "periods 5;" in content


def test_to_dynare_with_det_spec_multiple_regimes():
    """Test DetSpec with 3+ regimes and cumulative timing."""
    mod = create_simple_rbc_model()

    # Create DetSpec with 3 regimes
    # Regime 0: delta=0.15, starts at t=1
    # Regime 1: delta=0.2, starts at t=11
    # Regime 2: delta=0.1 (baseline), starts at t=31
    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15})
    spec.add_regime(1, preset_par_regime={"delta": 0.2}, time_regime=11)
    spec.add_regime(2, preset_par_regime={"delta": 0.1}, time_regime=31)

    content = mod.to_dynare(det_spec=spec)

    # Check timing in endval blocks
    assert "endval(learnt_in=1);" in content  # Regime 0 starts at t=1
    assert "endval(learnt_in=11);" in content  # Regime 1 starts at t=11
    assert "endval(learnt_in=31);" in content  # Regime 2 starts at t=31

    # Check that each regime has correct delta deviation
    # Regime 0: 0.15 - 0.1 = 0.05
    # Regime 1: 0.2 - 0.1 = 0.1
    # Regime 2: 0.1 - 0.1 = 0.0


def test_to_dynare_det_spec_with_steady_irfs():
    """Test that steady, compute_irfs, and det_spec can coexist in correct order."""
    mod = create_simple_rbc_model()

    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15}, time_regime=10)

    content = mod.to_dynare(steady=True, compute_irfs=True, det_spec=spec)

    # All three should be present
    assert "steady;" in content
    assert "stoch_simul(order=1);" in content
    assert "endval(learnt_in=1);" in content

    # Check ordering: steady, then stoch_simul, then perfect foresight
    steady_idx = content.find("steady;")
    stoch_simul_idx = content.find("stoch_simul(order=1);")
    endval_idx = content.find("endval(learnt_in=1);")
    pf_setup_idx = content.find("perfect_foresight_with_expectation_errors_setup")

    assert steady_idx < stoch_simul_idx < endval_idx < pf_setup_idx


def test_tv_parameter_equation_replacement():
    """Test that parameters are correctly replaced with (param + TV_param)."""
    mod = create_simple_rbc_model()

    spec = DetSpec(preset_par_init={"delta": 0.1, "alp": 0.6})
    spec.add_regime(0, preset_par_regime={"delta": 0.15, "alp": 0.65}, time_regime=10)

    content = mod.to_dynare(det_spec=spec)

    # Both delta and alp should be replaced
    assert "(delta + TV_delta)" in content
    assert "(alp + TV_alp)" in content

    # Parameter assignments should NOT be replaced
    assert "delta = 0.1" in content
    assert "alp = 0.6" in content

    # Verify specific equations are correctly transformed
    # Original: K_new = I + (1.0 - delta) * K
    # Should become: K_new = I + (1.0 - (delta + TV_delta)) * K
    assert "K_new = I + (1.0 - (delta + TV_delta)) * K;" in content

    # Original: fk = alp * Z * (K ** (alp - 1.0))
    # Should become: fk = (alp + TV_alp) * Z * (K ^ ((alp + TV_alp) - 1.0))
    assert "(alp + TV_alp)" in content


def test_tv_parameter_infrastructure_creation():
    """Test that TV_ infrastructure is correctly created."""
    mod = create_simple_rbc_model()

    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15}, time_regime=10)

    content = mod.to_dynare(det_spec=spec)

    # Check var block includes TV_delta
    var_block_start = content.find("var ")
    var_block_end = content.find(";", var_block_start)
    var_block = content[var_block_start:var_block_end]
    assert "TV_delta" in var_block

    # Check varexo block includes e_TV_delta
    varexo_block_start = content.find("varexo ")
    varexo_block_end = content.find(";", varexo_block_start)
    varexo_block = content[varexo_block_start:varexo_block_end]
    assert "e_TV_delta" in varexo_block

    # Check parameters block includes mu_TV_delta, rho_TV_delta, sig_TV_delta
    assert "mu_TV_delta = 0.0000000000000000;" in content
    assert "rho_TV_delta = 0.0000000000000000;" in content
    assert "sig_TV_delta = 1.0000000000000000;" in content

    # Check initval block includes TV_delta = 0
    initval_start = content.find("initval;")
    initval_end = content.find("end;", initval_start)
    initval_block = content[initval_start:initval_end]
    assert "TV_delta = 0;" in initval_block


def test_det_spec_existing_tv_variable_error():
    """Test that an error is raised if TV_<param> already exists."""
    mod = Model()

    mod.params.update({"delta": 0.1, "alp": 0.6})
    mod.steady_guess.update({"I": 0.5, "log_K": np.log(6.0)})

    mod.rules["intermediate"] += [
        ("K", "np.exp(log_K)"),
    ]

    mod.rules["transition"] += [
        ("log_K", "np.log(I)"),
    ]

    mod.rules["optimality"] += [
        ("I", "K - 1.0"),
    ]

    # Add a TV_delta exogenous variable
    mod.add_exog("TV_delta", pers=0.9, vol=0.1)

    mod.finalize()

    # Create DetSpec that tries to change delta
    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15}, time_regime=10)

    # Should raise ValueError
    with pytest.raises(ValueError, match="TV_delta already exists"):
        mod.to_dynare(det_spec=spec)


def test_det_spec_solver_kwargs():
    """Test that det_spec_solver_kwargs are correctly formatted."""
    mod = create_simple_rbc_model()

    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15}, time_regime=10)

    content = mod.to_dynare(
        det_spec=spec,
        det_spec_solver_kwargs={"homotopy_initial_step_size": 0.5, "maxit": 50},
    )

    # Check that solver kwargs are formatted correctly
    assert (
        "perfect_foresight_with_expectation_errors_solver(homotopy_initial_step_size=0.5,maxit=50);"
        in content
    )


def test_det_spec_filename_with_label():
    """Test that DetSpec label is included in the output filename."""
    mod = create_simple_rbc_model()

    # Test with custom label
    spec = DetSpec(preset_par_init={"delta": 0.1}, label="boom")
    spec.add_regime(0, preset_par_regime={"delta": 0.15})

    # Without explicit output_path, should use label
    mod.to_dynare(det_spec=spec)

    from equilibrium.settings import get_settings

    settings = get_settings()
    expected_path = settings.paths.debug_dir / f"{mod.label}_boom.mod"
    assert expected_path.exists()

    # Test with IRFs and DetSpec
    spec2 = DetSpec(preset_par_init={"delta": 0.1}, label="recession")
    spec2.add_regime(0, preset_par_regime={"delta": 0.05})

    mod.to_dynare(compute_irfs=True, det_spec=spec2)
    expected_path2 = settings.paths.debug_dir / f"{mod.label}_irfs_recession.mod"
    assert expected_path2.exists()


def test_det_spec_default_label():
    """Test that default DetSpec label (_default) is used when no label specified."""
    mod = create_simple_rbc_model()

    # DetSpec without explicit label (defaults to "_default")
    spec = DetSpec(preset_par_init={"delta": 0.1})
    spec.add_regime(0, preset_par_regime={"delta": 0.15})

    mod.to_dynare(det_spec=spec)

    from equilibrium.settings import get_settings

    settings = get_settings()
    expected_path = settings.paths.debug_dir / f"{mod.label}__default.mod"
    assert expected_path.exists()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
