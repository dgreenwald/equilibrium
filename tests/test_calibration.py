#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for unified calibration interface.

This module tests calibration for:
- Deterministic path matching
- Linear IRF matching
- Linear sequence matching
- Just-identified cases (root finding)
- Over-identified cases (minimization)
- Scalar and vector parameter cases
- PointTarget and FunctionalTarget specifications
"""

import numpy as np
import pytest

from equilibrium import Model
from equilibrium.solvers.calibration import (
    CalibrationResult,
    FunctionalTarget,
    PointTarget,
    calibrate,
)
from equilibrium.solvers.det_spec import DetSpec
from equilibrium.solvers.linear_spec import LinearSpec


def create_simple_model(label="test_calib", beta=0.95):
    """Create a simple RBC model for testing."""
    mod = Model(label=label)

    mod.params.update(
        {
            "alp": 0.6,
            "bet": beta,
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


class TestPointTarget:
    """Tests for PointTarget specification."""

    def test_create_point_target(self):
        """Test creating a PointTarget."""
        target = PointTarget(variable="I", time=10, value=1.05)
        assert target.variable == "I"
        assert target.time == 10
        assert target.value == 1.05
        assert target.shock is None
        assert target.weight == 1.0  # Default weight

    def test_point_target_with_shock(self):
        """Test creating a PointTarget with shock specification."""
        target = PointTarget(variable="y", time=5, value=0.98, shock="Z_til")
        assert target.variable == "y"
        assert target.time == 5
        assert target.value == 0.98
        assert target.shock == "Z_til"
        assert target.weight == 1.0

    def test_point_target_with_weight(self):
        """Test creating a PointTarget with custom weight."""
        target = PointTarget(variable="I", time=10, value=1.05, weight=2.5)
        assert target.variable == "I"
        assert target.weight == 2.5

    def test_point_target_validation(self):
        """Test PointTarget validation."""
        with pytest.raises(ValueError, match="time must be non-negative"):
            PointTarget(variable="I", time=-1, value=1.0)

        with pytest.raises(ValueError, match="weight must be positive"):
            PointTarget(variable="I", time=10, value=1.0, weight=0.0)

        with pytest.raises(ValueError, match="weight must be positive"):
            PointTarget(variable="I", time=10, value=1.0, weight=-1.0)


class TestFunctionalTarget:
    """Tests for FunctionalTarget specification."""

    def test_create_functional_target(self):
        """Test creating a FunctionalTarget."""

        def my_loss(result):
            return 0.0

        target = FunctionalTarget(func=my_loss, description="Test functional target")
        assert target.func is my_loss
        assert target.description == "Test functional target"
        assert target.weights is None  # Default no weights

    def test_functional_target_with_weights(self):
        """Test creating a FunctionalTarget with weights."""

        def multi_loss(result):
            return np.array([0.1, 0.2, 0.3])

        target = FunctionalTarget(func=multi_loss, weights=[1.0, 2.0, 0.5])
        assert target.weights is not None
        assert len(target.weights) == 3
        assert target.weights[1] == 2.0

    def test_functional_target_weight_validation(self):
        """Test FunctionalTarget weight validation."""

        def my_loss(result):
            return 0.0

        with pytest.raises(ValueError, match="All weights must be positive"):
            FunctionalTarget(func=my_loss, weights=[1.0, -0.5, 1.0])

        with pytest.raises(ValueError, match="All weights must be positive"):
            FunctionalTarget(func=my_loss, weights=[0.0])

    def test_functional_target_callable(self):
        """Test that FunctionalTarget can be called on a result."""

        def avg_consumption(result):
            # c is an intermediate variable - check in Y
            c_idx = result.y_names.index("c")
            return np.mean(result.Y[:10, c_idx]) - 0.95

        target = FunctionalTarget(func=avg_consumption)

        # Create mock result
        from equilibrium.solvers.results import DeterministicResult

        result = DeterministicResult(
            UX=np.random.randn(50, 2),
            Z=np.zeros((50, 1)),
            Y=np.random.randn(50, 7),  # Include Y for intermediate variables
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
            y_names=["K", "Z", "K_new", "fk", "y", "c", "uc"],
        )

        # Should be callable
        error = target.func(result)
        assert isinstance(error, (float, np.ndarray))


class TestCalibrationScalarJustIdentified:
    """Tests for scalar just-identified calibration."""

    def test_scalar_root_linear_irf(self):
        """Test scalar parameter calibration using linear IRF."""
        base_model = create_simple_model()

        # Define parameter mapping: calibrate shock size
        def param_to_model(params):
            shock_size = params[0]
            # Return model and LinearSpec
            return base_model, LinearSpec(
                shock_name="Z_til", shock_size=shock_size, Nt=50
            )

        # Target: IRF of investment (I) at time 5 should be close to specific value
        # First, compute reference IRF to get a target using existing machinery
        ref_spec = LinearSpec(shock_name="Z_til", shock_size=0.01, Nt=50)
        ref_irf_dict = base_model.linear_mod.compute_irfs(ref_spec.Nt)
        ref_irf = ref_irf_dict[ref_spec.shock_name]

        # Scale by shock size
        ref_irf_scaled_UX = ref_irf.UX * ref_spec.shock_size

        # Use I which is in UX (control variable)
        I_idx = ref_irf.var_names.index("I")
        target_value = ref_irf_scaled_UX[5, I_idx]

        targets = [PointTarget(variable="I", time=5, value=target_value)]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.005]),
            solver="linear_irf",
            bounds=[(0.0001, 0.1)],
            tol=1e-6,
        )

        # Should find the correct shock size
        assert result.success, f"Calibration failed: {result.message}"
        assert np.isclose(
            result.parameters_array[0], 0.01, rtol=0.01
        ), f"Expected shock size ~0.01, got {result.parameters_array[0]}"
        assert result.method == "root_scalar"

    def test_scalar_root_deterministic(self):
        """Test scalar parameter calibration using deterministic solver."""
        base_model = create_simple_model()

        # Define parameter mapping: calibrate a shock value
        def param_to_model(params):
            shock_val = params[0]
            spec = DetSpec(Nt=50)
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock_val)
            return base_model, spec

        # Target: investment (I) at time 10
        targets = [PointTarget(variable="I", time=10, value=0.55)]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01]),
            solver="deterministic",
            bounds=[(0.0, 0.1)],
            tol=1e-4,
        )

        assert result.success or result.residual < 1e-3, (
            f"Calibration did not converge well: {result.message}, "
            f"residual={result.residual}"
        )
        assert result.method == "root_scalar"


class TestCalibrationVectorJustIdentified:
    """Tests for vector just-identified calibration."""

    def test_vector_root_linear_sequence(self):
        """Test vector parameter calibration using linear sequence solver."""
        base_model = create_simple_model()

        # Define parameter mapping: calibrate two shock values
        def param_to_model(params):
            shock1, shock2 = params
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock1)
            spec.add_regime(1, time_regime=20)
            spec.add_shock(1, "Z_til", shock_per=0, shock_val=shock2)
            return base_model, spec

        # Get reference values
        ref_spec = DetSpec()
        ref_spec.add_regime(0)
        ref_spec.add_shock(0, "Z_til", shock_per=0, shock_val=0.01)
        ref_spec.add_regime(1, time_regime=20)
        ref_spec.add_shock(1, "Z_til", shock_per=0, shock_val=0.02)

        from equilibrium.solvers.linear import solve_sequence_linear

        ref_result = solve_sequence_linear(ref_spec, base_model, 50)
        ref_spliced = ref_result.splice(50)
        I_idx = ref_spliced.var_names.index("I")

        # Target: investment at two different times
        targets = [
            PointTarget(variable="I", time=10, value=ref_spliced.UX[10, I_idx]),
            PointTarget(variable="I", time=30, value=ref_spliced.UX[30, I_idx]),
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.005, 0.015]),
            solver="linear_sequence",
            tol=1e-5,
        )

        assert result.success, f"Calibration failed: {result.message}"
        assert result.method == "root"
        # Check parameters are close to reference
        assert np.allclose(
            result.parameters_array, [0.01, 0.02], rtol=0.1
        ), f"Expected [0.01, 0.02], got {result.parameters_array}"


class TestCalibrationOverIdentified:
    """Tests for over-identified calibration (minimization)."""

    def test_scalar_minimize(self):
        """Test scalar parameter calibration with multiple targets."""
        base_model = create_simple_model()

        # Define parameter mapping
        def param_to_model(params):
            shock_val = params[0]
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock_val)
            return base_model, spec

        # Multiple targets for single parameter (over-identified)
        targets = [
            PointTarget(variable="I", time=5, value=0.545),
            PointTarget(variable="I", time=10, value=0.546),
            PointTarget(variable="I", time=15, value=0.543),
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01]),
            solver="linear_sequence",
            bounds=[(0.0, 0.1)],
            tol=1e-5,
        )

        # Should use minimization
        assert result.method == "minimize_scalar"
        # Should find some reasonable solution
        assert result.parameters_array[0] > 0

    def test_vector_minimize(self):
        """Test vector parameter calibration with more targets than parameters."""
        base_model = create_simple_model()

        # Define parameter mapping: 2 parameters
        def param_to_model(params):
            shock1, shock2 = params
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock1)
            spec.add_regime(1, time_regime=20)
            spec.add_shock(1, "Z_til", shock_per=0, shock_val=shock2)
            return base_model, spec

        # 3 targets for 2 parameters (over-identified)
        targets = [
            PointTarget(variable="I", time=5, value=0.545),
            PointTarget(variable="I", time=15, value=0.546),
            PointTarget(variable="I", time=30, value=0.543),
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01, 0.02]),
            solver="linear_sequence",
            tol=1e-4,
        )

        # Should use minimization
        assert result.method == "minimize"
        # Should find some solution
        assert len(result.parameters_array) == 2


class TestFunctionalTargetCalibration:
    """Tests for calibration with FunctionalTarget."""

    def test_functional_target_scalar(self):
        """Test calibration with a functional target."""
        base_model = create_simple_model()

        # Define parameter mapping
        def param_to_model(params):
            shock_val = params[0]
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock_val)
            return base_model, spec

        # Functional target: average consumption over first 10 periods
        def avg_consumption_error(result):
            # c is intermediate variable in Y
            if result.Y is None:
                return 1e10  # Return large error if Y not available
            c_idx = result.y_names.index("c")
            avg_c = np.mean(result.Y[:10, c_idx])
            target_avg = 0.84  # Roughly the steady state value
            return avg_c - target_avg

        targets = [
            FunctionalTarget(
                func=avg_consumption_error,
                description="Average consumption = 0.84",
            )
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01]),
            solver="linear_sequence",
            bounds=[(0.0, 0.1)],
            tol=1e-4,
        )

        # Should work
        assert result.method in ["root_scalar", "minimize_scalar"]
        assert result.parameters_array[0] > 0

    def test_mixed_targets(self):
        """Test calibration with both PointTarget and FunctionalTarget."""
        base_model = create_simple_model()

        # Define parameter mapping
        def param_to_model(params):
            shock1, shock2 = params
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock1)
            spec.add_regime(1, time_regime=20)
            spec.add_shock(1, "Z_til", shock_per=0, shock_val=shock2)
            return base_model, spec

        # Mixed targets
        def std_investment_error(result):
            I_idx = result.var_names.index("I")
            std_I = np.std(result.UX[:, I_idx])
            return std_I - 0.005

        targets = [
            PointTarget(variable="I", time=10, value=0.545),
            FunctionalTarget(func=std_investment_error, description="Std(I) = 0.005"),
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01, 0.02]),
            solver="linear_sequence",
            tol=1e-4,
        )

        # Should work
        assert result.method == "root"
        assert len(result.parameters_array) == 2


class TestCalibrationValidation:
    """Tests for calibration input validation."""

    def test_no_targets_error(self):
        """Test that calibration raises error with no targets."""
        base_model = create_simple_model()

        def param_to_model(params):
            return base_model, DetSpec()

        with pytest.raises(ValueError, match="At least one target"):
            calibrate(
                model=base_model,
                targets=[],
                param_to_model=param_to_model,
                initial_params=np.array([0.01]),
                solver="linear_irf",
            )

    def test_underidentified_error(self):
        """Test that calibration raises error for under-identified problems."""
        base_model = create_simple_model()

        def param_to_model(params):
            return base_model, DetSpec()

        # More parameters than targets
        targets = [PointTarget(variable="I", time=10, value=0.6)]

        with pytest.raises(ValueError, match="under-identified"):
            calibrate(
                model=base_model,
                targets=targets,
                param_to_model=param_to_model,
                initial_params=np.array([0.01, 0.02, 0.03]),  # 3 params, 1 target
                solver="linear_irf",
            )

    def test_invalid_solver_error(self):
        """Test that calibration raises error for invalid solver."""
        base_model = create_simple_model()

        def param_to_model(params):
            return base_model, DetSpec()

        targets = [PointTarget(variable="I", time=10, value=0.6)]

        with pytest.raises(ValueError, match="Unknown solver"):
            calibrate(
                model=base_model,
                targets=targets,
                param_to_model=param_to_model,
                initial_params=np.array([0.01]),
                solver="invalid_solver",
            )


class TestCalibrationResult:
    """Tests for CalibrationResult container."""

    def test_calibration_result_creation(self):
        """Test creating a CalibrationResult."""
        result = CalibrationResult(
            parameters={"beta": 0.96},
            parameters_array=np.array([0.96]),
            success=True,
            residual=1e-8,
            iterations=10,
            message="Converged",
            method="root_scalar",
        )

        assert result.parameters == {"beta": 0.96}
        assert result.success
        assert result.residual == 1e-8
        assert result.iterations == 10
        assert result.method == "root_scalar"

    def test_calibration_result_with_solution(self):
        """Test CalibrationResult with solution."""
        from equilibrium.solvers.results import DeterministicResult

        solution = DeterministicResult(
            UX=np.random.randn(50, 2),
            Z=np.zeros((50, 1)),
            var_names=["I", "log_K"],
            exog_names=["Z_til"],
        )

        result = CalibrationResult(
            parameters={"shock": 0.01},
            parameters_array=np.array([0.01]),
            success=True,
            solution=solution,
        )

        assert result.solution is not None
        assert result.solution.UX.shape == (50, 2)


class TestWeightedCalibration:
    """Tests for weighted over-identified calibration."""

    def test_weighted_point_targets(self):
        """Test calibration with weighted PointTargets."""
        base_model = create_simple_model()

        # Define parameter mapping
        def param_to_model(params):
            shock_val = params[0]
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock_val)
            return base_model, spec

        # Multiple targets with different weights (over-identified)
        targets = [
            PointTarget(variable="I", time=5, value=0.545, weight=1.0),
            PointTarget(
                variable="I", time=10, value=0.546, weight=5.0
            ),  # Higher weight
            PointTarget(variable="I", time=15, value=0.543, weight=1.0),
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01]),
            solver="linear_sequence",
            bounds=[(0.0, 0.1)],
            tol=1e-5,
        )

        # Should use minimization
        assert result.method == "minimize_scalar"
        # Should find some reasonable solution
        assert result.parameters_array[0] > 0

    def test_weighted_functional_target(self):
        """Test calibration with weighted FunctionalTarget."""
        base_model = create_simple_model()

        # Define parameter mapping
        def param_to_model(params):
            shock_val = params[0]
            spec = DetSpec()
            spec.add_regime(0)
            spec.add_shock(0, "Z_til", shock_per=0, shock_val=shock_val)
            return base_model, spec

        # Functional target that returns multiple values with weights
        def multi_moment_error(result):
            I_idx = result.var_names.index("I")
            mean_I = np.mean(result.UX[:20, I_idx])
            std_I = np.std(result.UX[:20, I_idx])
            return np.array([mean_I - 0.545, std_I - 0.003])

        targets = [
            FunctionalTarget(
                func=multi_moment_error,
                weights=[1.0, 10.0],  # Weight std more heavily
                description="Match mean and std of I",
            )
        ]

        # Calibrate
        result = calibrate(
            model=base_model,
            targets=targets,
            param_to_model=param_to_model,
            initial_params=np.array([0.01]),
            solver="linear_sequence",
            bounds=[(0.0, 0.1)],
            tol=1e-4,
        )

        # Should work
        assert result.method in ["minimize_scalar"]
        assert result.parameters_array[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
