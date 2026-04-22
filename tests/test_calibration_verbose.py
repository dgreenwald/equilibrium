import logging

import numpy as np

from equilibrium import Model
from equilibrium.solvers.calibration import (
    FunctionalTarget,
    ModelParam,
    PointTarget,
    calibrate,
)
from equilibrium.solvers.det_spec import DetSpec


def create_simple_model():
    """Create a very simple model for testing calibration."""
    mod = Model(label="test_verbose")
    mod.params.update({"alpha": 0.35, "beta": 0.96})
    mod.steady_guess.update({"y": 1.0, "c": 0.8})

    mod.rules["optimality"] += [
        ("y", "y - alpha * 2.0"),
        ("c", "c - (y - 0.2)"),
    ]
    mod.finalize()
    mod.solve_steady()
    return mod


def test_calibrate_verbose(caplog):
    model = create_simple_model()

    # We want to calibrate alpha such that y = 1.0
    # Currently alpha=0.35 -> y=0.7
    # We need alpha=0.5 -> y=1.0

    targets = [
        PointTarget(variable="y", time=0, value=1.0),
        PointTarget(variable="c", time=0, value=0.8),
    ]

    calib_params = [
        ModelParam("alpha", initial=0.35, bounds=(0.1, 0.9)),
    ]

    spec = DetSpec(n_regimes=1, Nt=10)

    caplog.set_level(logging.INFO)

    result = calibrate(
        model=model,
        targets=targets,
        calib_params=calib_params,
        spec=spec,
        verbose=True,
        progress_every=1,
    )

    assert result.success
    assert np.isclose(result.parameters["alpha"], 0.5)

    # Check if verbose output is in logs
    assert "Target details:" in caplog.text
    assert "y at t=0: model=" in caplog.text
    assert "c at t=0: model=" in caplog.text
    assert "target=1" in caplog.text
    assert "target=0.8" in caplog.text


def test_calibrate_verbose_functional(caplog):
    model = create_simple_model()

    def my_func(solution):
        y_val = solution.UX[0, solution.var_names.index("y")]
        return y_val - 1.0

    targets = [
        FunctionalTarget(func=my_func, description="Match y to 1.0"),
    ]

    calib_params = [
        ModelParam("alpha", initial=0.35, bounds=(0.1, 0.9)),
    ]

    spec = DetSpec(n_regimes=1, Nt=10)

    caplog.set_level(logging.INFO)

    result = calibrate(
        model=model,
        targets=targets,
        calib_params=calib_params,
        spec=spec,
        verbose=True,
        progress_every=1,
    )

    assert result.success
    assert "Target details:" in caplog.text
    assert "Match y to 1.0: error=" in caplog.text


def test_calibrate_verbose_functional_vector(caplog):
    model = create_simple_model()

    def my_func_vector(solution):
        y_val = solution.UX[0, solution.var_names.index("y")]
        c_val = solution.UX[0, solution.var_names.index("c")]
        return np.array([y_val - 1.0, c_val - 0.8])

    targets = [
        FunctionalTarget(
            func=my_func_vector, description="Match y and c", weights=[1.0, 2.0]
        ),
    ]

    calib_params = [
        ModelParam("alpha", initial=0.35, bounds=(0.1, 0.9)),
    ]

    spec = DetSpec(n_regimes=1, Nt=10)

    caplog.set_level(logging.INFO)

    result = calibrate(
        model=model,
        targets=targets,
        calib_params=calib_params,
        spec=spec,
        verbose=True,
        progress_every=1,
    )

    assert result.success
    assert "Target details:" in caplog.text
    # Check for the vector output format
    assert "Match y and c: error=" in caplog.text
    assert "['" in caplog.text  # List representation


def test_calibrate_overidentified_large_residual():
    """
    Test that an over-identified problem with a large residual (due to
    being over-identified) still reports success if the optimizer converged.
    """
    model = create_simple_model()

    # y = alpha * 2, c = y - 0.2
    # Targets: y=1.0, c=0.9
    # alpha=0.5 -> y=1.0, c=0.8 (errors: 0, -0.1)
    # alpha=0.55 -> y=1.1, c=0.9 (errors: 0.1, 0)
    # alpha=0.525 -> y=1.05, c=0.85 (errors: 0.05, -0.05) -> residual = sqrt(0.05^2 + 0.05^2) = 0.0707

    targets = [
        PointTarget(variable="y", time=0, value=1.0),
        PointTarget(variable="c", time=0, value=0.9),
    ]

    calib_params = [
        ModelParam("alpha", initial=0.35, bounds=(0.1, 0.9)),
    ]

    spec = DetSpec(n_regimes=1, Nt=10)

    # We set a very small tolerance to ensure optimizer thinks it's converged
    result = calibrate(
        model=model,
        targets=targets,
        calib_params=calib_params,
        spec=spec,
        tol=1e-8,
        progress_every=0,
    )

    assert result.success
    # residual should be around 0.0707, which is > 10 * 1e-8
    assert result.residual > 0.01
    assert np.isclose(result.parameters["alpha"], 0.525, atol=1e-4)
