import numpy as np

from equilibrium.solvers.calibration import _solve_scalar_root


def test_solve_scalar_root_supports_explicit_secant_method():
    """Secant should use x0/x1 instead of passing only a bracket."""
    result = _solve_scalar_root(
        func=lambda x: x - 2.0,
        x0=1.5,
        bounds=[(0.0, 4.0)],
        method="secant",
        tol=1e-10,
        maxiter=20,
    )

    assert result.success, result.message
    assert result.method == "root_scalar"
    assert np.isclose(result.parameters_array[0], 2.0)
