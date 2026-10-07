"""Trust-region calibration regressions using inexpensive model solutions."""

import numpy as np
import pytest

from equilibrium.solvers import calibration
from equilibrium.solvers.calibration import FunctionalTarget, ModelParam
from equilibrium.solvers.linear_spec import LinearSpec


class StubModel:
    def __init__(self, params):
        self.params = params

    def update_copy(self, params):
        return StubModel({**self.params, **params})

    def solve_steady(self, **kwargs):
        pass

    def linearize(self):
        pass


@pytest.fixture(params=["calibrate", "calibrate_custom"])
def run_calibration(request, monkeypatch):
    def solution(model, *args):
        return np.array(list(model.params.values()), dtype=float)

    monkeypatch.setattr(calibration, "_compute_irf_from_linear_model", solution)

    def run(target, n=1, initial=0.5, param_bounds=(-np.inf, np.inf), **kwargs):
        params = [ModelParam(f"p{i}", initial, param_bounds) for i in range(n)]
        common = dict(
            model=StubModel({p.name: p.initial for p in params}),
            calib_params=params,
            targets=[target],
            method="trf",
            tol=1e-10,
            progress_every=0,
        )
        common.update(kwargs)
        if request.param == "calibrate":
            return calibration.calibrate(
                **common,
                solver="linear_irf",
                spec=LinearSpec(shock_name="z", Nt=2),
            )
        return calibration.calibrate_custom(**common, solution_builder=solution)

    return run


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("over_identified", [False, True])
def test_trf_problem_shapes(run_calibration, n, over_identified):
    expected = np.arange(1, n + 1, dtype=float)

    def errors(x):
        residuals = x - expected
        return np.concatenate([residuals, residuals]) if over_identified else residuals

    result = run_calibration(FunctionalTarget(errors), n=n, return_solution=True)
    assert result.success, result.message
    assert result.method == "least_squares"
    assert result.parameters_array.shape == (n,)
    assert np.issubdtype(result.parameters_array.dtype, np.floating)
    np.testing.assert_allclose(result.parameters_array, expected, atol=1e-8)
    np.testing.assert_allclose(result.solution, expected, atol=1e-8)
    assert result.parameters == result.model.params
    assert result.residual < 1e-8
    assert result.iterations > 0
    assert result.message


def test_trf_weighted_optimum(run_calibration):
    result = run_calibration(
        FunctionalTarget(lambda x: [x[0] - 1, x[0] - 3], weights=[1, 3])
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.parameters_array, [2.5], atol=1e-8)
    assert result.residual == pytest.approx(np.sqrt(3))


def test_trf_active_bound(run_calibration):
    result = run_calibration(
        FunctionalTarget(lambda x: [x[0] - 2, x[0] - 3]),
        param_bounds=(0, 1),
    )
    assert result.success, result.message
    assert result.parameters_array[0] <= 1
    assert result.parameters_array[0] == pytest.approx(1, abs=1e-8)
    assert result.residual == pytest.approx(np.sqrt(5))


def test_trf_rejects_false_convergence_and_ignores_exact_weights(run_calibration):
    result = run_calibration(
        FunctionalTarget(lambda x: 1.0, weights=[1e-30]),
    )
    assert not result.success
    assert result.residual == pytest.approx(1)
    assert "exceeds tolerance" in result.message


@pytest.mark.parametrize("bounds", [[(1, 1)], [(2, 3)]])
def test_trf_invalid_bounds_return_failure(run_calibration, bounds):
    result = run_calibration(FunctionalTarget(lambda x: x - 1), bounds=bounds)
    assert not result.success
    assert result.method == "least_squares"
    np.testing.assert_array_equal(result.parameters_array, [0.5])
    assert result.message


def test_trf_evaluation_budget(run_calibration):
    result = run_calibration(FunctionalTarget(lambda x: x - 2), maxiter=1)
    assert not result.success
    assert result.iterations == 1
    assert result.residual == pytest.approx(1.5)


def test_trf_still_rejects_under_identification(run_calibration):
    with pytest.raises(ValueError, match="under-identified"):
        run_calibration(FunctionalTarget(lambda x: x[0] - 1), n=2)


def test_trf_save_and_warm_start(run_calibration, tmp_path):
    target = FunctionalTarget(lambda x: x - 2)
    first = run_calibration(target, label="trf", save_dir=tmp_path)
    assert first.success, first.message
    restored = run_calibration(
        target,
        initialize_from_saved=True,
        load_label="trf",
        save_dir=tmp_path,
        maxiter=1,
        return_solution=True,
    )
    assert restored.success, restored.message
    np.testing.assert_allclose(restored.solution, [2], atol=1e-8)


def test_trf_unsuccessful_result_is_not_saved(run_calibration, tmp_path):
    with pytest.raises(RuntimeError, match="Calibration failed"):
        run_calibration(
            FunctionalTarget(lambda x: 1.0), label="failed", save_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["trf", "Nelder-Mead"])
def test_public_optimizer_options(run_calibration, method):
    options = (
        {"max_nfev": 500, "x_scale": "jac"}
        if method == "trf"
        else {"maxiter": 500, "adaptive": True, "xatol": 1e-9}
    )
    result = run_calibration(
        FunctionalTarget(lambda x: [x[0] - 1, x[1] - 2, x[0] + x[1] - 3]),
        n=2,
        method=method,
        maxiter=1,
        optimizer_options=options,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.parameters_array, [1, 2], atol=1e-6)


@pytest.mark.parametrize(
    "options,exception",
    [
        ([], TypeError),
        ({"gradient_kwargs": []}, TypeError),
        *[
            ({key: None}, ValueError)
            for key in (
                "fun",
                "f",
                "x0",
                "x1",
                "args",
                "kwargs",
                "method",
                "bounds",
                "bracket",
                "options",
            )
        ],
    ],
)
def test_public_invalid_optimizer_options(run_calibration, options, exception):
    with pytest.raises(exception, match="optimizer_options"):
        run_calibration(FunctionalTarget(lambda x: x - 1), optimizer_options=options)


def test_optimizer_tolerance_does_not_relax_acceptance(run_calibration):
    result = run_calibration(
        FunctionalTarget(lambda x: 1.0), optimizer_options={"gtol": 0.1}
    )
    assert not result.success
    assert "exceeds tolerance" in result.message
