"""Grid initialization regressions using inexpensive deterministic solutions."""

import numpy as np
import pytest

from equilibrium import GridSearchResult
from equilibrium.solvers import calibration as cal
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
def run(request, monkeypatch):
    def invoke(
        errors,
        *,
        initial=(0.5,),
        bounds=None,
        builder=None,
        weights=None,
        effective_bounds=None,
        **kwargs,
    ):
        params = [
            cal.ModelParam(f"p{i}", x, (0.0, 1.0) if bounds is None else bounds[i])
            for i, x in enumerate(initial)
        ]
        solution = builder or (lambda model: np.array(list(model.params.values())))
        common = dict(
            model=StubModel({p.name: p.initial for p in params}),
            calib_params=params,
            targets=[cal.FunctionalTarget(errors, weights=weights)],
            method="trf",
            progress_every=0,
        )
        if effective_bounds is not None:
            common["bounds"] = effective_bounds
        common.update(kwargs)
        if request.param == "calibrate":
            monkeypatch.setattr(
                cal, "_compute_irf_from_linear_model", lambda m, s: solution(m)
            )
            return cal.calibrate(
                **common, solver="linear_irf", spec=LinearSpec(shock_name="z", Nt=2)
            )
        return cal.calibrate_custom(**common, solution_builder=solution)

    return invoke


@pytest.fixture
def capture_start(monkeypatch):
    starts = []

    def solve(fun, x0, *args):
        starts.append(x0.copy())
        return cal.CalibrationResult(
            parameters_array=x0.copy(),
            success=True,
            residual=float(np.linalg.norm(fun(x0))),
            iterations=7,
        )

    monkeypatch.setattr(cal, "_solve_least_squares", solve)
    return starts


@pytest.mark.parametrize(
    "grid",
    [
        5,
        {"p0": 5},
        {"p0": [0.0, 0.25, 0.75]},
        [(0.0,), (0.25,), (0.75,)],
        np.array([[0.0], [0.25], [0.75]]),
    ],
)
def test_winner_handoff(run, capture_start, grid):
    result = run(lambda x: x - 0.75, grid_search=grid)
    np.testing.assert_array_equal(capture_start[0], [0.75])
    assert isinstance(result.grid_search, GridSearchResult)
    assert result.grid_search.best_score == 0
    assert result.grid_search.failures == 0
    assert result.iterations == 7
    assert result.grid_search.best_params.shape == (1,)
    assert result.grid_search.best_params.dtype.kind == "f"


def test_exact_rows_and_deduplication(run, capture_start):
    seen = []

    def error(x):
        seen.append(tuple(x))
        return x - [0.2, 0.8]

    result = run(
        error,
        initial=(0.5, 0.5),
        grid_search=[(0.2, 0.8), (0.8, 0.2), (0.2, 0.8), (0.5, 0.5)],
    )
    assert seen[:3] == [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)]
    assert result.grid_search.evaluations == 3
    np.testing.assert_array_equal(capture_start[0], [0.2, 0.8])


def test_partial_mapping_and_overridden_bounds(run, capture_start):
    result = run(
        lambda x: x - [0.7, 0.4],
        initial=(0.2, 0.4),
        grid_search={"p0": 2},
        bounds=[(0.0, 0.7), (0.0, 1.0)],
    )
    np.testing.assert_array_equal(result.grid_search.best_params, [0.7, 0.4])
    assert result.grid_search.evaluations == 3


@pytest.mark.parametrize("weighted", [False, True])
def test_scoring(run, capture_start, weighted):
    errors = (lambda x: [x[0], x[0] - 1]) if weighted else (lambda x: x - 1)
    weights = [1, 9] if weighted else [1e-20]
    result = run(errors, weights=weights, initial=(0.5,), grid_search=[(0.0,), (1.0,)])
    np.testing.assert_array_equal(result.grid_search.best_params, [1])
    assert result.grid_search.best_score == (1 if weighted else 0)


def test_baseline_and_ties(run, capture_start):
    result = run(lambda x: [1.0], initial=(0.5,), grid_search=[(0.0,), (1.0,)])
    np.testing.assert_array_equal(result.grid_search.best_params, [0.5])
    result = run(
        lambda x: [abs(x[0] - 0.5)], initial=(1.0,), grid_search=[(0.25,), (0.75,)]
    )
    np.testing.assert_array_equal(result.grid_search.best_params, [0.25])


def test_saved_baseline(run, capture_start, monkeypatch):
    monkeypatch.setattr(
        "equilibrium.utils.io.read_calibrated_params", lambda *a, **k: {"p0": 0.8}
    )
    result = run(
        lambda x: x - 0.8, grid_search=3, initialize_from_saved=True, load_label="saved"
    )
    np.testing.assert_array_equal(result.grid_search.initial_params, [0.8])
    np.testing.assert_array_equal(capture_start[0], [0.8])


def test_failed_baseline_vector_target_and_nonfinite(run, capture_start):
    def errors(x):
        if x[0] == 0.5:
            raise RuntimeError("bad initial guess")
        if x[0] == 0:
            return [np.nan, np.nan]
        return x - [0.75, 0.25]

    result = run(errors, initial=(0.5, 0.5), grid_search=[(0, 0), (0.75, 0.25)])
    assert result.grid_search.failures == 2
    assert result.grid_search.evaluations == 3
    np.testing.assert_array_equal(capture_start[0], [0.75, 0.25])


def test_all_failed_and_dimension_mismatch(run):
    with pytest.raises(RuntimeError, match="all 3 candidates"):
        run(lambda x: [np.nan], grid_search=2)
    with pytest.raises(ValueError, match="dimensionality"):
        run(lambda x: [1.0] if x[0] == 0.5 else [1.0, 2.0], grid_search=2)


@pytest.mark.parametrize(
    "grid",
    [
        True,
        1,
        {},
        {"missing": 2},
        {"p0": []},
        {"p0": [np.inf]},
        {"p0": [-1]},
        [],
        [0.2, 0.3],
        [(0.2, 0.3)],
        [(0.2,), (0.3, 0.4)],
        [(np.nan,)],
        [(2.0,)],
    ],
)
def test_invalid_grid_rejected_before_evaluation(run, grid):
    def never(x):
        pytest.fail("invalid grid should not evaluate targets")

    with pytest.raises((ValueError, TypeError)):
        run(never, grid_search=grid)


def test_unbounded_counts_and_explicit_rows(run, capture_start):
    with pytest.raises(ValueError, match="finite bounds"):
        run(lambda x: x, bounds=[(-np.inf, np.inf)], grid_search=3)
    result = run(lambda x: x, bounds=[(-np.inf, np.inf)], grid_search=[(0,)])
    assert result.grid_search.best_score == 0


@pytest.mark.parametrize("method", [None, "brentq", "bounded", "golden"])
def test_incompatible_scalar_method(run, method):
    with pytest.raises(ValueError, match="accepts an initial guess"):
        run(lambda x: x, method=method, grid_search=3)


def test_secant_identification(run):
    with pytest.raises(ValueError, match="one target"):
        run(lambda x: [x[0], x[0]], method="secant", grid_search=2)
    result = run(lambda x: x - 0.3, method="secant", grid_search=3)
    assert result.success
    np.testing.assert_allclose(result.parameters_array, [0.3])


def test_disabled_and_multimodal(run):
    # The residual derivative vanishes at zero; TRF cannot escape this start.
    plain = run(lambda x: x**2 - 0.64, initial=(0.0,), bounds=[(-1.0, 1.0)])
    searched = run(
        lambda x: x**2 - 0.64,
        initial=(0.0,),
        bounds=[(-1.0, 1.0)],
        grid_search=[(0.7,)],
    )
    assert plain.grid_search is None
    assert not plain.success
    assert searched.success
    np.testing.assert_allclose(searched.parameters_array, [0.8], atol=1e-5)


def test_transforms(run, capture_start):
    class Solution:
        def __init__(self, x):
            self.x = x

        def transform(self, **kwargs):
            return self.x * 2

    result = run(
        lambda x: x - 1.5,
        builder=lambda m: Solution(np.array(list(m.params.values()))),
        default_transform={},
        grid_search=[(0.25,), (0.75,)],
    )
    np.testing.assert_array_equal(result.grid_search.best_params, [0.75])


def test_mixed_parameter_column_order(monkeypatch):
    params = [
        cal.ShockParam("z", 0.2, (0, 1)),
        cal.ModelParam("alpha", 0.5, (0, 1)),
        cal.RegimeParam("tau", 1, 0.4, (0, 1)),
    ]
    captured = {}

    def loop(**kwargs):
        captured.update(kwargs)
        rows, _ = cal._prepare_grid(
            kwargs["grid_search"],
            kwargs["initial_params"],
            kwargs["bounds"],
            kwargs["param_names"],
            kwargs["grid_column_order"],
        )
        np.testing.assert_array_equal(list(rows), [[0.7, 0.8, 0.3]])
        return cal.CalibrationResult()

    monkeypatch.setattr(cal, "_run_calibration_loop", loop)
    model = StubModel({"alpha": 0.5})
    model.exog_list = ["z"]
    cal.calibrate(
        model,
        [cal.FunctionalTarget(lambda x: x)],
        params,
        method="trf",
        grid_search=[(0.3, 0.7, 0.8)],
        spec=cal.DetSpec(Nt=2),
    )
    assert captured["param_names"] == ["alpha", "regime_tau_r1", "shock_z_r0_t0"]


def test_qualified_names_and_ambiguous_names():
    names = ["alpha", "regime_tau_r1", "shock_z_r0_t0"]
    rows, count = cal._prepare_grid(
        {names[1]: [0.2, 0.4], names[2]: [0.3]}, np.zeros(3), [(0, 1)] * 3, names, None
    )
    assert list(rows) == [(0, 0.2, 0.3), (0, 0.4, 0.3)]
    assert count == 2
    with pytest.raises(ValueError, match="ambiguous"):
        cal._prepare_grid({"p": 2}, np.zeros(2), [(0, 1)] * 2, ["p", "p"], None)


@pytest.mark.parametrize("method", [None, "newton", "Nelder-Mead"])
def test_other_optimizer_handoff(run, monkeypatch, method):
    starts = []

    def solve(fun, x0, *args):
        starts.append(x0.copy())
        return cal.CalibrationResult(
            parameters_array=x0.copy(), success=True, residual=0
        )

    monkeypatch.setattr(cal, "_solve_vector_root", solve)
    monkeypatch.setattr(cal, "_solve_vector_minimize", solve)
    result = run(
        lambda x: x - [0.2, 0.8],
        initial=(0.5, 0.5),
        method=method,
        grid_search=[(0.2, 0.8)],
    )
    np.testing.assert_array_equal(starts[0], [0.2, 0.8])
    assert result.grid_search.best_score == 0


@pytest.mark.parametrize(
    "method, expected", [("trf", [0.8, 0.0]), ("Nelder-Mead", [0.0, 1.0])]
)
def test_square_system_weighting(run, monkeypatch, capture_start, method, expected):
    monkeypatch.setattr(
        cal,
        "_solve_vector_minimize",
        lambda fun, x0, *args: cal.CalibrationResult(parameters_array=x0),
    )
    result = run(
        lambda x: x,
        initial=(1.0, 1.0),
        weights=[100, 1],
        method=method,
        grid_search=[(0.8, 0.0), (0.0, 1.0)],
    )
    np.testing.assert_array_equal(result.grid_search.best_params, expected)


def test_nonfinite_weights_and_overflow(run):
    with pytest.raises(RuntimeError, match="all 3 candidates"):
        run(lambda x: x, weights=[np.nan], grid_search=2)
    with pytest.raises(RuntimeError, match="all 3 candidates"):
        run(lambda x: [1e300], grid_search=2)


def test_diagnostics_logging(run, capture_start, caplog):
    with caplog.at_level("INFO", logger=cal.__name__):
        result = run(lambda x: x, grid_search={"p0": [0, 0, 1]}, progress_every=2)
    assert result.grid_search.evaluations == 3
    assert "Grid search eval 2" in caplog.text
    assert "Grid search eval 1:" not in caplog.text
    assert "Grid search selected" in caplog.text


def test_effective_bounds_override(run, capture_start):
    result = run(lambda x: x - 0.6, grid_search=2, effective_bounds=[(0.2, 0.6)])
    np.testing.assert_array_equal(result.grid_search.best_params, [0.6])
    with pytest.raises(ValueError, match="effective bounds"):
        run(lambda x: x, grid_search=[(0.8,)], effective_bounds=[(0.2, 0.6)])


def test_saved_float_with_integer_initial(run, capture_start, monkeypatch):
    monkeypatch.setattr(
        "equilibrium.utils.io.read_calibrated_params", lambda *a, **k: {"p0": 0.8}
    )
    result = run(
        lambda x: x - 0.8,
        initial=(0,),
        grid_search=2,
        initialize_from_saved=True,
        load_label="saved",
    )
    np.testing.assert_array_equal(result.grid_search.initial_params, [0.8])
    np.testing.assert_array_equal(capture_start[0], [0.8])
