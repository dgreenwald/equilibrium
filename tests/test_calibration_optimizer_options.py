"""Backend routing and compatibility for flat calibration tuning options."""

from types import SimpleNamespace

import numpy as np
import pytest

from equilibrium.solvers import calibration as cal
from equilibrium.solvers import newton


@pytest.mark.parametrize("method,budget", [(None, "maxfev"), ("lm", "maxiter")])
def test_root_options(monkeypatch, method, budget):
    options = {budget: 123, "factor": 0.5}

    def root(fun, x0, **kwargs):
        assert kwargs["options"] == options
        return SimpleNamespace(x=x0, success=True, fun=fun(x0), nfev=1, message="ok")

    monkeypatch.setattr(cal.opt, "root", root)
    result = cal._solve_vector_root(
        lambda x: x, np.ones(2), method, 1e-6, 10, optimizer_options=options
    )
    assert result.success, result.message
    assert options == {budget: 123, "factor": 0.5}


def test_lm_default_budget(monkeypatch):
    def root(fun, x0, **kwargs):
        assert kwargs["options"] == {"maxiter": 42}
        return SimpleNamespace(x=x0, success=True, fun=fun(x0), message="ok")

    monkeypatch.setattr(cal.opt, "root", root)
    assert cal._solve_vector_root(lambda x: x, np.ones(2), "lm", 1e-6, 42).success


@pytest.mark.parametrize("scalar", [False, True])
def test_minimize_options(monkeypatch, scalar):
    options = {"maxiter": 200, "xatol": 1e-9}

    def minimize(fun, **kwargs):
        assert kwargs["options"] == options
        x = 0.0 if scalar else np.zeros(2)
        return SimpleNamespace(x=x, success=True, fun=fun(x), nfev=1, message="ok")

    if scalar:
        monkeypatch.setattr(cal.opt, "minimize_scalar", minimize)
        result = cal._solve_scalar_minimize(
            lambda x: (np.array([x]), np.ones(1)),
            0.5,
            [(0, 1)],
            None,
            1e-6,
            10,
            options,
        )
    else:
        monkeypatch.setattr(
            cal.opt, "minimize", lambda fun, x0, **kw: minimize(fun, **kw)
        )
        result = cal._solve_vector_minimize(
            lambda x: (x, np.ones(2)),
            np.ones(2),
            [(0, 1)] * 2,
            "Nelder-Mead",
            1e-6,
            10,
            options,
        )
    assert result.success, result.message
    assert options == {"maxiter": 200, "xatol": 1e-9}


def test_scalar_root_options_survive_fallback(monkeypatch):
    calls = []
    options = {"xtol": 1e-9, "rtol": 1e-8, "maxiter": 200, "k": 2}

    def root(fun, **kwargs):
        calls.append(kwargs)
        assert kwargs["xtol"] == 1e-9
        assert kwargs["rtol"] == 1e-8
        assert kwargs["maxiter"] == 200
        assert kwargs["options"] == {"k": 2}
        kwargs["options"].clear()
        if kwargs["method"] == "brentq":
            raise ValueError("different signs required")
        return SimpleNamespace(root=1.0, converged=True, iterations=2, flag="ok")

    monkeypatch.setattr(cal.opt, "root_scalar", root)
    result = cal._solve_scalar_root(
        lambda x: x - 1, 0.5, [(0, 2)], None, 1e-6, 10, options
    )
    assert result.success, result.message
    assert [c["method"] for c in calls] == ["brentq", "secant"]
    assert options == {"xtol": 1e-9, "rtol": 1e-8, "maxiter": 200, "k": 2}


def test_trf_options(monkeypatch):
    options = {"ftol": 1e-9, "max_nfev": 200, "x_scale": "jac"}

    def least_squares(fun, x0, **kwargs):
        assert kwargs["method"] == "trf"
        assert kwargs["ftol"] == 1e-9
        assert kwargs["xtol"] == kwargs["gtol"] == 1e-6
        assert kwargs["max_nfev"] == 200
        assert kwargs["x_scale"] == "jac"
        return SimpleNamespace(x=x0, success=True, fun=fun(x0), nfev=1, message="ok")

    monkeypatch.setattr(cal.opt, "least_squares", least_squares)
    result = cal._solve_least_squares(lambda x: x, np.ones(2), None, 1e-6, 10, options)
    assert result.success, result.message
    assert options == {"ftol": 1e-9, "max_nfev": 200, "x_scale": "jac"}


@pytest.mark.parametrize("use_new", [False, True])
def test_newton_legacy_gradient_merge(monkeypatch, use_new):
    legacy = {"step": 1e-4, "method": "central"}
    options = {"max_iterations": 200, "max_backstep_iterations": 30}
    if use_new:
        options["gradient_kwargs"] = {"step": 1e-5}

    def root(fun, x0, **kwargs):
        assert kwargs["max_iterations"] == 200
        assert kwargs["max_backstep_iterations"] == 30
        assert kwargs["gradient_kwargs"] == {
            "step": 1e-5 if use_new else 1e-4,
            "method": "central",
        }
        kwargs["gradient_kwargs"].clear()
        return SimpleNamespace(x=x0, success=True, dist=0)

    monkeypatch.setattr(newton, "root", root)
    result = cal._solve_vector_root(
        lambda x: x, np.ones(2), "newton", 1e-6, 10, legacy, options
    )
    assert result.success, result.message
    assert legacy == {"step": 1e-4, "method": "central"}
    if use_new:
        assert options["gradient_kwargs"] == {"step": 1e-5}
