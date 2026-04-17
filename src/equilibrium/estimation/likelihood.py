"""Likelihood helpers bridging linearized models to state-space estimation."""

from __future__ import annotations

import numpy as np

from .state_space import StateSpaceEstimates, StateSpaceModel


def _measurement_covariance(obs_names, meas_err):
    ny = len(obs_names)

    if meas_err is None:
        return np.zeros((ny, ny), dtype=float)

    if isinstance(meas_err, dict):
        H = np.zeros((ny, ny), dtype=float)
        for idx, name in enumerate(obs_names):
            stdev = float(meas_err.get(name, 0.0))
            H[idx, idx] = stdev**2
        return H

    H = np.asarray(meas_err, dtype=float)
    if H.shape != (ny, ny):
        raise ValueError(
            f"Measurement error covariance must have shape {(ny, ny)}, got {H.shape}."
        )
    return H


def _measurement_matrices(model, observables, in_deviations=False):
    linear_model = model.linear_mod
    if linear_model is None or linear_model.A_s is None or linear_model.B_s is None:
        raise RuntimeError(
            "Model must be linearized before building a state-space model."
        )

    if not observables:
        raise ValueError("At least one observable must be provided.")

    if not getattr(model, "_steady_solved", False):
        raise RuntimeError(
            "Model steady state must be solved before building a state-space model."
        )

    model.steady_state_derivatives()

    n_u = model.N["u"]
    n_x = model.N["x"]
    n_z = model.N["z"]
    n_state = n_u + n_x + n_z

    offsets = {"u": 0, "x": n_u, "z": n_u + n_x}
    category_map = {}
    for category in ["u", "x", "z", "intermediate", "read_expectations"]:
        for name in model.var_lists.get(category, []):
            category_map[name] = category

    Z_rows = []
    b = []
    for name in observables:
        if name not in category_map:
            raise ValueError(
                f"Observable '{name}' must be an existing model variable in "
                "'u', 'x', 'z', 'intermediate', or 'read_expectations'."
            )

        category = category_map[name]
        row = np.zeros(n_state, dtype=float)
        if category in offsets:
            idx = model.var_lists[category].index(name)
            row[offsets[category] + idx] = 1.0
        elif category == "intermediate":
            idx = model.var_lists["intermediate"].index(name)
            row = np.hstack(
                [
                    np.asarray(
                        model.derivatives["intermediates"][var][idx, :], dtype=float
                    )
                    for var in ["u", "x", "z"]
                ]
            )
        else:
            if linear_model.L is None:
                raise RuntimeError(
                    "Linear model does not expose the read_expectations mapping."
                )
            idx = model.var_lists["read_expectations"].index(name)
            row = np.asarray(linear_model.L[idx, :], dtype=float)

        Z_rows.append(row)
        if in_deviations:
            b.append(0.0)
        else:
            b.append(float(model.steady_dict[name]))

    return np.vstack(Z_rows), np.asarray(b, dtype=float)


def build_state_space(
    model, observables, meas_err=None, in_deviations=False
) -> StateSpaceModel:
    """Build a state-space model from a linearized equilibrium model."""
    linear_model = getattr(model, "linear_mod", None)
    if linear_model is None or linear_model.A_s is None or linear_model.B_s is None:
        raise RuntimeError(
            "Model must be linearized before building a state-space model."
        )

    obs_names = list(observables)
    Z, b = _measurement_matrices(model, obs_names, in_deviations=in_deviations)
    Q = np.diag([float(model.params[f"VOL_{shock}"]) ** 2 for shock in model.exog_list])
    H = _measurement_covariance(obs_names, meas_err)

    return StateSpaceModel(
        A=np.asarray(linear_model.A_s, dtype=float),
        R=np.asarray(linear_model.B_s, dtype=float),
        Q=Q,
        Z=Z,
        H=H,
        b=b,
    )


def log_likelihood_ssm(ssm, data, *, fixed_init=None) -> float:
    """Evaluate the Gaussian log-likelihood for a pre-built state-space model."""
    try:
        y = np.asarray(data, dtype=float)
        if y.ndim != 2:
            raise ValueError(f"Data must be 2-D, got shape {y.shape}.")
        if y.shape[1] != ssm.Ny:
            raise ValueError(f"Data must have {ssm.Ny} columns, got {y.shape[1]}.")

        estimates = StateSpaceEstimates(ssm, y, fixed_init=fixed_init)
        if not estimates.valid:
            return -1e10

        estimates.kalman_filter()
        log_like = float(estimates.log_like)
        if not np.isfinite(log_like):
            return -1e10
        return log_like
    except Exception:
        return -1e10


def log_likelihood(
    model, data, *, observables, meas_err=None, fixed_init=None, in_deviations=False
) -> float:
    """Evaluate the Gaussian log-likelihood for observed model data."""
    try:
        y = np.asarray(data, dtype=float)
        if y.ndim != 2:
            raise ValueError(f"Data must be 2-D, got shape {y.shape}.")

        obs_names = list(observables)
        expected_ny = len(obs_names)
        if y.shape[1] != expected_ny:
            raise ValueError(
                f"Data must have {expected_ny} observable columns, got {y.shape[1]}."
            )

        ssm = build_state_space(
            model, observables=obs_names, meas_err=meas_err, in_deviations=in_deviations
        )
        return log_likelihood_ssm(ssm, y, fixed_init=fixed_init)
    except Exception:
        return -1e10
