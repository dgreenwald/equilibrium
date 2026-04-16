"""Tests for estimation/state_space.py."""

import numpy as np

from equilibrium.estimation.state_space import (
    StateSpaceEstimates,
    StateSpaceModel,
    init_to_val,
)


def _make_scalar_ssm():
    return StateSpaceModel(
        A=np.array([[0.5]]),
        R=np.array([[1.0]]),
        Q=np.array([[1.0]]),
        Z=np.array([[1.0]]),
        H=np.array([[0.25]]),
        b=np.array([1.0]),
    )


def _make_vector_ssm():
    return StateSpaceModel(
        A=np.array([[0.6, 0.0], [0.2, 0.4]]),
        R=np.eye(2),
        Q=np.diag([0.5, 0.25]),
        Z=np.array([[1.0, 0.5], [0.0, 1.0]]),
        H=np.diag([0.1, 0.2]),
        b=np.array([0.25, -0.5]),
    )


def test_init_to_val():
    arr = init_to_val((2, 3), np.nan)
    assert arr.shape == (2, 3)
    assert np.isnan(arr).all()


def test_unconditional_cov_scalar_model():
    ssm = _make_scalar_ssm()
    cov = ssm.unconditional_cov()
    np.testing.assert_allclose(cov, np.array([[4.0 / 3.0]]))


def test_simulate_and_kalman_filter():
    ssm = _make_scalar_ssm()
    shocks = np.array([[0.2], [-0.1], [0.05]])
    meas_err = np.zeros((4, 1))
    y, x = ssm.simulate(x_1=np.array([0.0]), shocks=shocks, meas_err=meas_err)

    est = StateSpaceEstimates(ssm, y)
    est.kalman_filter()

    assert est.valid is True
    assert np.isfinite(est.log_like)
    assert est.x_pred.shape == (4, 1)
    assert est.P_pred.shape == (4, 1, 1)
    assert x.shape == (4, 1)


def test_kalman_filter_handles_missing_observations():
    ssm = _make_scalar_ssm()
    y = np.array([[1.0], [np.nan], [0.8], [1.1]])
    est = StateSpaceEstimates(ssm, y)
    est.kalman_filter()

    assert np.isfinite(est.log_like)
    assert est.ix.shape == y.shape
    assert not est.ix[1, 0]


def test_smoothers_and_state_draw_shapes():
    np.random.seed(0)
    ssm = _make_vector_ssm()
    y, _ = ssm.simulate(x_1=np.zeros(2), Nt=6)

    est = StateSpaceEstimates(ssm, y)
    est.kalman_filter()
    est.disturbance_smoother()
    est.state_smoother()
    est.shock_smoother()
    est.meas_err_smoother()
    est.draw_states(draw_shocks=True, draw_meas_err=True)

    assert est.r.shape == (6, 2)
    assert est.x_smooth.shape == (6, 2)
    assert est.shocks_smooth.shape == (5, 2)
    assert est.meas_err_smooth.shape == (6, 2)
    assert est.state_draw.shape == (6, 2)
    assert est.shock_draw.shape == (5, 2)
    assert est.meas_err_draw.shape == (6, 2)


def test_decomposition_helpers_have_expected_shapes_and_identities():
    ssm = _make_vector_ssm()
    shocks = np.array(
        [
            [0.1, -0.2],
            [0.0, 0.3],
            [-0.1, 0.05],
            [0.2, 0.0],
        ]
    )
    meas_err = np.zeros((5, 2))
    y, states = ssm.simulate(
        x_1=np.array([0.2, -0.1]), shocks=shocks, meas_err=meas_err
    )

    shock_components, det_component = ssm.decompose_by_shock(shocks, states)
    np.testing.assert_allclose(shock_components.sum(axis=0) + det_component, states)

    y_shock_only, y_shock_removed = ssm.decompose_y_by_shock(shocks, states, y=y)
    assert y_shock_only.shape == (2, 5, 2)
    assert y_shock_removed.shape == (2, 5, 2)

    shock_only_components = y_shock_only - (det_component @ ssm.Z.T)[np.newaxis, :, :]
    np.testing.assert_allclose(
        y_shock_removed + shock_only_components,
        np.repeat(y[np.newaxis, :, :], ssm.Ne, axis=0),
    )

    y_state_only, y_state_removed = ssm.decompose_y_by_state(states, y=y)
    assert y_state_only.shape == (2, 5, 2)
    assert y_state_removed.shape == (2, 5, 2)

    state_only_components = y_state_only - ssm.b[np.newaxis, np.newaxis, :]
    np.testing.assert_allclose(
        y_state_removed + state_only_components,
        np.repeat(y[np.newaxis, :, :], ssm.Nx, axis=0),
    )

    y_shock_only_offset, y_shock_removed_offset = ssm.decompose_y_by_shock(
        shocks, states, y=y, start_ix=1
    )
    expected_prefix = np.repeat(y[np.newaxis, :1, :], ssm.Ne, axis=0)
    np.testing.assert_allclose(y_shock_only_offset[:, :1, :], expected_prefix)
    np.testing.assert_allclose(y_shock_removed_offset[:, :1, :], expected_prefix)

    y_state_only_offset, y_state_removed_offset = ssm.decompose_y_by_state(
        states, y=y, start_ix=1
    )
    expected_prefix = np.repeat(y[np.newaxis, :1, :], ssm.Nx, axis=0)
    np.testing.assert_allclose(y_state_only_offset[:, :1, :], expected_prefix)
    np.testing.assert_allclose(y_state_removed_offset[:, :1, :], expected_prefix)
