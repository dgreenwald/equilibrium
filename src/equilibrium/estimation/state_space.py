"""Linear Gaussian state space tools used by estimation.

Ported from ``py_tools.time_series.state_space`` with local numerical helpers
and without the ``py_tools.stats`` dependency.
"""

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn

from . import numerical as nm


def init_to_val(shape, val):
    """Create an array of a given shape filled with a specified value."""
    x = np.empty(shape)
    x[:] = val
    return x


class StateSpaceModel:
    """Linear Gaussian state space model."""

    def __init__(self, A, R, Q, Z, H, c=None, b=None):
        self.A = A
        self.R = R
        self.Q = Q
        self.Z = Z
        self.H = H

        self.Nx, self.Ne = R.shape

        if self.Z is not None:
            self.Ny, _ = Z.shape
        else:
            self.Ny = 0

        assert all([var == self.Nx for var in [self.A.shape[0], self.A.shape[1]]])
        assert all([var == self.Ne for var in [self.Q.shape[0], self.Q.shape[1]]])

        if b is None:
            self.b = np.zeros(self.Ny)
        else:
            assert len(b) == self.Ny
            self.b = b

        if c is not None:
            assert len(c) == self.Nx
            x_bar = np.linalg.solve(np.eye(self.Nx) - self.A, c)
            self.b -= np.dot(Z, x_bar)

        self.QR = self.Q @ self.R.T
        self.RQR = self.R @ self.QR

        self.CQT = nm.robust_cholesky(self.Q, min_eig=0.0).T
        if H is not None:
            self.CHT = nm.robust_cholesky(self.H, min_eig=0.0).T

    def unconditional_cov(self, fixed_init=None):
        """Compute the unconditional covariance of the state vector."""
        if fixed_init is None:
            fixed_init = []

        if not fixed_init:
            try:
                return sp.linalg.solve_discrete_lyapunov(self.A, self.RQR)
            except Exception:
                return None

        free_init = [ii for ii in range(self.Nx) if ii not in fixed_init]
        A_trunc = self.A[free_init, :][:, free_init]
        RQR_trunc = self.RQR[free_init, :][:, free_init]

        V_trunc = sp.linalg.solve_discrete_lyapunov(A_trunc, RQR_trunc)
        V_full = np.zeros((self.Nx, self.Nx))
        for itrunc, ifull in enumerate(free_init):
            V_full[ifull, free_init] = V_trunc[itrunc, :]

        return V_full

    def simulate(self, x_1=None, Nt=None, shocks=None, meas_err=None, ix=None, use_b=True):
        """Simulate observations and states from the model."""
        if shocks is None:
            if meas_err is None:
                assert Nt is not None
            else:
                Nt = meas_err.shape[0]
        else:
            Nt = shocks.shape[0] + 1

        if shocks is None:
            shocks = self.draw_shocks(Nt - 1)

        if meas_err is None:
            meas_err = self.draw_meas_err(Nt)

        assert shocks.shape == (Nt - 1, self.Ne)
        assert meas_err.shape == (Nt, self.Ny)

        if x_1 is None:
            Sig0 = self.unconditional_cov()
            C = np.linalg.cholesky(Sig0)
            x_1 = np.dot(C, np.random.randn(self.Nx))

        if ix is None:
            ix = np.ones((Nt, self.Ny), dtype=bool)
        else:
            assert ix.shape == (Nt, self.Ny)

        y_sim = init_to_val((Nt, self.Ny), np.nan)
        x_sim = np.zeros((Nt, self.Nx))

        x_t = x_1

        for tt in range(Nt):
            ix_t = ix[tt, :]
            y_sim[tt, ix_t] = np.dot(self.Z[ix_t, :], x_t) + meas_err[tt, ix_t]
            if use_b:
                y_sim[tt, ix_t] += self.b[ix_t]
            x_sim[tt, :] = x_t

            if tt < Nt - 1:
                x_t = np.dot(self.A, x_t) + np.dot(self.R, shocks[tt, :])

        return (y_sim, x_sim)

    def draw_shocks(self, Nt):
        """Draw random structural shocks from N(0, Q)."""
        return np.dot(np.random.randn(Nt, self.Ne), self.CQT)

    def draw_meas_err(self, Nt):
        """Draw random measurement errors from N(0, H)."""
        return np.dot(np.random.randn(Nt, self.Ny), self.CHT)

    def decompose_by_shock(self, shocks, states, start_ix=0):
        """Decompose a state history into contributions from each shock."""
        shock_components_samp, det_component_samp = self.decompose_by_shock_init(
            shocks[start_ix:, :], states[start_ix, :]
        )

        if start_ix > 0:
            shock_components = np.concatenate(
                (np.zeros((self.Ne, start_ix, self.Nx)), shock_components_samp), axis=1
            )
            det_component = np.concatenate(
                (states[:start_ix, :], det_component_samp), axis=0
            )
            return shock_components, det_component

        return shock_components_samp, det_component_samp

    def decompose_by_shock_init(self, shocks, x1):
        """Decompose states into shock and deterministic components."""
        Nt = shocks.shape[0] + 1

        det_component = np.zeros((Nt, self.Nx))
        det_component[0, :] = x1

        for tt in range(1, Nt):
            det_component[tt, :] = self.A @ det_component[tt - 1, :]

        shock_components = np.zeros((Nt, self.Nx, self.Ne))
        Ri_shock_list = [
            shocks[:, ishock][:, np.newaxis] @ self.R[:, ishock][np.newaxis, :]
            for ishock in range(self.Ne)
        ]
        R_shocks = np.concatenate([x[:, :, np.newaxis] for x in Ri_shock_list], axis=2)

        for tt in range(Nt - 1):
            shock_components[tt + 1, :, :] = (
                self.A @ shock_components[tt, :, :] + R_shocks[tt, :, :]
            )

        shock_components = np.moveaxis(shock_components, [0, 1, 2], [1, 2, 0])
        return shock_components, det_component

    def decompose_y_by_shock(self, shocks, states, y=None, start_ix=0):
        """Decompose observables into contributions from each shock."""
        if y is None:
            y = states @ self.Z.T + self.b[np.newaxis, :]

        shock_components_samp, det_component_samp = self.decompose_by_shock_init(
            shocks[start_ix:, :], states[start_ix, :]
        )

        y_shock_components_samp = np.stack(
            [
                shock_components_samp[ishock, :, :] @ self.Z.T
                for ishock in range(self.Ne)
            ],
            axis=0,
        )
        y_det_component_samp = det_component_samp @ self.Z.T

        y_shock_only_samp = y_shock_components_samp + y_det_component_samp[np.newaxis, :, :]
        y_shock_removed_samp = y[np.newaxis, start_ix:, :] - y_shock_components_samp

        if start_ix > 0:
            this_y = np.repeat(y[np.newaxis, :start_ix, :], self.Ne, axis=0)
            y_shock_only = np.concatenate((this_y, y_shock_only_samp), axis=1)
            y_shock_removed = np.concatenate((this_y, y_shock_removed_samp), axis=1)
            return y_shock_only, y_shock_removed

        return y_shock_only_samp, y_shock_removed_samp

    def decompose_y_by_state(self, states, y=None, start_ix=0):
        """Decompose observables into contributions from each state."""
        if y is None:
            y = states @ self.Z.T + self.b[np.newaxis, :]

        y_state_components_samp = np.concatenate(
            [
                (
                    states[start_ix:, istate][:, np.newaxis]
                    @ self.Z[:, istate][np.newaxis, :]
                )[np.newaxis, :, :]
                for istate in range(self.Nx)
            ],
            axis=0,
        )

        y_state_only_samp = (
            y_state_components_samp + self.b[np.newaxis, np.newaxis, :]
        )
        y_state_removed_samp = y[np.newaxis, start_ix:, :] - y_state_components_samp

        if start_ix > 0:
            this_y = np.repeat(y[np.newaxis, :start_ix, :], self.Nx, axis=0)
            y_state_only = np.concatenate((this_y, y_state_only_samp), axis=1)
            y_state_removed = np.concatenate((this_y, y_state_removed_samp), axis=1)
            return y_state_only, y_state_removed

        return y_state_only_samp, y_state_removed_samp


class StateSpaceEstimates:
    """Estimated states from applying a state space model to a dataset."""

    def __init__(self, ssm, y, x_init=None, P_init=None, fixed_init=None):
        if fixed_init is None:
            fixed_init = []

        self.fixed_init = fixed_init

        self.set_data(y)
        self.set_ssm(ssm)

        if x_init is None:
            x_init = np.zeros(self.Nx)

        if P_init is None:
            P_init = self.ssm.unconditional_cov(self.fixed_init)
            self.valid = P_init is not None
        else:
            self.valid = True

        self.x_init = x_init
        self.P_init = P_init

        self.r = None
        self.state_draw = None
        self.shock_draw = None
        self.meas_err_draw = None

    def set_data(self, y):
        """Read in a new set of observables."""
        self.y = y
        self.Nt, self.Ny = self.y.shape
        self.ix = np.isfinite(self.y)

    def set_ssm(self, ssm):
        """Assign a new state space model."""
        self.ssm = ssm
        self.Nx, self.Ne = self.ssm.R.shape
        assert self.Ny == ssm.Ny

        self.y_til = self.y - self.ssm.b[np.newaxis, :]

    def kalman_filter(self, x_init=None, P_init=None, overwrite_r=True, y_til=None):
        """Run the Kalman filter on the observed data."""
        if overwrite_r:
            self.r = None

        if x_init is not None:
            self.x_init = x_init
        if P_init is not None:
            self.P_init = P_init

        if y_til is None:
            self.base_data_results = True
            y_til = self.y_til
        else:
            self.base_data_results = False

        x_pred_t = self.x_init
        P_pred_t = self.P_init

        self.err = np.zeros((self.Nt, self.Ny))
        self.x_pred = np.zeros((self.Nt, self.Nx))
        self.P_pred = np.zeros((self.Nt, self.Nx, self.Nx))
        self.ZFi = np.zeros((self.Nt, self.Nx, self.Ny))
        self.K = np.zeros((self.Nt, self.Nx, self.Ny))
        self.G = np.zeros((self.Nt, self.Nx, self.Nx))
        self.log_like = 0.0

        for tt in range(self.Nt):
            self.x_pred[tt, :] = x_pred_t
            self.P_pred[tt, :, :] = P_pred_t

            ix_t = self.ix[tt, :]
            Z_t = self.ssm.Z[ix_t, :]
            err_t = y_til[tt, ix_t] - np.dot(Z_t, x_pred_t)

            H_t = self.ssm.H[ix_t, :][:, ix_t]
            PZ = np.dot(P_pred_t, Z_t.T)
            F_t = np.dot(Z_t, PZ) + H_t

            if np.any(ix_t):
                try:
                    self.log_like += mvn.logpdf(
                        err_t, mean=np.zeros(np.sum(ix_t)), cov=F_t
                    )
                except Exception:
                    self.log_like = -1e10
                    return None

            ZFi_t = nm.rsolve(Z_t.T, F_t)
            AP_t = np.dot(self.ssm.A, P_pred_t)
            K_t = np.dot(AP_t, ZFi_t)
            G_t = self.ssm.A - np.dot(K_t, Z_t)

            x_pred_t = np.dot(self.ssm.A, x_pred_t) + np.dot(K_t, err_t)
            P_pred_t = np.dot(AP_t, G_t.T) + self.ssm.RQR

            self.ix[tt, :] = ix_t
            self.err[tt, ix_t] = err_t
            self.ZFi[tt, :, ix_t] = ZFi_t.T
            self.K[tt, :, ix_t] = K_t.T
            self.G[tt, :, :] = G_t

        return None

    def disturbance_smoother(self):
        """Run the disturbance smoother to compute the auxiliary variable r."""
        self.r = np.zeros((self.Nt, self.Nx))
        r_t = np.zeros(self.Nx)

        for tt in range(self.Nt - 1, -1, -1):
            ix_t = self.ix[tt, :]
            r_t = np.dot(self.ZFi[tt, :, ix_t].T, self.err[tt, ix_t]) + np.dot(
                self.G[tt, :, :].T, r_t
            )
            self.r[tt, :] = r_t

        return None

    def state_smoother(self, disturbance_smooth=False):
        """Compute smoothed state estimates."""
        if (self.r is None) or disturbance_smooth:
            self.disturbance_smoother()

        self.x_smooth = np.zeros((self.Nt, self.Nx))
        self.x_smooth[0, :] = self.x_init + np.dot(self.P_init, self.r[0, :])

        for tt in range(1, self.Nt):
            self.x_smooth[tt, :] = np.dot(
                self.ssm.A, self.x_smooth[tt - 1, :]
            ) + np.dot(self.ssm.RQR, self.r[tt, :])

        return None

    def shock_smoother(self, disturbance_smooth=False):
        """Compute smoothed structural shock estimates."""
        if (self.r is None) or disturbance_smooth:
            self.disturbance_smoother()

        self.shocks_smooth = self.r[1:, :] @ self.ssm.QR.T
        return None

    def meas_err_smoother(self, disturbance_smooth=False):
        """Compute smoothed measurement error estimates."""
        if (self.r is None) or disturbance_smooth:
            self.disturbance_smoother()

        self.meas_err_smooth = init_to_val((self.Nt, self.Ny), np.nan)

        for tt in range(self.Nt):
            ix_t = self.ix[tt, :]
            H_t = self.ssm.H[ix_t, :][:, ix_t]
            Z_t = self.ssm.Z[ix_t, :]
            F_t = Z_t @ (self.P_pred[tt, :, :] @ Z_t.T)

            HFi_t = nm.rsolve(H_t, F_t)

            self.meas_err_smooth[tt, ix_t] = (
                HFi_t @ self.err[tt, ix_t] - (H_t @ self.K[tt, :, ix_t]) @ self.r[tt, :]
            )

        return None

    def draw_states(self, draw_shocks=False, draw_meas_err=False):
        """Draw a sample of states using the simulation smoother."""
        shocks = self.ssm.draw_shocks(self.Nt - 1)
        meas_err = self.ssm.draw_meas_err(self.Nt)
        x_1 = self.x_init + np.linalg.cholesky(self.P_init) @ np.random.randn(self.P_init.shape[0])

        y_plus, x_plus = self.ssm.simulate(
            x_1, shocks=shocks, meas_err=meas_err, ix=self.ix, use_b=False
        )

        y_star = self.y_til - y_plus

        self.kalman_filter(x_init=self.x_init, P_init=self.P_init, y_til=y_star)
        self.state_smoother(disturbance_smooth=True)

        self.state_draw = x_plus + self.x_smooth

        if draw_shocks:
            self.shock_smoother()
            self.shock_draw = shocks + self.shocks_smooth

        if draw_meas_err:
            self.meas_err_smoother()
            self.meas_err_draw = meas_err + self.meas_err_smooth

        return None

    def get_shock_components(self, redraw_shocks=False, start_ix=0):
        """Decompose the state path into contributions from structural shocks."""
        if redraw_shocks or (self.shock_draw is None) or (self.state_draw is None):
            self.draw_states(draw_shocks=True)

        return self.ssm.decompose_by_shock(
            self.shock_draw, self.state_draw, start_ix=start_ix
        )
