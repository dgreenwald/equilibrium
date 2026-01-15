#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:43:08 2022

@author: dan
"""

import numpy as np
from scipy.linalg import ordqz

# from py_tools.aim import AimObj
# from py_tools.utilities import tic, toc


def split_four_ways(X, n_pre):
    """
    Split a matrix into four blocks based on predetermined variable count.

    Parameters
    ----------
    X : array_like
        Matrix to be split.
    n_pre : int
        Number of rows/columns for the upper-left block.

    Returns
    -------
    X11, X12, X21, X22 : array_like
        Four blocks of the original matrix.
    """
    X11 = X[:n_pre, :n_pre]
    X12 = X[:n_pre, n_pre:]
    X21 = X[n_pre:, :n_pre]
    X22 = X[n_pre:, n_pre:]

    return X11, X12, X21, X22


def solve_klein(A, B, C, Phi, n_pre):
    """
    Solve linearized DSGE model using Klein (2000) method.

    This implements the generalized Schur decomposition approach for solving
    linear rational expectations models with predetermined and non-predetermined
    variables.

    Parameters
    ----------
    A : array_like
        Coefficient matrix on current period variables.
    B : array_like
        Coefficient matrix on lagged variables.
    C : array_like
        Coefficient matrix on exogenous shocks.
    Phi : array_like
        Transition matrix for exogenous processes.
    n_pre : int
        Number of predetermined (state) variables.

    Returns
    -------
    G_x : array_like
        Policy function for predetermined variables.
    H_x : array_like
        State transition matrix for predetermined variables.
    G_z : array_like
        Impact of shocks on jump variables.
    H_z : array_like
        Impact of shocks on state variables.
    """
    n_uns = A.shape[0] - n_pre
    n_exog = Phi.shape[0]

    # Generalized Schur decomposition, sorted by outside unit circle
    S, T, alp, bet, Q, Z = ordqz(A, B, sort="ouc", output="complex")

    Qstar = np.asmatrix(Q).H
    Q1 = Qstar[:n_pre, :]
    Q2 = Qstar[n_pre:, :]

    Z11, Z12, Z21, Z22 = split_four_ways(Z, n_pre)
    S11, S12, _, S22 = split_four_ways(S, n_pre)
    T11, T12, _, T22 = split_four_ways(T, n_pre)

    Z11_inv = np.linalg.inv(Z11)
    S11_inv_T11 = np.linalg.solve(S11, T11)

    # Policy functions for predetermined variables
    G_x = (Z21 @ Z11_inv).real
    H_x = (Z11 @ S11_inv_T11 @ Z11_inv).real

    # Solve for impact of exogenous shocks
    PhiST = np.kron(Phi.T, S22) - np.kron(np.eye(n_exog), T22)
    q2C = np.asarray(Q2 @ C).flatten(order="F")
    M = np.linalg.solve(PhiST, q2C).reshape((n_uns, n_exog), order="F")

    Z11_inv_Z12 = Z11_inv @ Z12

    G_z = ((Z22 - Z21 @ Z11_inv_Z12) @ M).real
    H_z = (
        -Z11 @ S11_inv_T11 @ Z11_inv_Z12 @ M
        + Z11 @ np.linalg.solve(S11, T12 @ M - S12 @ M @ Phi + Q1 @ C)
        + Z12 @ M @ Phi
    ).real

    return G_x, H_x, G_z, H_z


def solve_aim(H, nlead=1, tol=1e-10):
    """
    Solve linear rational expectations model using Anderson-Moore AIM algorithm.

    Parameters
    ----------
    H : array_like
        Structural form matrix with all leads and lags.
    nlead : int, optional
        Number of lead periods. Default is 1.
    tol : float, optional
        Numerical tolerance for detecting zero rows/columns. Default is 1e-10.

    Returns
    -------
    AimObj
        Solver object containing the reduced form solution.
    """
    aim_obj = AimObj(H, nlead=nlead, tol=tol)
    aim_obj.solve()
    return aim_obj


class AimObj:
    """
    Solver for AIM (Anderson-Moore Algorithm) system.

    This class implements the Anderson-Moore algorithm for solving linear
    rational expectations models. The algorithm transforms the structural
    form into a reduced form through a series of QR decompositions and
    eigenvalue analysis.
    """

    def __init__(self, H, nlead=1, tol=1e-10):
        """
        Initialize AIM solver.

        Parameters
        ----------
        H : array_like
            Structural form coefficient matrix.
        nlead : int, optional
            Number of lead periods. Default is 1.
        tol : float, optional
            Numerical tolerance. Default is 1e-10.
        """
        self.H = H
        self.neq, self.hcols = H.shape
        self.periods = self.hcols // self.neq
        self.nlead = nlead
        self.nlag = int(self.periods - self.nlead) - 1
        self.tol = tol
        # self.eig_bnd = 1.0

        self.left = np.arange(self.hcols - self.neq)
        self.right = np.arange(self.hcols - self.neq, self.hcols)

        self.iz = 0
        self.zrows = int(self.neq * self.nlead)
        self.zcols = int(self.neq * (self.periods - 1))
        self.Q = np.zeros((self.zrows, self.zcols))

    def solve(self):
        """
        Execute the complete AIM solution algorithm.

        This orchestrates the solution steps: exact shift, numeric shift,
        companion matrix construction, eigensystem analysis, and reduced form.
        """
        self.exact_shift()
        self.numeric_shift()
        self.build_companion()
        self.eigensystem()
        self.reduced_form()

        return None

    def shift_right(self, x):
        """
        Shift matrix columns to the right by one block.

        Parameters
        ----------
        x : array_like
            Matrix to shift.

        Returns
        -------
        array_like
            Matrix with columns shifted right by neq positions.
        """
        x_shift = np.zeros(x.shape)
        x_shift[:, self.neq :] = x[:, : -self.neq]

        return x_shift

    def shuffle(self, ix):
        """
        Move rows with exact zeros in right block to auxiliary Q matrix.

        Parameters
        ----------
        ix : array_like
            Indices of rows to shuffle.
        """
        nz = len(ix)
        self.Q[self.iz : self.iz + nz, :] = self.H[ix, :][:, self.left]
        self.H[ix, :] = self.shift_right(self.H[ix, :])
        self.iz += nz

        return None

    def exact_shift(self):
        """
        Identify and shift rows with exact zeros in the right block.

        This step handles equations that have no current-dated variables,
        moving them to the Q matrix and shifting the remaining equations.
        """
        zerorows = np.sum(np.abs(self.H[:, self.right]), axis=1) < self.tol
        while np.any(zerorows) and self.iz < self.zrows:
            ix = np.arange(self.neq)[zerorows]
            self.shuffle(ix)
            zerorows = np.sum(np.abs(self.H[:, self.right]), axis=1) < self.tol

        return None

    def numeric_shift(self):
        """
        Use QR decomposition to identify numerically dependent rows.

        This step handles equations that are numerically linearly dependent
        on previously processed equations, using QR factorization to detect
        near-zero pivots in the R matrix.
        """
        q, r = np.linalg.qr(self.H[:, self.right])
        zerorows = np.abs(np.diag(r)) < self.tol
        while np.any(zerorows) and self.iz < self.zrows:
            ix = np.arange(self.neq)[zerorows]
            self.H = np.dot(q.T, self.H)
            self.shuffle(ix)
            q, r = np.linalg.qr(self.H[:, self.right])
            zerorows = np.abs(np.diag(r)) < self.tol

        return None

    def build_companion(self):
        """
        Construct companion form matrix from reduced H matrix.

        This builds the first-order companion matrix representation and
        removes inessential lags (variables that don't affect the dynamics).
        """
        self.A = np.zeros((self.zcols, self.zcols))
        if self.zcols > self.neq:
            self.A[: -self.neq, self.neq :] = np.eye(self.zcols - self.neq)

        Gam = -np.linalg.solve(self.H[:, self.right], self.H[:, self.left])
        self.A[-self.neq :, :] = Gam

        # Delete inessential lags
        self.js = np.arange(self.zcols)
        drop = np.sum(np.abs(self.A), axis=0) < self.tol

        while np.any(drop):
            self.A = self.A[:, ~drop][~drop, :]
            self.js = self.js[~drop]
            drop = np.sum(np.abs(self.A), axis=0) < self.tol

        return None

    def eigensystem(self):
        """
        Compute eigenvectors of companion matrix and fill Q matrix.

        This step computes the eigenvectors of the companion matrix and
        uses them to complete the Q matrix needed for the reduced form solution.
        Eigenvectors are sorted by eigenvalue magnitude.
        """
        vals, vecs = np.linalg.eig(self.A.T)
        ix = np.flipud(np.argsort(np.abs(vals)))

        sorted_vecs = vecs[:, ix]

        self.Q[self.iz :, self.js] = sorted_vecs[:, : (self.zrows - self.iz)].T

        return None

    def reduced_form(self):
        """
        Compute the reduced form policy function from Q matrix.

        This final step solves for Z and extracts the matrix B which defines
        the reduced form representation of the model dynamics.
        """
        self.Z = -np.linalg.solve(
            self.Q[:, self.zcols - self.zrows :], self.Q[:, : self.zcols - self.zrows]
        )
        self.B = self.Z[: self.neq, : self.neq * self.nlag]

        return None
