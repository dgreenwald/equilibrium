"""Random-walk Metropolis-Hastings MCMC sampler.

Ported from py_tools/bayesian/mcmc.py.  The SMC sampler and all MPI/parallel
paths are omitted (deferred).

IO changes vs. original:
- ``out_dir`` / ``suffix`` replaced by ``model_label`` / ``estimation_label``;
  output directory is ``settings.paths.save_dir / "estimation" / model_label /
  estimation_label`` and is created on first save.
- ``save_chain(chain_no)`` writes ``chain{N}.npz`` containing ``draws``,
  ``post_sim``, ``acc_rate``, and ``jump_scale``.
- ``save_metadata()`` writes ``mode.npz`` and ``hessian.npz``.
  config.json is written by ``estimation.io.save_estimation`` (Step 7).
- ``load_chain(chain_no)`` reads the corresponding ``chain{N}.npz``.
- ``save_list`` / ``load_list`` / ``save_all`` / ``load_all`` /
  ``save_item`` / ``load_item`` are removed.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.optimize as opt
from scipy.stats import multivariate_normal as mv

from . import numerical as nm
from .prior import Prior


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def adapt_jump_scale(acc_rate, adapt_sens, adapt_target, adapt_range):
    """Compute a multiplicative scaling factor for the MCMC jump scale.

    Uses a logistic function to map the acceptance rate to a scaling
    factor in the interval ``(1 - adapt_range/2, 1 + adapt_range/2)``.

    Parameters
    ----------
    acc_rate : float
        Current acceptance rate, in ``[0, 1]``.
    adapt_sens : float
        Sensitivity parameter controlling the steepness of the logistic curve.
    adapt_target : float
        Target acceptance rate (the midpoint of the logistic curve).
    adapt_range : float
        Total range of the scaling factor (half on each side of 1.0).

    Returns
    -------
    float
        Multiplicative scaling factor to apply to the jump scale.
    """
    e_term = np.exp(adapt_sens * (acc_rate - adapt_target))
    return (1.0 - 0.5 * adapt_range) + adapt_range * (e_term / (1.0 + e_term))


def randomize_blocks(nx, nblock):
    """Randomly partition *nx* indices into *nblock* roughly equal boolean blocks.

    Parameters
    ----------
    nx : int
        Total number of parameter indices.
    nblock : int
        Number of blocks to create.

    Returns
    -------
    list of ndarray of bool
        List of length *nblock*, where each element is a boolean mask of
        length *nx* indicating which indices belong to that block.
    """
    ix_all = np.random.permutation(nx)
    block_size = int(np.ceil(nx / nblock))

    blocks = []
    cutoffs = np.arange(0, nx, block_size)

    for ii in range(nblock):
        start_ix = cutoffs[ii]
        if ii < nblock - 1:
            end_ix = cutoffs[ii + 1]
        else:
            end_ix = nx
        blocks.append(ix_all[start_ix:end_ix])

    return numerical_to_bool_blocks(blocks, nx)


def partition_C(C, blocks):
    """Extract sub-matrices of *C* corresponding to each block.

    Parameters
    ----------
    C : ndarray of shape ``(nx, nx)``
        Full covariance (or Cholesky) matrix.
    blocks : list of ndarray of bool
        Boolean block masks as returned by :func:`randomize_blocks`.

    Returns
    -------
    list of ndarray
        List where element *i* is the sub-matrix of *C* restricted to
        the rows and columns indicated by ``blocks[i]``.
    """
    C_list = []
    for block in blocks:
        C_list.append(C[block, :][:, block])
    return C_list


def numerical_to_bool_blocks(blocks, nx):
    """Convert a list of numerical index arrays to boolean mask arrays.

    Parameters
    ----------
    blocks : list of array-like of int
        Each element contains the numerical (integer) indices that belong
        to that block.
    nx : int
        Total number of parameters (length of each output mask).

    Returns
    -------
    list of ndarray of bool
        List of boolean masks of length *nx*, one per block.
    """
    bool_blocks = []
    for block in blocks:
        this_block = np.zeros(nx, dtype=bool)
        this_block[block] = True
        bool_blocks.append(this_block)
    return bool_blocks


def check_bounds(x, lb, ub):
    """Check whether all elements of *x* lie within ``[lb, ub]``.

    Parameters
    ----------
    x : array-like
        Parameter vector to check.
    lb : array-like
        Element-wise lower bounds.
    ub : array-like
        Element-wise upper bounds.

    Returns
    -------
    bool
        ``True`` if ``lb[i] <= x[i] <= ub[i]`` for every *i*;
        ``False`` otherwise.
    """
    return np.all(x >= lb) and np.all(x <= ub)


def print_mesg(mesg, fid=None):
    """Print or write a message to a log file.

    Parameters
    ----------
    mesg : str
        Message to output.
    fid : file-like object, optional
        Open file handle.  If provided, the message is written there
        (with a trailing newline) and flushed.  If ``None``, the message
        is printed to stdout with ``flush=True``.
    """
    if fid is not None:
        fid.write(mesg + "\n")
        fid.flush()
    else:
        print(mesg, flush=True)


def metropolis_step(fcn, x, x_try, post=None, log_u=None, args=()):
    """Perform a single Metropolis-Hastings accept/reject step.

    Parameters
    ----------
    fcn : callable
        Log-posterior function with signature ``fcn(x, *args)``.
    x : ndarray
        Current parameter vector.
    x_try : ndarray
        Proposed parameter vector.
    post : float, optional
        Log-posterior at *x*.  Computed from *fcn* if not provided.
    log_u : float, optional
        Log of a uniform random draw, used for the accept/reject decision.
        Drawn from ``log(Uniform(0, 1))`` if not provided.
    args : tuple, optional
        Additional arguments passed to *fcn*.

    Returns
    -------
    x_new : ndarray
        Accepted parameter vector (either *x_try* or *x*).
    post_new : float
        Log-posterior at *x_new*.
    accepted : bool
        ``True`` if the proposal was accepted.
    """
    if post is None:
        post = fcn(x, *args)

    post_try = fcn(x_try, *args)

    if log_u is None:
        log_u = np.log(np.random.rand())

    if log_u < post_try - post:
        return (x_try, post_try, True)
    else:
        return (x, post, False)


def importance_sample(fcn, dist, Nsim, Nx, args=()):
    """Draw importance-weighted samples from a posterior (serial).

    Draws *Nsim* proposals from *dist* and evaluates *fcn* (the
    log-posterior) at each draw.

    Parameters
    ----------
    fcn : callable
        Log-posterior function with signature ``fcn(x, *args)``.
    dist : scipy.stats frozen distribution
        Proposal distribution with ``rvs`` and ``logpdf`` methods.
    Nsim : int
        Number of importance samples.
    Nx : int
        Dimensionality of the parameter space.
    args : tuple, optional
        Additional arguments forwarded to *fcn*.

    Returns
    -------
    draws : ndarray of shape ``(Nsim, Nx)``
        Importance-sample draws.
    log_weights : ndarray of shape ``(Nsim,)``
        Log importance weights (log-posterior minus log-proposal).
    """
    draws = np.atleast_2d(dist.rvs(Nsim))
    if draws.shape[0] != Nsim:
        draws = draws.reshape(Nsim, Nx)
    post = np.zeros(Nsim)
    for jj in range(Nsim):
        post[jj] = fcn(draws[jj, :], *args)

    p_proposal = dist.logpdf(draws)
    log_weights = post - p_proposal
    return draws, log_weights


def rwmh(
    posterior,
    x_init,
    jump_scale=1.0,
    C_list=None,
    Nstep=1,
    blocks=None,
    block_sizes=None,
    post_init=None,
    e=None,
    log_u=None,
    quiet=True,
):
    """Run a random-walk Metropolis-Hastings sampler for *Nstep* steps.

    Parameters
    ----------
    posterior : callable
        Log-posterior function with signature ``posterior(x)``.
    x_init : ndarray
        Starting parameter vector.
    jump_scale : float, optional
        Global scaling factor for the proposal covariance.  Default is ``1.0``.
    C_list : list of ndarray, optional
        Cholesky factors of the proposal covariance for each block.  If
        ``None``, identity matrices are used.
    Nstep : int, optional
        Number of MCMC steps.  Default is ``1``.
    blocks : list of ndarray of bool, optional
        Boolean block masks.  If ``None``, a single block covering all
        parameters is used.
    block_sizes : list of int, optional
        Number of parameters in each block.  Inferred from *blocks* if
        ``None``.
    post_init : float, optional
        Log-posterior at *x_init*.  Evaluated if ``None``.
    e : ndarray of shape ``(Nstep, Nx)``, optional
        Pre-drawn standard normal innovations.  Drawn if ``None``.
    log_u : ndarray of shape ``(Nstep, Nblock)``, optional
        Pre-drawn log-uniform acceptance thresholds.  Drawn if ``None``.
    quiet : bool, optional
        If ``False``, print *x*, *x_try*, and *post* at every step.
        Default is ``True``.

    Returns
    -------
    x_store : ndarray of shape ``(Nstep, Nx)``
        Accepted parameter vectors at each step.
    post_store : ndarray of shape ``(Nstep,)``
        Log-posterior values at each accepted parameter vector.
    acc_rate : float
        Fraction of proposals accepted, averaged over all steps and blocks.
    """
    Nx = len(x_init)

    if blocks is None:
        blocks = [np.ones(Nx, dtype=bool)]
    Nblock = len(blocks)

    if block_sizes is None:
        block_sizes = [np.sum(block) for block in blocks]

    if e is None:
        e = np.random.randn(Nstep, Nx)

    if C_list is None:
        C_list = [np.eye(block_size) for block_size in block_sizes]

    if log_u is None:
        log_u = np.log(np.random.rand(Nstep, Nblock))

    if post_init is None:
        post_init = posterior(x_init)

    x_store = np.zeros((Nstep, Nx))
    post_store = np.zeros(Nstep)
    acc_rate = 0

    x = x_init.copy()
    post = post_init

    for istep in range(Nstep):
        for iblock, block in enumerate(blocks):
            x_try = x.copy()
            x_try[block] += jump_scale * np.dot(C_list[iblock], e[istep, block])
            x, post, acc = metropolis_step(
                posterior, x, x_try, post=post, log_u=log_u[istep, iblock]
            )
            acc_rate += acc

            if not quiet:
                print("x: " + repr(x))
                print("x_try: " + repr(x_try))
                print("post: " + repr(post))

        x_store[istep, :] = x
        post_store[istep] = post

    acc_rate /= Nstep * Nblock

    return (x_store, post_store, acc_rate)


# ---------------------------------------------------------------------------
# MonteCarlo base class
# ---------------------------------------------------------------------------


class MonteCarlo:
    """Base class for Monte Carlo samplers.

    Handles parameter bounds, log-posterior evaluation, mode finding,
    Hessian computation, and file I/O.

    Parameters
    ----------
    log_like : callable, optional
        Log-likelihood function with signature ``log_like(vals, *args)``.
    prior : Prior, optional
        Bayesian prior.  Defaults to an empty (flat) :class:`Prior`.
    args : tuple, optional
        Extra arguments forwarded to *log_like*.
    lb : array-like, optional
        Element-wise lower bounds on parameters.
    ub : array-like, optional
        Element-wise upper bounds on parameters.
    names : list of str, optional
        Human-readable parameter names.
    bounds_dict : dict, optional
        Mapping from parameter name to ``(lb, ub)`` pair.  Used to fill
        in missing entries of *lb* / *ub* when *names* is provided.
    model_label : str, optional
        Label for the model; used to construct the output directory.
    estimation_label : str, optional
        Label for this estimation run; used to construct the output directory.
        Output directory is
        ``settings.paths.save_dir / "estimation" / model_label /
        estimation_label``.
    Nx : int, optional
        Number of parameters.  Inferred from *lb*, *ub*, or *names* when
        possible.

    Attributes
    ----------
    log_like : callable or None
    prior : Prior
    args : tuple
    names : list of str or None
    lb : ndarray or None
    ub : ndarray or None
    Nx : int or None
    x_mode : ndarray or None
        Parameter vector at the posterior mode (set by :meth:`find_mode`).
    post_mode : float or None
        Log-posterior at the mode.
    H : ndarray or None
        Negative Hessian of the log-posterior at the mode.
    H_inv : ndarray or None
        Pseudo-inverse of ``H``.
    CH_inv : ndarray or None
        Cholesky factor of ``H_inv``.
    model_label : str or None
    estimation_label : str or None
    out_dir : Path or None
        Computed output directory, or ``None`` if labels are not set.
    """

    def __init__(
        self,
        log_like=None,
        prior=None,
        args=(),
        lb=None,
        ub=None,
        names=None,
        bounds_dict=None,
        model_label=None,
        estimation_label=None,
        Nx=None,
    ):
        if bounds_dict is None:
            bounds_dict = {}

        self.log_like = log_like
        if prior is None:
            prior = Prior()
        self.prior = prior
        self.args = args
        self.names = names
        self.x_mode = None
        self.post_mode = None
        self.H = None
        self.H_inv = None
        self.CH_inv = None
        self.lb = None
        self.ub = None

        if lb is not None:
            self.Nx = len(lb)
        elif ub is not None:
            self.Nx = len(ub)
        elif names is not None:
            self.Nx = len(names)
        elif Nx is not None:
            self.Nx = Nx
        else:
            self.Nx = None

        if self.Nx is not None:
            lb_missing = lb is None
            ub_missing = ub is None

            if lb_missing:
                self.lb = -np.inf * np.ones(self.Nx)
            else:
                self.lb = np.array(lb, copy=True)

            if ub_missing:
                self.ub = np.inf * np.ones(self.Nx)
            else:
                self.ub = np.array(ub, copy=True)

            if self.names is not None and (lb_missing or ub_missing):
                for ii, name in enumerate(self.names):
                    lb_i, ub_i = bounds_dict.get(name, (-np.inf, np.inf))

                    if lb_i is None:
                        lb_i = -np.inf
                    if ub_i is None:
                        ub_i = np.inf

                    if lb_missing:
                        self.lb[ii] = lb_i
                    if ub_missing:
                        self.ub[ii] = ub_i

        self.model_label = model_label
        self.estimation_label = estimation_label

        if model_label is not None and estimation_label is not None:
            from ..settings import get_settings

            settings = get_settings()
            self.out_dir = (
                Path(settings.paths.save_dir)
                / "estimation"
                / model_label
                / estimation_label
            )
        else:
            self.out_dir = None

    # ------------------------------------------------------------------
    # Posterior evaluation
    # ------------------------------------------------------------------

    def posterior(self, params):
        """Evaluate the log-posterior for a parameter vector.

        Returns ``-1e+10`` when any bound is violated.

        Parameters
        ----------
        params : ndarray
            Parameter vector.

        Returns
        -------
        float
            Log-posterior value (log-likelihood + log-prior).
        """
        if (
            (self.lb is None)
            or (self.ub is None)
            or check_bounds(params.ravel(), self.lb, self.ub)
        ):
            return self.log_like(params, *self.args) + self.prior.logpdf(params)
        else:
            return -1e10

    def min_objfcn(self, unbdd_params):
        """Objective function for minimisation (negative log-posterior).

        Applies the inverse bound transform before evaluating the posterior,
        allowing unconstrained optimisation.

        Parameters
        ----------
        unbdd_params : ndarray
            Unbounded (transformed) parameter vector.

        Returns
        -------
        float
            Negative log-posterior at the corresponding bounded parameters.
        """
        params = self.bound_transform(unbdd_params, to_bdd=True)
        return -self.posterior(params)

    # ------------------------------------------------------------------
    # Mode finding
    # ------------------------------------------------------------------

    def find_mode(
        self,
        x0,
        tol=1e-8,
        basinhopping=False,
        method="bfgs",
        iterate=False,
        iter_tol=1e-6,
        disp_iterate=True,
        **kwargs,
    ):
        """Find the posterior mode via numerical optimisation.

        Minimises the negative log-posterior using
        :func:`scipy.optimize.minimize` (or
        :func:`~scipy.optimize.basinhopping`).  Sets :attr:`x_mode` and
        :attr:`post_mode` on completion.

        Parameters
        ----------
        x0 : ndarray
            Starting point for the optimiser.
        tol : float, optional
            Convergence tolerance.  Default is ``1e-8``.
        basinhopping : bool, optional
            If ``True``, use basin-hopping global optimisation.  Default is
            ``False``.
        method : str, optional
            Optimisation method passed to ``scipy.optimize``.  Default is
            ``'bfgs'``.
        iterate : bool, optional
            If ``True``, repeatedly optimise until improvement falls below
            *iter_tol*.  Default is ``False``.
        iter_tol : float, optional
            Convergence criterion for iterative mode-finding.  Default is
            ``1e-6``.
        disp_iterate : bool, optional
            If ``True`` (and *iterate* is ``True``), print progress at each
            iteration.  Default is ``True``.
        **kwargs
            Additional keyword arguments forwarded to the optimiser.

        Returns
        -------
        res : OptimizeResult
            Result object returned by the scipy optimiser.
        """
        x0 = x0.ravel()

        post_start = None
        done = False
        count = 0
        while not done:
            count += 1
            if iterate and (post_start is None):
                post_start = self.posterior(x0)

            x0_u = self.bound_transform(x0, to_bdd=False)
            if basinhopping:
                minimizer_kwargs = kwargs.get("minimizer_kwargs", {})
                if "method" not in minimizer_kwargs:
                    minimizer_kwargs["method"] = method
                res = opt.basinhopping(
                    self.min_objfcn, x0_u, minimizer_kwargs=minimizer_kwargs, **kwargs
                )
            else:
                res = opt.minimize(
                    self.min_objfcn, x0_u, method=method, tol=tol, **kwargs
                )

            self.x_mode = self.bound_transform(res.x, to_bdd=True)
            self.post_mode = -res.fun

            if iterate:
                if disp_iterate:
                    print(
                        "Iteration {0:d}: starting posterior = {1:g}, "
                        "ending posterior = {2:g}".format(
                            count, post_start, self.post_mode
                        ),
                        flush=True,
                    )
                    if self.names is not None:
                        these_params = {
                            self.names[ii]: self.x_mode[ii]
                            for ii in range(len(self.names))
                        }
                    else:
                        these_params = {
                            "param{:d}".format(ii): self.x_mode[ii]
                            for ii in range(len(self.x_mode))
                        }
                    print("Params: " + repr(these_params), flush=True)
                done = np.abs(self.post_mode - post_start) < iter_tol
                if not done:
                    x0 = self.x_mode
                    post_start = self.post_mode
            else:
                done = True

        return res

    def find_mode_de(self, bounds, **kwargs):
        """Find the posterior mode using differential evolution.

        Parameters
        ----------
        bounds : sequence of ``(min, max)`` pairs
            Bounds for each parameter.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`~scipy.optimize.differential_evolution`.

        Returns
        -------
        res : OptimizeResult
        """
        res = opt.differential_evolution(self.min_objfcn, bounds, **kwargs)
        self.x_mode = res.x
        self.post_mode = -res.fun
        return res

    # ------------------------------------------------------------------
    # Transforms and Hessian
    # ------------------------------------------------------------------

    def bound_transform(self, vals, *args, **kwargs):
        """Apply the bound transform to *vals* using this object's bounds.

        Parameters
        ----------
        vals : ndarray
            Parameter vector to transform.
        *args, **kwargs
            Forwarded to :func:`~equilibrium.estimation.numerical.bound_transform`.

        Returns
        -------
        ndarray
            Transformed parameter vector.
        """
        return nm.bound_transform(vals, self.lb, self.ub, *args, **kwargs)

    def compute_hessian(self, x0=None, cholesky=True, robust=True, **kwargs):
        """Compute the (negative) Hessian of the log-posterior at *x0*.

        Sets :attr:`H`, :attr:`H_inv`, and optionally :attr:`CH_inv`.

        Parameters
        ----------
        x0 : ndarray, optional
            Point at which to compute the Hessian.  Defaults to
            :attr:`x_mode`.
        cholesky : bool, optional
            If ``True`` (default), also compute and store
            ``CH_inv = chol(H_inv)``.
        robust : bool, optional
            If ``True`` (default) and *cholesky* is ``True``, use
            :func:`~equilibrium.estimation.numerical.robust_cholesky` to
            handle near-singular matrices.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`~equilibrium.estimation.numerical.hessian`.
        """
        if x0 is None:
            x0 = self.x_mode.copy()

        self.H = -nm.hessian(self.posterior, x0, **kwargs)
        self.H_inv = np.linalg.pinv(self.H)

        if cholesky:
            if robust:
                self.CH_inv = nm.robust_cholesky(self.H_inv)
            else:
                self.CH_inv = np.linalg.cholesky(self.H_inv)

    def set_CH_inv(self, CH_inv):
        """Manually set the Cholesky factor of the inverse Hessian.

        Parameters
        ----------
        CH_inv : ndarray
            Lower-triangular Cholesky factor to store as :attr:`CH_inv`.
        """
        self.CH_inv = CH_inv

    # ------------------------------------------------------------------
    # Metropolis helper
    # ------------------------------------------------------------------

    def metro(self, x, post, x_try, **kwargs):
        """Perform a single Metropolis-Hastings step using the stored posterior.

        Parameters
        ----------
        x : ndarray
            Current parameter vector.
        post : float
            Log-posterior at *x*.
        x_try : ndarray
            Proposed parameter vector.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`metropolis_step`.

        Returns
        -------
        x_new : ndarray
        post_new : float
        accepted : bool
        """
        return metropolis_step(self.posterior, x, x_try, post=post, **kwargs)

    # ------------------------------------------------------------------
    # Importance sampling
    # ------------------------------------------------------------------

    def importance_sample(self, Nsim, resample=True, offset=None):
        """Run importance sampling centred on the posterior mode.

        Uses a multivariate normal proposal with covariance :attr:`H_inv`
        (optionally inflated by *offset*).

        Parameters
        ----------
        Nsim : int
            Number of importance samples.
        resample : bool, optional
            If ``True`` (default), resample with replacement using the
            importance weights so all returned log-weights are zero.
        offset : float, optional
            If provided, add ``offset * I`` to the proposal covariance.

        Returns
        -------
        draws : ndarray of shape ``(Nsim, Nx)``
        log_weights : ndarray of shape ``(Nsim,)``
        ess : float
            Effective sample size before resampling.
        """
        assert self.x_mode is not None
        assert self.H_inv is not None

        cov = self.H_inv.copy()
        if offset is not None:
            cov += np.diag(offset * np.ones(self.Nx))

        dist = mv(mean=self.x_mode, cov=cov)
        draws, log_weights = importance_sample(self.posterior, dist, Nsim, self.Nx)

        probs = np.exp(log_weights - np.amax(log_weights))
        probs /= np.sum(probs)

        W_til = Nsim * probs
        ess = len(W_til) / np.mean(W_til**2)

        if resample:
            probs = np.exp(log_weights - np.amax(log_weights))
            probs /= np.sum(probs)
            ix = np.random.choice(Nsim, size=Nsim, p=probs)
            draws = draws[ix, :]
            log_weights = np.zeros(log_weights.shape)

        return draws, log_weights, ess

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def open_log(self, filename):
        """Open a text log file for recording sampler output.

        Sets ``self.fid`` to the open file handle, or to ``None`` if
        :attr:`out_dir` is not set.

        Parameters
        ----------
        filename : str
            Name of the log file (e.g. ``'log_chain0.txt'``).
        """
        if self.out_dir is None:
            self.fid = None
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fid = open(self.out_dir / filename, "wt")

    def print_log(self, mesg):
        """Write *mesg* to the log file (or stdout if no file is open).

        Parameters
        ----------
        mesg : str
            Message to record.
        """
        print_mesg(mesg, fid=self.fid)

    def close_log(self):
        """Close the log file handle if one is open."""
        if self.fid is not None:
            self.fid.close()

    # ------------------------------------------------------------------
    # Metadata I/O  (config.json is written by estimation.io — Step 7)
    # ------------------------------------------------------------------

    def save_metadata(self):
        """Save mode and Hessian arrays to the output directory.

        Writes:
        - ``mode.npz``    — ``x_mode``, ``post_mode``
        - ``hessian.npz`` — ``H``, ``H_inv``, ``CH_inv`` (skips ``None`` arrays)

        The full ``config.json`` (param names, priors, bounds, data shape)
        is written by ``estimation.io.save_estimation`` (Step 7).
        """
        if self.out_dir is None:
            raise ValueError("Cannot save: model_label and estimation_label must be set.")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.x_mode is not None:
            np.savez(
                self.out_dir / "mode.npz",
                x_mode=self.x_mode,
                post_mode=np.atleast_1d(self.post_mode),
            )

        hess_arrays = {
            k: getattr(self, k)
            for k in ("H", "H_inv", "CH_inv")
            if getattr(self, k) is not None
        }
        if hess_arrays:
            np.savez(self.out_dir / "hessian.npz", **hess_arrays)

    def load_metadata(self):
        """Load mode and Hessian arrays from the output directory."""
        if self.out_dir is None:
            raise ValueError("Cannot load: model_label and estimation_label must be set.")

        mode_path = self.out_dir / "mode.npz"
        if mode_path.exists():
            data = np.load(mode_path)
            self.x_mode = data["x_mode"]
            self.post_mode = float(data["post_mode"])

        hess_path = self.out_dir / "hessian.npz"
        if hess_path.exists():
            data = np.load(hess_path)
            for k in ("H", "H_inv", "CH_inv"):
                if k in data:
                    setattr(self, k, data[k])


# ---------------------------------------------------------------------------
# RWMC
# ---------------------------------------------------------------------------


class RWMC(MonteCarlo):
    """Random-Walk Markov Chain Monte Carlo sampler.

    Inherits from :class:`MonteCarlo` and adds a Metropolis-Hastings
    random-walk sampler with adaptive jump scaling and optional block
    updates.

    Parameters
    ----------
    rwmc_chains : list of RWMC, optional
        If provided, merge the draws and posteriors from multiple completed
        chains into this object.
    *args, **kwargs
        Forwarded to :class:`MonteCarlo`.

    Attributes
    ----------
    draws : ndarray or None
        Stored MCMC draws, shape ``(Nsim, Nx)``.
    post_sim : ndarray or None
        Log-posterior at each draw, shape ``(Nsim,)``.
    acc_rate : float or None
        Overall acceptance rate.
    jump_scale : float or None
        Jump scale at the end of the last sampling run.
    """

    def __init__(self, rwmc_chains=None, *args, **kwargs):
        MonteCarlo.__init__(self, *args, **kwargs)

        self.draws = None
        self.post_sim = None
        self.acc_rate = None
        self.jump_scale = None

        if rwmc_chains is not None:
            self.post_sim = np.hstack([chain.post_sim for chain in rwmc_chains])
            self.draws = np.vstack([chain.draws for chain in rwmc_chains])

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(
        self,
        x0=None,
        jump_scale=None,
        jump_mult=1.0,
        stride=1,
        C=None,
        C_list=None,
        blocks="none",
        bool_blocks=False,
        n_blocks=None,
        adapt_sens=16.0,
        adapt_range=0.1,
        adapt_target=0.25,
    ):
        """Set up the sampler before calling :meth:`sample`.

        Parameters
        ----------
        x0 : ndarray, optional
            Starting point.  Defaults to :attr:`x_mode`.
        jump_scale : float, optional
            Initial proposal scaling factor.  If ``None``, defaults to
            ``jump_mult * 2.4 / sqrt(Nx)``.
        jump_mult : float, optional
            Multiplier for the default jump scale.  Default is ``1.0``.
        stride : int, optional
            Only every *stride*-th step is recorded.  Default is ``1``.
        C : ndarray, optional
            Cholesky factor of the proposal covariance (shared across blocks).
            Defaults to :attr:`CH_inv`.
        C_list : list of ndarray, optional
            Per-block Cholesky factors.  Overrides *C* when provided.
        blocks : {``'none'``, ``'random'``, list}, optional
            Block structure.  ``'none'`` uses a single block.  ``'random'``
            requires *n_blocks*.  Default is ``'none'``.
        bool_blocks : bool, optional
            If ``True``, the *blocks* list contains boolean masks; otherwise
            numerical index arrays.  Default is ``False``.
        n_blocks : int, optional
            Number of random blocks.  Required when ``blocks='random'``.
        adapt_sens : float, optional
            Sensitivity of the adaptive jump-scale update.  Default is ``16.0``.
        adapt_range : float, optional
            Range of the adaptive scaling factor.  Default is ``0.1``.
        adapt_target : float, optional
            Target acceptance rate for adaptation.  Default is ``0.25``.
        """
        self.stride = stride

        if x0 is None:
            self.x0 = self.x_mode
        else:
            self.x0 = x0

        if self.Nx is None:
            self.Nx = len(self.x0)
        else:
            assert self.Nx == len(self.x0)

        if jump_scale is None:
            self.jump_scale = jump_mult * 2.4 / np.sqrt(self.Nx)
        else:
            self.jump_scale = jump_scale

        if blocks == "none":
            self.blocks = [np.ones(self.Nx, dtype=bool)]
        elif blocks == "random":
            assert n_blocks is not None
            self.blocks = randomize_blocks(self.Nx, n_blocks)
        elif bool_blocks:
            self.blocks = blocks
        else:
            self.blocks = numerical_to_bool_blocks(blocks, self.Nx)

        assert (sum([np.sum(block) for block in self.blocks])) == len(self.x0)

        if C_list is not None:
            self.C_list = C_list
        else:
            if C is None:
                C = self.CH_inv.copy()
            self.C_list = []
            for block in self.blocks:
                self.C_list.append(C[block, :][:, block])

        self.adapt_sens = adapt_sens
        self.adapt_range = adapt_range
        self.adapt_target = adapt_target

        self.Nblock = len(self.blocks)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        Nsim,
        n_print=None,
        n_recov=None,
        n_save=None,
        log=True,
        cov_offset=0.0,
        min_recov=0,
        n_retune=None,
        chain_no=0,
        *args,
        **kwargs,
    ):
        """Run the RWMC chain for *Nsim* draws.

        Parameters
        ----------
        Nsim : int
            Number of draws to store (after thinning by :attr:`stride`).
        n_print : int, optional
            Print progress every *n_print* stored draws.
        n_recov : int, optional
            Recompute the proposal covariance from the empirical sample
            covariance every *n_recov* stored draws.
        n_save : int, optional
            Save intermediate output every *n_save* stored draws.
            Requires :attr:`out_dir` to be set.
        log : bool, optional
            If ``True`` (default), write a text log file.
        cov_offset : float, optional
            Regularisation added to the diagonal of the empirical covariance
            during recomputation.  Default is ``0.0``.
        min_recov : int, optional
            Minimum stored draws before covariance recomputation starts.
            Default is ``0``.
        n_retune : int, optional
            Re-tune the jump scale every *n_retune* stored draws.
        chain_no : int, optional
            Chain index used in output filenames.  Default is ``0``.
        *args, **kwargs
            Accepted but unused (for API flexibility).
        """
        if (n_save is not None) and (self.out_dir is None):
            raise ValueError("RWMC.sample requires model_label/estimation_label when n_save is set.")

        self.Nsim = Nsim

        x = self.x0.copy()
        post = self.posterior(x)

        if self.Nx is None:
            self.Nx = len(x)

        if log:
            self.open_log(f"log_chain{chain_no}.txt")
        else:
            self.fid = None

        Nstep = Nsim * self.stride
        Ntot = Nstep * self.Nblock

        self.draws = np.zeros((self.Nsim, self.Nx))
        self.post_sim = np.zeros(self.Nsim)
        self.acc = 0

        acc_last_retune = 0
        istep_last_retune = 0

        e = [np.random.randn(Nstep, np.sum(block)) for block in self.blocks]
        log_u = np.log(np.random.rand(Nstep, self.Nblock))

        self.max_x = 1.0 * x
        self.max_post = 1.0 * post

        self.print_log("Jump scale is {}".format(self.jump_scale))

        for istep in range(Nstep):
            for iblock, block in enumerate(self.blocks):
                x_try = x.copy()
                x_try[block] += self.jump_scale * np.dot(
                    self.C_list[iblock], e[iblock][istep, :]
                )
                x, post, acc = self.metro(x, post, x_try, log_u=log_u[istep, iblock])

                if post > self.max_post:
                    self.max_post = 1.0 * post
                    self.max_x = 1.0 * x

                self.acc += acc

            if (istep + 1) % self.stride == 0:
                self.acc_rate = self.acc / ((istep + 1) * self.Nblock)
                jstep = (istep + 1) // self.stride - 1

                self.draws[jstep, :] = x
                self.post_sim[jstep] = post

                if n_print is not None:
                    if (jstep + 1) % n_print == 0:
                        self.print_log(
                            "Draw {0:d}. Acceptance rate: {1:4.3f}. Max posterior = {2:4.3f}".format(
                                jstep + 1, self.acc_rate, self.max_post
                            )
                        )

                if n_recov is not None:
                    if (jstep + 1 >= min_recov) and (
                        ((jstep + 1) - min_recov) % n_recov == 0
                    ):
                        self.print_log("Recomputing covariance")
                        for iblock, block in enumerate(self.blocks):
                            sample_cov = np.cov(
                                self.draws[: jstep + 1, block], rowvar=False
                            ) + cov_offset * np.eye(np.sum(block))
                            self.C_list[iblock] = np.linalg.cholesky(sample_cov)

                if n_retune is not None:
                    if (jstep + 1) % n_retune == 0:
                        acc_since_retune = self.acc - acc_last_retune
                        steps_since_retune = (istep + 1) - istep_last_retune
                        acc_rate_since_retune = acc_since_retune / (
                            steps_since_retune * self.Nblock
                        )

                        self.print_log(
                            "Acceptance rate for last {0:d} draws: {1:4.3f}".format(
                                steps_since_retune, acc_rate_since_retune
                            )
                        )
                        self.print_log(
                            "Retuning: old jump scale = {:7.6f}".format(self.jump_scale)
                        )
                        self.jump_scale *= adapt_jump_scale(
                            acc_rate_since_retune,
                            self.adapt_sens,
                            self.adapt_target,
                            self.adapt_range,
                        )
                        self.print_log(
                            "Retuning: new jump scale = {:7.6f}".format(self.jump_scale)
                        )

                        acc_last_retune = self.acc
                        istep_last_retune = istep

                if n_save is not None:
                    if (jstep + 1) % n_save == 0:
                        self.print_log("Saving intermediate output")
                        self.save_chain(chain_no=chain_no)

        self.acc_rate = self.acc / Ntot

        self.close_log()

    # ------------------------------------------------------------------
    # Chain I/O
    # ------------------------------------------------------------------

    def save_chain(self, chain_no=0):
        """Save draws, log-posteriors, acceptance rate, and jump scale for one chain.

        Writes ``chain{N}.npz`` containing ``draws``, ``post_sim``,
        ``acc_rate``, and ``jump_scale`` to :attr:`out_dir`.

        Parameters
        ----------
        chain_no : int, optional
            Chain index.  Default is ``0``.
        """
        if self.out_dir is None:
            raise ValueError("Cannot save: model_label and estimation_label must be set.")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.out_dir / f"chain{chain_no}.npz",
            draws=self.draws,
            post_sim=self.post_sim,
            acc_rate=np.atleast_1d(self.acc_rate),
            jump_scale=np.atleast_1d(self.jump_scale),
        )

    def load_chain(self, chain_no=0):
        """Load draws, log-posteriors, acceptance rate, and jump scale for one chain.

        Reads ``chain{N}.npz`` from :attr:`out_dir` and populates
        :attr:`draws`, :attr:`post_sim`, :attr:`acc_rate`, and
        :attr:`jump_scale`.

        Parameters
        ----------
        chain_no : int, optional
            Chain index.  Default is ``0``.
        """
        if self.out_dir is None:
            raise ValueError("Cannot load: model_label and estimation_label must be set.")

        path = self.out_dir / f"chain{chain_no}.npz"
        data = np.load(path)
        self.draws = data["draws"]
        self.post_sim = data["post_sim"]
        self.acc_rate = float(data["acc_rate"])
        self.jump_scale = float(data["jump_scale"])

    def load_chains(self, chains):
        """Load multiple chains and store them in list attributes.

        Populates :attr:`draws_list`, :attr:`post_sim_list`, and
        :attr:`acc_rate_list` from disk.

        Parameters
        ----------
        chains : iterable of int
            Chain indices to load.
        """
        self.draws_list = []
        self.post_sim_list = []
        self.acc_rate_list = []

        for chain_no in chains:
            self.load_chain(chain_no)
            self.draws_list.append(self.draws)
            self.post_sim_list.append(self.post_sim)
            self.acc_rate_list.append(self.acc_rate)

        self.draws = None
        self.post_sim = None
        self.acc_rate = None

    def stack_chains(self, burn_in=0, stride=1):
        """Concatenate draws and log-posteriors from all loaded chains.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial draws to discard from each chain.  Default is ``0``.
        stride : int, optional
            Keep every *stride*-th draw after burn-in.  Default is ``1``.

        Returns
        -------
        draws_all : ndarray of shape ``(total_draws, Nx)``
        post_sim_all : ndarray of shape ``(total_draws,)``
        """
        draws_all = np.vstack([draws[burn_in::stride, :] for draws in self.draws_list])
        post_sim_all = np.hstack(
            [post_sim[burn_in::stride] for post_sim in self.post_sim_list]
        )
        return draws_all, post_sim_all

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run_all(
        self,
        x0,
        Nsim,
        mode_kwargs=None,
        hess_kwargs=None,
        init_kwargs=None,
        sample_kwargs=None,
    ):
        """Find mode, compute Hessian, initialise, run chain, and save.

        Convenience method that chains :meth:`~MonteCarlo.find_mode`,
        :meth:`~MonteCarlo.compute_hessian`, :meth:`initialize`,
        :meth:`sample`, and (when :attr:`out_dir` is set)
        :meth:`~MonteCarlo.save_metadata` + :meth:`save_chain`.

        Parameters
        ----------
        x0 : ndarray
            Starting point for mode-finding.
        Nsim : int
            Number of MCMC draws to collect.
        mode_kwargs : dict, optional
        hess_kwargs : dict, optional
        init_kwargs : dict, optional
        sample_kwargs : dict, optional
        """
        if mode_kwargs is None:
            mode_kwargs = {}
        if hess_kwargs is None:
            hess_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        self.find_mode(x0, **mode_kwargs)
        self.compute_hessian(**hess_kwargs)
        self.initialize(**init_kwargs)
        chain_no = sample_kwargs.get("chain_no", 0)
        self.sample(Nsim, **sample_kwargs)

        if self.out_dir is not None:
            self.save_metadata()
            self.save_chain(chain_no=chain_no)
