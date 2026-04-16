"""Tests for estimation/mcmc.py."""

import numpy as np
import pytest

from equilibrium.estimation.mcmc import (
    RWMC,
    MonteCarlo,
    adapt_jump_scale,
    check_bounds,
    importance_sample,
    metropolis_step,
    numerical_to_bool_blocks,
    partition_C,
    randomize_blocks,
    rwmh,
)
from equilibrium.estimation.prior import Prior

# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def test_adapt_jump_scale_range():
    lo = adapt_jump_scale(0.0, adapt_sens=16.0, adapt_target=0.25, adapt_range=0.1)
    hi = adapt_jump_scale(1.0, adapt_sens=16.0, adapt_target=0.25, adapt_range=0.1)
    assert 0.95 <= lo <= 1.05
    assert 0.95 <= hi <= 1.05
    assert hi > lo


def test_adapt_jump_scale_at_target():
    # At the target rate the scale factor should be exactly 1.0
    val = adapt_jump_scale(0.25, adapt_sens=16.0, adapt_target=0.25, adapt_range=0.1)
    assert abs(val - 1.0) < 1e-10


def test_randomize_blocks_covers_all_indices():
    blocks = randomize_blocks(nx=7, nblock=3)
    assert len(blocks) == 3
    covered = np.zeros(7, dtype=int)
    for block in blocks:
        covered += block.astype(int)
    np.testing.assert_array_equal(covered, np.ones(7, dtype=int))


def test_numerical_to_bool_blocks():
    bool_blocks = numerical_to_bool_blocks([np.array([0, 2]), np.array([1, 3])], nx=4)
    assert len(bool_blocks) == 2
    np.testing.assert_array_equal(bool_blocks[0], [True, False, True, False])
    np.testing.assert_array_equal(bool_blocks[1], [False, True, False, True])


def test_partition_C():
    C = np.arange(16, dtype=float).reshape(4, 4)
    blocks = numerical_to_bool_blocks([np.array([0, 2]), np.array([1, 3])], nx=4)
    C_list = partition_C(C, blocks)
    assert len(C_list) == 2
    assert C_list[0].shape == (2, 2)


def test_check_bounds_pass_and_fail():
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 2.0])
    assert check_bounds(np.array([0.0, 1.0]), lb, ub)
    assert not check_bounds(np.array([2.0, 1.0]), lb, ub)
    assert not check_bounds(np.array([0.0, -0.1]), lb, ub)


def test_metropolis_step_force_reject():
    def f(z):
        return -np.sum(z * z)

    x = np.array([0.0])
    x_try = np.array([1.0])
    x_new, post_new, acc = metropolis_step(f, x, x_try, post=f(x), log_u=0.0)
    np.testing.assert_array_equal(x_new, x)
    assert acc is False


def test_metropolis_step_force_accept():
    def f(z):
        return -np.sum(z * z)

    x = np.array([0.0])
    x_try = np.array([1.0])
    x_new, post_new, acc = metropolis_step(f, x, x_try, post=f(x), log_u=-100.0)
    np.testing.assert_array_equal(x_new, x_try)
    assert acc is True


def test_importance_sample_shapes():
    from scipy.stats import multivariate_normal as mv

    def f(z):
        return -0.5 * np.sum(z * z)

    dist = mv(mean=np.zeros(2), cov=np.eye(2))

    draws, lw = importance_sample(f, dist, Nsim=1, Nx=2)
    assert draws.shape == (1, 2)
    assert lw.shape == (1,)

    draws, lw = importance_sample(f, dist, Nsim=5, Nx=2)
    assert draws.shape == (5, 2)
    assert lw.shape == (5,)


def test_rwmh_smoke():
    def post(z):
        return -0.5 * np.sum(z * z)

    x_store, p_store, acc = rwmh(post, np.array([0.0]), Nstep=4)
    assert x_store.shape == (4, 1)
    assert p_store.shape == (4,)
    assert 0.0 <= acc <= 1.0


def test_rwmh_two_blocks():
    def post(z):
        return -0.5 * np.sum(z * z)

    x0 = np.zeros(4)
    blocks = randomize_blocks(4, 2)
    x_store, p_store, acc = rwmh(post, x0, blocks=blocks, Nstep=6)
    assert x_store.shape == (6, 4)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# MonteCarlo class
# ---------------------------------------------------------------------------


def _make_mc(**kwargs):
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0)
    return MonteCarlo(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, **kwargs)


def test_montecarlo_bounds_default_to_inf():
    mc = _make_mc(Nx=2)
    assert np.all(np.isneginf(mc.lb))
    assert np.all(np.isposinf(mc.ub))


def test_montecarlo_bounds_from_bounds_dict():
    mc = MonteCarlo(
        log_like=lambda x: -0.5 * np.sum(x * x),
        names=["a", "b"],
        bounds_dict={"a": (0.0, 1.0), "b": (-np.inf, np.inf)},
    )
    assert mc.lb[0] == 0.0
    assert mc.ub[0] == 1.0
    assert np.isneginf(mc.lb[1])


def test_montecarlo_posterior_within_bounds():
    mc = _make_mc(lb=np.array([-1.0]), ub=np.array([1.0]))
    assert np.isfinite(mc.posterior(np.array([0.0])))


def test_montecarlo_posterior_outside_bounds():
    mc = _make_mc(lb=np.array([-1.0]), ub=np.array([1.0]))
    assert mc.posterior(np.array([2.0])) == -1e10


def test_montecarlo_find_mode():
    mc = _make_mc(Nx=1)
    mc.find_mode(np.array([0.5]), method="Nelder-Mead")
    assert np.isfinite(mc.post_mode)
    np.testing.assert_allclose(mc.x_mode, [0.0], atol=1e-4)


def test_montecarlo_find_mode_iterate_no_names():
    mc = _make_mc(Nx=1)
    mc.find_mode(np.array([0.2]), iterate=True, disp_iterate=True, method="Nelder-Mead")
    assert np.isfinite(mc.post_mode)


def test_montecarlo_compute_hessian():
    mc = _make_mc(Nx=2)
    mc.x_mode = np.zeros(2)
    mc.compute_hessian()
    assert mc.H is not None
    assert mc.H_inv is not None
    assert mc.CH_inv is not None
    assert mc.CH_inv.shape == (2, 2)


def test_montecarlo_importance_sample():
    mc = _make_mc(Nx=1)
    mc.x_mode = np.array([0.0])
    mc.H_inv = np.array([[1.0]])
    draws, lw, ess = mc.importance_sample(20)
    assert draws.shape == (20, 1)
    assert lw.shape == (20,)
    assert np.isfinite(ess)


def test_montecarlo_save_metadata(tmp_path):
    mc = MonteCarlo(
        log_like=lambda x: -0.5 * np.sum(x * x),
        Nx=2,
        model_label="test_model",
        estimation_label="test_run",
    )
    # Override out_dir to use tmp_path
    mc.out_dir = tmp_path
    mc.x_mode = np.array([0.1, 0.2])
    mc.post_mode = -0.5
    mc.H = np.eye(2)
    mc.H_inv = np.eye(2)
    mc.CH_inv = np.eye(2)

    mc.save_metadata()

    assert (tmp_path / "mode.npz").exists()
    assert (tmp_path / "hessian.npz").exists()

    mc2 = MonteCarlo(log_like=lambda x: 0.0, Nx=2)
    mc2.out_dir = tmp_path
    mc2.load_metadata()
    np.testing.assert_allclose(mc2.x_mode, mc.x_mode)
    assert abs(mc2.post_mode - mc.post_mode) < 1e-12
    np.testing.assert_allclose(mc2.CH_inv, mc.CH_inv)


def test_montecarlo_save_metadata_raises_without_labels():
    mc = MonteCarlo(log_like=lambda x: 0.0, Nx=1)
    mc.x_mode = np.array([0.0])
    mc.post_mode = 0.0
    with pytest.raises(ValueError):
        mc.save_metadata()


# ---------------------------------------------------------------------------
# RWMC class
# ---------------------------------------------------------------------------


def _make_rw(tmp_path=None, **kwargs):
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0, name="theta")
    kw = dict(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, Nx=1)
    kw.update(kwargs)
    rw = RWMC(**kw)
    if tmp_path is not None:
        rw.out_dir = tmp_path
    return rw


def test_rwmc_sample_without_out_dir_raises_on_n_save():
    rw = _make_rw()
    rw.initialize(x0=np.array([0.0]), C=np.eye(1))
    with pytest.raises(ValueError):
        rw.sample(Nsim=4, n_save=2, log=False)


def test_rwmc_sample_basic(tmp_path):
    rw = _make_rw(tmp_path=tmp_path)
    rw.x_mode = np.array([0.0])
    rw.CH_inv = np.eye(1)
    rw.initialize(stride=1)
    rw.sample(Nsim=10, log=False)
    assert rw.draws.shape == (10, 1)
    assert rw.post_sim.shape == (10,)
    assert 0.0 <= rw.acc_rate <= 1.0


def test_rwmc_save_and_load_chain(tmp_path):
    rw = _make_rw(tmp_path=tmp_path)
    rw.draws = np.random.randn(5, 1)
    rw.post_sim = np.random.randn(5)
    rw.acc_rate = 0.3
    rw.jump_scale = 0.5

    rw.save_chain(chain_no=0)
    assert (tmp_path / "chain0.npz").exists()

    rw2 = _make_rw(tmp_path=tmp_path)
    rw2.load_chain(chain_no=0)
    np.testing.assert_allclose(rw2.draws, rw.draws)
    np.testing.assert_allclose(rw2.post_sim, rw.post_sim)
    assert abs(rw2.acc_rate - rw.acc_rate) < 1e-12
    assert abs(rw2.jump_scale - rw.jump_scale) < 1e-12


def test_rwmc_load_chains_and_stack(tmp_path):
    for chain_no in (0, 1):
        rw = _make_rw(tmp_path=tmp_path)
        rw.draws = np.ones((4, 1)) * chain_no
        rw.post_sim = np.zeros(4)
        rw.acc_rate = 0.25
        rw.jump_scale = 0.5
        rw.save_chain(chain_no=chain_no)

    rw_all = _make_rw(tmp_path=tmp_path)
    rw_all.load_chains([0, 1])
    assert len(rw_all.draws_list) == 2

    draws_all, post_all = rw_all.stack_chains(burn_in=1)
    assert draws_all.shape == (6, 1)  # 3 draws per chain after burn-in
    assert post_all.shape == (6,)


def test_rwmc_run_all(tmp_path):
    rw = _make_rw(tmp_path=tmp_path)
    rw.run_all(
        np.array([0.1]),
        Nsim=10,
        mode_kwargs={"method": "Nelder-Mead"},
        init_kwargs={"stride": 1},
        sample_kwargs={"log": False},
    )
    assert rw.draws.shape == (10, 1)
    assert rw.post_sim.shape == (10,)
    assert (tmp_path / "mode.npz").exists()
    assert (tmp_path / "hessian.npz").exists()
    assert (tmp_path / "chain0.npz").exists()


def test_rwmc_n_save_writes_intermediate(tmp_path):
    rw = _make_rw(tmp_path=tmp_path)
    rw.x_mode = np.array([0.0])
    rw.CH_inv = np.eye(1)
    rw.initialize(stride=1)
    rw.sample(Nsim=10, n_save=5, log=False, chain_no=0)
    assert (tmp_path / "chain0.npz").exists()


def test_rwmc_n_retune_runs(tmp_path):
    rw = _make_rw(tmp_path=tmp_path)
    rw.x_mode = np.array([0.0])
    rw.CH_inv = np.eye(1)
    rw.initialize(stride=1)
    # Just confirm it doesn't raise
    rw.sample(Nsim=10, n_retune=5, log=False)
    assert 0.0 <= rw.jump_scale


def test_rwmc_merge_chains():
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0)

    def log_like(x):
        return -0.5 * np.sum(x * x)

    chain_a = RWMC(log_like=log_like, prior=prior, Nx=1)
    chain_a.draws = np.ones((5, 1))
    chain_a.post_sim = np.zeros(5)

    chain_b = RWMC(log_like=log_like, prior=prior, Nx=1)
    chain_b.draws = np.ones((5, 1)) * 2
    chain_b.post_sim = np.ones(5)

    merged = RWMC(rwmc_chains=[chain_a, chain_b], log_like=log_like, prior=prior, Nx=1)
    assert merged.draws.shape == (10, 1)
    assert merged.post_sim.shape == (10,)
