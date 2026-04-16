"""Tests for estimation/prior.py."""

import numpy as np
import pytest
from scipy import stats as st

from equilibrium.estimation.prior import (
    BETA,
    GAMMA,
    INV_GAMMA,
    NORM,
    TRUNC_NORM,
    Prior,
    get_prior,
)

# ---------------------------------------------------------------------------
# get_prior
# ---------------------------------------------------------------------------


def test_get_prior_none_returns_none():
    assert get_prior(None) is None


def test_get_prior_string_aliases():
    for name in ("beta", "gamma", "inv_gamma", "norm", "trunc_norm"):
        dist = get_prior(name, mean=0.5, sd=0.1)
        assert dist is not None


def test_get_prior_int_constants():
    for code, mean, sd in [
        (BETA, 0.5, 0.1),
        (GAMMA, 2.0, 0.5),
        (INV_GAMMA, 2.0, 0.5),
        (NORM, 0.0, 1.0),
        (TRUNC_NORM, 0.5, 0.1),
    ]:
        dist = get_prior(code, mean=mean, sd=sd)
        assert dist is not None


def test_get_prior_invalid_string_raises():
    with pytest.raises(ValueError, match="Unknown prior type"):
        get_prior("not_a_prior", mean=0.0, sd=1.0)


def test_get_prior_invalid_type_raises():
    with pytest.raises(TypeError):
        get_prior(3.14, mean=0.0, sd=1.0)


def test_get_prior_requires_mean_and_sd():
    with pytest.raises(ValueError, match="mean and sd"):
        get_prior("norm", mean=0.0, sd=None)
    with pytest.raises(ValueError, match="mean and sd"):
        get_prior(NORM, mean=None, sd=1.0)


def test_get_prior_beta_logpdf_finite():
    dist = get_prior(BETA, mean=0.5, sd=0.1)
    assert np.isfinite(dist.logpdf(0.5))


def test_get_prior_gamma_logpdf_finite():
    dist = get_prior(GAMMA, mean=2.0, sd=0.5)
    assert np.isfinite(dist.logpdf(2.0))


def test_get_prior_inv_gamma_logpdf_finite():
    dist = get_prior(INV_GAMMA, mean=2.0, sd=0.5)
    assert np.isfinite(dist.logpdf(2.0))


def test_get_prior_norm_matches_scipy():
    mean, sd = 1.0, 0.5
    dist = get_prior(NORM, mean=mean, sd=sd)
    expected = st.norm(loc=mean, scale=sd).logpdf(1.2)
    np.testing.assert_allclose(dist.logpdf(1.2), expected)


def test_get_prior_trunc_norm_in_support():
    dist = get_prior(TRUNC_NORM, mean=0.5, sd=0.2)
    # Must be positive in (0, 1)
    assert dist.logpdf(0.5) > -np.inf
    # Must be -inf outside support
    assert not np.isfinite(dist.logpdf(-0.1))
    assert not np.isfinite(dist.logpdf(1.1))


# ---------------------------------------------------------------------------
# Prior class
# ---------------------------------------------------------------------------


def test_prior_names_auto_generated():
    p = Prior()
    p.add(NORM, mean=0.0, sd=1.0)
    p.add(NORM, mean=0.0, sd=1.0)
    assert p.names == ["param1", "param2"]


def test_prior_names_explicit():
    p = Prior()
    p.add(NORM, name="alpha", mean=0.0, sd=1.0)
    assert p.names == ["alpha"]
    assert p.non_flat_names == ["alpha"]


def test_prior_flat_component_not_in_non_flat_names():
    p = Prior()
    p.add(NORM, name="a", mean=0.0, sd=1.0)
    p.add(None, name="b")
    assert "b" not in p.non_flat_names
    assert "b" in p.names


def test_prior_logpdf_finite():
    p = Prior()
    p.add(NORM, mean=0.0, sd=1.0, name="a")
    p.add(GAMMA, mean=2.0, sd=0.5, name="b")
    lp = p.logpdf(np.array([0.2, 1.5]))
    assert np.isfinite(lp)


def test_prior_logpdf_ignores_flat_component():
    p = Prior()
    p.add(NORM, mean=0.0, sd=1.0)
    p.add(None)
    lp = p.logpdf(np.array([0.1, 999.0]))
    assert np.isfinite(lp)


def test_prior_logpdf_all_flat_returns_zero():
    p = Prior()
    p.add(None)
    p.add(None)
    assert p.logpdf(np.array([1.0, 2.0])) == 0.0


def test_prior_sample_shape():
    p = Prior()
    p.add(NORM, mean=0.0, sd=1.0, name="a")
    p.add(GAMMA, mean=2.0, sd=0.5, name="b")
    draws = p.sample(10)
    assert draws.shape == (2, 10)


def test_prior_sample_raises_with_flat_component():
    p = Prior()
    p.add(NORM, mean=0.0, sd=1.0)
    p.add(None)
    with pytest.raises(ValueError, match="flat"):
        p.sample(5)
