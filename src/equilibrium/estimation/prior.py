"""Bayesian prior distributions for MCMC estimation.

Ported verbatim from py_tools/bayesian/prior.py.
"""

import numpy as np
from scipy import stats as st

BETA = 1
GAMMA = 2
INV_GAMMA = 3
NORM = 4
TRUNC_NORM = 5


def get_prior(prior_type, mean=None, sd=None):
    """Return a scipy frozen distribution for a named prior type.

    Parameters
    ----------
    prior_type : {None, int, str}
        Identifier for the prior family.  Accepted string values are
        ``'beta'``, ``'gamma'``, ``'inv_gamma'``, ``'norm'``, and
        ``'trunc_norm'``.  Integer constants ``BETA``, ``GAMMA``,
        ``INV_GAMMA``, ``NORM``, and ``TRUNC_NORM`` defined in this module
        can be used instead.  Pass ``None`` to obtain a flat (improper)
        prior represented by ``None``.
    mean : float, optional
        Prior mean.  Required when *prior_type* is not ``None``.
    sd : float, optional
        Prior standard deviation.  Required when *prior_type* is not ``None``.

    Returns
    -------
    dist : scipy.stats frozen distribution or None
        A frozen scipy distribution whose ``logpdf`` and ``rvs`` methods
        can be used for Bayesian inference, or ``None`` for a flat prior.

    Raises
    ------
    ValueError
        If *prior_type* is a string that is not one of the recognised names.
    TypeError
        If *prior_type* is not ``None``, an ``int``, or a ``str``.
    """

    prior_num_dict = {
        "beta": BETA,
        "gamma": GAMMA,
        "inv_gamma": INV_GAMMA,
        "norm": NORM,
        "trunc_norm": TRUNC_NORM,
    }

    if prior_type is None:
        return None
    elif isinstance(prior_type, int):
        prior_num = prior_type
    elif isinstance(prior_type, str):
        if prior_type not in prior_num_dict:
            raise ValueError("Unknown prior type: {}".format(prior_type))
        prior_num = prior_num_dict[prior_type]
    else:
        raise TypeError(
            "prior_type must be None, int, or str; got {}".format(
                type(prior_type).__name__
            )
        )

    if mean is None or sd is None:
        raise ValueError(
            "mean and sd must both be provided for prior_type={}".format(prior_type)
        )

    if prior_num == BETA:
        alp = (1.0 - mean) * ((mean / sd) ** 2) - mean
        bet = (1.0 - mean) * alp / mean
        return st.beta(alp, bet)
    elif prior_num == GAMMA:
        the = (sd**2) / mean
        k = mean / the
        return st.gamma(k, scale=the)
    elif prior_num == INV_GAMMA:
        alp = 2 + ((mean / sd) ** 2)
        bet = mean * (alp - 1)
        return st.invgamma(alp, scale=bet)
    elif prior_num == NORM:
        return st.norm(loc=mean, scale=sd)
    elif prior_num == TRUNC_NORM:
        a = (0.0 - mean) / sd
        b = (1.0 - mean) / sd
        return st.truncnorm(a, b, mean, sd)
    else:
        raise RuntimeError("Unexpected prior code")


class Prior:
    """Container for a collection of independent Bayesian prior distributions.

    Each component is added via :meth:`add` and can be a proper
    (non-flat) distribution or a flat (improper) prior represented by
    ``None``.

    Attributes
    ----------
    dists : list of scipy.stats frozen distributions or None
        Ordered list of component distributions.
    names : list of str
        Parameter names corresponding to each component.
    non_flat_names : list of str
        Names of parameters that have a non-flat prior.
    """

    def __init__(self):
        """Initialise an empty prior with no components."""
        self.dists = []
        self.names = []
        self.non_flat_names = []
        self._non_flat_indices = []

    def add(self, prior_type, name=None, *args, **kwargs):
        """Add a single prior component.

        Parameters
        ----------
        prior_type : {None, int, str}
            Prior family identifier passed to :func:`get_prior`.
        name : str, optional
            Human-readable name for the parameter.  If omitted a name of
            the form ``'param<n>'`` is generated automatically.
        *args, **kwargs
            Additional positional and keyword arguments forwarded to
            :func:`get_prior` (typically *mean* and *sd*).
        """
        this_prior = get_prior(prior_type, *args, **kwargs)
        self.dists.append(this_prior)

        if name is None:
            name = "param{:d}".format(len(self.dists))
        self.names.append(name)

        if this_prior is not None:
            self.non_flat_names.append(name)
            self._non_flat_indices.append(len(self.dists) - 1)

    def logpdf(self, vals):
        """Compute the total log probability density for a set of parameter values.

        Flat (``None``) components are skipped; they contribute zero to the
        sum.

        Parameters
        ----------
        vals : array-like
            Sequence of parameter values with the same length as
            :attr:`dists`.

        Returns
        -------
        float
            Sum of ``logpdf`` values across all non-flat components, or
            ``0.0`` if every component is flat.
        """
        if not self._non_flat_indices:
            return 0.0
        total = 0.0
        for idx in self._non_flat_indices:
            total += self.dists[idx].logpdf(vals[idx])
        return total

    def sample(self, n_samp):
        """Draw independent samples from all prior components.

        Parameters
        ----------
        n_samp : int
            Number of samples to draw from each component.

        Returns
        -------
        ndarray of shape ``(n_components, n_samp)``
            Array where row *i* contains *n_samp* draws from component *i*.

        Raises
        ------
        ValueError
            If any component is flat (``None``), because a flat prior has
            no well-defined sampling distribution.
        """
        if any(dist is None for dist in self.dists):
            raise ValueError("Cannot sample from Prior with flat (None) components.")
        return np.vstack([dist.rvs(n_samp) for dist in self.dists])
