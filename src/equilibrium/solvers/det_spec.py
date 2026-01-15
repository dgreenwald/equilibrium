#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic (perfect-foresight) scenario specification utilities.

This module provides the DetSpec class, which describes a sequence of
parameter regimes and shocks, to be used with equilibrium's deterministic solver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _pad_list(
    lst: List[Any],
    target_length: int,
    default_factory,
) -> List[Any]:
    """
    Pad a list to the desired length using a factory function to create
    a fresh default object for each appended entry.

    Parameters
    ----------
    lst : list
        List to be padded in-place.
    target_length : int
        Desired final length.
    default_factory : callable
        Called with no arguments to construct a new default element.

    Returns
    -------
    list
        The same list instance, after padding.
    """
    while len(lst) < target_length:
        lst.append(default_factory())
    return lst


@dataclass
class DetSpec:
    """
    Specification of a deterministic (perfect-foresight) scenario with
    piecewise-constant parameter regimes and regime-specific shocks.

    The key idea is:
        - There are `n_regimes` regimes.
        - Parameters can differ by regime.
        - Each regime can have its own set of exogenous shock specifications.
        - `time_list[k]` is the time index at which we transition from regime k
          to regime k+1.

    Attributes
    ----------
    preset_par_init : dict
        Baseline parameter set used to compute the initial steady state.
        Parameter overrides for each regime are relative to this baseline.
    preset_par_list : list of dict
        Length `n_regimes`. Element r is a dict of parameter overrides
        (relative to preset_par_init) for regime r.
    shocks : list of list of tuple
        Length `n_regimes`. `shocks[r]` is the list of shock tuples for regime r,
        where each tuple is of the form (var, per, val):
            - var (str): shock variable name
            - per (float): persistence parameter
            - val (float): shock value (e.g. innovation size)
    time_list : list of int
        Length `n_regimes - 1`. `time_list[k]` is the time index (integer)
        at which we switch FROM regime k TO regime k+1.
    n_regimes : int
        Number of regimes (>= 0). By construction:
            len(preset_par_list) == n_regimes
            len(shocks)          == n_regimes
            len(time_list)       == max(n_regimes - 1, 0)
    z_init : np.ndarray, optional
        Initial values for exogenous states. If None, defaults to zeros.
        Can be overridden when calling solvers.
    ux_init : np.ndarray, optional
        Initial values for endogenous states and controls. If None, defaults
        to steady state. Can be overridden when calling solvers.
    Nt : int, optional
        Horizon (number of time periods) for simulation. Default is 100.
    label : str, optional
        Label for this experiment/scenario. Used for identifying and saving results.
        Default is "_default".

    Notes
    -----
    This class is *just* the specification. You will typically add an adapter
    method that takes a `Model` and horizon `Nt` and then:
        - builds parameter paths and exogenous paths;
        - calls the deterministic solver sequentially.
    """

    preset_par_init: Dict[str, Any] = field(default_factory=dict)
    preset_par_list: List[Dict[str, Any]] = field(default_factory=list)
    shocks: List[List[Tuple[str, float, float]]] = field(default_factory=list)
    time_list: List[int] = field(default_factory=list)
    n_regimes: int = 0
    z_init: Optional[np.ndarray] = None
    ux_init: Optional[np.ndarray] = None
    Nt: int = 100
    label: str = "_default"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    def __init__(
        self,
        preset_par_list: Optional[Sequence[Dict[str, Any]]] = None,
        shocks: Optional[Sequence[Sequence[Tuple[str, float, float]]]] = None,
        time_list: Optional[Sequence[int]] = None,
        preset_par_start: Optional[Dict[str, Any]] = None,
        preset_par_init: Optional[Dict[str, Any]] = None,
        n_regimes: Optional[int] = None,
        z_init: Optional[np.ndarray] = None,
        ux_init: Optional[np.ndarray] = None,
        Nt: int = 100,
        label: str = "_default",
    ) -> None:
        # Initialize baseline parameters (used for initial steady state)
        # preset_par_init takes precedence over preset_par_start for backward compatibility
        if preset_par_init is not None:
            self.preset_par_init = copy.deepcopy(preset_par_init)
        elif preset_par_start is not None:
            self.preset_par_init = copy.deepcopy(preset_par_start)
        else:
            self.preset_par_init = {}

        # Initialize parameter presets for regimes
        if preset_par_list is None:
            self.preset_par_list = []
            self.n_regimes = 0
        else:
            # Deep-copy to insulate from user modifications
            self.preset_par_list = [copy.deepcopy(d) for d in preset_par_list]
            self.n_regimes = len(self.preset_par_list)

        # Initialize shocks
        if shocks is not None:
            self.shocks = [list(regime_shocks) for regime_shocks in shocks]
            # Update n_regimes to match shocks length
            self.n_regimes = max(self.n_regimes, len(self.shocks))
        else:
            self.shocks = []

        # Initialize times
        self.time_list = list(time_list) if time_list is not None else []

        # Ensure n_regimes is at least what user requested
        if n_regimes is not None:
            self.n_regimes = max(self.n_regimes, n_regimes)

        # Set initial conditions
        self.z_init = z_init if z_init is None else np.asarray(z_init)
        self.ux_init = ux_init if ux_init is None else np.asarray(ux_init)

        # Set Nt
        self.Nt = Nt

        # Set label
        self.label = label

        # Normalize internal lengths
        self.update_n_regimes(self.n_regimes)

    # ------------------------------------------------------------------
    # Core invariants
    # ------------------------------------------------------------------
    def update_n_regimes(self, n_regimes: Optional[int] = None) -> "DetSpec":
        """
        Ensure internal lists are consistent with `n_regimes`.

        After this call:
            len(preset_par_list) == n_regimes
            len(shocks)          == n_regimes
            len(time_list)       == max(n_regimes - 1, 0)
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        if n_regimes < 0:
            raise ValueError("n_regimes must be non-negative")

        # Pad parameters: each regime uses a copy of preset_par_init as default
        _pad_list(
            self.preset_par_list,
            n_regimes,
            default_factory=lambda: copy.deepcopy(self.preset_par_init),
        )

        # Pad shock containers; each regime gets its own fresh list
        _pad_list(self.shocks, n_regimes, default_factory=list)

        # Pad transition times: one fewer than regimes
        target_time_len = max(n_regimes - 1, 0)
        _pad_list(self.time_list, target_time_len, default_factory=lambda: 0)

        # Update stored n_regimes
        self.n_regimes = max(self.n_regimes, n_regimes)
        return self

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add_regime(
        self,
        regime: Optional[int] = None,
        *,
        preset_par_regime: Optional[Dict[str, Any]] = None,
        shocks_regime: Optional[Sequence[Tuple[str, float, float]]] = None,
        time_regime: Optional[int] = None,
    ) -> "DetSpec":
        """
        Add or overwrite a regime specification.

        Parameters
        ----------
        regime : int or None
            Regime index to modify. If None, appends a new regime at the end,
            i.e. regime == current n_regimes.
        preset_par_regime : dict, optional
            Parameter overrides for this regime (relative to preset_par_init).
            These are merged into `preset_par_list[regime]`.
        shocks_regime : sequence of tuples, optional
            List of shock tuples (var, per, val) for this regime.
            If provided, REPLACES any existing shocks for this regime.
        time_regime : int, optional
            Time index at which we transition FROM regime (regime-1) TO this
            `regime`. Ignored if `regime == 0`. For regime r > 0 we set:
                time_list[r - 1] = time_regime
        """
        if regime is None:
            regime = self.n_regimes  # append new regime at the end
        if regime < 0:
            raise ValueError("regime index must be non-negative")

        # Ensure we have at least regime+1 regimes defined
        self.update_n_regimes(regime + 1)

        # Parameters: regime r uses index r
        if preset_par_regime is None:
            preset_par_regime = {}
        self.preset_par_list[regime].update(preset_par_regime)

        # Shocks: replace entries for this regime if provided
        if shocks_regime is not None:
            self.shocks[regime] = list(shocks_regime)

        # Timing for transitions: regime 0 has no incoming transition
        if regime > 0 and time_regime is not None:
            # transition from regime (regime-1) to regime occurs at time_regime
            idx = regime - 1
            if idx >= len(self.time_list):
                # Grow if needed (e.g. user changed n_regimes earlier)
                self.update_n_regimes(regime + 1)
            self.time_list[idx] = int(time_regime)

        return self

    def add_shock(
        self,
        regime: int,
        shock_var: str,
        shock_per: float,
        shock_val: float,
    ) -> "DetSpec":
        """
        Add a single shock to an existing regime.
        """
        if regime < 0:
            raise ValueError("regime index must be non-negative")

        self.update_n_regimes(regime + 1)
        self.shocks[regime].append((shock_var, float(shock_per), float(shock_val)))
        return self

    def add_shocks(
        self,
        regime: int,
        *,
        shocks: Optional[Sequence[Tuple[str, float, float]]] = None,
    ) -> "DetSpec":
        """
        Add multiple shocks to a regime at once.

        Parameters
        ----------
        regime : int
            Regime index to add shocks to.
        shocks : sequence of tuples, optional
            List of shock tuples (var, per, val) to add.
        """
        if regime < 0:
            raise ValueError("regime index must be non-negative")

        self.update_n_regimes(regime + 1)

        if shocks is not None:
            self.shocks[regime] += list(shocks)

        return self

    # ------------------------------------------------------------------
    # Validation and utilities
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Validate internal consistency. Raises ValueError on problems.
        """
        if self.n_regimes < 0:
            raise ValueError("n_regimes must be non-negative")

        # Check lengths
        if len(self.preset_par_list) != self.n_regimes:
            raise ValueError(
                f"preset_par_list length {len(self.preset_par_list)} "
                f"!= n_regimes ({self.n_regimes})"
            )
        if len(self.shocks) != self.n_regimes:
            raise ValueError(
                f"shocks length {len(self.shocks)} != n_regimes ({self.n_regimes})"
            )

        expected_time_len = max(self.n_regimes - 1, 0)
        if len(self.time_list) != expected_time_len:
            raise ValueError(
                f"time_list length {len(self.time_list)} "
                f"!= n_regimes - 1 ({expected_time_len})"
            )

        # Optional: check that each shock tuple has the correct structure
        for r in range(self.n_regimes):
            for i, shock in enumerate(self.shocks[r]):
                if not isinstance(shock, tuple) or len(shock) != 3:
                    raise ValueError(
                        f"Shock {i} in regime {r} must be a tuple of length 3 (var, per, val), "
                        f"got {shock}"
                    )
                var, per, val = shock
                if not isinstance(var, str):
                    raise ValueError(
                        f"Shock variable in regime {r}, shock {i} must be a string, got {type(var)}"
                    )
                if not isinstance(per, (int, float)):
                    raise ValueError(
                        f"Shock persistence in regime {r}, shock {i} must be numeric, got {type(per)}"
                    )
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Shock value in regime {r}, shock {i} must be numeric, got {type(val)}"
                    )

    # ------------------------------------------------------------------
    # Future integration point
    # ------------------------------------------------------------------
    def build_exog_paths(
        self,
        mod,
        Nt: int,
        regime: int = 0,
        z_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build exogenous paths from this spec for a given regime.

        Parameters
        ----------
        mod : equilibrium.Model
            The model instance.
        Nt : int
            Horizon (number of periods) for the deterministic simulation.
        regime : int, optional
            Index for which regime we are creating the paths for. Default is 0.
        z_init : array-like, optional
            Initial values for the exogenous states. If None, set to a vector
            of zeros of appropriate size (equal to number of exogenous variables).

        Returns
        -------
        Z_path : np.ndarray
            Exogenous variable paths, shape (Nt, n_exog).
        """
        # Validate regime parameter
        if regime < 0:
            raise ValueError("regime must be non-negative")

        # Number of exogenous variables
        n_exog = len(mod.exog_list)

        # Set default z_init to zeros if not provided
        if z_init is None:
            z_init = np.zeros(n_exog)
        else:
            z_init = np.asarray(z_init)

        # Initialize innovations array: (Nt, n_exog)
        innovations = np.zeros((Nt, n_exog))

        # Fill innovations from shocks for the specified regime
        if regime < len(self.shocks):
            for var, per, val in self.shocks[regime]:
                # Find the index of this variable in exog_list
                if var in mod.exog_list:
                    i_var = mod.exog_list.index(var)
                    # Note: per+1 because initial value Z[0,:] is the starting state
                    # before the shock arrives
                    per_idx = int(per) + 1
                    if 0 <= per_idx < Nt:
                        innovations[per_idx, i_var] = val

        # Build Z_path
        Z_path = np.zeros((Nt, n_exog))
        Z_path[0, :] = z_init

        # Loop over time to build Z_path
        z = z_init.copy()
        for tt in range(1, Nt):
            z = (
                mod.linear_mod.Phi @ z
                + mod.linear_mod.impact_matrix @ innovations[tt, :]
            )
            Z_path[tt, :] = z

        return Z_path
