#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear IRF specification utilities.

This module provides the LinearSpec class for specifying linear impulse
response function computations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearSpec:
    """
    Specification for linear impulse response function (IRF) computation.

    This class encapsulates the parameters needed to compute an IRF using
    the linearized model dynamics.

    Attributes
    ----------
    shock_name : str
        Name of the shock variable (must be in model.exog_list).
    shock_size : float, default 1.0
        Size of the shock impulse.
    Nt : int, default 100
        Horizon (number of time periods) for the IRF.

    Examples
    --------
    >>> # Basic IRF specification
    >>> spec = LinearSpec(shock_name="Z_til", shock_size=0.01, Nt=50)
    >>>
    >>> # IRF with default shock size
    >>> spec = LinearSpec(shock_name="tfp_shock", Nt=40)
    """

    shock_name: str
    shock_size: float = 1.0
    Nt: int = 100

    def __post_init__(self):
        """Validate specification."""
        if not isinstance(self.shock_name, str) or not self.shock_name:
            raise ValueError("shock_name must be a non-empty string")
        if self.shock_size == 0:
            raise ValueError("shock_size cannot be zero")
        if self.Nt <= 0:
            raise ValueError(f"Nt must be positive, got {self.Nt}")
