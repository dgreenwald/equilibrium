"""
Plotting utilities for Equilibrium.

The module currently exposes :func:`plot_paths`, which renders IRFs or other
time-series panels with pagination support, :func:`plot_deterministic_results`, which
plots DeterministicResult or SequenceResult objects, :func:`plot_model_irfs`, which
plots IRFs from multiple Model objects for multiple shocks, and :func:`plot_irf_results`,
which plots IRFs from IrfResult dictionaries.
"""

from .plot import (
    PlotSpec,
    plot_deterministic_results,
    plot_irf_results,
    plot_model_irfs,
    plot_paths,
)

__all__ = [
    "plot_paths",
    "PlotSpec",
    "plot_deterministic_results",
    "plot_model_irfs",
    "plot_irf_results",
]
