"""
Plotting utilities for Equilibrium.

The module currently exposes :func:`plot_paths`, which renders IRFs or other
time-series panels with pagination support, :func:`plot_deterministic_results`, which
plots DeterministicResult or SequenceResult objects, :func:`plot_model_irfs`, which
plots IRFs from multiple Model objects for multiple shocks, :func:`plot_irf_results`,
which plots IRFs from IrfResult dictionaries, and :func:`overlay_to_result`, which
converts external data to DeterministicResult format for overlay plotting.
"""

from .plot import (
    PlotSpec,
    overlay_to_result,
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
    "overlay_to_result",
]
