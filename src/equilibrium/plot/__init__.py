"""
Plotting utilities for Equilibrium.

The module exposes plotting functions for rendering time-series and IRF plots, as well
as data preparation utilities for custom analysis workflows.

Plotting Functions
------------------
- :func:`plot_paths`: Low-level rendering of time-series panels with pagination
- :func:`plot_deterministic_results`: Plot DeterministicResult or SequenceResult objects
- :func:`plot_model_irfs`: Plot IRFs from multiple Model objects
- :func:`plot_irf_results`: Plot IRFs from IrfResult dictionaries
- :class:`PlotSpec`: Styling container for plot customization

Data Preparation
----------------
- :func:`prepare_deterministic_paths`: Prepare harmonized paths for plotting or analysis
- :func:`overlay_to_result`: Convert external data to DeterministicResult format
- :class:`PreparedPaths`: Container for prepared deterministic paths
"""

from .plot import (
    PlotSpec,
    plot_deterministic_results,
    plot_irf_results,
    plot_model_irfs,
    plot_paths,
)
from .preparation import (
    PreparedPaths,
    overlay_to_result,
    prepare_deterministic_paths,
)

__all__ = [
    # Plotting functions
    "plot_paths",
    "plot_deterministic_results",
    "plot_model_irfs",
    "plot_irf_results",
    "PlotSpec",
    # Data preparation
    "prepare_deterministic_paths",
    "PreparedPaths",
    "overlay_to_result",
]
