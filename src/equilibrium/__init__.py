"""
Equilibrium – Dynamic general-equilibrium solver in JAX
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

# Initialize settings early to configure JAX (compilation cache, x64 precision)
# before any JAX code is imported
from .settings import get_settings as _get_settings

_get_settings()

from . import blocks
from .io import load_results, resolve_output_path, save_results
from .model import LinearModel, Model
from .plot.plot import (
    plot_deterministic_results,
    plot_irf_results,
    plot_model_irfs,
    plot_paths,
)
from .solvers.calibration import (
    CalibrationResult,
    FunctionalTarget,
    PointTarget,
    calibrate,
)
from .solvers.det_spec import DetSpec
from .solvers.linear_spec import LinearSpec
from .solvers.results import (
    DeterministicResult,
    IrfResult,
    PathResult,
    SequenceResult,
    SeriesTransform,
)
from .utils.io import (
    load_deterministic_result,
    load_model_irfs,
    load_sequence_result,
    read_steady_value,
    read_steady_values,
)

try:  # when installed (pip install equilibrium or -e .)
    __version__ = _v(__name__)
except PackageNotFoundError:  # running from a Git checkout w/out install
    __version__ = "0.0.0"

__all__: list[str] = [
    "__version__",
    "Model",
    "LinearModel",
    "plot_paths",
    "plot_deterministic_results",
    "plot_model_irfs",
    "plot_irf_results",
    "resolve_output_path",
    "save_results",
    "load_results",
    "read_steady_value",
    "read_steady_values",
    "load_model_irfs",
    "load_deterministic_result",
    "load_sequence_result",
    "blocks",
    "PathResult",
    "IrfResult",
    "DeterministicResult",
    "SequenceResult",
    "SeriesTransform",
    "DetSpec",
    "LinearSpec",
    "calibrate",
    "CalibrationResult",
    "PointTarget",
    "FunctionalTarget",
]
