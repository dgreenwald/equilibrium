"""
Equilibrium – Dynamic general-equilibrium solver in JAX
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

# Initialize settings early to configure JAX (compilation cache, x64 precision)
# before any JAX code is imported
from .settings import get_settings as _get_settings

_get_settings()

from . import blocks  # noqa: E402
from .io import load_results, resolve_output_path, save_results  # noqa: E402
from .model import LinearModel, Model  # noqa: E402
from .plot.plot import (  # noqa: E402
    plot_deterministic_results,
    plot_irf_results,
    plot_model_irfs,
    plot_paths,
)
from .solvers.calibration import (  # noqa: E402
    CalibrationResult,
    FunctionalTarget,
    ModelParam,
    PointTarget,
    RegimeParam,
    ShockParam,
    calibrate,
)
from .solvers.det_spec import DetSpec  # noqa: E402
from .solvers.linear_spec import LinearSpec  # noqa: E402
from .solvers.results import (  # noqa: E402
    DeterministicResult,
    IrfResult,
    PathResult,
    SequenceResult,
    SeriesTransform,
)
from .utils.io import (  # noqa: E402
    load_deterministic_result,
    load_model_irfs,
    load_sequence_result,
    read_calibrated_param,
    read_calibrated_params,
    read_steady_value,
    read_steady_values,
    save_calibrated_params,
    save_calibrated_params_to_latex,
    save_steady_values_to_latex,
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
    "read_calibrated_param",
    "read_calibrated_params",
    "save_calibrated_params",
    "save_steady_values_to_latex",
    "save_calibrated_params_to_latex",
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
    "ModelParam",
    "ShockParam",
    "RegimeParam",
]
