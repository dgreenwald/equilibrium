"""
Solvers module for deterministic and linear path solving.

This module provides functions for computing deterministic transition paths
using both nonlinear and linearized model dynamics, as well as a unified
calibration interface.
"""

from .calibration import (
    CalibrationResult,
    FunctionalTarget,
    ModelParam,
    PointTarget,
    RegimeParam,
    ShockParam,
    calibrate,
)
from .det_spec import DetSpec
from .linear_spec import LinearSpec
from .quadrature import (
    ExogenousProcess,
    JaxExogenousProcess,
    JaxQuadratureRule,
    QuadratureRule,
    deterministic_quadrature,
    exogenous_process_from_model,
    gauss_hermite_normal,
    next_exogenous_states_jax,
    smolyak_gauss_hermite,
    tensor_gauss_hermite,
)
from .results import DeterministicResult, SequenceResult, SeriesTransform

__all__ = [
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
    "QuadratureRule",
    "JaxQuadratureRule",
    "ExogenousProcess",
    "JaxExogenousProcess",
    "deterministic_quadrature",
    "gauss_hermite_normal",
    "tensor_gauss_hermite",
    "smolyak_gauss_hermite",
    "exogenous_process_from_model",
    "next_exogenous_states_jax",
]
