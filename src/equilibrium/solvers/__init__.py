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
]
