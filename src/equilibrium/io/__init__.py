"""
I/O module for saving and loading model results.

This module provides utilities for saving and loading linearized model
solutions, impulse response functions, and simulation results.
"""

from .results import load_results, resolve_output_path, save_results

__all__ = ["resolve_output_path", "save_results", "load_results"]
