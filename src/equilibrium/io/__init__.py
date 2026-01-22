"""
I/O module for saving and loading model results and exporting to other formats.

This module provides utilities for:
- Saving and loading linearized model solutions, IRFs, and simulation results
- Exporting models to external formats (Dynare)
"""

from .dynare import export_to_dynare, format_var_list
from .results import load_results, resolve_output_path, save_results

__all__ = [
    # Results I/O
    "resolve_output_path",
    "save_results",
    "load_results",
    # Dynare export
    "export_to_dynare",
    "format_var_list",
]
