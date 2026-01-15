"""
Results I/O functions for saving and loading model outputs.

This module provides functions for:
- Resolving output paths with smart defaults
- Saving results in multiple formats (npz, csv, json)
- Loading results from saved files
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..settings import get_settings

logger = logging.getLogger(__name__)


def _convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object to convert. Can be a dict, list, numpy type, or other.

    Returns
    -------
    Any
        Object with numpy types converted to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def resolve_output_path(
    filepath: Optional[str | Path] = None,
    *,
    result_type: str = "results",
    model_label: str = "_default",
    experiment_label: Optional[str] = None,
    save_dir: Optional[str | Path] = None,
    timestamp: bool = False,
    suffix: str = ".npz",
) -> Path:
    """
    Resolve the output path for saving results.

    If filepath is provided, use it directly. Otherwise, construct a default
    path using settings.paths.save_dir, result_type subdirectory, and model_label.

    Parameters
    ----------
    filepath : str or Path, optional
        Explicit path to use. If provided, returns this path directly.
    result_type : str, default "results"
        Subdirectory name under save_dir (e.g., "irfs", "simulations", "paths").
    model_label : str, default "_default"
        Model label to use in the filename.
    experiment_label : str, optional
        Experiment/scenario label to append to filename. If provided, filename
        becomes "{model_label}_{experiment_label}".
    save_dir : str or Path, optional
        Base directory to use instead of settings.paths.save_dir.
    timestamp : bool, default False
        If True and filepath is None, append timestamp to filename.
    suffix : str, default ".npz"
        File extension to use (e.g., ".npz", ".json", ".csv").

    Returns
    -------
    Path
        The resolved output path.
    """
    if filepath is not None:
        resolved = Path(filepath)
        logger.debug("Using explicit filepath: %s", resolved)
        return resolved

    base_dir = Path(save_dir) if save_dir is not None else get_settings().paths.save_dir
    base_dir = base_dir / result_type
    base_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    filename = model_label
    if experiment_label:
        filename = f"{filename}_{experiment_label}"
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{ts}"

    # Ensure suffix starts with a dot
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    resolved = base_dir / f"{filename}{suffix}"
    logger.debug("Resolved output path: %s", resolved)
    return resolved


def save_results(
    data: dict[str, Any],
    filepath: str | Path,
    *,
    format: str = "npz",
    metadata: Optional[dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Save results to a file.

    Parameters
    ----------
    data : dict
        Dictionary of results to save. Values can be arrays, scalars, etc.
    filepath : str or Path
        Path to save the results to.
    format : str, default "npz"
        Output format. Supported: 'npz', 'csv', 'json'.
    metadata : dict, optional
        Additional metadata to include (e.g., model label, parameters).
    overwrite : bool, default False
        If False and file exists, raise FileExistsError.

    Returns
    -------
    Path
        The path to the saved file.

    Raises
    ------
    FileExistsError
        If file exists and overwrite is False.
    ValueError
        If format is not supported.
    """
    path = Path(filepath)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Use overwrite=True to replace."
        )

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        _save_npz(data, path, metadata)
    elif format == "json":
        _save_json(data, path, metadata)
    elif format == "csv":
        _save_csv(data, path, metadata)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz', 'json', or 'csv'.")

    logger.info("Saved results to %s", path)
    return path


def _save_npz(
    data: dict[str, Any], path: Path, metadata: Optional[dict] = None
) -> None:
    """Save data as compressed NumPy archive."""
    save_dict = {}

    for key, value in data.items():
        if value is not None:
            save_dict[key] = np.asarray(value)

    if metadata:
        metadata_native = _convert_to_native(metadata)
        # Store metadata as a JSON string in a special key
        save_dict["__metadata__"] = np.array(json.dumps(metadata_native))

    np.savez_compressed(path, **save_dict)


def _save_json(
    data: dict[str, Any], path: Path, metadata: Optional[dict] = None
) -> None:
    """Save data as JSON file."""
    save_dict = {}

    for key, value in data.items():
        if value is None:
            save_dict[key] = None
        else:
            save_dict[key] = _convert_to_native(value)

    if metadata:
        save_dict["__metadata__"] = _convert_to_native(metadata)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)


def _save_csv(
    data: dict[str, Any], path: Path, metadata: Optional[dict] = None
) -> None:
    """Save data as CSV file(s).

    For each 2D array, saves a separate CSV file with column names based on keys.
    For 1D arrays and scalars, creates a summary CSV.
    """
    import csv

    # Handle 2D arrays - save each as separate CSV
    arrays_2d = {}
    other_data = {}

    for key, value in data.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 2:
            arrays_2d[key] = arr
        else:
            other_data[key] = arr

    # If there's exactly one 2D array, save it to the main path
    if len(arrays_2d) == 1:
        key, arr = next(iter(arrays_2d.items()))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header if we have column names from metadata
            if metadata and "var_names" in metadata:
                writer.writerow(metadata["var_names"])
            for row in arr:
                writer.writerow(row)
    elif arrays_2d:
        # Multiple 2D arrays - save each to a separate file
        for key, arr in arrays_2d.items():
            arr_path = path.with_name(f"{path.stem}_{key}.csv")
            with open(arr_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in arr:
                    writer.writerow(row)

    # Save scalars/1D arrays to main path or summary file
    if other_data or (not arrays_2d and metadata):
        summary_path = (
            path if not arrays_2d else path.with_name(f"{path.stem}_summary.csv")
        )
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in other_data.items():
                arr = np.asarray(value)
                if arr.ndim == 0:
                    writer.writerow([key, float(arr)])
                elif arr.ndim == 1:
                    writer.writerow([key, arr.tolist()])
            if metadata:
                writer.writerow(["__metadata__", json.dumps(metadata)])


def load_results(filepath: str | Path) -> dict[str, Any]:
    """
    Load results from a saved file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to load.

    Returns
    -------
    dict
        Dictionary of loaded data. If metadata was saved, it will be in
        the "__metadata__" key.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npz":
        return _load_npz(path)
    elif suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use '.npz' or '.json'.")


def _load_npz(path: Path) -> dict[str, Any]:
    """Load data from NumPy archive."""
    with np.load(path, allow_pickle=True) as data:
        result = {}
        for key in data.files:
            if key == "__metadata__":
                # Decode metadata from JSON string
                meta_str = str(data[key])
                result[key] = json.loads(meta_str)
            else:
                result[key] = data[key]
        return result


def _load_json(path: Path) -> dict[str, Any]:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for key, value in data.items():
        if key == "__metadata__":
            result[key] = value
        elif isinstance(value, list):
            result[key] = np.array(value)
        else:
            result[key] = value

    return result
