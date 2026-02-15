#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result container classes for path-based solver outputs.

This module provides dataclass containers for storing and serializing
results from deterministic path solvers and IRF computations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

import numpy as np

from ..io import load_results, resolve_output_path, save_results

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesTransform:
    """
    Transform specification for a single output series.

    Attributes
    ----------
    diff : bool
        If True, difference the series against the base index.
    log_to_level : bool
        If True, exponentiate the series before differencing.
    scale : float
        Multiplicative scale factor applied after differencing.
    to_percent : bool
        If True, multiply by 100 after differencing and scaling.
    base_index : int
        Index used as the baseline when differencing.
    """

    diff: bool = False
    log_to_level: bool = False
    scale: float = 1.0
    to_percent: bool = False
    base_index: int = 0

    def apply(self, series: np.ndarray) -> np.ndarray:
        """Apply the transform to a 1D series array."""
        if series.ndim != 1:
            raise ValueError("SeriesTransform expects a 1D array.")
        if not 0 <= self.base_index < series.shape[0]:
            raise ValueError("SeriesTransform base_index is out of bounds.")

        transformed = series.copy()

        if self.log_to_level:
            transformed = np.exp(transformed)

        if self.diff:
            base_val = transformed[self.base_index]
            if self.log_to_level:
                transformed = (transformed / base_val) - 1.0
            else:
                transformed = transformed - base_val

        scale_factor = self.scale * (100.0 if self.to_percent else 1.0)
        if scale_factor != 1.0:
            transformed = transformed * scale_factor

        return transformed


def _normalize_series_transform(
    transform: SeriesTransform | Mapping[str, Any] | None,
) -> SeriesTransform:
    if transform is None:
        return SeriesTransform()
    if isinstance(transform, SeriesTransform):
        return transform
    if isinstance(transform, Mapping):
        return SeriesTransform(**transform)
    raise TypeError("series_transforms must map to SeriesTransform or dict values.")


def _apply_series_transforms(
    data: np.ndarray,
    names: list[str],
    series_transforms: Optional[Mapping[str, SeriesTransform | Mapping[str, Any]]],
    default_transform: Optional[SeriesTransform | Mapping[str, Any]],
) -> np.ndarray:
    if series_transforms is None and default_transform is None:
        return data

    transformed = np.array(data, copy=True)
    default_spec = (
        _normalize_series_transform(default_transform)
        if default_transform is not None
        else None
    )

    for idx, name in enumerate(names):
        spec = series_transforms.get(name) if series_transforms else None
        if spec is None:
            if default_spec is None:
                continue
            spec = default_spec
        spec = _normalize_series_transform(spec)
        transformed[:, idx] = spec.apply(transformed[:, idx])

    return transformed


@dataclass
class PathResult:
    """
    Base container for path-based results (deterministic paths or IRFs).

    This dataclass stores solution arrays along with metadata. It serves as
    a base class for both deterministic paths and impulse response functions.

    Attributes
    ----------
    UX : np.ndarray
        Control and state variables, shape (Nt, N_ux).
    Z : np.ndarray
        Exogenous variables path, shape (Nt, N_z).
    Y : np.ndarray, optional
        Intermediate variables, shape (Nt, N_y). None if not computed.
    model_label : str
        Label of the model used.
    var_names : list[str]
        Names of variables in UX columns.
    exog_names : list[str]
        Names of exogenous variables in Z columns.
    y_names : list[str]
        Names of intermediate variables in Y columns.
    """

    UX: np.ndarray
    Z: np.ndarray
    Y: Optional[np.ndarray] = None
    model_label: str = "_default"
    var_names: list[str] = field(default_factory=list)
    exog_names: list[str] = field(default_factory=list)
    y_names: list[str] = field(default_factory=list)

    def save(
        self,
        filepath: Optional[Union[str, Path]] = None,
        *,
        format: str = "npz",
        overwrite: bool = False,
        timestamp: bool = False,
        result_type: str = "paths",
    ) -> Path:
        """
        Save the result to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save the result. If None, uses default path from settings.
        format : str, default "npz"
            Format for saving. Supported: 'npz', 'json'.
        overwrite : bool, default False
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.
        result_type : str, default "paths"
            Subdirectory for saving (e.g., "paths", "irfs").

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = resolve_output_path(
            filepath,
            result_type=result_type,
            model_label=self.model_label,
            timestamp=timestamp,
            suffix=f".{format}",
        )

        data = {
            "UX": self.UX,
            "Z": self.Z,
        }
        if self.Y is not None:
            data["Y"] = self.Y

        metadata = self._get_metadata()

        result_path = save_results(
            data, path, format=format, metadata=metadata, overwrite=overwrite
        )
        logger.info("Saved %s to %s", self.__class__.__name__, result_path)
        return result_path

    def transform(
        self,
        *,
        series_transforms: Optional[
            Mapping[str, SeriesTransform | Mapping[str, Any]]
        ] = None,
        default_transform: Optional[SeriesTransform | Mapping[str, Any]] = None,
    ) -> "PathResult":
        """
        Return a new result with per-series transformations applied.

        Parameters
        ----------
        series_transforms : dict[str, SeriesTransform or dict], optional
            Per-series transform specifications keyed by series name. Applies
            across UX, Z, and Y names. Dict values are expanded into
            SeriesTransform.
        default_transform : SeriesTransform or dict, optional
            Transform to apply to any series without an explicit entry in
            series_transforms.
        """
        if series_transforms is None and default_transform is None:
            return self

        transformed_UX = _apply_series_transforms(
            self.UX, self.var_names, series_transforms, default_transform
        )
        transformed_Z = _apply_series_transforms(
            self.Z, self.exog_names, series_transforms, default_transform
        )
        transformed_Y = None
        if self.Y is not None:
            transformed_Y = _apply_series_transforms(
                self.Y, self.y_names, series_transforms, default_transform
            )

        return replace(self, UX=transformed_UX, Z=transformed_Z, Y=transformed_Y)

    def __getitem__(self, name: str) -> np.ndarray:
        """
        Return a 1D series by name from UX, Z, or Y.

        Parameters
        ----------
        name : str
            Series name from var_names, exog_names, or y_names.

        Returns
        -------
        np.ndarray
            1D array of the requested series.

        Raises
        ------
        KeyError
            If the series name is not found, is ambiguous, or Y data is missing.
        TypeError
            If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Series name must be a string.")

        matches: list[tuple[str, int]] = []
        if name in self.var_names:
            matches.append(("UX", self.var_names.index(name)))
        if name in self.exog_names:
            matches.append(("Z", self.exog_names.index(name)))
        if name in self.y_names:
            matches.append(("Y", self.y_names.index(name)))

        if not matches:
            raise KeyError(f"Series '{name}' not found.")
        if len(matches) > 1:
            raise KeyError(f"Series '{name}' is ambiguous across multiple namespaces.")

        source, idx = matches[0]
        if source == "UX":
            return self.UX[:, idx]
        if source == "Z":
            return self.Z[:, idx]
        if self.Y is None:
            raise KeyError(f"Series '{name}' is unavailable because Y is None.")
        return self.Y[:, idx]

    def to_df(self) -> "pd.DataFrame":
        """
        Convert result to a pandas DataFrame with time index.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for all variables (UX, Z, and optionally Y).
            Index is a RangeIndex representing time periods.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> result = model.solve_steady()
        >>> df = result.to_df()
        >>> df['consumption'].plot()
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_df(). " "Install it with: pip install pandas"
            ) from e

        # Build the dataframe
        data = {}

        # Add UX columns
        for i, name in enumerate(self.var_names):
            data[name] = self.UX[:, i]

        # Add Z columns
        for i, name in enumerate(self.exog_names):
            data[name] = self.Z[:, i]

        # Add Y columns if present
        if self.Y is not None:
            for i, name in enumerate(self.y_names):
                data[name] = self.Y[:, i]

        return pd.DataFrame(data)

    def _get_metadata(self) -> dict:
        """Get metadata dictionary for saving. Override in subclasses."""
        return {
            "model_label": self.model_label,
            "var_names": self.var_names,
            "exog_names": self.exog_names,
            "y_names": self.y_names,
            "result_type": self.__class__.__name__,
        }

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PathResult":
        """
        Load a PathResult from a saved file.

        Parameters
        ----------
        filepath : str or Path
            Path to the file to load.

        Returns
        -------
        PathResult
            Loaded result object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported or result type mismatch.
        """
        loaded = load_results(filepath)
        metadata = loaded.get("__metadata__", {})

        # Verify this is the correct result type
        result_type = metadata.get("result_type", "")
        if result_type and result_type != cls.__name__:
            raise ValueError(
                f"Expected {cls.__name__}, got {result_type}. "
                f"Use the appropriate load method for {result_type}."
            )

        return cls(
            UX=np.asarray(loaded["UX"]),
            Z=np.asarray(loaded["Z"]),
            Y=np.asarray(loaded["Y"]) if "Y" in loaded else None,
            model_label=metadata.get("model_label", "_default"),
            var_names=metadata.get("var_names", []),
            exog_names=metadata.get("exog_names", []),
            y_names=metadata.get("y_names", []),
        )


@dataclass
class DeterministicResult(PathResult):
    """
    Container for results from a deterministic path solve.

    This dataclass extends PathResult with deterministic solver-specific
    metadata about convergence and terminal conditions.

    Attributes
    ----------
    UX : np.ndarray
        Control and state variables, shape (Nt, N_ux).
    Z : np.ndarray
        Exogenous variables path, shape (Nt, N_z).
    Y : np.ndarray, optional
        Intermediate variables, shape (Nt, N_y). None if not computed.
    model_label : str
        Label of the model used.
    var_names : list[str]
        Names of variables in UX columns.
    exog_names : list[str]
        Names of exogenous variables in Z columns.
    y_names : list[str]
        Names of intermediate variables in Y columns.
    terminal_condition : str
        Terminal condition used ('stable' or 'steady').
    converged : bool
        Whether the solver converged.
    final_residual : float
        Final residual norm.
    """

    terminal_condition: str = "stable"
    converged: bool = True
    final_residual: float = 0.0

    def _get_metadata(self) -> dict:
        """Get metadata dictionary including deterministic-specific fields."""
        metadata = super()._get_metadata()
        metadata.update(
            {
                "terminal_condition": self.terminal_condition,
                "converged": self.converged,
                "final_residual": self.final_residual,
            }
        )
        return metadata

    def save(
        self,
        filepath: Optional[Union[str, Path]] = None,
        *,
        format: str = "npz",
        overwrite: bool = False,
        timestamp: bool = False,
        experiment_label: Optional[str] = None,
    ) -> Path:
        """
        Save the result to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save the result. If None, uses default path from settings
            with result_type="paths".
        format : str, default "npz"
            Format for saving. Supported: 'npz', 'json'.
        overwrite : bool, default False
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.
        experiment_label : str, optional
            Experiment/scenario label to include in filename. If provided,
            filename becomes "{model_label}_{experiment_label}".

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = resolve_output_path(
            filepath,
            result_type="paths",
            model_label=self.model_label,
            experiment_label=experiment_label,
            timestamp=timestamp,
            suffix=f".{format}",
        )

        data = {
            "UX": self.UX,
            "Z": self.Z,
        }
        if self.Y is not None:
            data["Y"] = self.Y

        metadata = self._get_metadata()
        if experiment_label:
            metadata["experiment_label"] = experiment_label

        result_path = save_results(
            data, path, format=format, metadata=metadata, overwrite=overwrite
        )
        logger.info("Saved %s to %s", self.__class__.__name__, result_path)
        return result_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DeterministicResult":
        """
        Load a DeterministicResult from a saved file.

        Parameters
        ----------
        filepath : str or Path
            Path to the file to load.

        Returns
        -------
        DeterministicResult
            Loaded result object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported or not a DeterministicResult.
        """
        loaded = load_results(filepath)
        metadata = loaded.get("__metadata__", {})

        # Verify this is a DeterministicResult
        result_type = metadata.get("result_type", "")
        if result_type and result_type != "DeterministicResult":
            raise ValueError(
                f"Expected DeterministicResult, got {result_type}. "
                f"Use the appropriate load method for {result_type}."
            )

        return cls(
            UX=np.asarray(loaded["UX"]),
            Z=np.asarray(loaded["Z"]),
            Y=np.asarray(loaded["Y"]) if "Y" in loaded else None,
            model_label=metadata.get("model_label", "_default"),
            var_names=metadata.get("var_names", []),
            exog_names=metadata.get("exog_names", []),
            y_names=metadata.get("y_names", []),
            terminal_condition=metadata.get("terminal_condition", "stable"),
            converged=metadata.get("converged", True),
            final_residual=metadata.get("final_residual", 0.0),
        )


@dataclass
class IrfResult(PathResult):
    """
    Container for impulse response function results.

    This dataclass extends PathResult with IRF-specific metadata about
    the shock that generated the response.

    Attributes
    ----------
    UX : np.ndarray
        Control and state variables, shape (Nt, N_ux).
    Z : np.ndarray
        Exogenous variables path, shape (Nt, N_z).
    Y : np.ndarray, optional
        Intermediate variables, shape (Nt, N_y). None if not computed.
    model_label : str
        Label of the model used.
    var_names : list[str]
        Names of variables in UX columns.
    exog_names : list[str]
        Names of exogenous variables in Z columns.
    y_names : list[str]
        Names of intermediate variables in Y columns.
    shock_name : str
        Name of the shock that generated this IRF.
    shock_size : float
        Size of the shock impulse.
    """

    shock_name: str = ""
    shock_size: float = 1.0

    def _get_metadata(self) -> dict:
        """Get metadata dictionary including IRF-specific fields."""
        metadata = super()._get_metadata()
        metadata.update(
            {
                "shock_name": self.shock_name,
                "shock_size": self.shock_size,
            }
        )
        return metadata

    def save(
        self,
        filepath: Optional[Union[str, Path]] = None,
        *,
        format: str = "npz",
        overwrite: bool = False,
        timestamp: bool = False,
    ) -> Path:
        """
        Save the IRF result to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save the result. If None, uses default path from settings
            with result_type="irfs".
        format : str, default "npz"
            Format for saving. Supported: 'npz', 'json'.
        overwrite : bool, default False
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.

        Returns
        -------
        Path
            Path to the saved file.
        """
        return super().save(
            filepath,
            format=format,
            overwrite=overwrite,
            timestamp=timestamp,
            result_type="irfs",
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IrfResult":
        """
        Load an IrfResult from a saved file.

        Parameters
        ----------
        filepath : str or Path
            Path to the file to load.

        Returns
        -------
        IrfResult
            Loaded result object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported or not an IrfResult.
        """
        loaded = load_results(filepath)
        metadata = loaded.get("__metadata__", {})

        # Verify this is an IrfResult
        result_type = metadata.get("result_type", "")
        if result_type and result_type != "IrfResult":
            raise ValueError(
                f"Expected IrfResult, got {result_type}. "
                f"Use the appropriate load method for {result_type}."
            )

        return cls(
            UX=np.asarray(loaded["UX"]),
            Z=np.asarray(loaded["Z"]),
            Y=np.asarray(loaded["Y"]) if "Y" in loaded else None,
            model_label=metadata.get("model_label", "_default"),
            var_names=metadata.get("var_names", []),
            exog_names=metadata.get("exog_names", []),
            y_names=metadata.get("y_names", []),
            shock_name=metadata.get("shock_name", ""),
            shock_size=metadata.get("shock_size", 1.0),
        )


@dataclass
class SequenceResult:
    """
    Container for results from a multi-regime deterministic solve sequence.

    This dataclass stores results for each regime along with metadata
    about the sequence, and provides methods for saving and loading results.

    Attributes
    ----------
    regimes : list[DeterministicResult]
        Results for each regime.
    time_list : list[int]
        Transition times between regimes.
    model_label : str
        Base model label.
    experiment_label : str
        Experiment/scenario label for this sequence.
    """

    regimes: list[DeterministicResult] = field(default_factory=list)
    time_list: list[int] = field(default_factory=list)
    model_label: str = "_default"
    experiment_label: str = "_default"

    @property
    def n_regimes(self) -> int:
        """Number of regimes in the sequence."""
        return len(self.regimes)

    def splice(
        self,
        T_max: Optional[int] = None,
    ) -> "DeterministicResult":
        """
        Splice together regime results into a continuous DeterministicResult.

        This function combines the UX, Z, and Y paths from each regime in the
        sequence into a single continuous path, handling the timing carefully
        to avoid duplication at regime transitions.

        Parameters
        ----------
        T_max : int, optional
            The total length of the spliced path (number of time periods).
            If None, use a default computed from the regime lengths and
            transition times.

        Returns
        -------
        DeterministicResult
            A single result containing the spliced UX, Z, and Y paths.

        Raises
        ------
        ValueError
            If the sequence has no regimes, or if metadata (var_names,
            exog_names, y_names, terminal_condition) is inconsistent across regimes.

        Notes
        -----
        Each regime is initialized by the previous regime, so the transition
        point from regime i to regime i+1 occurs at time_list[i]. The first
        period of regime i+1 corresponds to the same time point as period
        time_list[i] in regime i, so we skip it to avoid duplication.
        """
        if T_max is None:
            T_max = self._compute_default_T_max()
        if self.n_regimes == 0:
            raise ValueError("Cannot splice an empty SequenceResult")

        # Get reference metadata from first regime
        ref_result = self.regimes[0]
        ref_var_names = ref_result.var_names
        ref_exog_names = ref_result.exog_names
        ref_y_names = ref_result.y_names
        ref_terminal_condition = ref_result.terminal_condition

        # Check if any regime has Y data
        has_y_data = any(r.Y is not None for r in self.regimes)

        # Verify all regimes have consistent metadata
        for i, regime in enumerate(self.regimes[1:], start=1):
            if regime.var_names != ref_var_names:
                raise ValueError(
                    f"Regime {i} var_names {regime.var_names} differs from "
                    f"regime 0 var_names {ref_var_names}"
                )
            if regime.exog_names != ref_exog_names:
                raise ValueError(
                    f"Regime {i} exog_names {regime.exog_names} differs from "
                    f"regime 0 exog_names {ref_exog_names}"
                )
            if regime.y_names != ref_y_names:
                raise ValueError(
                    f"Regime {i} y_names {regime.y_names} differs from "
                    f"regime 0 y_names {ref_y_names}"
                )
            if regime.terminal_condition != ref_terminal_condition:
                raise ValueError(
                    f"Regime {i} terminal_condition '{regime.terminal_condition}' "
                    f"differs from regime 0 terminal_condition '{ref_terminal_condition}'"
                )

        # Build spliced paths
        # time_list contains transition times: [t_1, t_2, ...]
        # Regime 0 runs from t=0 to t=t_1-1 (inclusive), contributing t_1 periods
        # Regime 1 runs from t=t_1 to t=t_2-1, but first period is duplicate, so skip
        # etc.

        spliced_UX_parts = []
        spliced_Z_parts = []
        spliced_Y_parts = [] if has_y_data else None
        current_time = 0

        for i, regime in enumerate(self.regimes):
            if i < len(self.time_list):
                # Not the last regime: take periods up to and including transition
                end_time = self.time_list[i]
                if i == 0:
                    # First regime: include all periods from 0 to transition time
                    n_periods = min(
                        end_time + 1, regime.UX.shape[0], T_max - current_time
                    )
                    start_idx = 0
                else:
                    # Subsequent regimes: skip period 0 (duplicate) and take up to transition
                    n_periods = min(
                        end_time, regime.UX.shape[0] - 1, T_max - current_time
                    )
                    start_idx = 1
            else:
                # Last regime: take remaining periods needed
                n_periods_available = regime.UX.shape[0]
                if i == 0:
                    n_periods = min(n_periods_available, T_max - current_time)
                    start_idx = 0
                else:
                    n_periods = min(n_periods_available - 1, T_max - current_time)
                    start_idx = 1

            if n_periods <= 0:
                break

            end_idx = start_idx + n_periods
            spliced_UX_parts.append(regime.UX[start_idx:end_idx, :])
            spliced_Z_parts.append(regime.Z[start_idx:end_idx, :])
            if has_y_data and regime.Y is not None:
                spliced_Y_parts.append(regime.Y[start_idx:end_idx, :])
            current_time += n_periods

            if current_time >= T_max:
                break

        # Concatenate all parts
        spliced_UX = np.vstack(spliced_UX_parts)
        spliced_Z = np.vstack(spliced_Z_parts)
        spliced_Y = np.vstack(spliced_Y_parts) if spliced_Y_parts else None

        # Determine convergence: all regimes must have converged
        all_converged = all(r.converged for r in self.regimes)

        # Use max final residual
        max_residual = max(r.final_residual for r in self.regimes)

        return DeterministicResult(
            UX=spliced_UX,
            Z=spliced_Z,
            Y=spliced_Y,
            model_label=self.model_label,
            var_names=ref_var_names,
            exog_names=ref_exog_names,
            y_names=ref_y_names,
            terminal_condition=ref_terminal_condition,
            converged=all_converged,
            final_residual=max_residual,
        )

    def _compute_default_T_max(self) -> int:
        """
        Compute a reasonable default T_max for splicing a SequenceResult.

        This returns the sum of contributions from each regime:
        - Regime 0 contributes time_list[0] + 1 periods (from t=0 to t=time_list[0])
        - Subsequent regimes contribute time_list[i] - time_list[i-1] periods
        - Last regime contributes its remaining periods (after skipping the first)

        Returns
        -------
        int
            The default T_max value.
        """
        if self.n_regimes == 0:
            return 0

        if self.n_regimes == 1:
            # Single regime: return its full length
            return self.regimes[0].UX.shape[0]

        total = 0
        for i, regime in enumerate(self.regimes):
            if i < len(self.time_list):
                if i == 0:
                    # First regime: include up to transition time (inclusive)
                    total += self.time_list[i] + 1
                else:
                    # Middle regimes: from previous transition +1 to current transition
                    # But skip first period (duplicate), so:
                    # periods = time_list[i] - time_list[i-1]
                    total += self.time_list[i] - self.time_list[i - 1]
            else:
                # Last regime: remaining periods (minus first which is duplicate)
                if i > 0:
                    total += regime.UX.shape[0] - 1
                else:
                    total += regime.UX.shape[0]

        return total

    def save(
        self,
        filepath: Optional[Union[str, Path]] = None,
        *,
        format: str = "npz",
        overwrite: bool = False,
        timestamp: bool = False,
    ) -> Path:
        """
        Save the sequence result to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save the result. If None, uses default path from settings
            with result_type="sequences".
        format : str, default "npz"
            Format for saving. Supported: 'npz', 'json'.
        overwrite : bool, default False
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = resolve_output_path(
            filepath,
            result_type="sequences",
            model_label=self.model_label,
            experiment_label=self.experiment_label,
            timestamp=timestamp,
            suffix=f".{format}",
        )

        # Build data dict with regime arrays
        data = {}
        regime_metadata = []

        for i, regime in enumerate(self.regimes):
            data[f"UX_regime_{i}"] = regime.UX
            data[f"Z_regime_{i}"] = regime.Z
            if regime.Y is not None:
                data[f"Y_regime_{i}"] = regime.Y
            regime_metadata.append(
                {
                    "model_label": regime.model_label,
                    "var_names": regime.var_names,
                    "exog_names": regime.exog_names,
                    "y_names": regime.y_names,
                    "terminal_condition": regime.terminal_condition,
                    "converged": regime.converged,
                    "final_residual": regime.final_residual,
                }
            )

        metadata = {
            "model_label": self.model_label,
            "experiment_label": self.experiment_label,
            "n_regimes": self.n_regimes,
            "time_list": self.time_list,
            "regime_metadata": regime_metadata,
            "result_type": "SequenceResult",
        }

        result_path = save_results(
            data, path, format=format, metadata=metadata, overwrite=overwrite
        )
        logger.info("Saved SequenceResult to %s", result_path)
        return result_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SequenceResult":
        """
        Load a SequenceResult from a saved file.

        Parameters
        ----------
        filepath : str or Path
            Path to the file to load.

        Returns
        -------
        SequenceResult
            Loaded result object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported or not a SequenceResult.
        """
        loaded = load_results(filepath)
        metadata = loaded.get("__metadata__", {})

        # Verify this is a SequenceResult
        result_type = metadata.get("result_type", "")
        if result_type and result_type != "SequenceResult":
            raise ValueError(
                f"Expected SequenceResult, got {result_type}. "
                f"Use the appropriate load method for {result_type}."
            )

        n_regimes = metadata.get("n_regimes", 0)
        time_list = metadata.get("time_list", [])
        regime_metadata = metadata.get("regime_metadata", [])
        model_label = metadata.get("model_label", "_default")
        experiment_label = metadata.get("experiment_label", "_default")

        regimes = []
        for i in range(n_regimes):
            regime_meta = regime_metadata[i] if i < len(regime_metadata) else {}
            y_key = f"Y_regime_{i}"
            regimes.append(
                DeterministicResult(
                    UX=np.asarray(loaded.get(f"UX_regime_{i}")),
                    Z=np.asarray(loaded.get(f"Z_regime_{i}")),
                    Y=np.asarray(loaded[y_key]) if y_key in loaded else None,
                    model_label=regime_meta.get("model_label", model_label),
                    var_names=regime_meta.get("var_names", []),
                    exog_names=regime_meta.get("exog_names", []),
                    y_names=regime_meta.get("y_names", []),
                    terminal_condition=regime_meta.get("terminal_condition", "stable"),
                    converged=regime_meta.get("converged", True),
                    final_residual=regime_meta.get("final_residual", 0.0),
                )
            )

        return cls(
            regimes=regimes,
            time_list=time_list,
            model_label=model_label,
            experiment_label=experiment_label,
        )

    def to_df(self, T_max: Optional[int] = None) -> "pd.DataFrame":
        """
        Convert the spliced sequence result to a pandas DataFrame.

        This is a convenience method that first splices the regimes together
        and then converts the result to a DataFrame.

        Parameters
        ----------
        T_max : int, optional
            The total length of the spliced path (number of time periods).
            If None, use a default computed from the regime lengths and
            transition times.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for all variables from the spliced result.
            Index is a RangeIndex representing time periods.

        Raises
        ------
        ImportError
            If pandas is not installed.
        ValueError
            If the sequence has no regimes, or if metadata is inconsistent
            across regimes.

        Examples
        --------
        >>> seq_result = deterministic.solve_sequence(spec, model, Nt=100)
        >>> df = seq_result.to_df()
        >>> df[['consumption', 'output']].plot()
        """
        spliced = self.splice(T_max=T_max)
        return spliced.to_df()
