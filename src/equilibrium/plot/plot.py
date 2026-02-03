from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from ..model.model import Model
    from ..solvers.results import (
        DeterministicResult,
        IrfResult,
        SequenceResult,
        SeriesTransform,
    )


# Overlay styling presets
OVERLAY_STYLE_PRESETS = {
    "dashdot": {"linestyle": "-.", "linewidth": 2.0, "use_markers": False},
    "dashed": {"linestyle": "--", "linewidth": 2.0, "use_markers": False},
    "dotted": {"linestyle": ":", "linewidth": 2.0, "use_markers": False},
    "markers": {
        "linestyle": "none",
        "marker": "o",
        "use_markers": True,
        "markersize": 5.0,
    },
}
DEFAULT_OVERLAY_COLOR = "black"
DEFAULT_OVERLAY_STYLE = "dashdot"  # Dash-dot line ('-.')


@dataclass(frozen=True)
class PlotSpec:
    """Container for commonly reused plot styling options."""

    var_titles: Optional[Dict[str, str]] = None
    group_colors: Optional[Dict[str, str]] = None
    group_styles: Optional[Dict[str, str]] = None
    marker_styles: Optional[Dict[str, str]] = None
    group_titles: Optional[Dict[str, str]] = None
    legend_include: Optional[Sequence[str]] = None
    legend_exclude: Optional[Sequence[str]] = None
    band_groups: Optional[Dict[str, Sequence[str]]] = None
    band_colors: Optional[Dict[str, str]] = None
    band_alphas: Optional[Dict[str, float]] = None
    zero_line: Optional[bool] = None
    zero_line_kwargs: Optional[Dict[str, Any]] = None
    series_transforms: Optional[
        Mapping[str, Union["SeriesTransform", Mapping[str, Any]]]
    ] = None
    plot_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.var_titles is not None:
            kwargs["var_titles"] = self.var_titles
        if self.group_colors is not None:
            kwargs["group_colors"] = self.group_colors
        if self.group_styles is not None:
            kwargs["group_styles"] = self.group_styles
        if self.marker_styles is not None:
            kwargs["marker_styles"] = self.marker_styles
        if self.group_titles is not None:
            kwargs["group_titles"] = self.group_titles
        if self.legend_include is not None:
            kwargs["legend_include"] = self.legend_include
        if self.legend_exclude is not None:
            kwargs["legend_exclude"] = self.legend_exclude
        if self.band_groups is not None:
            kwargs["band_groups"] = self.band_groups
        if self.band_colors is not None:
            kwargs["band_colors"] = self.band_colors
        if self.band_alphas is not None:
            kwargs["band_alphas"] = self.band_alphas
        if self.zero_line is not None:
            kwargs["zero_line"] = self.zero_line
        if self.zero_line_kwargs is not None:
            kwargs["zero_line_kwargs"] = self.zero_line_kwargs
        if self.plot_kwargs:
            kwargs.update(self.plot_kwargs)
        return kwargs


def plot_paths(
    path_vals: np.ndarray,
    full_list: Sequence[str],
    include_list: Sequence[str],
    title_str: Optional[str],
    x_str: str,
    prefix: str,
    group_names: Optional[Sequence[str]],
    plot_dir: Union[str, Path],
    *,
    plot_type: str = "pdf",
    alp: float = 1.0,
    style: str = "irf",
    max_per_page: int = 8,
    grid: Optional[np.ndarray] = None,
    n_per_row: int = 2,
    xbins: int = 5,
    ybins: Optional[int] = None,
    drop_obs: int = 0,
    group_colors: Optional[Dict[str, str]] = None,
    group_styles: Optional[Dict[str, str]] = None,
    var_titles: Optional[Dict[str, str]] = None,
    group_titles: Optional[Dict[str, str]] = None,
    bigfont: int = 12,
    smallfont: int = 10,
    leg_outside: bool = False,
    show_grid: bool = False,
    irf_limits: bool = False,
    plot_size: Optional[Tuple[float, float]] = None,
    use_markers: bool = True,
    marker_styles: Optional[Dict[str, str]] = None,
    fillstyle: str = "none",
    markersize: float = 3.0,
    mew: float = 1.5,
    lw: Optional[float] = None,
    marker_stride: Optional[int] = None,
    legend_include: Optional[Sequence[str]] = None,
    legend_exclude: Optional[Sequence[str]] = None,
    legend_loc: str = "best",
    zero_line: bool = False,
    zero_line_kwargs: Optional[Dict] = None,
    band_groups: Optional[Dict[str, Sequence[str]]] = None,
    band_colors: Optional[Dict[str, str]] = None,
    band_alphas: Optional[Dict[str, float]] = None,
    var_in_title: bool = False,
    clear_previous: bool = True,
) -> List[Path]:
    """
    Plot time-series paths by variable, optionally grouped with confidence bands.

    Parameters
    ----------
    path_vals : np.ndarray
        Array of shape ``(groups, periods, variables)``. Lower-dimensional input
        is automatically expanded so variables remain on the last axis.
    full_list : Sequence[str]
        Master list of variable names aligned with the third axis of
        ``path_vals``.
    include_list : Sequence[str]
        Ordered subset of variables from ``full_list`` to plot.
    title_str : str, optional
        Page title applied to the top row of subplots when ``var_in_title`` is
        ``False``.
    x_str : str
        Label for the horizontal axis.
    prefix : str
        Filename prefix for the paginated plots.
    group_names : Sequence[str], optional
        Names corresponding to the first axis of ``path_vals``. Used for
        legends, bands, and styling.
    plot_dir : Union[str, Path]
        Output directory for the rendered figures.
    plot_type : str, default "pdf"
        File extension for saved plots (e.g., "png", "pdf").
    alp : float, default 1.0
        Line transparency applied to plotted series.
    style : str, default ``\"irf\"``
        Named style switch used to toggle defaults for linewidth and layout.
    max_per_page : int, default 8
        Maximum number of subplots per page before starting a new figure.
    grid : np.ndarray, optional
        Custom x-axis grid. Defaults to ``np.arange(drop_obs, Nt)``.
    drop_obs : int, default 0
        Number of initial observations to omit from plotted series.
    band_groups : dict, optional
        Mapping from band labels to pairs of group names whose values bound the
        filled region.
    group_colors, group_styles, marker_styles : dict, optional
        Styling dictionaries keyed by entries in ``group_names``.
    legend_include, legend_exclude : Sequence[str], optional
        Control which groups appear in the legend. Only one of the arguments may
        be supplied.
    zero_line : bool, default False
        Whether to overlay a horizontal reference line at zero.
    clear_previous : bool, default True
        Remove previously generated ``prefix``-matching files before saving new
        pages.

    Returns
    -------
    list[Path]
        Paths to the saved plot files.

    Raises
    ------
    ValueError
        If a requested variable or band specification cannot be resolved.
    """

    # --- defaults / normalization ---
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plot_dir.resolve()}")

    if not plot_type.startswith("."):
        plot_type = "." + plot_type

    group_colors = group_colors or {}
    group_styles = group_styles or {}
    var_titles = var_titles or {}
    marker_styles = marker_styles or {}
    zero_line_kwargs = {
        "color": "black",
        "linewidth": 1,
        "linestyle": ":",
    } | (zero_line_kwargs or {})
    band_groups = band_groups or {}
    band_colors = band_colors or {}
    band_alphas = band_alphas or {}

    # shape: ensure (G, T, V)
    arr = np.asarray(path_vals)
    while arr.ndim < 3:
        arr = arr[None, ...]
    n_group, nt, n_vars = arr.shape

    if grid is None:
        grid = np.arange(drop_obs, nt, dtype=int)
    grid = np.asarray(grid)

    if lw is None:
        lw = 1.5 if style == "irf" else 1.0

    if marker_stride is None:
        marker_stride = max(1, round(len(grid) / 20))

    # --- legend include/exclude and bands filtering ---
    # Flatten band component groups to filter legend entries
    bands_list = [g for groups in band_groups.values() for g in groups]
    if group_names:
        base_group_names = [g for g in group_names if g not in bands_list]
    else:
        base_group_names = None

    if base_group_names is not None:
        if group_titles is None:
            group_titles = {g: g for g in base_group_names}
        if legend_exclude is not None and legend_include is not None:
            raise ValueError(
                "Provide either legend_include or legend_exclude, not both."
            )
        if legend_exclude is not None:
            legend_include = [
                g for g in base_group_names if g not in set(legend_exclude)
            ]
        elif legend_include is None:
            legend_include = list(base_group_names)
    add_legend = bool(
        base_group_names
        and any(
            g in (group_titles or {}) and g in (legend_include or [])
            for g in base_group_names
        )
    )

    # --- indices (precompute) ---
    var_indices: List[int] = []
    for v in include_list:
        if v not in full_list:
            raise ValueError(f"Variable '{v}' not found in full_list.")
        var_indices.append(full_list.index(v))

    group_index = {}
    if group_names is not None:
        group_index = {g: group_names.index(g) for g in group_names}

    # --- layout helpers ---
    max_per_page = min(max_per_page, len(include_list))
    n_plots_total = len(include_list)

    if ybins is None:
        ybins = 5 if max_per_page < 5 else 4

    # --- clear previously saved pages, if requested ---
    saved_paths: List[Path] = []
    if clear_previous:
        for f in plot_dir.glob(f"{prefix}_page_*{plot_type}"):
            try:
                f.unlink()
            except OSError:
                pass

    # --- pagination loop ---
    total_done = 0
    page = 1
    while total_done < n_plots_total:
        remaining = n_plots_total - total_done
        n_this_page = min(max_per_page, remaining)
        rows_this_page = ((n_this_page - 1) // n_per_row) + 1

        # figure
        default_plot_size = (3.0 * n_per_row, 3.0 * rows_this_page)
        figsize = plot_size if plot_size else default_plot_size
        fig, axes = plt.subplots(
            nrows=math.ceil(n_this_page / n_per_row),
            ncols=n_per_row,
            figsize=figsize,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()

        # draw panels for this page
        for i in range(n_this_page):
            ax = axes[i]
            vname = include_list[total_done + i]
            vidx = var_indices[total_done + i]
            title_text = var_titles.get(vname, vname)

            # compute y-lims if irf_limits requested (collect across groups we will plot)
            if irf_limits:
                ydata = []
                if base_group_names is None:
                    # no named groups; plot the first group's series
                    ydata.append(arr[0, drop_obs:, vidx])
                else:
                    for g in base_group_names:
                        ig = group_index[g]
                        ydata.append(arr[ig, drop_obs:, vidx])
                        if g in band_groups:
                            comp = band_groups[g]
                            if len(comp) != 2:
                                raise ValueError(
                                    f"Band for '{g}' must have exactly 2 component groups."
                                )
                            for gg in comp:
                                ydata.append(arr[group_index[gg], drop_obs:, vidx])
                ymin = np.min([np.min(y) for y in ydata]) if ydata else -1e-6
                ymax = np.max([np.max(y) for y in ydata]) if ydata else 1e-6
                if ymax - ymin < 1e-12:
                    ymin -= 1e-6
                    ymax += 1e-6
                spread = ymax - ymin
                ylim = (ymin - 0.1 * spread, ymax + 0.1 * spread)
            else:
                ylim = None

            # zero line
            if zero_line:
                ax.plot(grid, np.zeros_like(grid, dtype=float), **zero_line_kwargs)

            # draw lines per (base) group
            if base_group_names is not None:
                for count_group, g in enumerate(base_group_names):
                    ig = group_index[g]
                    color = group_colors.get(g, None)
                    ls = group_styles.get(g, "-")
                    marker = marker_styles.get(g, None) if use_markers else None
                    label = (
                        group_titles.get(g)
                        if (legend_include and g in legend_include)
                        else None
                    )

                    ax.plot(
                        grid,
                        arr[ig, drop_obs:, vidx],
                        linewidth=lw,
                        color=color,
                        linestyle=ls,
                        alpha=alp,
                        marker=marker,
                        mew=mew,
                        markersize=markersize,
                        fillstyle=fillstyle,
                        label=label,
                        markevery=marker_stride,
                    )

                    # bands (two series to fill between)
                    if g in band_groups:
                        comp = band_groups[g]
                        if len(comp) != 2:
                            raise ValueError(
                                f"Band for '{g}' must have exactly 2 component groups."
                            )
                        i0, i1 = (group_index[comp[0]], group_index[comp[1]])
                        y0 = arr[i0, drop_obs:, vidx]
                        y1 = arr[i1, drop_obs:, vidx]
                        band_color = band_colors.get(
                            g, color if color is not None else "gray"
                        )
                        band_alpha = band_alphas.get(g, 0.25)
                        ax.fill_between(
                            grid, y0, y1, color=band_color, alpha=band_alpha
                        )
            else:
                # no named groups; draw group 0
                ax.plot(
                    grid,
                    arr[0, drop_obs:, vidx],
                    linewidth=lw,
                    linestyle="-",
                    alpha=alp,
                )

            # labels & limits
            if var_in_title:
                ax.set_title(title_text, fontsize=smallfont)
            else:
                ax.set_ylabel(title_text, fontsize=smallfont)

            ax.set_xlim(grid.min(), grid.max())
            if ylim is not None:
                ax.set_ylim(*ylim)

            if show_grid:
                ax.grid(True)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))
            ax.ticklabel_format(useOffset=False)

            # page-level title on top row if requested
            if (title_str is not None) and (not var_in_title):
                # top row panels
                n_cols = n_per_row
                if i < n_cols:
                    ax.set_title(title_str, fontsize=bigfont)

            # bottom row: x label
            n_rows = math.ceil(n_this_page / n_per_row)
            last_row_start = (n_rows - 1) * n_per_row
            if i >= last_row_start:
                ax.set_xlabel(x_str, fontsize=smallfont)

        # remove any unused axes on the last page
        for j in range(n_this_page, len(axes)):
            fig.delaxes(axes[j])

        # legend (once per page)
        if add_legend:
            # choose a visible axis to anchor the legend bbox if outside
            host_ax = axes[min(n_this_page - 1, len(axes) - 1)]
            if leg_outside:
                lgd = host_ax.legend(fontsize=smallfont - 2, bbox_to_anchor=(1.5, 1.1))
                save_kwargs = {"bbox_extra_artists": (lgd,), "bbox_inches": "tight"}
            else:
                host_ax.legend(fontsize=smallfont - 2, loc=legend_loc)
                save_kwargs = {}
        else:
            save_kwargs = {}

        # save this page
        outpath = plot_dir / f"{prefix}_page_{page}{plot_type}"
        fig.savefig(outpath, **save_kwargs)
        plt.close(fig)
        saved_paths.append(outpath)

        total_done += n_this_page
        page += 1

    return saved_paths


def overlay_to_result(
    overlay_data: Union[np.ndarray, Mapping[str, np.ndarray]],
    overlay_var_names: Optional[Sequence[str]] = None,
    overlay_name: str = "Data",
    reference_result: Optional["DeterministicResult"] = None,
    fill_missing: bool = True,
) -> "DeterministicResult":
    """
    Convert external overlay data to DeterministicResult format for plotting.

    This helper function enables overlaying empirical data, alternative model outputs,
    or other external time series on plots generated by plot_deterministic_results().
    The returned DeterministicResult can be appended to a results list for comparison.

    Parameters
    ----------
    overlay_data : np.ndarray or Mapping[str, np.ndarray]
        External data to overlay. Either a 2D array of shape (n_periods, n_vars)
        or a dictionary mapping variable names to 1D arrays of length n_periods.
    overlay_var_names : Sequence[str], optional
        Variable names corresponding to columns of overlay_data when overlay_data
        is an array. Required if overlay_data is an array; ignored if it's a dict.
    overlay_name : str, default "Data"
        Label for the overlay series (used in legends and saved result metadata).
    reference_result : DeterministicResult, optional
        Reference result used to align variable names and fill missing columns.
        If provided and fill_missing=True, the overlay will be padded with NaN
        columns for variables in the reference that are not in the overlay.
    fill_missing : bool, default True
        Whether to add NaN columns for variables present in reference_result but
        not in overlay_data. Only used when reference_result is provided.

    Returns
    -------
    DeterministicResult
        Result container with:
        - UX: overlay data (possibly padded with NaN for missing variables)
        - Z: zeros placeholder of shape (n_periods, 1)
        - var_names: variable names from overlay
        - model_label: overlay_name
        - Minimal metadata (converged=True, final_residual=0.0)

    Raises
    ------
    ValueError
        If overlay_data is an array but overlay_var_names is not provided, or if
        arrays in a dict have inconsistent lengths.
    TypeError
        If overlay_data is neither an array nor a dict.

    Examples
    --------
    >>> # Create overlay from dict
    >>> empirical = {'consumption': np.array([1.0, 1.1, 1.2]),
    ...              'output': np.array([2.0, 2.1, 2.2])}
    >>> overlay_result = overlay_to_result(empirical, overlay_name="Empirical")
    >>> # Create overlay from array
    >>> data_array = np.random.randn(50, 3)
    >>> overlay_result = overlay_to_result(
    ...     data_array,
    ...     overlay_var_names=['c', 'y', 'k'],
    ...     overlay_name="Historical",
    ... )
    >>> # Align with reference result
    >>> overlay_result = overlay_to_result(
    ...     empirical,
    ...     overlay_name="Data",
    ...     reference_result=simulation_result,
    ...     fill_missing=True,
    ... )
    """
    from ..solvers.results import DeterministicResult

    # Convert overlay_data to dict format
    if isinstance(overlay_data, dict):
        # Convert all values to numeric arrays, handling pandas object dtypes
        overlay_dict = {}
        for key, value in overlay_data.items():
            try:
                overlay_dict[key] = np.asarray(value, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert overlay variable '{key}' to numeric. "
                    f"Ensure all overlay data is numeric. Original error: {e}"
                )

        # Validate consistent lengths
        if overlay_dict:
            lengths = {k: len(v) for k, v in overlay_dict.items()}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                raise ValueError(
                    f"All arrays in overlay_data dict must have the same length. "
                    f"Got lengths: {lengths}"
                )
        var_names = list(overlay_dict.keys())
    elif isinstance(overlay_data, np.ndarray):
        if overlay_var_names is None:
            raise ValueError(
                "overlay_var_names must be provided when overlay_data is an array"
            )
        try:
            overlay_arr = np.asarray(overlay_data, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert overlay_data array to numeric. "
                f"Ensure all overlay data is numeric. Original error: {e}"
            )
        if overlay_arr.ndim == 1:
            overlay_arr = overlay_arr[:, None]
        elif overlay_arr.ndim != 2:
            raise ValueError(
                f"overlay_data array must be 1D or 2D, got {overlay_arr.ndim}D"
            )
        n_periods, n_vars = overlay_arr.shape
        if len(overlay_var_names) != n_vars:
            raise ValueError(
                f"overlay_var_names length ({len(overlay_var_names)}) must match "
                f"number of columns in overlay_data ({n_vars})"
            )
        var_names = list(overlay_var_names)
        overlay_dict = {name: overlay_arr[:, i] for i, name in enumerate(var_names)}
    else:
        raise TypeError(
            f"overlay_data must be a numpy array or dict, got {type(overlay_data)}"
        )

    # Align with reference result if provided
    if reference_result is not None and fill_missing:
        # Get all variable names from reference (UX and Y)
        ref_var_names = list(reference_result.var_names)
        if reference_result.Y is not None:
            ref_var_names.extend(reference_result.y_names)

        # Build aligned array with reference variable order
        n_periods = len(next(iter(overlay_dict.values())))
        aligned_var_names = []
        aligned_arrays = []

        for ref_var in ref_var_names:
            aligned_var_names.append(ref_var)
            if ref_var in overlay_dict:
                # Force numeric conversion to handle pandas object dtypes
                try:
                    aligned_arrays.append(
                        np.asarray(overlay_dict[ref_var], dtype=np.float64)
                    )
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot convert overlay variable '{ref_var}' to numeric. "
                        f"Ensure all overlay data is numeric. Original error: {e}"
                    )
            else:
                # Fill missing with NaN
                aligned_arrays.append(np.full(n_periods, np.nan))

        # Add overlay variables not in reference (append at end)
        for overlay_var in var_names:
            if overlay_var not in ref_var_names:
                aligned_var_names.append(overlay_var)
                try:
                    aligned_arrays.append(
                        np.asarray(overlay_dict[overlay_var], dtype=np.float64)
                    )
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot convert overlay variable '{overlay_var}' to numeric. "
                        f"Ensure all overlay data is numeric. Original error: {e}"
                    )

        UX = np.column_stack(aligned_arrays)
        var_names = aligned_var_names
    else:
        # No alignment, just stack in original order
        arrays = []
        for name in var_names:
            try:
                arrays.append(np.asarray(overlay_dict[name], dtype=np.float64))
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert overlay variable '{name}' to numeric. "
                    f"Ensure all overlay data is numeric. Original error: {e}"
                )
        UX = np.column_stack(arrays) if len(arrays) > 1 else arrays[0][:, None]

    # Ensure UX is 2D
    if UX.ndim == 1:
        UX = UX[:, None]

    n_periods = UX.shape[0]

    # Create placeholder Z (zeros)
    Z = np.zeros((n_periods, 1))

    # Create DeterministicResult
    return DeterministicResult(
        UX=UX,
        Z=Z,
        var_names=var_names,
        exog_names=["_placeholder"],
        y_names=[],
        Y=None,
        model_label=overlay_name,
        converged=True,
        final_residual=0.0,
    )


def _build_overlay_plot_spec(
    overlay_name: str,
    overlay_style: Optional[str],
    overlay_color: Optional[str],
    overlay_kwargs: Optional[Dict[str, Any]],
    overlay_spec: Optional[PlotSpec],
) -> Dict[str, Any]:
    """
    Build plot styling specifications for overlay data.

    Handles priority resolution: overlay_spec > overlay_kwargs > overlay_style/color > defaults.

    Parameters
    ----------
    overlay_name : str
        Name of the overlay group (used as key in styling dicts).
    overlay_style : str, optional
        Style preset name ('dashdot', 'dashed', 'dotted', 'markers').
    overlay_color : str, optional
        Color for the overlay series.
    overlay_kwargs : dict, optional
        Direct styling kwargs (e.g., {'linestyle': '-.', 'linewidth': 3.0}).
    overlay_spec : PlotSpec, optional
        Full PlotSpec with complete control over styling.

    Returns
    -------
    dict
        Dictionary with 'group_colors', 'group_styles', and 'marker_styles' keys,
        each mapping overlay_name to appropriate styling values.
    """
    result: Dict[str, Any] = {
        "group_colors": {},
        "group_styles": {},
        "marker_styles": {},
    }

    # Priority 1: overlay_spec (full control)
    if overlay_spec is not None:
        if overlay_spec.group_colors and overlay_name in overlay_spec.group_colors:
            result["group_colors"][overlay_name] = overlay_spec.group_colors[
                overlay_name
            ]
        if overlay_spec.group_styles and overlay_name in overlay_spec.group_styles:
            result["group_styles"][overlay_name] = overlay_spec.group_styles[
                overlay_name
            ]
        if overlay_spec.marker_styles and overlay_name in overlay_spec.marker_styles:
            result["marker_styles"][overlay_name] = overlay_spec.marker_styles[
                overlay_name
            ]
        return result

    # Priority 2: overlay_kwargs (direct control)
    if overlay_kwargs is not None:
        # Extract relevant styling from kwargs
        if "color" in overlay_kwargs:
            result["group_colors"][overlay_name] = overlay_kwargs["color"]
        elif overlay_color is not None:
            result["group_colors"][overlay_name] = overlay_color
        else:
            result["group_colors"][overlay_name] = DEFAULT_OVERLAY_COLOR

        if "linestyle" in overlay_kwargs:
            result["group_styles"][overlay_name] = overlay_kwargs["linestyle"]

        if "marker" in overlay_kwargs:
            result["marker_styles"][overlay_name] = overlay_kwargs["marker"]

        # Note: Other kwargs like linewidth, alpha, markersize will be passed
        # through via the main kwargs dict, not through group-specific styling
        return result

    # Priority 3: overlay_style preset + overlay_color
    style_preset = OVERLAY_STYLE_PRESETS.get(
        overlay_style or DEFAULT_OVERLAY_STYLE,
        OVERLAY_STYLE_PRESETS[DEFAULT_OVERLAY_STYLE],
    )

    result["group_colors"][overlay_name] = overlay_color or DEFAULT_OVERLAY_COLOR

    if "linestyle" in style_preset:
        result["group_styles"][overlay_name] = style_preset["linestyle"]

    if "marker" in style_preset:
        result["marker_styles"][overlay_name] = style_preset["marker"]

    return result


def plot_deterministic_results(
    results: Optional[Sequence[Union["DeterministicResult", "SequenceResult"]]] = None,
    include_list: Optional[Sequence[str]] = None,
    plot_dir: Optional[Union[str, Path]] = None,
    subdir_name: Optional[str] = None,
    *,
    T_max: Optional[int] = None,
    series_transforms: Optional[
        Mapping[str, Union[SeriesTransform, Mapping[str, Any]]]
    ] = None,
    result_labels: Optional[Sequence[tuple[str, Optional[str]]]] = None,
    result_kind: str = "sequence",
    save_dir: Optional[Union[str, Path]] = None,
    result_names: Optional[Sequence[str]] = None,
    prefix: str = "results",
    plot_type: str = "pdf",
    title_str: Optional[str] = None,
    x_str: str = "Period",
    plot_spec: Optional[PlotSpec] = None,
    overlay_data: Optional[Union[np.ndarray, Mapping[str, np.ndarray]]] = None,
    overlay_var_names: Optional[Sequence[str]] = None,
    overlay_name: str = "Data",
    overlay_style: Optional[str] = None,
    overlay_color: Optional[str] = None,
    overlay_kwargs: Optional[Dict[str, Any]] = None,
    overlay_spec: Optional[PlotSpec] = None,
    **kwargs,
) -> List[Path]:
    """
    Plot a list of DeterministicResult or SequenceResult objects.

    For each SequenceResult, the regimes are spliced together into a continuous
    DeterministicResult before plotting. The variable names across all results
    are harmonized; if a variable is missing in a particular result, its values
    are filled with NaN.

    Parameters
    ----------
    results : Sequence[DeterministicResult | SequenceResult], optional
        List of result objects to plot. SequenceResult objects will be spliced
        into DeterministicResult objects using the splice method.
    result_labels : Sequence[tuple[str, Optional[str]]], optional
        List of (model_label, experiment_label) pairs to load and plot. Loaded
        results are appended after any explicit ``results``.
    result_kind : str, default "sequence"
        Type of labeled results to load: "sequence" or "deterministic".
    save_dir : str or Path, optional
        Base directory used to load labeled results. Defaults to settings.
    include_list : Sequence[str], optional
        List of variable names to plot. Variables not present in a result will
        be filled with NaN. If None, defaults to all variables present in
        the results.
    plot_dir : str or Path, optional
        Directory where plot files will be saved. If None, defaults to a
        subdirectory called "deterministic" inside the plot_dir specified
        in Settings. A label-based subdirectory is appended to the base
        directory (whether defaulted or user-provided).
    subdir_name : str, optional
        Subdirectory name appended to the base plot directory. Defaults to a
        label-derived name when not provided.
    T_max : int, optional
        Maximum number of time periods to include when splicing SequenceResult
        objects. If None, SequenceResults use a default computed from their
        regime structure (full splice length), and the final plot uses the
        minimum path length across all processed results.
    series_transforms : dict[str, SeriesTransform or dict], optional
        Per-series transform specifications keyed by series name. Applies
        across UX, Z, and Y names for each result.
    result_names : Sequence[str], optional
        Names for each result (used in legends). If None, defaults to
        explicit "Result {i}" entries followed by label-derived names
        for any ``result_labels``.
    prefix : str, default "results"
        Filename prefix for the saved plots.
    plot_type : str, default "pdf"
        File extension for saved plots (e.g., "png", "pdf").
    title_str : str, optional
        Title string to display on plots.
    x_str : str, default "Period"
        Label for the x-axis.
    plot_spec : PlotSpec, optional
        Bundle of plot styling options passed through to plot_paths, plus
        optional series_transforms when not provided directly.
    overlay_data : np.ndarray or dict[str, np.ndarray], optional
        External data to overlay on plots (e.g., empirical data for comparison).
        Either a 2D array of shape (n_periods, n_vars) or a dict mapping variable
        names to 1D arrays. When provided, this data is converted to a
        DeterministicResult and appended to the results list with special styling.
    overlay_var_names : Sequence[str], optional
        Variable names for overlay_data when it's an array. Required if
        overlay_data is an array; ignored if it's a dict.
    overlay_name : str, default "Data"
        Name for the overlay series (used in legends).
    overlay_style : str, optional
        Style preset for overlay: 'dashdot' (default), 'dashed', 'dotted', or
        'markers'. Determines linestyle and marker behavior.
    overlay_color : str, optional
        Color for overlay series. Defaults to 'black'.
    overlay_kwargs : dict, optional
        Direct styling kwargs for overlay (e.g., {'linestyle': '-.', 'linewidth':
        3.0, 'alpha': 0.8}). Has higher priority than overlay_style/overlay_color.
    overlay_spec : PlotSpec, optional
        Full PlotSpec control for overlay styling. Has highest priority, overriding
        all other overlay styling parameters.
    **kwargs
        Additional keyword arguments passed to plot_paths.

    Returns
    -------
    list[Path]
        Paths to the saved plot files.

    Raises
    ------
    ValueError
        If results is empty, if overlay_data is an array but overlay_var_names
        is not provided, or if arrays in overlay_data dict have inconsistent lengths.

    Examples
    --------
    >>> from equilibrium.plot import plot_deterministic_results
    >>> from equilibrium.solvers.results import DeterministicResult, SequenceResult
    >>> # Plot deterministic results with defaults (all variables, default plot_dir)
    >>> paths = plot_deterministic_results([det_result1, det_result2])
    >>> # Plot with specific variables and custom directory
    >>> paths = plot_deterministic_results(
    ...     [det_result1, det_result2],
    ...     include_list=["consumption", "output"],
    ...     plot_dir="/tmp/plots",
    ...     result_names=["Baseline", "Alternative"],
    ... )
    >>> # Plot sequence results (will be spliced automatically)
    >>> paths = plot_deterministic_results(
    ...     [seq_result],
    ...     include_list=["consumption", "output"],
    ...     plot_dir="/tmp/plots",
    ...     T_max=100,
    ... )
    >>> # Overlay empirical data (default style: black dash-dot line)
    >>> empirical = {'consumption': np.array([...]), 'output': np.array([...])}
    >>> paths = plot_deterministic_results(
    ...     [simulation_result],
    ...     overlay_data=empirical,
    ...     overlay_name="Empirical Data",
    ...     result_names=["Model"],
    ... )
    >>> # Overlay with custom styling via kwargs
    >>> paths = plot_deterministic_results(
    ...     [simulation_result],
    ...     overlay_data=empirical,
    ...     overlay_name="Data",
    ...     overlay_kwargs={'linestyle': ':', 'linewidth': 2.5, 'color': 'red'},
    ... )
    >>> # Overlay with style presets
    >>> paths = plot_deterministic_results(
    ...     [simulation_result],
    ...     overlay_data=data_array,
    ...     overlay_var_names=['c', 'y', 'k'],
    ...     overlay_style='markers',  # or 'dashed', 'dotted', 'dashdot'
    ...     overlay_color='navy',
    ... )
    >>> # Overlay standalone (no simulation results)
    >>> paths = plot_deterministic_results(
    ...     overlay_data=empirical,
    ...     overlay_name="Historical Data",
    ... )
    """
    # Import here to avoid circular imports
    from ..settings import get_settings
    from ..solvers.results import DeterministicResult, SequenceResult
    from ..utils.io import load_deterministic_result, load_sequence_result

    results_list: List[DeterministicResult] = []
    auto_names: List[str] = []
    label_parts: List[str] = []

    if results is not None:
        for result in results:
            if isinstance(result, SequenceResult):
                results_list.append(result.splice(T_max=T_max))
            elif isinstance(result, DeterministicResult):
                results_list.append(result)
            else:
                raise TypeError(
                    "results must contain DeterministicResult or SequenceResult, "
                    f"got {type(result).__name__}."
                )
        auto_names.extend([f"Result {i}" for i in range(len(results_list))])

    if result_labels:
        for model_label, experiment_label in result_labels:
            if result_kind == "sequence":
                loaded = load_sequence_result(
                    model_label,
                    experiment_label,
                    save_dir=save_dir,
                    splice=True,
                    T_max=T_max,
                )
            elif result_kind == "deterministic":
                loaded = load_deterministic_result(
                    model_label, experiment_label, save_dir=save_dir
                )
            else:
                raise ValueError(
                    "result_kind must be 'sequence' or 'deterministic', "
                    f"got '{result_kind}'."
                )
            results_list.append(loaded)
            label_str = (
                f"{model_label}_{experiment_label}" if experiment_label else model_label
            )
            auto_names.append(label_str)
            label_parts.append(label_str)

    # Process overlay data if provided
    if overlay_data is not None:
        # Get reference result for alignment (first result if available)
        reference_result = results_list[0] if results_list else None

        # Convert overlay to DeterministicResult
        overlay_result = overlay_to_result(
            overlay_data=overlay_data,
            overlay_var_names=overlay_var_names,
            overlay_name=overlay_name,
            reference_result=reference_result,
            fill_missing=True,
        )

        # Append overlay result to results list
        results_list.append(overlay_result)
        auto_names.append(overlay_name)

        # Build overlay styling specification
        overlay_style_spec = _build_overlay_plot_spec(
            overlay_name=overlay_name,
            overlay_style=overlay_style,
            overlay_color=overlay_color,
            overlay_kwargs=overlay_kwargs,
            overlay_spec=overlay_spec,
        )

        # Merge overlay styling with existing plot_spec
        if plot_spec is not None:
            # Convert plot_spec to kwargs and merge with overlay styling
            plot_spec_kwargs = plot_spec.to_kwargs()

            # Merge group_colors
            if "group_colors" in plot_spec_kwargs or overlay_style_spec["group_colors"]:
                existing_colors = plot_spec_kwargs.get("group_colors", {}) or {}
                plot_spec_kwargs["group_colors"] = {
                    **existing_colors,
                    **overlay_style_spec["group_colors"],
                }

            # Merge group_styles
            if "group_styles" in plot_spec_kwargs or overlay_style_spec["group_styles"]:
                existing_styles = plot_spec_kwargs.get("group_styles", {}) or {}
                plot_spec_kwargs["group_styles"] = {
                    **existing_styles,
                    **overlay_style_spec["group_styles"],
                }

            # Merge marker_styles
            if (
                "marker_styles" in plot_spec_kwargs
                or overlay_style_spec["marker_styles"]
            ):
                existing_markers = plot_spec_kwargs.get("marker_styles", {}) or {}
                plot_spec_kwargs["marker_styles"] = {
                    **existing_markers,
                    **overlay_style_spec["marker_styles"],
                }

            # Update kwargs with merged plot_spec
            kwargs = {**plot_spec_kwargs, **kwargs}
            # Set plot_spec to None since we've merged it into kwargs
            plot_spec = None
        else:
            # No existing plot_spec, just add overlay styling to kwargs
            for key, value in overlay_style_spec.items():
                if value:  # Only add non-empty dicts
                    if key in kwargs:
                        # Merge with existing kwargs
                        kwargs[key] = {**kwargs[key], **value}
                    else:
                        kwargs[key] = value

    if not results_list:
        raise ValueError("results must be a non-empty sequence")

    # Set default plot_dir from Settings
    if subdir_name is not None:
        subdir_name = str(subdir_name).strip() or None

    if plot_dir is None:
        settings = get_settings()
        base_plot_dir = settings.paths.plot_dir / "deterministic"
    else:
        base_plot_dir = Path(plot_dir)

    if subdir_name is None:
        if label_parts:
            subdir_name = "_".join(label_parts)
        else:
            subdir_name = "_default"

    plot_dir = base_plot_dir / subdir_name if subdir_name else base_plot_dir

    # Convert SequenceResults to DeterministicResults via splice
    processed_results: List[DeterministicResult] = []
    for i, result in enumerate(results_list):
        if isinstance(result, DeterministicResult):
            processed_results.append(result)
        else:
            raise TypeError(
                f"Result at index {i} has unsupported type {type(result).__name__}. "
                "Expected DeterministicResult."
            )

    if series_transforms is None and plot_spec is not None:
        series_transforms = plot_spec.series_transforms

    if series_transforms:
        processed_results = [
            result.transform(series_transforms=series_transforms)
            for result in processed_results
        ]

    # Set default result names
    if result_names is None:
        result_names = auto_names
    else:
        # Validate result_names length
        # If overlay was added, result_names should match original results (not including overlay)
        expected_length = len(processed_results) - (
            1 if overlay_data is not None else 0
        )
        if len(result_names) != expected_length:
            raise ValueError(
                f"result_names length ({len(result_names)}) must match "
                f"number of non-overlay results ({expected_length})"
            )
        # Append overlay name if overlay was added
        if overlay_data is not None:
            result_names = list(result_names) + [overlay_name]

    # Determine the union of all variable names that exist in any result
    # Include both UX variables (var_names) and intermediate variables (y_names)
    all_var_names: List[str] = []
    for result in processed_results:
        for name in result.var_names:
            if name not in all_var_names:
                all_var_names.append(name)

    for result in processed_results:
        if isinstance(result, SequenceResult):
            label = f"{result.model_label}_{result.experiment_label}"
        else:
            label = result.model_label
        label_parts.append(label)
        for name in result.y_names:
            if name not in all_var_names:
                all_var_names.append(name)

    # Set default include_list to all variables if not provided
    if include_list is None:
        include_list = all_var_names

    # Filter to only variables in include_list (preserving include_list order)
    full_list = [v for v in include_list if v in all_var_names]

    if not full_list:
        raise ValueError(
            f"None of the variables in include_list {list(include_list)} "
            f"are present in any result. Available variables: {all_var_names}"
        )

    # Determine the common time dimension (min across all results)
    n_periods = min(r.UX.shape[0] for r in processed_results)

    # Build path_vals array: shape (n_results, n_periods, n_vars)
    n_results = len(processed_results)
    n_vars = len(full_list)
    path_vals = np.full((n_results, n_periods, n_vars), np.nan)

    for i, result in enumerate(processed_results):
        for j, var_name in enumerate(full_list):
            if var_name in result.var_names:
                var_idx = result.var_names.index(var_name)
                # Take up to n_periods from this result
                path_vals[i, :, j] = result.UX[:n_periods, var_idx]
            elif var_name in result.y_names and result.Y is not None:
                # Variable is in intermediate variables (Y)
                var_idx = result.y_names.index(var_name)
                path_vals[i, :, j] = result.Y[:n_periods, var_idx]

    # Call plot_paths with the harmonized data
    plot_kwargs = plot_spec.to_kwargs() if plot_spec is not None else {}
    merged_kwargs = {**plot_kwargs, **kwargs}

    return plot_paths(
        path_vals=path_vals,
        full_list=full_list,
        include_list=full_list,
        title_str=title_str,
        x_str=x_str,
        prefix=prefix,
        group_names=list(result_names),
        plot_dir=plot_dir,
        plot_type=plot_type,
        **merged_kwargs,
    )


def plot_model_irfs(
    models: Optional[Sequence["Model"]] = None,
    shock: Optional[str] = None,
    shocks: Optional[Sequence[str]] = None,
    include_list: Optional[Sequence[str]] = None,
    plot_dir: Optional[Union[str, Path]] = None,
    *,
    model_labels: Optional[Sequence[str]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    model_names: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
    plot_type: str = "pdf",
    title_str: Optional[str] = None,
    x_str: str = "Period",
    n_periods: Optional[int] = None,
    shock_size: Union[float, dict[str, float]] = 1.0,
    plot_spec: Optional[PlotSpec] = None,
    **kwargs,
) -> List[Path]:
    """
    Plot impulse response functions from multiple Model objects for multiple shocks.

    If a shock is not present in a model's exog_list, NaN values are used for
    that model's IRFs.

    Parameters
    ----------
    models : Sequence[Model], optional
        List of Model objects to plot IRFs from. Each model must have been
        linearized with ``model.linearize()`` and have IRFs computed via
        ``model.compute_linear_irfs()``.
    model_labels : Sequence[str], optional
        Model labels to load IRFs from disk and plot. Used when ``models`` is
        not provided.
    save_dir : str or Path, optional
        Base directory used to load labeled IRFs. Defaults to settings.
    shock : str, optional
        Name of a single exogenous shock to plot IRFs for. Cannot be used
        together with ``shocks``.
    shocks : Sequence[str], optional
        List of exogenous shock names to plot IRFs for. If both ``shock`` and
        ``shocks`` are None, defaults to the union of all shocks (exog_list)
        from all models. Cannot be used together with ``shock``.
    include_list : Sequence[str], optional
        List of variable names to plot. Variables not present in a model will
        be filled with NaN. If None, defaults to all variables present in
        the first model.
    plot_dir : str or Path, optional
        Directory where plot files will be saved. If None, defaults to
        ``settings.paths.plot_dir / "irfs" / "{sorted_model_labels}"`` where
        model labels are sorted alphabetically and joined with underscores.
    model_names : Sequence[str], optional
        Names for each model (used in legends). If None, uses each model's
        ``label`` attribute.
    prefix : str, optional
        Filename prefix for the saved plots. If None, defaults to
        "irf_to_{shock}" for each shock.
    plot_type : str, default "pdf"
        File extension for saved plots (e.g., "png", "pdf").
    title_str : str, optional
        Title string to display on plots. If None, defaults to
        "Impulse Response to {shock}" for each shock.
    x_str : str, default "Period"
        Label for the x-axis.
    n_periods : int, optional
        Number of periods to plot. If None, uses the minimum IRF horizon
        across all models.
    shock_size : float or dict[str, float], default 1.0
        Scaling factor(s) applied to the IRFs before plotting. If a float,
        the same size is used for all shocks. If a dict, maps shock names
        to their sizes. For shocks not in the dict, defaults to the shock's
        standard deviation from the first model's parameters (SIG_<shock>).
    plot_spec : PlotSpec, optional
        Bundle of plot styling options passed through to plot_paths.
    **kwargs
        Additional keyword arguments passed to plot_paths.

    Returns
    -------
    list[Path]
        Paths to all saved plot files across all shocks.

    Raises
    ------
    ValueError
        If models/model_labels is empty, if both shock and shocks are provided,
        or if model_names length doesn't match models length.
    RuntimeError
        If any model has not been linearized or lacks computed IRFs.

    Examples
    --------
    >>> from equilibrium import Model
    >>> from equilibrium.plot import plot_model_irfs
    >>> # Assume model1 and model2 are set up and linearized
    >>> model1.linearize()
    >>> model1.compute_linear_irfs(50)
    >>> model2.linearize()
    >>> model2.compute_linear_irfs(50)
    >>> # Plot IRFs for both models responding to all shocks
    >>> paths = plot_model_irfs(
    ...     [model1, model2],
    ...     include_list=["consumption", "output"],
    ...     model_names=["Baseline", "Alternative"],
    ... )
    >>> # Plot IRFs for a single shock
    >>> paths = plot_model_irfs(
    ...     [model1, model2],
    ...     shock="Z_til",
    ...     include_list=["consumption", "output"],
    ...     model_names=["Baseline", "Alternative"],
    ... )
    >>> # Plot IRFs for specific shocks
    >>> paths = plot_model_irfs(
    ...     [model1, model2],
    ...     shocks=["Z_til", "G_til"],
    ...     include_list=["consumption", "output"],
    ...     model_names=["Baseline", "Alternative"],
    ... )
    """
    # Import here to avoid circular imports
    from ..model.model import Model
    from ..settings import get_settings
    from ..utils.io import load_model_irfs

    if models is None:
        models = []

    if model_labels and models:
        raise ValueError("Provide either models or model_labels, not both.")

    if model_labels:
        if plot_dir is None:
            settings = get_settings()
            sorted_labels = sorted(model_labels)
            subdir_name = "_".join(sorted_labels)
            plot_dir = settings.paths.plot_dir / "irfs" / subdir_name
        irf_dicts = [
            load_model_irfs(label, save_dir=save_dir) for label in model_labels
        ]
        plot_kwargs = plot_spec.to_kwargs() if plot_spec is not None else {}
        merged_kwargs = {**plot_kwargs, **kwargs}
        return plot_irf_results(
            irf_results=irf_dicts,
            include_list=include_list,
            plot_dir=plot_dir,
            result_names=model_names or list(model_labels),
            shocks=shocks if shocks is not None else ([shock] if shock else None),
            prefix=prefix,
            plot_type=plot_type,
            title_str=title_str,
            x_str=x_str,
            n_periods=n_periods,
            shock_size=shock_size,
            **merged_kwargs,
        )

    if not models:
        raise ValueError("models must be a non-empty sequence")

    # Handle backward compatibility: shock (singular) vs shocks (plural)
    if shock is not None and shocks is not None:
        raise ValueError(
            "Cannot specify both 'shock' and 'shocks' parameters. "
            "Use 'shock' for a single shock or 'shocks' for multiple shocks."
        )

    if shock is not None:
        # Single shock provided
        shocks = [shock]

    # Validate all inputs are Model objects
    for i, model in enumerate(models):
        if not isinstance(model, Model):
            raise TypeError(
                f"Model at index {i} has unsupported type {type(model).__name__}. "
                "Expected Model."
            )

    # Validate each model has been linearized and has IRFs
    for i, model in enumerate(models):
        if model.linear_mod is None:
            raise RuntimeError(
                f"Model at index {i} (label='{model.label}') has not been linearized. "
                "Call model.linearize() first."
            )
        if model.linear_mod.irfs is None:
            raise RuntimeError(
                f"Model at index {i} (label='{model.label}') has no computed IRFs. "
                "Call model.compute_linear_irfs() first."
            )

    # Determine shocks to plot
    if shocks is None:
        # Use union of all shocks from all models
        all_shocks: List[str] = []
        for model in models:
            for shock_name in model.exog_list:
                if shock_name not in all_shocks:
                    all_shocks.append(shock_name)
        shocks = all_shocks

    if not shocks:
        raise ValueError(
            "No shocks found. Either specify shock/shocks parameter or ensure models have exog_list."
        )

    plot_kwargs = plot_spec.to_kwargs() if plot_spec is not None else {}
    merged_kwargs = {**plot_kwargs, **kwargs}

    # Extract var_titles from kwargs to use for shock labels
    var_titles = merged_kwargs.get("var_titles", {}) or {}

    # Build shock_sizes dict
    if isinstance(shock_size, dict):
        shock_sizes = shock_size.copy()
    else:
        # If shock_size is a float, use it for all shocks
        shock_sizes = {}

    # For any shock not in shock_sizes, default to SIG_<shock> from first model
    for shock_name in shocks:
        if shock_name not in shock_sizes:
            # Try to get SIG_<shock> from the first model's params
            sig_key = f"SIG_{shock_name}"
            if hasattr(models[0], "params") and sig_key in models[0].params:
                shock_sizes[shock_name] = models[0].params[sig_key]
            else:
                # Fall back to 1.0 if SIG_<shock> not found
                shock_sizes[shock_name] = (
                    shock_size if isinstance(shock_size, float) else 1.0
                )

    # Set default model names from model labels
    if model_names is None:
        model_names = [model.label for model in models]
    elif len(model_names) != len(models):
        raise ValueError(
            f"model_names length ({len(model_names)}) must match "
            f"models length ({len(models)})"
        )

    # Set default plot_dir from Settings with subdirectory based on model labels
    if plot_dir is None:
        settings = get_settings()
        # Create subdirectory name from sorted model labels to avoid file congestion
        sorted_labels = sorted(model.label for model in models)
        subdir_name = "_".join(sorted_labels)
        plot_dir = settings.paths.plot_dir / "irfs" / subdir_name

    # Determine the union of all variable names that exist in any model
    all_var_names: List[str] = []
    for model in models:
        for name in model.all_vars:
            if name not in all_var_names:
                all_var_names.append(name)

    # Set default include_list to first model's variables if not provided
    if include_list is None:
        include_list = list(models[0].all_vars)

    # Filter to only variables in include_list (preserving include_list order)
    full_list = [v for v in include_list if v in all_var_names]

    if not full_list:
        raise ValueError(
            f"None of the variables in include_list {list(include_list)} "
            f"are present in any model. Available variables: {all_var_names}"
        )

    # Determine the common time dimension (min across all models)
    n_periods_available = [model.linear_mod.irfs.shape[1] for model in models]
    min_periods = min(n_periods_available)

    if n_periods is not None:
        if n_periods > min_periods:
            raise ValueError(
                f"Requested n_periods={n_periods} exceeds minimum available "
                f"IRF horizon={min_periods}"
            )
        actual_periods = n_periods
    else:
        actual_periods = min_periods

    # Loop through shocks and plot each
    all_paths: List[Path] = []
    for shock_name in shocks:
        # Validate shock exists in at least one model and get shock indices
        shock_indices = []
        shock_found = False
        for i, model in enumerate(models):
            if shock_name in model.exog_list:
                shock_indices.append(model.exog_list.index(shock_name))
                shock_found = True
            else:
                shock_indices.append(None)  # Mark as missing

        if not shock_found:
            raise ValueError(
                f"Shock '{shock_name}' not found in any model. "
                f"Available shocks across models: {[model.exog_list for model in models]}"
            )

        # Set shock-specific defaults
        shock_prefix = prefix if prefix is not None else f"irf_to_{shock_name}"
        # Use var_titles to get a friendly name for the shock if available
        shock_label = var_titles.get(shock_name, shock_name)
        shock_title = (
            title_str if title_str is not None else f"Impulse Response to {shock_label}"
        )

        # Get shock-specific size
        current_shock_size = shock_sizes.get(shock_name, 1.0)

        # Build path_vals array: shape (n_models, n_periods, n_vars)
        n_models = len(models)
        n_vars = len(full_list)
        path_vals = np.full((n_models, actual_periods, n_vars), np.nan)

        for i, model in enumerate(models):
            shock_idx = shock_indices[i]
            if shock_idx is not None:
                # Shock exists in this model
                irfs = model.linear_mod.irfs[
                    shock_idx, :actual_periods, :
                ]  # (actual_periods, n_state_vars)

                for j, var_name in enumerate(full_list):
                    if var_name in model.all_vars:
                        var_idx = model.all_vars.index(var_name)
                        path_vals[i, :, j] = current_shock_size * irfs[:, var_idx]
            # else: shock_idx is None, so path_vals[i, :, :] remains NaN

        # Call plot_paths with the harmonized data
        paths = plot_paths(
            path_vals=path_vals,
            full_list=full_list,
            include_list=full_list,
            title_str=shock_title,
            x_str=x_str,
            prefix=shock_prefix,
            group_names=list(model_names),
            plot_dir=plot_dir,
            plot_type=plot_type,
            **merged_kwargs,
        )
        all_paths.extend(paths)

    return all_paths


def plot_irf_results(
    irf_results: Optional[
        dict[str, "IrfResult"] | Sequence[dict[str, "IrfResult"]]
    ] = None,
    include_list: Optional[Sequence[str]] = None,
    plot_dir: Optional[Union[str, Path]] = None,
    *,
    model_labels: Optional[Sequence[str]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    result_names: Optional[Sequence[str]] = None,
    shocks: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
    plot_type: str = "pdf",
    title_str: Optional[str] = None,
    x_str: str = "Period",
    n_periods: Optional[int] = None,
    shock_size: Union[float, dict[str, float]] = 1.0,
    **kwargs,
) -> List[Path]:
    """
    Plot impulse response functions from IrfResult dictionaries.

    This function accepts either a single dict of IrfResults (for one model) or
    a sequence of dicts (for multiple models), and plots the IRFs for each shock.

    Parameters
    ----------
    irf_results : dict[str, IrfResult] or Sequence[dict[str, IrfResult]], optional
        IRF results to plot. Can be a single dictionary mapping shock names to
        IrfResults (for one model), or a sequence of such dictionaries (for
        multiple models to compare).
    model_labels : Sequence[str], optional
        Model labels to load IRFs from disk and plot. Loaded results are
        appended after any explicit ``irf_results``.
    save_dir : str or Path, optional
        Base directory used to load labeled IRFs. Defaults to settings.
    include_list : Sequence[str], optional
        List of variable names to plot. Variables not present in a result will
        be filled with NaN. If None, defaults to all variables present in the
        first result.
    plot_dir : str or Path, optional
        Directory where plot files will be saved. If None, defaults to a
        subdirectory called "irfs" inside the plot_dir specified in Settings.
    result_names : Sequence[str], optional
        Names for each set of results (used in legends when multiple dicts are
        provided). If None, defaults to explicit "Result {i}" entries followed
        by model labels for any ``model_labels``.
    shocks : Sequence[str], optional
        List of shock names to plot. If None, plots all shocks present in any
        result dict.
    prefix : str, optional
        Filename prefix for the saved plots. If None, defaults to
        "irf_to_{shock}" for each shock.
    plot_type : str, default "pdf"
        File extension for saved plots (e.g., "png", "pdf").
    title_str : str, optional
        Title string template to display on plots. If None, defaults to
        "Impulse Response to {shock}".
    x_str : str, default "Period"
        Label for the x-axis.
    n_periods : int, optional
        Number of periods to plot. If None, uses the minimum IRF horizon
        across all results.
    shock_size : float or dict[str, float], default 1.0
        Scaling factor(s) applied to the IRFs before plotting. If a float,
        the same size is used for all shocks. If a dict, maps shock names
        to their sizes. For shocks not in the dict, defaults to 1.0.
    **kwargs
        Additional keyword arguments passed to plot_paths.

    Returns
    -------
    list[Path]
        Paths to all saved plot files across all shocks.

    Raises
    ------
    ValueError
        If irf_results is empty, if result_names length doesn't match number
        of result dicts, or if no shocks are found.

    Examples
    --------
    >>> from equilibrium import Model
    >>> from equilibrium.plot import plot_irf_results
    >>> # Compute IRFs for a model
    >>> model.linearize()
    >>> irf_dict = model.compute_linear_irfs(50)
    >>> # Plot all IRFs
    >>> paths = plot_irf_results(irf_dict, include_list=["c", "k", "y"])
    >>> # Compare IRFs from two models
    >>> irf_dict1 = model1.compute_linear_irfs(50)
    >>> irf_dict2 = model2.compute_linear_irfs(50)
    >>> paths = plot_irf_results(
    ...     [irf_dict1, irf_dict2],
    ...     include_list=["c", "k"],
    ...     result_names=["Baseline", "Alternative"],
    ... )
    """
    from ..settings import get_settings
    from ..solvers.results import IrfResult
    from ..utils.io import load_model_irfs

    irf_dicts: List[dict[str, IrfResult]] = []
    auto_names: List[str] = []

    if irf_results is not None:
        if isinstance(irf_results, dict):
            irf_dicts = [irf_results]
        else:
            irf_dicts = list(irf_results)
        auto_names.extend([f"Result {i}" for i in range(len(irf_dicts))])

    if model_labels:
        for label in model_labels:
            irf_dicts.append(load_model_irfs(label, save_dir=save_dir))
            auto_names.append(label)

    if not irf_dicts:
        raise ValueError("irf_results must be non-empty")

    # Validate all dicts contain IrfResult objects
    for i, irf_dict in enumerate(irf_dicts):
        if not isinstance(irf_dict, dict):
            raise TypeError(
                f"irf_results element at index {i} must be a dict, got {type(irf_dict)}"
            )
        for shock_name, irf_result in irf_dict.items():
            if not isinstance(irf_result, IrfResult):
                raise TypeError(
                    f"irf_results[{i}]['{shock_name}'] must be an IrfResult, "
                    f"got {type(irf_result).__name__}"
                )

    # Set default plot_dir from Settings
    if plot_dir is None:
        settings = get_settings()
        plot_dir = settings.paths.plot_dir / "irfs"

    # Set default result names
    if result_names is None:
        result_names = auto_names
    elif len(result_names) != len(irf_dicts):
        raise ValueError(
            f"result_names length ({len(result_names)}) must match "
            f"irf_results length ({len(irf_dicts)})"
        )

    # Determine shocks to plot
    if shocks is None:
        # Use union of all shocks from all result dicts
        all_shocks: List[str] = []
        for irf_dict in irf_dicts:
            for shock_name in irf_dict.keys():
                if shock_name not in all_shocks:
                    all_shocks.append(shock_name)
        shocks = all_shocks

    if not shocks:
        raise ValueError("No shocks found in irf_results")

    # Extract var_titles from kwargs to use for shock labels
    var_titles = kwargs.get("var_titles", {}) or {}

    # Build shock_sizes dict
    if isinstance(shock_size, dict):
        shock_sizes = shock_size.copy()
    else:
        # If shock_size is a float, use it for all shocks
        shock_sizes = {shock_name: shock_size for shock_name in shocks}

    # Determine union of all variable names
    all_var_names: List[str] = []
    for irf_dict in irf_dicts:
        for irf_result in irf_dict.values():
            for name in irf_result.var_names:
                if name not in all_var_names:
                    all_var_names.append(name)
            for name in irf_result.y_names:
                if name not in all_var_names:
                    all_var_names.append(name)

    # Set default include_list
    if include_list is None:
        include_list = all_var_names

    # Filter to only variables in include_list
    full_list = [v for v in include_list if v in all_var_names]

    if not full_list:
        raise ValueError(
            f"None of the variables in include_list {list(include_list)} "
            f"are present in any result. Available variables: {all_var_names}"
        )

    # Plot each shock
    all_paths: List[Path] = []

    for shock_name in shocks:
        # Use var_titles to get a friendly name for the shock if available
        shock_label = var_titles.get(shock_name, shock_name)

        # Set default title for this shock
        shock_title = (
            title_str.format(shock=shock_label)
            if title_str and "{shock}" in title_str
            else (title_str if title_str else f"Impulse Response to {shock_label}")
        )

        # Set default prefix for this shock
        shock_prefix = prefix if prefix else f"irf_to_{shock_name}"

        # Get shock-specific size
        current_shock_size = shock_sizes.get(shock_name, 1.0)

        # Collect results for this shock across all result dicts
        shock_results = []
        result_labels = []

        for i, irf_dict in enumerate(irf_dicts):
            if shock_name in irf_dict:
                shock_results.append(irf_dict[shock_name])
                result_labels.append(result_names[i])

        if not shock_results:
            # Skip shocks that don't exist in any dict
            continue

        # Determine common time dimension
        n_periods_available = [r.UX.shape[0] for r in shock_results]
        min_periods = min(n_periods_available)

        if n_periods is not None:
            if n_periods > min_periods:
                raise ValueError(
                    f"Requested n_periods={n_periods} exceeds minimum available "
                    f"IRF horizon={min_periods} for shock '{shock_name}'"
                )
            actual_periods = n_periods
        else:
            actual_periods = min_periods

        # Build path_vals array: shape (n_results, actual_periods, n_vars)
        n_results = len(shock_results)
        n_vars = len(full_list)
        path_vals = np.full((n_results, actual_periods, n_vars), np.nan)

        for i, irf_result in enumerate(shock_results):
            for j, var_name in enumerate(full_list):
                if var_name in irf_result.var_names:
                    var_idx = irf_result.var_names.index(var_name)
                    path_vals[i, :, j] = (
                        current_shock_size * irf_result.UX[:actual_periods, var_idx]
                    )
                elif var_name in irf_result.y_names and irf_result.Y is not None:
                    var_idx = irf_result.y_names.index(var_name)
                    path_vals[i, :, j] = (
                        current_shock_size * irf_result.Y[:actual_periods, var_idx]
                    )

        # Call plot_paths for this shock
        paths = plot_paths(
            path_vals=path_vals,
            full_list=full_list,
            include_list=full_list,
            title_str=shock_title,
            x_str=x_str,
            prefix=shock_prefix,
            group_names=result_labels,
            plot_dir=plot_dir,
            plot_type=plot_type,
            **kwargs,
        )
        all_paths.extend(paths)

    return all_paths
