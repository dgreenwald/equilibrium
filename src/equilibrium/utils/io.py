from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..settings import get_settings

if TYPE_CHECKING:
    from ..solvers.results import DeterministicResult, IrfResult, SequenceResult


def _json_dumps_canonical(obj: Any) -> str:
    # Canonicalize for stable hashing & diffs
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _atomic_write_text(path: Path, text: str) -> None:
    # Write to a temp file and atomically replace
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)  # atomic on POSIX/Windows


def _list_backups(backup_dir: Path, stem: str) -> list[Path]:
    # Backups look like: steady_YYYYmmdd-HHMMSS_sha1.json
    return sorted(backup_dir.glob(f"{stem}_*.json"))


def list_files_by_stem(
    parent_dir: Path, stem: str, *, suffix: Optional[str] = None
) -> list[Path]:
    pattern = f"{stem}_*{suffix or ''}"
    return sorted(parent_dir.glob(pattern))


def prune_files_by_stem(
    parent_dir: Path,
    stem: str,
    *,
    keep: Optional[int],
    suffix: Optional[str] = None,
) -> None:
    if keep is None or keep < 0:
        return
    if not parent_dir.exists():
        return

    files = list_files_by_stem(parent_dir, stem, suffix=suffix)
    if len(files) <= keep:
        return

    for old in files[:-keep]:
        try:
            old.unlink()
        except FileNotFoundError:
            pass


def save_json_with_backups(
    data: Any,
    main_path: Path,
    *,
    keep: int = 10,
    backup_dir: Optional[Path] = None,
    stem: str = "steady",
    always_backup: bool = True,  # set False to only back up when changed
) -> Path:
    """
    Save `data` to `main_path` (JSON) atomically and keep up to `keep` backups
    under `backup_dir` (default: main_path.parent / "steady_backups").
    Returns the path to the written main file.
    """
    backup_dir = backup_dir or (main_path.parent / "steady_backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Canonical JSON for hashing and file content
    payload = _json_dumps_canonical(data)
    new_hash = _sha1(payload)

    # If main exists and contents identical, optionally skip backup
    changed = True
    if main_path.exists():
        try:
            current = main_path.read_text(encoding="utf-8")
            changed = current != payload
        except Exception:
            changed = True  # if unreadable/corrupt, treat as changed

    # Write a timestamped backup (either always, or only if changed)
    if always_backup or changed:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        backup_name = f"{stem}_{ts}_{new_hash}.json"
        _atomic_write_text(backup_dir / backup_name, payload)

        # Enforce retention: keep most recent `keep`
        backups = _list_backups(backup_dir, stem)
        if keep is not None and keep >= 0 and len(backups) > keep:
            for old in backups[:-keep]:
                # best-effort cleanup
                try:
                    old.unlink()
                except FileNotFoundError:
                    pass

    # Update the main file atomically
    _atomic_write_text(main_path, payload)
    return main_path


def list_steady_backups(
    main_path: Path, stem: str = "steady", backup_dir: Optional[Path] = None
) -> list[Path]:
    backup_dir = backup_dir or (main_path.parent / "steady_backups")
    return _list_backups(backup_dir, stem)


def restore_steady_from_backup(backup_path: Path, main_path: Path) -> None:
    # Atomically restore main from a chosen backup
    text = backup_path.read_text(encoding="utf-8")
    _atomic_write_text(main_path, text)


def load_json(main_path: Path) -> Any:
    return json.loads(main_path.read_text(encoding="utf-8"))


def read_steady_value(
    label: str,
    variable: str,
    default: Optional[float] = None,
    save_dir: Optional[Path | str] = None,
) -> float:
    """
    Read a steady state value from a saved JSON file.

    This function reads a specific variable value from a model's saved steady state
    JSON file, enabling cross-model parameter sharing. For example, one model can
    use another model's steady state output as a parameter input.

    Parameters
    ----------
    label : str
        Model label identifying the steady state file (e.g., 'baseline', 'ltv_only').
        The file is expected to be named '{label}_steady_state.json'.
    variable : str
        Variable name to retrieve from the steady state (e.g., 'ltv_agg', 'K', 'C').
    default : float | None, optional
        Default value to return if the file or variable is not found.
        If None and the value cannot be found, raises an error.
    save_dir : Path | str | None, optional
        Directory containing steady state files. If None, uses the configured
        save directory from settings (typically ~/.local/share/EQUILIBRIUM/cache/).

    Returns
    -------
    float
        The steady state value for the specified variable.

    Raises
    ------
    FileNotFoundError
        If the steady state JSON file doesn't exist and no default is provided.
    KeyError
        If the variable is not found in the JSON and no default is provided.

    Examples
    --------
    >>> # Use baseline model's ltv_agg as ltv_target in another model
    >>> baseline_ltv = read_steady_value('baseline', 'ltv_agg', default=0.854)
    >>> params = {'ltv_target': baseline_ltv}

    >>> # Read without default (will raise error if not found)
    >>> capital_stock = read_steady_value('baseline', 'K')

    Notes
    -----
    - The source model must have been solved with `save=True` to create the JSON file.
    - Models should be solved in dependency order (e.g., solve baseline before
      models that reference its outputs).
    - This function does not trigger model solving; it only reads saved results.
    """
    # Determine save directory
    if save_dir is None:
        settings = get_settings()
        save_dir = settings.paths.save_dir
    else:
        save_dir = Path(save_dir)

    # Construct file path
    filepath = save_dir / f"{label}_steady_state.json"

    # Try to read the file
    try:
        data = load_json(filepath)
    except FileNotFoundError:
        if default is not None:
            return default
        raise FileNotFoundError(
            f"Steady state file not found: {filepath}. "
            f"Have you run solve_steady(save=True) for model '{label}'?"
        ) from None

    # Try to extract the variable
    try:
        value = data[variable]
    except KeyError:
        if default is not None:
            return default
        available = list(data.keys())
        raise KeyError(
            f"Variable '{variable}' not found in steady state for model '{label}'. "
            f"Available variables: {available}"
        ) from None

    # Ensure we return a float
    return float(value)


def read_steady_values(
    label: str,
    default: Optional[dict[str, float]] = None,
    save_dir: Optional[Path | str] = None,
) -> dict[str, float]:
    """
    Read all steady state values from a saved JSON file.

    Parameters
    ----------
    label : str
        Model label identifying the steady state file (e.g., 'baseline', 'ltv_only').
        The file is expected to be named '{label}_steady_state.json'.
    default : dict[str, float] | None, optional
        Default value to return if the file is not found.
        If None and the file cannot be found, raises an error.
    save_dir : Path | str | None, optional
        Directory containing steady state files. If None, uses the configured
        save directory from settings (typically ~/.local/share/EQUILIBRIUM/cache/).

    Returns
    -------
    dict[str, float]
        Mapping of steady state variable names to values.

    Raises
    ------
    FileNotFoundError
        If the steady state JSON file doesn't exist and no default is provided.

    Notes
    -----
    - The source model must have been solved with `save=True` to create the JSON file.
    - This function does not trigger model solving; it only reads saved results.
    """
    if save_dir is None:
        settings = get_settings()
        save_dir = settings.paths.save_dir
    else:
        save_dir = Path(save_dir)

    filepath = save_dir / f"{label}_steady_state.json"

    try:
        data = load_json(filepath)
    except FileNotFoundError:
        if default is not None:
            return default
        raise FileNotFoundError(
            f"Steady state file not found: {filepath}. "
            f"Have you run solve_steady(save=True) for model '{label}'?"
        ) from None

    return {key: float(value) for key, value in data.items()}


def read_calibrated_params(
    label: str,
    default: Optional[dict[str, float]] = None,
    save_dir: Optional[Path | str] = None,
    regime: Optional[int] = None,
) -> dict[str, float]:
    """
    Read all calibrated parameters from a saved JSON file.

    Parameters
    ----------
    label : str
        Label identifying the calibration file.
    default : dict[str, float] | None, optional
        Default dictionary to return if file not found.
    save_dir : Path | str | None, optional
        Directory containing calibration files.
    regime : int | None, optional
        If specified, filter parameters for this regime.
        - Global parameters are always included.
        - Regime-specific parameters (e.g., "regime_tau_r1") are included only
          if they match the regime, and are renamed to their base name (e.g., "tau").
        - Shock parameters are included only if they match the regime.

    Returns
    -------
    dict[str, float]
        Mapping of parameter names to calibrated values.
    """
    if save_dir is None:
        settings = get_settings()
        save_dir = settings.paths.save_dir
    else:
        save_dir = Path(save_dir)

    filepath = save_dir / f"{label}_calibrated_params.json"

    try:
        data = load_json(filepath)
    except FileNotFoundError:
        if default is not None:
            return default
        raise FileNotFoundError(
            f"Calibration file not found: {filepath}. "
            f"Have you saved parameters for label '{label}'?"
        ) from None

    # Cast values to float
    raw_params = {k: float(v) for k, v in data.items()}

    if regime is None:
        return raw_params

    # Filter by regime
    import re

    filtered_params = {}

    # Regex for regime params: regime_{name}_r{indices}
    # indices can be "1" or "1_2_3"
    regime_pattern = re.compile(r"^regime_(.+)_r([\d_]+)$")

    # Regex for shock params: shock_{name}_r{regime}_t{period}
    shock_pattern = re.compile(r"^shock_(.+)_r(\d+)_t(\d+)$")

    for key, value in raw_params.items():
        # Check for RegimeParam
        m_regime = regime_pattern.match(key)
        if m_regime:
            base_name = m_regime.group(1)
            indices_str = m_regime.group(2)
            indices = [int(x) for x in indices_str.split("_")]

            if regime in indices:
                filtered_params[base_name] = value
            continue

        # Check for ShockParam
        m_shock = shock_pattern.match(key)
        if m_shock:
            # shock_name = m_shock.group(1)
            shock_regime = int(m_shock.group(2))
            # period = int(m_shock.group(3))

            if shock_regime == regime:
                filtered_params[key] = value
            continue

        # Assume global param
        filtered_params[key] = value

    return filtered_params


def read_calibrated_param(
    label: str,
    param: str,
    default: Optional[float] = None,
    save_dir: Optional[Path | str] = None,
    regime: Optional[int] = None,
) -> float:
    """
    Read a calibrated parameter value from a saved JSON file.

    Parameters
    ----------
    label : str
        Label identifying the calibration file.
    param : str
        Parameter name to retrieve. If `regime` is specified, this should be
        the base name (e.g., "tau") rather than the full stored name
        (e.g., "regime_tau_r1").
    default : float | None, optional
        Default value to return if not found.
    save_dir : Path | str | None, optional
        Directory containing calibration files.
    regime : int | None, optional
        If specified, look up the parameter within the context of this regime
        (handling renaming of regime-specific parameters).

    Returns
    -------
    float
        The calibrated parameter value.
    """
    # Load all parameters (potentially filtered/renamed by regime)
    try:
        params = read_calibrated_params(label, save_dir=save_dir, regime=regime)
    except FileNotFoundError:
        if default is not None:
            return default
        raise

    try:
        return params[param]
    except KeyError:
        if default is not None:
            return default
        available = list(params.keys())
        raise KeyError(
            f"Parameter '{param}' not found in calibration '{label}' "
            f"(regime={regime}). Available parameters: {available}"
        ) from None


def save_calibrated_params(
    params: dict[str, float],
    label: str,
    save_dir: Optional[Path | str] = None,
) -> Path:
    """
    Save calibrated parameters to a JSON file.

    Parameters
    ----------
    params : dict[str, float]
        Dictionary of parameter names and values.
    label : str
        Label for the calibration file.
    save_dir : Path | str | None, optional
        Target directory.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if save_dir is None:
        settings = get_settings()
        save_dir = settings.paths.save_dir
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{label}_calibrated_params.json"

    # Use save_json_with_backups for consistency/safety
    return save_json_with_backups(params, filepath, stem=f"{label}_calib")


def load_model_irfs(
    model_label: str,
    shock: Optional[str] = None,
    *,
    save_dir: Optional[Path | str] = None,
) -> dict[str, "IrfResult"] | "IrfResult":
    """
    Load impulse response functions for a model by label.

    This function loads IRFs that were saved using model.save_linear_irfs().
    The IRFs are loaded from the file and converted to IrfResult objects for
    convenient plotting and analysis.

    Parameters
    ----------
    model_label : str
        Label of the model whose IRFs to load.
    shock : str, optional
        Specific shock name to load. If None, returns dict with all shocks.
    save_dir : Path or str, optional
        Directory to load from. Defaults to settings.paths.save_dir.

    Returns
    -------
    dict or IrfResult
        If shock is None: dict mapping shock names to IrfResult objects.
        If shock specified: single IrfResult for that shock.

    Raises
    ------
    FileNotFoundError
        If the IRF file for the model does not exist.
    KeyError
        If shock is specified but not found in the saved IRFs.

    Examples
    --------
    >>> # Load all IRFs for a model
    >>> irfs = load_model_irfs("baseline")
    >>> print(irfs.keys())  # ['Z_til', 'shock2', ...]

    >>> # Load specific shock
    >>> irf = load_model_irfs("baseline", shock="Z_til")
    >>> print(irf.UX.shape)  # (50, N_ux)

    Notes
    -----
    The model must have been solved and IRFs saved with:
    >>> mod.linearize()
    >>> mod.compute_linear_irfs(Nt_irf)
    >>> mod.save_linear_irfs()
    """
    from ..io import load_results, resolve_output_path
    from ..solvers.results import IrfResult

    # Resolve the file path
    filepath = resolve_output_path(
        None,
        result_type="irfs",
        model_label=model_label,
        save_dir=save_dir,
        suffix=".npz",
    )

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"IRF file not found: {filepath}. "
            f"Have you run save_linear_irfs() for model '{model_label}'?"
        )

    # Load the data
    data = load_results(filepath)
    metadata = data.get("__metadata__", {})

    # Extract IRF tensor and metadata
    irfs_tensor = data.get("irfs")
    if irfs_tensor is None:
        raise ValueError(f"No 'irfs' data found in file: {filepath}")

    shock_names = metadata.get("shock_names", [])
    ux_names = metadata.get("ux_names", []) or metadata.get("var_names", [])
    exog_names = metadata.get("exog_names", [])
    n_ux = metadata.get("n_ux")
    n_z = metadata.get("n_z")

    if not shock_names:
        raise ValueError(f"No shock names found in metadata: {filepath}")

    if n_ux is None:
        n_ux = len(ux_names) if ux_names else irfs_tensor.shape[2]

    if n_z is None:
        n_z = len(exog_names) if exog_names else 0

    # Convert to dict of IrfResult objects
    # irfs_tensor shape: (n_shocks, Nt, n_vars)
    irf_dict = {}
    for i, shock_name in enumerate(shock_names):
        # Check if we have per-shock UX, Z, Y data (new format)
        ux_key = f"UX_{shock_name}"
        z_key = f"Z_{shock_name}"
        y_key = f"Y_{shock_name}"

        if ux_key in data:
            # New format: load UX, Z, Y directly
            UX = data[ux_key]
            Z = data.get(z_key, np.zeros((UX.shape[0], 0)))
            Y = data.get(y_key, None)
            y_names_actual = metadata.get("y_names", []) if Y is not None else []
        else:
            # Old format: extract from irfs tensor
            shock_irf = irfs_tensor[i, :, :]

            # Separate into UX and Z using saved metadata where available.
            UX = shock_irf[:, :n_ux] if n_ux else shock_irf
            if n_z:
                Z = shock_irf[:, n_ux : n_ux + n_z]
            else:
                Z = np.zeros((shock_irf.shape[0], 0))
            Y = None
            y_names_actual = []

        irf_result = IrfResult(
            UX=UX,
            Z=Z,
            Y=Y,
            model_label=model_label,
            var_names=ux_names,
            exog_names=exog_names,
            y_names=y_names_actual,
            shock_name=shock_name,
            shock_size=1.0,  # Default, not stored in old format
        )
        irf_dict[shock_name] = irf_result

    # Return specific shock or full dict
    if shock is not None:
        if shock not in irf_dict:
            available = list(irf_dict.keys())
            raise KeyError(
                f"Shock '{shock}' not found in IRFs for model '{model_label}'. "
                f"Available shocks: {available}"
            )
        return irf_dict[shock]

    return irf_dict


def load_deterministic_result(
    model_label: str,
    experiment_label: Optional[str] = None,
    *,
    save_dir: Optional[Path | str] = None,
) -> "DeterministicResult":
    """
    Load a deterministic result by model and experiment labels.

    This function loads a DeterministicResult that was saved using
    result.save(experiment_label=...).

    Parameters
    ----------
    model_label : str
        Label of the model.
    experiment_label : str, optional
        Experiment/scenario label. If None, loads "{model_label}.npz".
        If provided, loads "{model_label}_{experiment_label}.npz".
    save_dir : Path or str, optional
        Directory to load from. Defaults to settings.paths.save_dir.

    Returns
    -------
    DeterministicResult
        The loaded deterministic result.

    Raises
    ------
    FileNotFoundError
        If the result file does not exist.

    Examples
    --------
    >>> # Load result with experiment label
    >>> result = load_deterministic_result("baseline", "pti_lib")

    >>> # Load result without experiment label
    >>> result = load_deterministic_result("baseline")

    Notes
    -----
    The result must have been saved with:
    >>> result.save(experiment_label="pti_lib")
    """
    from ..io import resolve_output_path
    from ..solvers.results import DeterministicResult

    # Resolve the file path
    filepath = resolve_output_path(
        None,
        result_type="paths",
        model_label=model_label,
        experiment_label=experiment_label,
        save_dir=save_dir,
        suffix=".npz",
    )

    # Check if file exists
    if not filepath.exists():
        label_str = (
            f"{model_label}_{experiment_label}" if experiment_label else model_label
        )
        raise FileNotFoundError(
            f"Deterministic result file not found: {filepath}. "
            f"Have you saved the result for '{label_str}'?"
        )

    # Load using DeterministicResult.load()
    return DeterministicResult.load(filepath)


def load_sequence_result(
    model_label: str,
    experiment_label: Optional[str] = None,
    *,
    save_dir: Optional[Path | str] = None,
    splice: bool = False,
    T_max: Optional[int] = None,
) -> "SequenceResult | DeterministicResult":
    """
    Load a sequence result by model and experiment labels.

    This function loads a SequenceResult that was saved after running
    solve_sequence() or solve_sequence_linear() with a labeled DetSpec.

    Parameters
    ----------
    model_label : str
        Label of the model.
    experiment_label : str, optional
        Experiment/scenario label (from DetSpec.label). If None, loads
        "{model_label}.npz". If provided, loads "{model_label}_{experiment_label}.npz".
    save_dir : Path or str, optional
        Directory to load from. Defaults to settings.paths.save_dir.
    splice : bool, default False
        If True, splice the loaded SequenceResult before returning.
    T_max : int, optional
        Total length for splicing. If None, SequenceResult uses its default.

    Returns
    -------
    SequenceResult or DeterministicResult
        The loaded sequence result, or a spliced DeterministicResult if
        splice is True.

    Raises
    ------
    FileNotFoundError
        If the result file does not exist.

    Examples
    --------
    >>> # Load sequence result with experiment label
    >>> result = load_sequence_result("baseline", "pti_lib")

    >>> # Load sequence result without experiment label
    >>> result = load_sequence_result("baseline")

    >>> # Load and splice in one step
    >>> result = load_sequence_result("baseline", splice=True, T_max=100)

    >>> # Use in plotting
    >>> results = [
    ...     load_sequence_result("baseline", "pti_lib", splice=True, T_max=100),
    ...     load_sequence_result("baseline", "ltv_lib", splice=True, T_max=100),
    ... ]
    >>> plot_deterministic_results(results, include_list=["c", "y"])

    Notes
    -----
    The result must have been saved from solve_sequence():
    >>> spec = DetSpec(label="pti_lib")
    >>> result = solve_sequence(spec, mod, Nt=100)  # Saves by default
    """
    from ..io import resolve_output_path
    from ..solvers.results import SequenceResult

    # Resolve the file path
    filepath = resolve_output_path(
        None,
        result_type="sequences",
        model_label=model_label,
        experiment_label=experiment_label,
        save_dir=save_dir,
        suffix=".npz",
    )

    # Check if file exists, fall back to "_default" when experiment label omitted
    if not filepath.exists() and experiment_label is None:
        fallback_path = resolve_output_path(
            None,
            result_type="sequences",
            model_label=model_label,
            experiment_label="_default",
            save_dir=save_dir,
            suffix=".npz",
        )
        if fallback_path.exists():
            filepath = fallback_path

    if not filepath.exists():
        label_str = (
            f"{model_label}_{experiment_label}" if experiment_label else model_label
        )
        raise FileNotFoundError(
            f"Sequence result file not found: {filepath}. "
            f"Have you saved the result for '{label_str}'?"
        )

    # Load using SequenceResult.load()
    result = SequenceResult.load(filepath)
    if splice:
        return result.splice(T_max=T_max)
    return result


def _to_camel_case(s: str) -> str:
    """Convert underscore_case to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def _write_latex_property_list(
    data: dict[str, float],
    path: Path,
    prop_name: str,
    command_str: str,
    prefix: Optional[str] = None,
    floatfmt: str = ".3f",
) -> None:
    """
    Write a LaTeX3 property list file defining key-value pairs.

    Parameters
    ----------
    data : dict[str, float]
        Dictionary of variable/parameter names to values.
    path : Path
        Output file path.
    prop_name : str
        Name of the LaTeX property list (e.g., "g_equilibrium_steady_prop").
    command_str : str
        Name of the accessor command (e.g., "steady").
    prefix : str, optional
        Prefix to add to all keys (e.g., "baseline" → "baseline_K").
    floatfmt : str, default ".3f"
        Python format specification for float values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\ExplSyntaxOn\n")
        # Only create prop/command if they don't exist
        f.write(f"\\prop_if_exist:NF \\{prop_name} {{ \\prop_new:N \\{prop_name} }}\n")

        for key, value in sorted(data.items()):
            # Add prefix if specified
            if prefix:
                key_with_prefix = f"{prefix}_{key}"
            else:
                key_with_prefix = key

            # Format the value
            val_str = format(float(value), floatfmt)

            # Escape special characters if needed
            val_str = val_str.replace("{", "\\{").replace("}", "\\}")

            f.write(
                f"\\prop_gput:Nnn \\{prop_name} {{{key_with_prefix}}} {{{val_str}}}\n"
            )

        # Only create command if it doesn't exist
        f.write(
            f"\\cs_if_exist:NF \\{command_str} {{ "
            f"\\cs_new:Npn \\{command_str} #1 {{ \\prop_item:Nn \\{prop_name} {{#1}} }} "
            f"}}\n"
        )
        f.write("\\ExplSyntaxOff\n")


def save_steady_values_to_latex(
    label: str,
    *,
    save_dir: Optional[Path | str] = None,
    tex_dir: Optional[Path | str] = None,
    floatfmt: str = ".3f",
    add_prefix: bool = True,
    prefix: Optional[str] = None,
    command_str: str = "steady",
    prop_name: str = "g_equilibrium_steady_prop",
) -> Path:
    """
    Export steady state values to a LaTeX property list file.

    Reads steady state values from the saved JSON file and exports them to a
    LaTeX file using LaTeX3 property lists. The generated file can be included
    in LaTeX documents with \\input{file.tex} and values accessed with the
    command defined by command_str (default: \\steady{label_variable}).

    Parameters
    ----------
    label : str
        Model label identifying the steady state file (e.g., 'baseline').
        The file '{label}_steady_state.json' will be read.
    save_dir : Path | str | None, optional
        Directory containing the steady state JSON file. If None, uses the
        configured save directory from settings.
    tex_dir : Path | str | None, optional
        Directory to write the .tex file. If None, uses {plot_dir}/tex/.
    floatfmt : str, default ".3f"
        Python format specification for float values (e.g., ".3f", ".4f", ".2e").
    add_prefix : bool, default True
        If True, prepend label to keys (e.g., "baseline_K"). If False, keys are
        just variable names (e.g., "K"). Default is True to allow multiple models
        to coexist in the same LaTeX document.
    prefix : str, optional
        Custom prefix for keys. If None and add_prefix=True, uses label.
    command_str : str, default "steady"
        Name of the LaTeX command for accessing values. All models share this
        command and use the same property list.
    prop_name : str, default "g_equilibrium_steady_prop"
        Name of the LaTeX property list. All models share this property list.

    Returns
    -------
    Path
        Path to the written .tex file.

    Raises
    ------
    FileNotFoundError
        If the steady state JSON file doesn't exist.

    Examples
    --------
    >>> # Basic usage (reads baseline_steady_state.json)
    >>> save_steady_values_to_latex('baseline')
    # Writes to {plot_dir}/tex/baseline_steady.tex

    >>> # In LaTeX document:
    # \\input{baseline_steady.tex}
    # The steady state capital stock is $K^* = \\steady{baseline_K}$.

    >>> # Multiple models with parameterized tables:
    >>> save_steady_values_to_latex('baseline')
    >>> save_steady_values_to_latex('alternative')
    # \\newcommand{\\tableRow}[1]{%
    #   \\steady{#1_K} & \\steady{#1_Y} & \\steady{#1_C} \\\\
    # }
    # \\tableRow{baseline}
    # \\tableRow{alternative}

    Notes
    -----
    - The source model must have been solved with save=True to create the JSON file.
    - Multiple models can share the same property list and command without conflicts.
    - Keys are prefixed with the label by default when add_prefix=True, but this is
      often redundant since the property list itself provides namespacing.
    """
    # Read steady state values from JSON
    steady_values = read_steady_values(label=label, save_dir=save_dir)

    # Determine output directory
    if tex_dir is None:
        settings = get_settings()
        tex_dir = settings.paths.plot_dir / "tex"
    else:
        tex_dir = Path(tex_dir)

    # Determine output file path
    output_path = tex_dir / f"{label}_steady.tex"

    # Determine prefix for keys
    if add_prefix:
        key_prefix = prefix if prefix is not None else label
    else:
        key_prefix = None

    # Write the LaTeX file
    _write_latex_property_list(
        data=steady_values,
        path=output_path,
        prop_name=prop_name,
        command_str=command_str,
        prefix=key_prefix,
        floatfmt=floatfmt,
    )

    return output_path


def save_calibrated_params_to_latex(
    label: str,
    *,
    regime: Optional[int] = None,
    save_dir: Optional[Path | str] = None,
    tex_dir: Optional[Path | str] = None,
    floatfmt: str = ".4f",
    add_prefix: bool = True,
    prefix: Optional[str] = None,
    command_str: str = "param",
    prop_name: str = "g_equilibrium_param_prop",
) -> Path:
    """
    Export calibrated parameter values to a LaTeX property list file.

    Reads calibrated parameter values from the saved JSON file and exports them
    to a LaTeX file using LaTeX3 property lists. The generated file can be included
    in LaTeX documents with \\input{file.tex} and values accessed with the
    command defined by command_str (default: \\param{label_parameter}).

    Parameters
    ----------
    label : str
        Label identifying the calibration file (e.g., 'baseline').
        The file '{label}_calibrated_params.json' will be read.
    regime : int, optional
        If specified, filter parameters for this regime and include regime suffix
        in the output filename. Output will be '{label}_params_r{regime}.tex'.
        If None, loads all parameters without filtering and outputs to
        '{label}_params.tex'.
    save_dir : Path | str | None, optional
        Directory containing the calibration JSON file. If None, uses the
        configured save directory from settings.
    tex_dir : Path | str | None, optional
        Directory to write the .tex file. If None, uses {plot_dir}/tex/.
    floatfmt : str, default ".4f"
        Python format specification for float values (e.g., ".3f", ".4f", ".2e").
    add_prefix : bool, default True
        If True, prepend label (and regime if applicable) to keys. If False, keys
        are just parameter names. Default is True to allow multiple models to
        coexist in the same LaTeX document.
    prefix : str, optional
        Custom prefix for keys. If None and add_prefix=True, uses label
        (with "_r{regime}" suffix if regime is specified).
    command_str : str, default "param"
        Name of the LaTeX command for accessing values. All models share this
        command and use the same property list.
    prop_name : str, default "g_equilibrium_param_prop"
        Name of the LaTeX property list. All models share this property list.

    Returns
    -------
    Path
        Path to the written .tex file.

    Raises
    ------
    FileNotFoundError
        If the calibration JSON file doesn't exist.

    Examples
    --------
    >>> # Basic usage (all parameters)
    >>> save_calibrated_params_to_latex('baseline')
    # Writes to {plot_dir}/tex/baseline_params.tex

    >>> # Regime-specific parameters
    >>> save_calibrated_params_to_latex('baseline', regime=0)
    # Writes to {plot_dir}/tex/baseline_params_r0.tex

    >>> # In LaTeX document:
    # \\input{baseline_params.tex}
    # The discount factor is $\\beta = \\param{baseline_bet}$.

    >>> # Multiple regimes:
    >>> save_calibrated_params_to_latex('baseline', regime=0)
    >>> save_calibrated_params_to_latex('baseline', regime=1)
    # \\input{baseline_params_r0.tex}
    # \\input{baseline_params_r1.tex}
    # Regime 0: $\\tau = \\param{baseline_r0_tau}$
    # Regime 1: $\\tau = \\param{baseline_r1_tau}$

    Notes
    -----
    - The calibration must have been saved with save_calibrated_params().
    - When regime is specified, regime-specific parameters are filtered and renamed
      (e.g., "regime_tau_r0" becomes "tau").
    - Multiple models and regimes can share the same property list and command.
    """
    # Read calibrated parameters from JSON
    calibrated_params = read_calibrated_params(
        label=label, save_dir=save_dir, regime=regime
    )

    # Determine output directory
    if tex_dir is None:
        settings = get_settings()
        tex_dir = settings.paths.plot_dir / "tex"
    else:
        tex_dir = Path(tex_dir)

    # Determine output file path (include regime suffix if specified)
    if regime is not None:
        output_path = tex_dir / f"{label}_params_r{regime}.tex"
    else:
        output_path = tex_dir / f"{label}_params.tex"

    # Determine prefix for keys
    if add_prefix:
        if prefix is not None:
            key_prefix = prefix
        elif regime is not None:
            key_prefix = f"{label}_r{regime}"
        else:
            key_prefix = label
    else:
        key_prefix = None

    # Write the LaTeX file
    _write_latex_property_list(
        data=calibrated_params,
        path=output_path,
        prop_name=prop_name,
        command_str=command_str,
        prefix=key_prefix,
        floatfmt=floatfmt,
    )

    return output_path
