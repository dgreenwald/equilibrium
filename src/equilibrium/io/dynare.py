"""
Dynare .mod file export for Equilibrium models.

This module provides functionality to export Model instances to Dynare .mod
format, enabling interoperability with the Dynare DSGE toolbox.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..settings import get_settings

if TYPE_CHECKING:
    from ..model import Model
    from ..solvers.det_spec import DetSpec

logger = logging.getLogger(__name__)


def _identify_changing_parameters(det_spec: "DetSpec", model: "Model") -> list[str]:
    """
    Identify parameters that change across regimes in a DetSpec.

    Parameters
    ----------
    det_spec : DetSpec
        The deterministic scenario specification
    model : Model
        The model instance (for validation)

    Returns
    -------
    list[str]
        List of parameter names that change in any regime

    Raises
    ------
    ValueError
        If a changing parameter already has TV_<param> in model.exog_list
    """
    changing_params = set()

    # Compare each regime against the baseline
    for regime_idx, regime_params in enumerate(det_spec.preset_par_list):
        for param, value in regime_params.items():
            # Check if this parameter differs from baseline
            baseline_value = det_spec.preset_par_init.get(param, None)
            if baseline_value is None:
                # Parameter is new in this regime (not in baseline)
                changing_params.add(param)
            elif value != baseline_value:
                # Parameter value differs from baseline
                changing_params.add(param)

    # Validate that TV_<param> doesn't already exist
    for param in changing_params:
        tv_var_name = f"TV_{param}"
        if tv_var_name in model.exog_list:
            raise ValueError(
                f"Parameter '{param}' changes in det_spec but {tv_var_name} already "
                f"exists as an exogenous variable. Remove {tv_var_name} from the "
                f"model or use a different parameter name in det_spec."
            )

    return sorted(changing_params)


def _create_tv_parameter_infrastructure(
    changing_params: list[str],
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    Create TV_ infrastructure for changing parameters.

    For each changing parameter, creates:
    - TV_<param>: endogenous variable (AR(1) process)
    - e_TV_<param>: exogenous shock
    - mu_TV_<param>, rho_TV_<param>, sig_TV_<param>: parameters

    Parameters
    ----------
    changing_params : list[str]
        List of parameters that change across regimes

    Returns
    -------
    tv_vars : list[str]
        TV_<param> endogenous variables to add
    tv_shocks : list[str]
        e_TV_<param> exogenous shocks to add
    tv_params : dict[str, float]
        Parameters to add (mu_TV_, rho_TV_, sig_TV_)
    """
    tv_vars = []
    tv_shocks = []
    tv_params = {}

    for param in changing_params:
        tv_var = f"TV_{param}"
        tv_shock = f"e_TV_{param}"

        tv_vars.append(tv_var)
        tv_shocks.append(tv_shock)

        # Add AR(1) parameters (all set to create a pure shock process)
        tv_params[f"mu_TV_{param}"] = 0.0  # Mean
        tv_params[f"rho_TV_{param}"] = 0.0  # Persistence (no autocorrelation)
        tv_params[f"sig_TV_{param}"] = 1.0  # Scale (unit shock)

    return tv_vars, tv_shocks, tv_params


def _replace_params_in_equation(
    equation: str,
    changing_params: list[str],
    all_params: set[str],
) -> str:
    """
    Replace parameter references with (param + TV_param) in an equation.

    Only replaces standalone parameter references (not in assignments).
    Uses word boundary matching to avoid partial replacements.

    Parameters
    ----------
    equation : str
        The equation string to modify
    changing_params : list[str]
        List of parameters to replace
    all_params : set[str]
        Set of all parameter names (to avoid replacing variables)

    Returns
    -------
    str
        Modified equation string

    Examples
    --------
    >>> _replace_params_in_equation("y = delta * K", ["delta"], {"delta"})
    'y = (delta + TV_delta) * K'
    >>> _replace_params_in_equation("delta = 0.1", ["delta"], {"delta"})
    'delta = 0.1'  # Don't replace in assignment
    """
    result = equation

    for param in changing_params:
        # Match parameter as a word boundary, but not if followed by '='
        # This pattern captures: param not followed by whitespace and '='
        # We need to be careful not to replace in "param = value" lines

        # First check if this is a parameter assignment line
        # Pattern: optional whitespace, param name, optional whitespace, equals
        assignment_pattern = rf"^\s*{re.escape(param)}\s*="
        if re.match(assignment_pattern, result):
            # This is an assignment to the parameter itself, don't modify
            continue

        # Replace all occurrences of the parameter with (param + TV_param)
        # Use word boundaries to avoid partial matches
        # Negative lookahead to avoid replacing in function names
        pattern = rf"\b{re.escape(param)}\b"

        def replacer(match):
            # Get the matched parameter name
            matched = match.group(0)
            # Return the replacement
            return f"({matched} + TV_{matched})"

        result = re.sub(pattern, replacer, result)

    return result


def _generate_regime_blocks(
    det_spec: "DetSpec",
    changing_params: list[str],
) -> list[str]:
    """
    Generate endval and shocks blocks for all regimes.

    Parameters
    ----------
    det_spec : DetSpec
        The deterministic scenario specification
    changing_params : list[str]
        List of parameters that change across regimes

    Returns
    -------
    list[str]
        List of Dynare code blocks (endval and shocks)
    """
    blocks = []

    # Calculate learnt_in times for each regime
    # Regime 0 starts at period 1
    # Regime r > 0 starts at time_list[r-1] (the transition time from regime r-1 to r)
    learnt_in_times = [1]  # First regime learned at t=1
    for t in det_spec.time_list:
        learnt_in_times.append(t)

    # Generate blocks for each regime
    for regime_idx in range(det_spec.n_regimes):
        learnt_in = learnt_in_times[regime_idx]
        regime_params = det_spec.preset_par_list[regime_idx]
        regime_shocks = (
            det_spec.shocks[regime_idx] if regime_idx < len(det_spec.shocks) else []
        )

        # Generate endval block for parameter changes (if any)
        if changing_params:
            endval_lines = [f"// Regime {regime_idx + 1}", ""]
            endval_lines.append(f"endval(learnt_in={learnt_in});")

            for param in changing_params:
                # Calculate absolute deviation from baseline
                baseline_value = det_spec.preset_par_init.get(param, 0.0)
                regime_value = regime_params.get(param, baseline_value)
                val_diff = regime_value - baseline_value

                endval_lines.append(f"e_TV_{param} = {val_diff:.16f};")

            endval_lines.append("end;")
            blocks.append("\n".join(endval_lines))

        # Generate shocks block for this regime (if any)
        if regime_shocks:
            for shock_var, shock_per, shock_val in regime_shocks:
                shock_lines = [f"// Regime {regime_idx + 1} shock to {shock_var}", ""]
                shock_lines.append(f"shocks(learnt_in={learnt_in});")
                shock_lines.append(f"    var e_{shock_var};")
                # shock_per is relative to regime start
                # Convert to absolute period by adding learnt_in
                absolute_period = int(shock_per + learnt_in - 1)
                shock_lines.append(f"    periods {absolute_period};")
                shock_lines.append(f"    values {shock_val:.16f};")
                shock_lines.append("end;")
                blocks.append("\n".join(shock_lines))

    return blocks


def export_to_dynare(
    model: "Model",
    output_path: str | Path | None = None,
    steady: bool = False,
    compute_irfs: bool = False,
    irf_var_list: list[str] | None = None,
    det_spec: "DetSpec | None" = None,
    det_spec_periods: int = 1000,
    det_spec_solver_kwargs: dict | None = None,
) -> str:
    """
    Export a finalized model to Dynare .mod file format.

    This is the main public function for Dynare export. It generates a complete
    Dynare .mod file including:
    - Parameter declarations and values
    - Variable declarations (endogenous, exogenous)
    - Model equations (AR(1) processes, intermediate, transition, expectations, optimality)
    - Initial values block (from steady state or guesses)
    - Shocks block with unit standard errors
    - Optionally, a steady state solver command
    - Optionally, a stoch_simul command for IRFs
    - Optionally, perfect foresight regime blocks and solver commands (when det_spec provided)

    Parameters
    ----------
    model : Model
        A finalized Model instance to export. Must have called model.finalize().
    output_path : str, Path, or None, optional
        Path to write the .mod file. If None, writes to settings.paths.debug_dir
        with filename pattern: <model.label>[_irfs][_<det_spec.label>].mod
        Examples: "_default.mod", "_default_irfs.mod", "_default_boom.mod",
        "_default_irfs_recession.mod"
    steady : bool, optional
        If True, add a "steady;" command after the initval block to compute
        the steady state in Dynare. Default is False.
    compute_irfs : bool, optional
        If True, add a "stoch_simul(order=1)" command after the shocks block.
        Default is False.
    irf_var_list : list[str] or None, optional
        Variables to include in the stoch_simul command. If None, omit
        variables so Dynare computes IRFs for all variables. Default is None.
    det_spec : DetSpec or None, optional
        Deterministic scenario specification for perfect foresight paths.
        If provided, generates time-varying parameter infrastructure and
        regime-specific endval/shocks blocks. Default is None.
    det_spec_periods : int, optional
        Horizon for perfect foresight simulation when det_spec is provided.
        Default is 1000.
    det_spec_solver_kwargs : dict or None, optional
        Additional kwargs for Dynare's perfect_foresight_with_expectation_errors_solver
        command (e.g., {"homotopy_initial_step_size": 0.5}). Default is None.

    Returns
    -------
    str
        The generated Dynare .mod file content.

    Raises
    ------
    RuntimeError
        If the model has not been finalized (model.var_lists is None).
    ValueError
        If det_spec specifies changing a parameter that already has TV_<param>
        as an exogenous variable in the model.

    Examples
    --------
    >>> model.finalize()
    >>> # Write to default location (debug_dir/<label>.mod)
    >>> dynare_code = export_to_dynare(model)
    >>> # Write to specific location
    >>> dynare_code = export_to_dynare(model, output_path="my_model.mod")
    >>> # Include steady state computation command
    >>> dynare_code = export_to_dynare(model, steady=True)
    >>> # Include IRFs for all variables
    >>> dynare_code = export_to_dynare(model, compute_irfs=True)
    >>> # Include IRFs for a subset of variables
    >>> dynare_code = export_to_dynare(
    ...     model, compute_irfs=True, irf_var_list=["y", "c"]
    ... )
    >>> # Include perfect foresight regime changes
    >>> from equilibrium.solvers.det_spec import DetSpec
    >>> spec = DetSpec(preset_par_init={"delta": 0.1})
    >>> spec.add_regime(0, preset_par_regime={"delta": 0.15}, time_regime=10)
    >>> dynare_code = export_to_dynare(model, det_spec=spec, det_spec_periods=100)

    Notes
    -----
    The generated file uses Dynare's standard timing notation:
    - x(-1) for lagged variables
    - x(+1) for forward-looking variables
    - x without timing for current period

    Exogenous shocks follow the convention e_<varname> with unit stderr.
    Actual shock scaling is handled via VOL_<var> parameters in AR(1) equations.

    When det_spec is provided, time-varying parameters are implemented using:
    - TV_<param>: endogenous AR(1) variable for each changing parameter
    - e_TV_<param>: exogenous shock to drive parameter changes
    - All references to <param> are replaced with (<param> + TV_<param>)
    - endval(learnt_in=t) blocks set parameter deviations from baseline
    """
    # Validation
    if model.var_lists is None:
        raise RuntimeError(
            "Model must be finalized before exporting to Dynare. "
            "Call model.finalize() first."
        )

    # DetSpec processing
    changing_params = []
    tv_infrastructure = {}
    regime_blocks = []

    if det_spec is not None:
        # Identify parameters that change across regimes
        changing_params = _identify_changing_parameters(det_spec, model)

        # Create TV_ infrastructure if there are changing parameters
        if changing_params:
            tv_vars, tv_shocks, tv_params = _create_tv_parameter_infrastructure(
                changing_params
            )
            tv_infrastructure = {
                "vars": tv_vars,
                "shocks": tv_shocks,
                "params": tv_params,
            }

        # Generate regime-specific blocks
        regime_blocks = _generate_regime_blocks(det_spec, changing_params)

    # Build the .mod file content
    blocks = []

    # Parameters block (include TV_ params if det_spec provided)
    params = list(model.var_lists["params"])
    if tv_infrastructure:
        params = params + list(tv_infrastructure["params"].keys())
    if params:
        param_str = format_var_list(params)
        blocks.append(f"parameters {param_str};")

    # Var block (endogenous, include TV_ vars if det_spec provided)
    endogenous = (
        model.var_lists["u"]
        + model.var_lists["x"]
        + model.var_lists["intermediate"]
        + model.var_lists["E"]
    )
    if tv_infrastructure:
        endogenous = endogenous + tv_infrastructure["vars"]
    if endogenous:
        var_str = format_var_list(endogenous)
        blocks.append(f"var {var_str};")

    # Varexo block (with "e_" prefix, include e_TV_ shocks if det_spec provided)
    exogenous = model.var_lists["z"]
    shock_names = [f"e_{var}" for var in exogenous]
    if tv_infrastructure:
        shock_names = shock_names + tv_infrastructure["shocks"]
    if shock_names:
        varexo_str = format_var_list(shock_names)
        blocks.append(f"varexo {varexo_str};")

    # Parameter assignments
    if params:
        param_assignments = ["// Parameters", ""]
        # First add model parameters
        for param in model.var_lists["params"]:
            value = model.params[param]
            param_assignments.append(f"{param} = {value:.16f};")
        # Then add TV_ parameters if present
        if tv_infrastructure:
            param_assignments.append("")
            param_assignments.append("// Time-varying parameter infrastructure")
            for param, value in tv_infrastructure["params"].items():
                param_assignments.append(f"{param} = {value:.16f};")
        blocks.append("\n".join(param_assignments))

    # Model block (with TV_ infrastructure if det_spec provided)
    model_equations = _generate_model_block(model, changing_params, tv_infrastructure)
    blocks.append("\n".join(model_equations))

    # Initval block (with TV_ vars if det_spec provided)
    initval_lines = _generate_initval_block(model, tv_infrastructure)
    if initval_lines:
        blocks.append("\n".join(initval_lines))

    # Shocks block
    shocks_lines = _generate_shocks_block(model)
    if shocks_lines:
        blocks.append("\n".join(shocks_lines))

    # Steady state command
    if steady:
        blocks.append("steady;")

    # Stochastic simulation (IRFs)
    if compute_irfs:
        blocks.append(_generate_stoch_simul_block(irf_var_list))

    # Perfect foresight regime blocks and commands (if det_spec provided)
    if det_spec is not None:
        # Add regime-specific endval and shocks blocks
        blocks.extend(regime_blocks)

        # Add perfect foresight setup command
        blocks.append(
            f"perfect_foresight_with_expectation_errors_setup(periods={det_spec_periods});"
        )

        # Add perfect foresight solver command
        if det_spec_solver_kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in det_spec_solver_kwargs.items())
            blocks.append(
                f"perfect_foresight_with_expectation_errors_solver({kwargs_str});"
            )
        else:
            blocks.append("perfect_foresight_with_expectation_errors_solver;")

    # Combine with blank lines between blocks
    content = "\n\n".join(blocks)
    if content:
        content += "\n"

    # Determine output path
    if output_path is None:
        settings = get_settings()
        suffix = ""
        if compute_irfs:
            suffix += "_irfs"
        if det_spec is not None:
            suffix += f"_{det_spec.label}"
        output_path = settings.paths.debug_dir / f"{model.label}{suffix}.mod"
    else:
        output_path = Path(output_path)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    logger.info(f"Dynare .mod file written to: {output_path}")

    return content


def format_var_list(var_list: list[str], indent: str = "  ") -> str:
    """
    Format a list of variables as comma-separated with line wrapping.

    This utility is exposed publicly as it may be useful for other export
    formats (e.g., GAMS, Julia).

    Parameters
    ----------
    var_list : list of str
        Variable names to format
    indent : str, optional
        Indentation for continuation lines. Default is "  " (two spaces).

    Returns
    -------
    str
        Formatted string with commas and line breaks (without trailing semicolon)

    Examples
    --------
    >>> vars = ["consumption", "capital", "labor", "output"]
    >>> print(format_var_list(vars))
    consumption, capital, labor, output

    >>> long_vars = [f"var_{i}" for i in range(20)]
    >>> print(format_var_list(long_vars))
    var_0, var_1, ..., var_8,
      var_9, var_10, ..., var_17,
      var_18, var_19
    """
    if not var_list:
        return ""

    max_line_length = 80
    lines = []
    current_line = var_list[0]

    for var in var_list[1:]:
        test_line = current_line + ", " + var
        if len(test_line) <= max_line_length:
            current_line = test_line
        else:
            lines.append(current_line + ",")
            current_line = indent + var

    # Add final line (no comma after last variable)
    lines.append(current_line)

    return "\n".join(lines)


def _format_space_separated_list(
    var_list: list[str], indent: str = "  ", max_line_length: int = 80
) -> str:
    if not var_list:
        return ""

    lines = []
    current_line = var_list[0]

    for var in var_list[1:]:
        test_line = current_line + " " + var
        if len(test_line) <= max_line_length:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = indent + var

    lines.append(current_line)
    return "\n".join(lines)


def _generate_stoch_simul_block(irf_var_list: list[str] | None) -> str:
    if not irf_var_list:
        return "stoch_simul(order=1);"

    var_str = _format_space_separated_list(irf_var_list)
    if "\n" in var_str:
        lines = var_str.splitlines()
        lines[0] = f"stoch_simul(order=1) {lines[0]}"
        lines[-1] = f"{lines[-1]};"
        return "\n".join(lines)

    return f"stoch_simul(order=1) {var_str};"


def _generate_model_block(
    model: "Model",
    changing_params: list[str] = None,
    tv_infrastructure: dict = None,
) -> list[str]:
    """
    Generate the model block with all equations.

    Parameters
    ----------
    model : Model
        The model instance to export
    changing_params : list[str], optional
        List of parameters that change across regimes (for DetSpec)
    tv_infrastructure : dict, optional
        Dictionary with TV_ infrastructure (vars, shocks, params)

    Returns
    -------
    list of str
        Lines for the model block
    """
    if changing_params is None:
        changing_params = []
    if tv_infrastructure is None:
        tv_infrastructure = {}

    # Get all parameter names for replacement logic
    all_params = set(model.var_lists["params"])

    model_equations = ["// Equilibrium conditions", "", "model;", ""]

    # 1. AR(1) exogenous processes
    for z in model.exog_list:
        pers = f"PERS_{z}"
        vol = f"VOL_{z}"
        eq = f"{z} = {pers} * {z}(-1) + {vol} * e_{z};"
        model_equations.append(eq)

    if model.exog_list:
        model_equations.append("")

    # 1b. TV_ AR(1) processes (if det_spec provided)
    if tv_infrastructure.get("vars"):
        model_equations.append("// Time-varying parameter processes")
        for tv_var in tv_infrastructure["vars"]:
            # Extract param name from TV_<param>
            param_name = tv_var[3:]  # Remove "TV_" prefix
            mu = f"mu_TV_{param_name}"
            rho = f"rho_TV_{param_name}"
            sig = f"sig_TV_{param_name}"
            shock = f"e_TV_{param_name}"
            eq = f"{tv_var} = (1.0 - {rho}) * {mu} + {rho} * {tv_var}(-1) + {sig} * {shock};"
            model_equations.append(eq)
        model_equations.append("")

    # 2. Intermediate equations (with parameter replacement if needed)
    for var, expr in model.rules["intermediate"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        # Replace parameters with (param + TV_param) if changing
        if changing_params:
            expr_dynare = _replace_params_in_equation(
                expr_dynare, changing_params, all_params
            )
        model_equations.append(f"{var} = {expr_dynare};")

    if model.rules["intermediate"]:
        model_equations.append("")

    # 3. Transition equations (with lags and parameter replacement if needed)
    for var, expr in model.rules["transition"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=True, model=model)
        # Replace parameters with (param + TV_param) if changing
        if changing_params:
            expr_dynare = _replace_params_in_equation(
                expr_dynare, changing_params, all_params
            )
        model_equations.append(f"{var} = {expr_dynare};")

    if model.rules["transition"]:
        model_equations.append("")

    # 4. Expectation equations (with parameter replacement if needed)
    for var, expr in model.rules["expectations"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        # Replace parameters with (param + TV_param) if changing
        if changing_params:
            expr_dynare = _replace_params_in_equation(
                expr_dynare, changing_params, all_params
            )
        model_equations.append(f"{var} = {expr_dynare};")

    if model.rules["expectations"]:
        model_equations.append("")

    # 5. Optimality equations (with parameter replacement if needed)
    for var, expr in model.rules["optimality"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        # Replace parameters with (param + TV_param) if changing
        if changing_params:
            expr_dynare = _replace_params_in_equation(
                expr_dynare, changing_params, all_params
            )
        model_equations.append(f"{var} = {expr_dynare};")

    model_equations.append("")
    model_equations.append("end;")

    return model_equations


def _generate_initval_block(
    model: "Model", tv_infrastructure: dict = None
) -> list[str]:
    """
    Generate the initval block for Dynare export.

    Parameters
    ----------
    model : Model
        The model instance to export
    tv_infrastructure : dict, optional
        Dictionary with TV_ infrastructure (vars, shocks, params)

    Returns
    -------
    list of str
        Lines for the initval block, or empty list if no initialization needed.
    """
    if tv_infrastructure is None:
        tv_infrastructure = {}

    initval_lines = ["// Initial values", "", "initval;", ""]

    # Check if steady state has been solved
    steady_solved = (
        hasattr(model, "steady_dict")
        and model.steady_dict
        and getattr(model, "res_steady", None) is not None
        and getattr(model.res_steady, "success", False)
    )

    # 1. Set exogenous variables to zero
    for z_var in model.exog_list:
        initval_lines.append(f"{z_var} = 0;")

    if model.exog_list:
        initval_lines.append("")  # Blank line after exogenous

    # 2. Set endogenous states (x) and policy controls (u)
    endogenous_vars = model.var_lists["u"] + model.var_lists["x"]

    if steady_solved:
        # Use values from steady_dict
        for var in endogenous_vars:
            if _steady_dict_has_key(model.steady_dict, var):
                value = float(model.steady_dict[var])
                initval_lines.append(f"{var} = {value:.16f};")
            elif var in model.init_dict:
                value = float(model.init_dict[var])
                initval_lines.append(f"{var} = {value:.16f};")
            else:
                initval_lines.append(f"{var} = 0;")
    else:
        # Use analytical_steady rules if available, otherwise steady_guess
        analytical_steady = model.rules.get("analytical_steady", {})

        for var in endogenous_vars:
            if var in analytical_steady:
                # Evaluate analytical_steady rule in steady state context
                expr = analytical_steady[var]
                # Remove _NEXT suffixes for steady state
                expr_steady = expr.replace("_NEXT", "")
                # Convert to Dynare syntax
                expr_dynare = _convert_to_dynare_syntax(expr_steady, add_lags=False)
                initval_lines.append(f"{var} = {expr_dynare};")
            elif var in model.init_dict:
                value = float(model.init_dict[var])
                initval_lines.append(f"{var} = {value:.16f};")
            else:
                initval_lines.append(f"{var} = 0;")

    if endogenous_vars:
        initval_lines.append("")  # Blank line after endogenous

    # 3. Set intermediate variables using their rules
    for var, expr in model.rules["intermediate"].items():
        # Convert to Dynare syntax without lags
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        initval_lines.append(f"{var} = {expr_dynare};")

    if model.rules["intermediate"]:
        initval_lines.append("")

    # 4. Set expectations variables using their rules with _NEXT removed
    for var, expr in model.rules["expectations"].items():
        # Remove _NEXT suffixes to flatten timing to steady state
        expr_steady = expr.replace("_NEXT", "")
        # Convert to Dynare syntax
        expr_dynare = _convert_to_dynare_syntax(expr_steady, add_lags=False)
        initval_lines.append(f"{var} = {expr_dynare};")

    if model.rules["expectations"]:
        initval_lines.append("")

    # 5. Set TV_ variables to zero (if det_spec provided)
    if tv_infrastructure.get("vars"):
        initval_lines.append("// Time-varying parameter variables")
        for tv_var in tv_infrastructure["vars"]:
            initval_lines.append(f"{tv_var} = 0;")
        initval_lines.append("")

    initval_lines.append("end;")

    return initval_lines


def _generate_shocks_block(model: "Model") -> list[str]:
    """
    Generate the shocks block for Dynare export.

    Each exogenous variable gets a shock with stderr = 1.0.
    The actual shock scaling happens in the AR(1) equations via VOL_<var>.

    Parameters
    ----------
    model : Model
        The model instance to export

    Returns
    -------
    list of str
        Lines for the shocks block, or empty list if no exogenous variables.
    """
    if not model.exog_list:
        return []

    shocks_lines = ["shocks;", ""]

    # Add each shock with stderr = 1.0
    for z_var in model.exog_list:
        shock_name = f"e_{z_var}"
        shocks_lines.append(f"var {shock_name}; stderr 1.0;")

    shocks_lines.append("")
    shocks_lines.append("end;")

    return shocks_lines


def _convert_to_dynare_syntax(
    expr: str,
    add_lags: bool = False,
    model: "Model" | None = None,
) -> str:
    """
    Convert Python expression to Dynare syntax.

    Transformations:
    - Replace ** with ^
    - Remove np. prefix from functions
    - Replace jax.scipy.special.erf with erf
    - Replace _NEXT suffix with (+1)
    - Optionally add (-1) timing to variables (for transition equations)

    Parameters
    ----------
    expr : str
        Python expression string
    add_lags : bool, optional
        If True, add (-1) timing to variables (for transition equations).
        Default is False.
    model : Model or None, optional
        Model instance required when add_lags=True

    Returns
    -------
    str
        Dynare expression string

    Raises
    ------
    ValueError
        If add_lags=True but model is None
    """
    # Basic syntax transformations
    result = expr.replace("**", "^")
    result = result.replace("np.exp(", "exp(")
    result = result.replace("np.log(", "log(")
    result = result.replace("np.sqrt(", "sqrt(")

    # Handle jax.scipy.special functions
    result = result.replace("jax.scipy.special.erf(", "erf(")

    # Remove other np. prefixes
    result = re.sub(r"np\.(\w+)", r"\1", result)

    # Replace _NEXT suffix with (+1)
    result = re.sub(r"(\w+)_NEXT\b", r"\1(+1)", result)

    # Add lags if requested
    if add_lags:
        if model is None:
            raise ValueError("Model required for add_lags=True")
        result = _add_timing_to_transition_rhs(result, model)

    return result


def _add_timing_to_transition_rhs(expr: str, model: "Model") -> str:
    """
    Add (-1) timing notation to variables on RHS of transition equations.

    Only adds timing to endogenous variables, not to parameters or function names.

    Parameters
    ----------
    expr : str
        Expression string (already with Dynare syntax)
    model : Model
        Model instance to get variable lists

    Returns
    -------
    str
        Expression with (-1) added to variable references
    """
    # Build sets of variables and parameters
    all_vars = set(
        model.var_lists["u"]
        + model.var_lists["x"]
        + model.var_lists["intermediate"]
        + model.var_lists["E"]
        + model.exog_list
    )
    params = set(model.var_lists["params"])

    # Functions that should not get timing
    functions = {
        "exp",
        "log",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "abs",
        "erf",
        "max",
        "min",
    }

    def replace_variable(match):
        var_name = match.group(1)

        if var_name in functions or var_name in params:
            return var_name
        if var_name in all_vars:
            return var_name + "(-1)"
        return var_name

    # Match word tokens, avoid if already followed by (
    result = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\()", replace_variable, expr)

    return result


def _steady_dict_has_key(steady_dict, key: str) -> bool:
    """
    Check if a key exists in steady_dict (works for dict and NamedTuple).

    Parameters
    ----------
    steady_dict : dict or NamedTuple
        The steady state dictionary
    key : str
        The key to check

    Returns
    -------
    bool
        True if key exists in steady_dict
    """
    if hasattr(steady_dict, "_fields"):
        return key in steady_dict._fields
    else:
        return key in steady_dict
