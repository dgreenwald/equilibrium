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

logger = logging.getLogger(__name__)


def export_to_dynare(
    model: "Model",
    output_path: str | Path | None = None,
    steady: bool = False,
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

    Parameters
    ----------
    model : Model
        A finalized Model instance to export. Must have called model.finalize().
    output_path : str, Path, or None, optional
        Path to write the .mod file. If None, writes to settings.paths.debug_dir
        as <model.label>.mod (e.g., "_default.mod").
    steady : bool, optional
        If True, add a "steady;" command after the initval block to compute
        the steady state in Dynare. Default is False.

    Returns
    -------
    str
        The generated Dynare .mod file content.

    Raises
    ------
    RuntimeError
        If the model has not been finalized (model.var_lists is None).

    Examples
    --------
    >>> model.finalize()
    >>> # Write to default location (debug_dir/<label>.mod)
    >>> dynare_code = export_to_dynare(model)
    >>> # Write to specific location
    >>> dynare_code = export_to_dynare(model, output_path="my_model.mod")
    >>> # Include steady state computation command
    >>> dynare_code = export_to_dynare(model, steady=True)

    Notes
    -----
    The generated file uses Dynare's standard timing notation:
    - x(-1) for lagged variables
    - x(+1) for forward-looking variables
    - x without timing for current period

    Exogenous shocks follow the convention e_<varname> with unit stderr.
    Actual shock scaling is handled via VOL_<var> parameters in AR(1) equations.
    """
    # Validation
    if model.var_lists is None:
        raise RuntimeError(
            "Model must be finalized before exporting to Dynare. "
            "Call model.finalize() first."
        )

    # Build the .mod file content
    blocks = []

    # Parameters block
    params = model.var_lists["params"]
    if params:
        param_str = format_var_list(params)
        blocks.append(f"parameters {param_str};")

    # Var block (endogenous)
    endogenous = (
        model.var_lists["u"]
        + model.var_lists["x"]
        + model.var_lists["intermediate"]
        + model.var_lists["E"]
    )
    if endogenous:
        var_str = format_var_list(endogenous)
        blocks.append(f"var {var_str};")

    # Varexo block (with "e_" prefix)
    exogenous = model.var_lists["z"]
    if exogenous:
        shock_names = [f"e_{var}" for var in exogenous]
        varexo_str = format_var_list(shock_names)
        blocks.append(f"varexo {varexo_str};")

    # Parameter assignments
    if params:
        param_assignments = ["// Parameters", ""]
        for param in params:
            value = model.params[param]
            param_assignments.append(f"{param} = {value:.16f};")
        blocks.append("\n".join(param_assignments))

    # Model block
    model_equations = _generate_model_block(model)
    blocks.append("\n".join(model_equations))

    # Initval block
    initval_lines = _generate_initval_block(model)
    if initval_lines:
        blocks.append("\n".join(initval_lines))

    # Shocks block
    shocks_lines = _generate_shocks_block(model)
    if shocks_lines:
        blocks.append("\n".join(shocks_lines))

    # Steady state command
    if steady:
        blocks.append("steady;")

    # Combine with blank lines between blocks
    content = "\n\n".join(blocks)
    if content:
        content += "\n"

    # Determine output path
    if output_path is None:
        settings = get_settings()
        output_path = settings.paths.debug_dir / f"{model.label}.mod"
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


def _generate_model_block(model: "Model") -> list[str]:
    """
    Generate the model block with all equations.

    Parameters
    ----------
    model : Model
        The model instance to export

    Returns
    -------
    list of str
        Lines for the model block
    """
    model_equations = ["// Equilibrium conditions", "", "model;", ""]

    # 1. AR(1) exogenous processes
    for z in model.exog_list:
        pers = f"PERS_{z}"
        vol = f"VOL_{z}"
        eq = f"{z} = {pers} * {z}(-1) + {vol} * e_{z};"
        model_equations.append(eq)

    if model.exog_list:
        model_equations.append("")

    # 2. Intermediate equations
    for var, expr in model.rules["intermediate"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        model_equations.append(f"{var} = {expr_dynare};")

    if model.rules["intermediate"]:
        model_equations.append("")

    # 3. Transition equations (with lags)
    for var, expr in model.rules["transition"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=True, model=model)
        model_equations.append(f"{var} = {expr_dynare};")

    if model.rules["transition"]:
        model_equations.append("")

    # 4. Expectation equations
    for var, expr in model.rules["expectations"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        model_equations.append(f"{var} = {expr_dynare};")

    if model.rules["expectations"]:
        model_equations.append("")

    # 5. Optimality equations
    for var, expr in model.rules["optimality"].items():
        expr_dynare = _convert_to_dynare_syntax(expr, add_lags=False)
        model_equations.append(f"{var} = {expr_dynare};")

    model_equations.append("")
    model_equations.append("end;")

    return model_equations


def _generate_initval_block(model: "Model") -> list[str]:
    """
    Generate the initval block for Dynare export.

    Parameters
    ----------
    model : Model
        The model instance to export

    Returns
    -------
    list of str
        Lines for the initval block, or empty list if no initialization needed.
    """
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
