"""
Symbolic preference block generator using SymPy for automatic differentiation.

This module uses SymPy to automatically compute marginal utilities through symbolic
differentiation, similar to the sym_fcns.py approach used in the model-generator
repository.

References
----------
Original implementation: https://github.com/dgreenwald/model-generator/blob/master/sym_fcns.py
"""

import sympy as sp
from sympy import diff, log, simplify, symbols

from ..model import ModelBlock, model_block


def _sympy_to_numpy(expr, agent: str = "AGENT", atype: str = "ATYPE") -> str:
    """Convert a SymPy expression to a NumPy-compatible string for Equilibrium code generation.

    This function converts SymPy symbolic expressions to string format compatible
    with Equilibrium's code generation system. It handles variable and parameter
    substitutions with agent/atype suffixes and replaces SymPy functions with
    NumPy equivalents.

    Parameters
    ----------
    expr : sympy.Expr
        The SymPy expression to convert.
    agent : str, default "AGENT"
        Agent identifier suffix for variables.
    atype : str, default "ATYPE"
        Agent type identifier suffix for parameters.

    Returns
    -------
    str
        A string representation suitable for Equilibrium code generation.

    Notes
    -----
    The function performs the following transformations:
    - Simplifies the expression using SymPy's simplify()
    - Replaces SymPy functions (log, exp, sqrt) with NumPy equivalents (np.log, np.exp, np.sqrt)
    - Adds agent/atype suffixes to variable and parameter names
    - Preserves power operations (** is already compatible)
    """
    import re

    # Simplify the expression first
    expr = simplify(expr)

    # Convert to string
    expr_str = str(expr)

    # Replace SymPy functions with NumPy equivalents using word boundaries
    # This ensures we only replace function names, not parts of other words
    expr_str = re.sub(r"\blog\b", "np.log", expr_str)
    expr_str = re.sub(r"\bexp\b", "np.exp", expr_str)
    expr_str = re.sub(r"\bsqrt\b", "np.sqrt", expr_str)

    # Add agent and atype suffixes to variables and parameters
    # Variables: c, h, n -> c_AGENT, h_AGENT, n_AGENT
    # Parameters: psi, varphi, eta, xi, eps_h -> psi_ATYPE, varphi_ATYPE, etc.

    var_names = ["c", "h", "n", "v"]
    param_names = ["varphi", "eps_h", "psi", "eta", "xi"]

    # Use word boundaries to ensure we only replace complete variable/parameter names
    # Sort by length (descending) to handle multi-character names first
    for param in sorted(param_names, key=len, reverse=True):
        expr_str = re.sub(r"\b" + param + r"\b", f"{param}_{atype}", expr_str)

    for var in sorted(var_names, key=len, reverse=True):
        expr_str = re.sub(r"\b" + var + r"\b", f"{var}_{agent}", expr_str)

    return expr_str


@model_block
def preference_block(
    block,
    *,
    agent: str = "AGENT",
    atype: str = "ATYPE",
    housing: bool = False,
    labor: bool = False,
    housing_spec: str = "cobb_douglas",
    util_type: str = "crra",
    nominal: bool = True,
) -> ModelBlock:
    """Generate a preference block using symbolic differentiation.

    This function uses SymPy to define utility functions symbolically with different
    specifications and automatically computes marginal utilities through symbolic
    differentiation. It supports various utility function types, housing aggregation
    methods, and labor disutility.

    Parameters
    ----------
    agent : str, default "AGENT"
        Agent identifier for variable naming (e.g., "borrower", "lender").
    atype : str, default "ATYPE"
        Agent type identifier for parameter naming (e.g., "b", "l").
    housing : bool, default False
        Whether to include housing in utility.
    labor : bool, default False
        Whether to include labor disutility.
    housing_spec : str, default "cobb_douglas"
        Housing aggregation specification. Options:
        - "cobb_douglas": x = h^xi * c^(1-xi) (separable) or h^xi * (c-v)^(1-xi) (GHH)
        - "ces": x = [xi*h^(1-eps_h) + (1-xi)*c^(1-eps_h)]^(1/(1-eps_h))
        - "h_exponent": x = h^xi * c (separable) or h^xi * (c-v) (GHH)
        - "substitutes": x = (1-xi)*c + xi*h (separable)
        - "ql_housing": x = (1-xi)*c + xi*h - v (variant for QL housing)
    util_type : str, default "crra"
        Utility function type. Options:
        - "crra": u = x^(1-psi)/(1-psi) - v
        - "unit_eis": u = log(x) - v (log utility, EIS = 1)
        - "risk_neutral": u = x - v
        - "ghh": Same utility forms but v incorporated into x
    nominal : bool, default True
        Whether to include nominal marginal utility (deflated by inflation).

    Returns
    -------
    ModelBlock
        A ModelBlock with intermediate rules for utility and marginal utilities:
        - uc_{agent}: Marginal utility of consumption
        - uh_{agent}: Marginal utility of housing (if housing=True)
        - n_un_{agent}: Marginal disutility of labor (if labor=True)
        - Lam_1_{agent}: Current period Lagrange multiplier
        - Lam_0_{agent}: Previous period Lagrange multiplier
        - Lam_1_nom_{agent}: Nominal Lagrange multiplier (if nominal=True)
        - x_{agent}: Composite consumption (if housing=True)
        - v_{agent}: Labor disutility (if labor=True)

    Raises
    ------
    ValueError
        If util_type or housing_spec is not a valid option.

    Examples
    --------
    Basic CRRA preferences without housing or labor:

    >>> block = preference_block_symbolic(util_type="crra")
    >>> 'uc_AGENT' in block.rules['intermediate']
    True

    Unit EIS (log) utility:

    >>> block = preference_block_symbolic(util_type="unit_eis")
    >>> 'uc_AGENT' in block.rules['intermediate']
    True

    GHH preferences with labor:

    >>> block = preference_block_symbolic(util_type="ghh", labor=True)
    >>> 'n_un_AGENT' in block.rules['intermediate']
    True

    Cobb-Douglas housing aggregation:

    >>> block = preference_block_symbolic(housing=True, housing_spec="cobb_douglas")
    >>> 'uh_AGENT' in block.rules['intermediate']
    True
    >>> 'x_AGENT' in block.rules['intermediate']
    True

    Notes
    -----
    The mathematical formulas for each specification:

    **Utility Functions:**

    CRRA: u = x^(1-psi) / (1-psi) - v

    Unit EIS: u = log(x) - v

    Risk Neutral: u = x - v

    GHH: Same forms but v is incorporated into x rather than subtracted

    **Housing Specifications:**

    Cobb-Douglas: x = h^xi * c^(1-xi) or h^xi * (c-v)^(1-xi) for GHH

    CES: x = [xi*h^(1-eps_h) + (1-xi)*c^(1-eps_h)]^(1/(1-eps_h))

    H-exponent: x = h^xi * c or h^xi * (c-v) for GHH

    Substitutes: x = (1-xi)*c + xi*h or (1-xi)*c + xi*h - v for ql_housing

    **Labor Disutility:**

    v = eta * n^(1+varphi) / (1+varphi)

    References
    ----------
    Original implementation: https://github.com/dgreenwald/model-generator/blob/master/sym_fcns.py
    """
    # Validate inputs
    valid_util_types = ["crra", "unit_eis", "risk_neutral", "ghh"]
    if util_type not in valid_util_types:
        raise ValueError(
            f"Invalid util_type: {util_type}. Valid options are: {valid_util_types}"
        )

    valid_housing_specs = [
        "cobb_douglas",
        "ces",
        "h_exponent",
        "substitutes",
        "ql_housing",
    ]
    if housing and housing_spec not in valid_housing_specs:
        raise ValueError(
            f"Invalid housing_spec: {housing_spec}. Valid options are: {valid_housing_specs}"
        )

    # Define symbolic variables
    c, h, n = symbols("c h n")

    # Define symbolic parameters
    psi, varphi, eta, xi, eps_h = symbols("psi varphi eta xi eps_h")

    # Build labor disutility if needed
    # We use v_sym as a symbol for code generation, and v_actual for derivatives
    v_sym = symbols("v")  # Symbol for code generation
    v_actual = sp.Integer(0)  # Actual expression for derivative computation

    if labor:
        v_actual = eta * (n ** (1 + varphi)) / (1 + varphi)
        # Add v to intermediate rules
        v_expr = _sympy_to_numpy(v_actual, agent, atype)
        block.rules["intermediate"] += [
            (f"v_{agent}", v_expr),
        ]

    # Build composite consumption x based on housing and housing_spec
    # We build two versions: x_codegen (for code generation) and x_deriv (for derivatives)
    x_codegen = c  # for code generation
    x_deriv = c  # for computing derivatives

    if housing:
        is_ghh = util_type == "ghh"

        if housing_spec == "cobb_douglas":
            if is_ghh and labor:
                x_codegen = (h**xi) * ((c - v_sym) ** (1 - xi))
                x_deriv = (h**xi) * ((c - v_actual) ** (1 - xi))
            else:
                x_codegen = (h**xi) * (c ** (1 - xi))
                x_deriv = x_codegen

        elif housing_spec == "ces":
            x_codegen = (xi * (h ** (1 - eps_h)) + (1 - xi) * (c ** (1 - eps_h))) ** (
                1 / (1 - eps_h)
            )
            x_deriv = x_codegen

        elif housing_spec == "h_exponent":
            if is_ghh and labor:
                x_codegen = (h**xi) * (c - v_sym)
                x_deriv = (h**xi) * (c - v_actual)
            else:
                x_codegen = (h**xi) * c
                x_deriv = x_codegen

        elif housing_spec == "substitutes":
            x_codegen = (1 - xi) * c + xi * h
            x_deriv = x_codegen

        elif housing_spec == "ql_housing":
            # For ql_housing variant with substitutes
            if labor:
                x_codegen = (1 - xi) * c + xi * h - v_sym
                x_deriv = (1 - xi) * c + xi * h - v_actual
            else:
                x_codegen = (1 - xi) * c + xi * h
                x_deriv = x_codegen

        # Add x to intermediate rules (using code generation version)
        x_expr = _sympy_to_numpy(x_codegen, agent, atype)
        block.rules["intermediate"] += [
            (f"x_{agent}", x_expr),
        ]

    # Build utility function based on util_type
    # We build two versions for consistency
    if util_type == "crra":
        u_codegen = (x_codegen ** (1 - psi)) / (1 - psi)
        u_deriv = (x_deriv ** (1 - psi)) / (1 - psi)
        if not (util_type == "ghh" and labor):
            u_codegen = u_codegen - v_sym if labor else u_codegen
            u_deriv = u_deriv - v_actual if labor else u_deriv

    elif util_type == "unit_eis":
        u_codegen = log(x_codegen)
        u_deriv = log(x_deriv)
        if not (util_type == "ghh" and labor):
            u_codegen = u_codegen - v_sym if labor else u_codegen
            u_deriv = u_deriv - v_actual if labor else u_deriv

    elif util_type == "risk_neutral":
        u_codegen = x_codegen
        u_deriv = x_deriv
        if not (util_type == "ghh" and labor):
            u_codegen = u_codegen - v_sym if labor else u_codegen
            u_deriv = u_deriv - v_actual if labor else u_deriv

    elif util_type == "ghh":
        # For GHH, v is incorporated into x, so we don't subtract it
        # Use the same utility functions but with x that already includes v
        # Actually, for GHH we need to handle it differently
        # The GHH case should use the same utility forms but x incorporates v

        # For true GHH, we need to be more careful
        # Let's use CRRA as the default form for GHH
        # But x already incorporates v when housing_spec does it
        # If no housing, we need to incorporate v into x
        if not housing and labor:
            x_codegen = c - v_sym
            x_deriv = c - v_actual
            # Update x in intermediate rules
            x_expr = _sympy_to_numpy(x_codegen, agent, atype)
            block.rules["intermediate"] += [
                (f"x_{agent}", x_expr),
            ]

        # Now apply the utility function to x (which already has v incorporated)
        # Default to CRRA form for GHH
        u_codegen = (x_codegen ** (1 - psi)) / (1 - psi)
        u_deriv = (x_deriv ** (1 - psi)) / (1 - psi)

    # Compute marginal utilities using symbolic differentiation
    # Use u_deriv for computing derivatives (it has the full expression)
    uc = diff(u_deriv, c)
    uc_expr = _sympy_to_numpy(uc, agent, atype)
    block.rules["intermediate"] += [
        (f"uc_{agent}", uc_expr),
    ]

    # Compute marginal utility of housing if housing is True
    if housing:
        uh = diff(u_deriv, h)
        uh_expr = _sympy_to_numpy(uh, agent, atype)
        block.rules["intermediate"] += [
            (f"uh_{agent}", uh_expr),
        ]

    # Compute marginal disutility of labor if labor is True
    if labor:
        n_un = -diff(u_deriv, n)  # negative marginal utility
        n_un_expr = _sympy_to_numpy(n_un, agent, atype)
        block.rules["intermediate"] += [
            (f"n_un_{agent}", n_un_expr),
        ]

    # Add components of the SDF
    block.rules["intermediate"] += [
        (f"Lam_1_{agent}", f"uc_{agent}"),
        (f"Lam_0_{agent}", f"uc_{agent} / bet_{atype}"),
    ]

    # Add nominal marginal utility if requested
    if nominal:
        block.rules["intermediate"] += [
            (f"Lam_1_nom_{agent}", f"Lam_1_{agent} / pi"),
        ]
