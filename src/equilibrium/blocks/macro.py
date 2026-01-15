"""
Structural blocks for macroeconomic models
"""

from equilibrium.model.model import ModelBlock, model_block


@model_block
def investment_block(
    block,
    *,
    tv_inv_efficiency: bool = False,
) -> ModelBlock:
    """Investment frictions"""

    block.rules["intermediate"] += [
        (
            "Phi_inv_AGENT",
            "tv_inv_efficiency * (inv_cost_0_ATYPE"
            " + inv_cost_1_ATYPE * pow(inv_rate_AGENT, 1.0 - inv_xi)"
            " / (1.0 - inv_xi))",
        ),
        (
            "d_Phi_inv_AGENT",
            "inv_cost_1_ATYPE * tv_inv_efficiency * pow(inv_rate_AGENT, -inv_xi)",
        ),
        ("Q_AGENT", "1.0 / d_Phi_inv_AGENT"),
        (
            "Q_bar_AGENT",
            "Q_AGENT + (Q_AGENT * Phi_inv_AGENT - inv_rate_AGENT) / (1.0 - delta)",
        ),
        ("inv_cost_1_AGENT", "delta ** inv_xi"),
        (
            "inv_cost_0_AGENT",
            "delta - inv_cost_1_AGENT * (delta ** (1.0 - inv_xi)) / (1.0 - inv_xi)",
        ),
    ]

    if not tv_inv_efficiency:
        block.rules["intermediate"] += [
            ("tv_inv_efficiency", "1.0"),
        ]

    # (no optimality eq here; inv_rate comes from firm FOC)


@model_block
def st_bond_block(
    block,
    *,
    include_lag: bool = False,
) -> ModelBlock:

    block.rules["expectations"] += [
        ("E_Lam_AGENT", "(Lam_1_nom_AGENT_NEXT / Lam_0_AGENT)"),
    ]

    block.rules["optimality"] += [
        ("R_new", "E_Lam_AGENT * R_new - 1.0"),
    ]

    block.rules["intermediate"] += [
        ("r_new", "R_new - 1.0"),
    ]

    block.rules["analytical_steady"] += [
        ("R_new", "pi_bar / bet_AGENT"),
    ]

    if include_lag:
        block.rules["transition"] += [
            ("R_lag", "R_new"),
        ]

        block.rules["intermediate"] += [
            ("r_lag", "R_lag - 1.0"),
        ]

        block.rules["analytical_steady"] += [
            ("R_lag", "R_new"),
        ]

    return block


@model_block
def debt_block(
    block,
    *,
    tv_prepayment: bool = False,
) -> ModelBlock:
    """Debt pricing"""

    if tv_prepayment:
        raise Exception

    block.rules["intermediate"] += [
        ("Om_denom_INSTRUMENT_AGENT", "1.0 - bet_pi_AGENT * frac_INSTRUMENT_remaining"),
    ]

    block.rules["optimality"] += [
        (
            "Om_principal_INSTRUMENT_AGENT",
            "E_Om_principal_INSTRUMENT_AGENT / Lam_0_AGENT - Om_principal_INSTRUMENT_AGENT",
        ),
        (
            "Om_spread_INSTRUMENT_AGENT",
            "E_Om_spread_INSTRUMENT_AGENT / Lam_0_AGENT - Om_spread_INSTRUMENT_AGENT",
        ),
    ]

    block.rules["expectations"] += [
        (
            "E_Om_principal_INSTRUMENT_AGENT",
            "Lam_1_nom_AGENT_NEXT * (marg_principal_flow_INSTRUMENT_AGENT_NEXT "
            "+ frac_INSTRUMENT_remaining * Om_principal_INSTRUMENT_AGENT_NEXT)",
        ),
        (
            "E_Om_spread_INSTRUMENT_AGENT",
            "Lam_1_nom_AGENT_NEXT * (marg_spread_flow_INSTRUMENT_AGENT_NEXT "
            "+ frac_INSTRUMENT_remaining * Om_spread_INSTRUMENT_AGENT_NEXT)",
        ),
    ]

    block.rules["analytical_steady"] += [
        (
            "Om_principal_INSTRUMENT_AGENT",
            "bet_pi_ATYPE * marg_principal_flow_INSTRUMENT_AGENT / Om_denom_INSTRUMENT_ATYPE",
        ),
        (
            "Om_spread_INSTRUMENT_AGENT",
            "bet_pi_ATYPE * marg_spread_flow_INSTRUMENT_AGENT / Om_denom_INSTRUMENT_ATYPE",
        ),
    ]
