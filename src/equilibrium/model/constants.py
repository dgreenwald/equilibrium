"""Model configuration constants shared across model modules.

This module provides shared constants used throughout the model package,
breaking circular dependencies between modules.
"""

# Standard rule categories for economic models
RULE_KEYS = (
    "intermediate",
    "read_expectations",
    "transition",
    "expectations",
    "optimality",
    "calibration",
    "analytical_steady",
)

# Rule keys requiring unique variable names across categories
UNIQUE_RULE_KEYS = (
    "intermediate",
    "read_expectations",
    "transition",
    "expectations",
    "optimality",
)
