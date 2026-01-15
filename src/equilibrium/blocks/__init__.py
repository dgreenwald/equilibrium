"""
Common model blocks that can be plugged into models.

This package contains reusable model blocks for common economic model components.
"""

from .macro import debt_block, investment_block, st_bond_block
from .symbolic import preference_block

__all__ = ["preference_block", "investment_block", "st_bond_block", "debt_block"]
