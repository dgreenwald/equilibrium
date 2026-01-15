#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model feature specification and labeling utilities.
"""

from collections.abc import Sequence
from typing import List


def get_mod_str(mod) -> str:
    """Return the canonical label for a model feature specification."""
    return ModSpec(mod).label


class ModSpec:
    """
    Normalize a model "feature" specification into a canonical label.

    Parameters
    ----------
    base_mod : None, str, or sequence of str
        - None      -> no features, empty label
        - "foo"     -> features = ["foo"], label = "foo"
        - ["a","b"] -> features = ["a","b"], label = "a_b"

    Notes
    -----
    If you want canonical, order-insensitive labels, consider sorting `features`
    (see comment in __init__).
    """

    def __init__(self, base_mod) -> None:
        if base_mod is None:
            features: List[str] = []
        elif isinstance(base_mod, str):
            features = [base_mod]
        elif isinstance(base_mod, Sequence):
            # Coerce each element to string explicitly
            features = [str(f) for f in base_mod]
        else:
            # Fallback: single feature from stringified object
            features = [str(base_mod)]

        # Option 1: preserve user order (current behavior)
        self.features = features

        # Option 2 (canonical): uncomment the next line instead
        # self.features = sorted(set(features))

        self.label = "_".join(self.features)

    def __contains__(self, key: str) -> bool:
        """Return True if `key` is one of the features."""
        return key in self.features

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return f"ModSpec(features={self.features!r}, label={self.label!r})"
