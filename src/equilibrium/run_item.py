#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run grouping utilities built around ModSpec normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .modspec import ModSpec

SpecInput = ModSpec | str | Iterable[str] | None


def _coerce_modspec(spec: SpecInput) -> ModSpec:
    """Normalize a raw specification into a ModSpec instance."""
    if isinstance(spec, ModSpec):
        return spec
    return ModSpec(spec)


@dataclass(frozen=True, init=False)
class RunItem:
    """
    Bundle a model specification together with zero or more experiment specs.

    Parameters
    ----------
    mod_spec : ModSpec, str, iterable of str, or None
        Base model feature specification.
    experiments : iterable, optional
        Iterable of experiment specifications. Each item is normalized into a
        ``ModSpec``. To represent one experiment with multiple features, pass a
        nested iterable such as ``[["low_rates", "boom"]]``.

    Notes
    -----
    ``RunItem`` is intentionally small. It standardizes labels and feature
    access for batch configuration, but leaves solver execution policy to user
    code.
    """

    mod_spec: ModSpec
    experiments: tuple[ModSpec, ...]

    def __init__(
        self,
        mod_spec: SpecInput,
        experiments: Iterable[SpecInput] | None = None,
    ) -> None:
        object.__setattr__(self, "mod_spec", _coerce_modspec(mod_spec))

        normalized_experiments = tuple(
            _coerce_modspec(experiment) for experiment in (experiments or ())
        )
        object.__setattr__(self, "experiments", normalized_experiments)

    @property
    def label(self) -> str:
        """Canonical label for the model specification."""
        return self.mod_spec.label

    @property
    def experiment_labels(self) -> tuple[str, ...]:
        """Canonical labels for each experiment specification."""
        return tuple(experiment.label for experiment in self.experiments)

    def with_experiments(self, experiments: Iterable[SpecInput]) -> "RunItem":
        """Return a new run item with the same model spec and new experiments."""
        return RunItem(self.mod_spec, experiments=experiments)

