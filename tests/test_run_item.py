#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ModSpec and RunItem normalization utilities.
"""

from equilibrium import ModSpec, RunItem


class TestModSpec:
    def test_string_input(self):
        spec = ModSpec("baseline")
        assert spec.features == ["baseline"]
        assert spec.label == "baseline"

    def test_sequence_input(self):
        spec = ModSpec(["baseline", "housing"])
        assert spec.features == ["baseline", "housing"]
        assert spec.label == "baseline_housing"

    def test_none_input(self):
        spec = ModSpec(None)
        assert spec.features == []
        assert spec.label == ""

    def test_contains(self):
        spec = ModSpec(["baseline", "housing"])
        assert "housing" in spec
        assert "pti_only" not in spec


class TestRunItem:
    def test_normalizes_model_spec(self):
        run = RunItem(["baseline", "housing"])
        assert isinstance(run.mod_spec, ModSpec)
        assert run.mod_spec.features == ["baseline", "housing"]
        assert run.label == "baseline_housing"
        assert run.experiments == ()

    def test_normalizes_experiments(self):
        run = RunItem(
            "baseline",
            experiments=[["low_rates", "boom"], "stress"],
        )
        assert run.label == "baseline"
        assert len(run.experiments) == 2
        assert run.experiment_labels == ("low_rates_boom", "stress")

    def test_preserves_existing_modspec_instances(self):
        base = ModSpec("baseline")
        exper = ModSpec(["low_rates", "boom"])
        run = RunItem(base, experiments=[exper])
        assert run.mod_spec is base
        assert run.experiments == (exper,)

    def test_with_experiments_returns_new_item(self):
        run = RunItem("baseline")
        updated = run.with_experiments([["low_rates"]])
        assert updated is not run
        assert updated.label == "baseline"
        assert updated.experiment_labels == ("low_rates",)
        assert run.experiments == ()
