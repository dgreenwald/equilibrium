"""Tests for the Index class."""

import numpy as np
import pytest

from equilibrium.approx import Index, SmolyakLevels, TensorProductLevels


class TestIndex:
    def _make_index(self):
        levels = (SmolyakLevels(level=3), SmolyakLevels(level=3))
        scheme = Index(
            levels=levels,
            max_total_level=3,
            max_levels=(2, 2),
        )
        return scheme

    def test_level_vectors_respect_constraints(self):
        scheme = self._make_index()
        expected = (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
        )
        assert scheme.level_vectors == expected

    def test_iter_blocks_produces_indices(self):
        scheme = self._make_index()
        blocks = list(scheme.iter_blocks())
        assert {block.levels for block in blocks} == set(scheme.level_vectors)

        for block in blocks:
            for dim, (level_value, level_obj) in enumerate(
                zip(block.levels, scheme.levels)
            ):
                expected_size = level_obj.level_size(level_value)
                assert block.basis_indices[dim].shape[0] == expected_size
                assert block.grid_indices[dim].shape[0] == expected_size
                np.testing.assert_array_equal(
                    block.basis_indices[dim], level_obj.basis_indices(level_value)
                )
                np.testing.assert_array_equal(
                    block.grid_indices[dim], level_obj.grid_indices(level_value)
                )

    def test_level_vectors_three_dimensions(self):
        levels = (
            SmolyakLevels(level=2),
            SmolyakLevels(level=2),
            SmolyakLevels(level=2),
        )
        scheme = Index(
            levels=levels,
            max_total_level=2,
            max_levels=(2, 2, 2),
        )
        expected = (
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 1, 0),
            (0, 1, 1),
            (0, 2, 0),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (2, 0, 0),
        )
        assert scheme.level_vectors == expected

    def test_invalid_configuration(self):
        level = SmolyakLevels(level=2)

        with pytest.raises(ValueError):
            Index([], max_total_level=1)

        with pytest.raises(ValueError):
            Index([level], max_total_level=-1)

        with pytest.raises(ValueError):
            Index([level], max_total_level=1, max_levels=5)

        with pytest.raises(ValueError):
            Index(
                [level],
                max_total_level=1,
                max_levels=[1, 1],
            )

        with pytest.raises(ValueError):
            Index(
                [level],
                max_total_level=1,
                max_levels=-1,
            )

    def test_size_property(self):
        """Test that size property returns correct number of sparse grid points."""
        scheme = self._make_index()
        # Manually count points from level_vectors
        total_points = 0
        for block in scheme.iter_blocks():
            n_points = 1
            for grid_idx in block.grid_indices:
                n_points *= len(grid_idx)
            total_points += n_points

        assert scheme.size == total_points

    def test_grid_ix_and_basis_ix_arrays(self):
        """Test that grid_ix and basis_ix arrays are built correctly."""
        scheme = self._make_index()

        # grid_ix and basis_ix should have size * dimension elements
        assert len(scheme.grid_ix) == scheme.size * scheme.dimension
        assert len(scheme.basis_ix) == scheme.size * scheme.dimension

        # All indices should be non-negative integers
        assert np.all(scheme.grid_ix >= 0)
        assert np.all(scheme.basis_ix >= 0)

    def test_tensor_product_levels_generate_full_grid(self):
        levels = (
            TensorProductLevels(n_points=2),
            TensorProductLevels(n_points=3),
        )
        max_levels = tuple(level.level for level in levels)
        max_total_level = sum(max_levels)

        scheme = Index(
            levels=levels,
            max_total_level=max_total_level,
            max_levels=max_levels,
        )

        expected_vectors = tuple(
            (i, j)
            for i in range(levels[0].n_points())
            for j in range(levels[1].n_points())
        )
        assert scheme.level_vectors == expected_vectors
        assert scheme.size == levels[0].n_points() * levels[1].n_points()
