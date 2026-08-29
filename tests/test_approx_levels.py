"""Tests for 1D indexing by levels."""

import numpy as np
import pytest

from equilibrium.approx import SmolyakInteriorLevels, SmolyakLevels, TensorProductLevels


class TestSmolyakLevels:
    def test_level_size(self):
        scheme = SmolyakLevels(level=4)
        assert scheme.level_size(0) == 1
        assert scheme.level_size(1) == 2
        assert scheme.level_size(2) == 2
        assert scheme.level_size(3) == 4
        assert scheme.level_size(4) == 8

    def test_basis_indices_sequential(self):
        scheme = SmolyakLevels(level=4)
        np.testing.assert_array_equal(scheme.basis_indices(0), np.array([0]))
        np.testing.assert_array_equal(scheme.basis_indices(1), np.array([1, 2]))
        np.testing.assert_array_equal(scheme.basis_indices(2), np.array([3, 4]))
        np.testing.assert_array_equal(scheme.basis_indices(3), np.array([5, 6, 7, 8]))
        np.testing.assert_array_equal(scheme.basis_indices(4), np.array([9, 10, 11, 12, 13, 14, 15, 16]))

    def test_grid_indices(self):
        scheme = SmolyakLevels(level=2)
        np.testing.assert_array_equal(scheme.grid_indices(0), np.array([2]))
        np.testing.assert_array_equal(scheme.grid_indices(1), np.array([0, 4]))
        np.testing.assert_array_equal(scheme.grid_indices(2), np.array([1, 3]))
        scheme = SmolyakLevels(level=3)
        np.testing.assert_array_equal(scheme.grid_indices(3), np.array([1, 3, 5, 7]))

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            SmolyakLevels(level=-1)
        scheme = SmolyakLevels(level=0)
        with pytest.raises(ValueError):
            scheme.grid_indices(-1)
        with pytest.raises(ValueError):
            scheme.basis_indices(-1)


class TestTensorProductLevels:
    def test_level_size_and_counts(self):
        levels = TensorProductLevels(n_points=4)
        assert levels.n_points() == 4
        for lv in range(4):
            assert levels.level_size(lv) == 1

    def test_indices_match_level(self):
        levels = TensorProductLevels(n_points=5)
        for lv in range(5):
            np.testing.assert_array_equal(levels.basis_indices(lv), np.array([lv]))
            np.testing.assert_array_equal(levels.grid_indices(lv), np.array([lv]))

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            TensorProductLevels(n_points=0)
        levels = TensorProductLevels(n_points=3)
        with pytest.raises(ValueError):
            levels.level_size(-1)
        with pytest.raises(ValueError):
            levels.grid_indices(3)

    def test_repr(self):
        levels = TensorProductLevels(n_points=2)
        assert "TensorProductLevels" in repr(levels)


class TestSmolyakInteriorLevels:
    def test_level_size(self):
        scheme = SmolyakInteriorLevels(level=3)
        assert scheme.level_size(0) == 1
        assert scheme.level_size(1) == 2
        assert scheme.level_size(2) == 4
        assert scheme.level_size(3) == 8

    def test_grid_indices(self):
        scheme = SmolyakInteriorLevels(level=1)
        np.testing.assert_array_equal(scheme.grid_indices(0), np.array([1]))
        np.testing.assert_array_equal(scheme.grid_indices(1), np.array([0, 2]))
        scheme = SmolyakInteriorLevels(level=2)
        np.testing.assert_array_equal(scheme.grid_indices(1), np.array([1, 5]))
        np.testing.assert_array_equal(scheme.grid_indices(2), np.array([0, 2, 4, 6]))

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            SmolyakInteriorLevels(level=-1)
        scheme = SmolyakInteriorLevels(level=0)
        with pytest.raises(ValueError):
            scheme.grid_indices(-1)
        with pytest.raises(ValueError):
            scheme.basis_indices(-1)
