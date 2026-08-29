"""Tests for 1D grid implementations."""

import numpy as np
import pytest

from equilibrium.approx import (
    ChebyshevLobattoGrid1d,
    Grid1d,
    UniformGrid1d,
    UniformGridWithBoundary1d,
)


class TestChebyshevLobattoGrid1d:
    """Tests for ChebyshevLobattoGrid1d."""

    def test_is_grid1d(self):
        grid = ChebyshevLobattoGrid1d()
        assert isinstance(grid, Grid1d)

    def test_properties(self):
        grid = ChebyshevLobattoGrid1d()
        assert grid.grid_type == "chebyshev_lobatto"
        assert grid.lb == -1.0
        assert grid.ub == 1.0

    def test_n_points_1(self):
        grid = ChebyshevLobattoGrid1d()
        result = grid.make_grid(1)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_n_points_2(self):
        grid = ChebyshevLobattoGrid1d()
        result = grid.make_grid(2)
        np.testing.assert_array_almost_equal(result, [-1.0, 1.0])

    def test_n_points_3(self):
        grid = ChebyshevLobattoGrid1d()
        result = grid.make_grid(3)
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])

    def test_n_points_5(self):
        grid = ChebyshevLobattoGrid1d()
        result = grid.make_grid(5)
        expected = [-1.0, -np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 1.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_bounds(self):
        """All points should be in [-1, 1]."""
        grid = ChebyshevLobattoGrid1d()
        for n in [1, 2, 3, 5, 10, 20]:
            result = grid.make_grid(n)
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)

    def test_ascending_order(self):
        """Points should be in ascending order."""
        grid = ChebyshevLobattoGrid1d()
        for n in [2, 3, 5, 10]:
            result = grid.make_grid(n)
            assert np.all(np.diff(result) > 0)

    def test_symmetry(self):
        """Grid should be symmetric around 0."""
        grid = ChebyshevLobattoGrid1d()
        for n in [3, 4, 5, 10, 11]:
            result = grid.make_grid(n)
            np.testing.assert_array_almost_equal(result, -result[::-1])

    def test_invalid_n_points(self):
        grid = ChebyshevLobattoGrid1d()
        with pytest.raises(ValueError):
            grid.make_grid(0)
        with pytest.raises(ValueError):
            grid.make_grid(-1)

    def test_repr(self):
        grid = ChebyshevLobattoGrid1d()
        assert repr(grid) == "ChebyshevLobattoGrid1d()"


class TestUniformGrid1d:
    """Tests for UniformGrid1d (interior points only)."""

    def test_is_grid1d(self):
        grid = UniformGrid1d()
        assert isinstance(grid, Grid1d)

    def test_properties(self):
        grid = UniformGrid1d()
        assert grid.grid_type == "uniform"
        assert grid.lb == 0.0
        assert grid.ub == 1.0

    def test_n_points_1(self):
        grid = UniformGrid1d()
        result = grid.make_grid(1)
        assert len(result) == 1
        assert result[0] == 0.5

    def test_n_points_3(self):
        grid = UniformGrid1d()
        result = grid.make_grid(3)
        expected = [0.25, 0.5, 0.75]
        np.testing.assert_array_almost_equal(result, expected)

    def test_n_points_5(self):
        grid = UniformGrid1d()
        result = grid.make_grid(5)
        # 5 interior points evenly spaced in (0, 1)
        delta = 1.0 / 6
        expected = [delta, 2 * delta, 3 * delta, 4 * delta, 5 * delta]
        np.testing.assert_array_almost_equal(result, expected)

    def test_strictly_interior(self):
        """All points should be in (0, 1) (strictly interior)."""
        grid = UniformGrid1d()
        for n in [1, 2, 5, 10]:
            result = grid.make_grid(n)
            assert np.all(result > 0.0)
            assert np.all(result < 1.0)

    def test_ascending_order(self):
        """Points should be in ascending order."""
        grid = UniformGrid1d()
        for n in [2, 5, 10]:
            result = grid.make_grid(n)
            assert np.all(np.diff(result) > 0)

    def test_invalid_n_points(self):
        grid = UniformGrid1d()
        with pytest.raises(ValueError):
            grid.make_grid(0)
        with pytest.raises(ValueError):
            grid.make_grid(-1)

    def test_repr(self):
        grid = UniformGrid1d()
        assert repr(grid) == "UniformGrid1d()"


class TestUniformGridWithBoundary1d:
    """Tests for UniformGridWithBoundary1d (includes boundary points)."""

    def test_is_grid1d(self):
        grid = UniformGridWithBoundary1d()
        assert isinstance(grid, Grid1d)

    def test_properties(self):
        grid = UniformGridWithBoundary1d()
        assert grid.grid_type == "uniform_with_boundary"
        assert grid.lb == 0.0
        assert grid.ub == 1.0

    def test_n_points_1(self):
        grid = UniformGridWithBoundary1d()
        result = grid.make_grid(1)
        assert len(result) == 1
        assert result[0] == 0.5

    def test_n_points_2(self):
        grid = UniformGridWithBoundary1d()
        result = grid.make_grid(2)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])

    def test_n_points_5(self):
        grid = UniformGridWithBoundary1d()
        result = grid.make_grid(5)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_includes_boundaries(self):
        """First and last points should be 0 and 1."""
        grid = UniformGridWithBoundary1d()
        for n in [2, 5, 10]:
            result = grid.make_grid(n)
            assert result[0] == 0.0
            assert result[-1] == 1.0

    def test_bounds(self):
        """All points should be in [0, 1]."""
        grid = UniformGridWithBoundary1d()
        for n in [1, 2, 5, 10]:
            result = grid.make_grid(n)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_ascending_order(self):
        """Points should be in ascending order."""
        grid = UniformGridWithBoundary1d()
        for n in [2, 5, 10]:
            result = grid.make_grid(n)
            assert np.all(np.diff(result) > 0)

    def test_invalid_n_points(self):
        grid = UniformGridWithBoundary1d()
        with pytest.raises(ValueError):
            grid.make_grid(0)
        with pytest.raises(ValueError):
            grid.make_grid(-1)

    def test_repr(self):
        grid = UniformGridWithBoundary1d()
        assert repr(grid) == "UniformGridWithBoundary1d()"


class TestUniformGridRelationship:
    """Tests verifying the relationship between uniform grid types."""

    def test_interior_is_bounded_without_endpoints(self):
        """UniformGrid1d(n) should equal UniformGridWithBoundary1d(n+2)[1:-1]."""
        interior = UniformGrid1d()
        bounded = UniformGridWithBoundary1d()

        for n in [1, 3, 5, 10]:
            interior_pts = interior.make_grid(n)
            bounded_pts = bounded.make_grid(n + 2)[1:-1]
            np.testing.assert_array_almost_equal(interior_pts, bounded_pts)
