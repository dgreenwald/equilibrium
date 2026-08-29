"""Tests for the Scheme class."""

import numpy as np
import pytest

from equilibrium.approx import (
    ChebyshevBasis1d,
    ChebyshevLobattoGrid1d,
    Scheme,
    SmolyakLevels,
)


class TestScheme:
    def _make_scheme(self):
        """Create a simple 2D Chebyshev scheme for testing."""
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=2)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            max_levels=(2, 2),
            auto_construct=False,
        )
        return scheme

    def test_initialization(self):
        """Test that a scheme can be initialized correctly."""
        scheme = self._make_scheme()
        assert scheme.dimension == 2
        assert len(scheme.grids) == 2
        assert len(scheme.bases) == 2
        assert scheme.index is not None

    def test_properties(self):
        """Test scheme properties."""
        scheme = self._make_scheme()
        assert len(scheme.grids) == 2
        assert len(scheme.bases) == 2
        assert scheme.dimension == 2
        assert scheme.index.dimension == 2

    def test_make_grids(self):
        """Test that make_grids produces the expected structure."""
        scheme = self._make_scheme()
        grids = scheme.make_grids()

        # Should have one grid tuple per level combination
        assert len(grids) == len(scheme.index.level_vectors)

        # Each grid tuple should have one array per dimension
        for grid_tuple in grids:
            assert len(grid_tuple) == scheme.dimension
            # Each element should be a numpy array
            for grid_array in grid_tuple:
                assert isinstance(grid_array, np.ndarray)
                assert grid_array.ndim == 1
                assert len(grid_array) > 0

    def test_evaluate_bases(self):
        """Test that evaluate_bases produces the expected structure."""
        scheme = self._make_scheme()
        scheme.construct()

        # Create test points as 2D array
        x1 = np.linspace(-1, 1, 5)
        x2 = np.linspace(-1, 1, 5)
        # Create meshgrid for all combinations
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        points = np.column_stack([X1.ravel(), X2.ravel()])  # (25, 2)

        bases = scheme.evaluate_bases(points)

        # Should be a 2D matrix
        assert isinstance(bases, np.ndarray)
        assert bases.ndim == 2
        # shape: (n_eval, n_basis)
        assert bases.shape[0] == 25  # number of evaluation points
        assert bases.shape[1] == scheme.index.size  # number of basis functions

    def test_evaluate_bases_wrong_dimension(self):
        """Test that evaluate_bases raises error with wrong dimension."""
        scheme = self._make_scheme()
        scheme.construct()

        # Try with wrong shape (3 dimensions instead of 2)
        points_3d = np.random.rand(5, 3)
        with pytest.raises(ValueError, match="points must have shape"):
            scheme.evaluate_bases(points_3d)

        # Try with 1D array that's the wrong length
        points_1d = np.array([1.0, 2.0, 3.0])  # length 3, but should be 2
        with pytest.raises(ValueError, match="points must have shape"):
            scheme.evaluate_bases(points_1d)

    def test_evaluate_bases_with_gradients(self):
        """evaluate_bases_with_gradients should match finite differences."""
        scheme = self._make_scheme()
        scheme.construct()

        points = np.array([[0.1, -0.2], [0.4, 0.3]])
        bases, grads = scheme.evaluate_bases_with_gradients(points)

        assert bases.shape == (points.shape[0], scheme.index.size)
        assert grads.shape == (points.shape[0], scheme.dimension, scheme.index.size)

        eps = 1e-6
        point = points[0]
        fd_vals = []
        for dim in range(scheme.dimension):
            plus = point.copy()
            minus = point.copy()
            plus[dim] += eps
            minus[dim] -= eps
            val_plus = scheme.evaluate_bases(plus)
            val_minus = scheme.evaluate_bases(minus)
            fd = (val_plus - val_minus) / (2 * eps)
            fd_vals.append(fd)

        fd_array = np.vstack(fd_vals)
        np.testing.assert_allclose(fd_array, grads[0], atol=1e-6, rtol=1e-4)

    def test_three_dimensional_scheme(self):
        """Test a 3D scheme."""
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=2)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            max_levels=(2, 2, 2),
            auto_construct=False,
        )

        assert scheme.dimension == 3
        scheme.construct()

        grids = scheme.make_grids()
        assert len(grids) == len(scheme.index.level_vectors)

        # Create 3D test points
        x = np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])  # 2 points in 3D
        bases = scheme.evaluate_bases(x)
        assert bases.shape == (2, scheme.index.size)

    def test_mixed_basis_types(self):
        """Test scheme with same basis type across dimensions."""
        # Note: With new interface, all dimensions use the same grid and basis
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=2)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            dimension=2,
        )

        assert scheme.dimension == 2
        grids = scheme.make_grids()
        assert len(grids) > 0

    def test_dimension_from_max_levels_sequence(self):
        """Test that dimension is inferred from max_levels sequence."""
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=2)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            max_levels=(2, 2, 2, 2),  # 4 dimensions
            auto_construct=False,
        )
        assert scheme.dimension == 4

    def test_dimension_parameter_required_for_int_max_levels(self):
        """Test that dimension is required when max_levels is int."""
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=2)

        # Should work with dimension provided
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            max_levels=2,
            dimension=3,
            auto_construct=False,
        )
        assert scheme.dimension == 3

    def test_invalid_configuration(self):
        """Test that invalid configurations raise errors."""
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=2)

        # Missing dimension when max_levels is int
        with pytest.raises(ValueError, match="dimension must be provided"):
            Scheme(
                grid=grid,
                basis=basis,
                levels=levels,
                max_total_level=2,
                max_levels=2,
            )

        # Conflicting dimension and max_levels length
        with pytest.raises(ValueError, match="conflicts with max_levels length"):
            Scheme(
                grid=grid,
                basis=basis,
                levels=levels,
                max_total_level=2,
                max_levels=(2, 2),
                dimension=3,
            )

        # Invalid dimension (< 1)
        with pytest.raises(ValueError, match="dimension must be >= 1"):
            Scheme(
                grid=grid,
                basis=basis,
                levels=levels,
                max_total_level=2,
                dimension=0,
            )

    def test_repr(self):
        """Test string representation."""
        scheme = self._make_scheme()
        repr_str = repr(scheme)
        assert "Scheme" in repr_str
        assert "dimension=2" in repr_str
        assert "max_total_level=2" in repr_str

    def test_construct_and_grid(self):
        """Test construct() and grid() methods."""
        scheme = self._make_scheme()

        # grid() should fail before construct()
        with pytest.raises(RuntimeError, match="Must call construct"):
            scheme.grid()

        # Call construct
        scheme.construct()

        # Now grid() should work
        grid_points = scheme.grid()
        assert isinstance(grid_points, np.ndarray)
        assert grid_points.ndim == 2
        assert grid_points.shape == (scheme.index.size, scheme.dimension)

        # All grid points should be in [-1, 1] for Chebyshev
        assert np.all(grid_points >= -1.0)
        assert np.all(grid_points <= 1.0)

    def test_basis_inverse(self):
        """Test basis_inverse property."""
        scheme = self._make_scheme()

        # Should fail before construct()
        with pytest.raises(RuntimeError, match="Must call construct"):
            _ = scheme.basis_inverse

        # Call construct
        scheme.construct()

        # Now should work
        basis_inv = scheme.basis_inverse
        assert isinstance(basis_inv, np.ndarray)
        assert basis_inv.ndim == 2
        assert basis_inv.shape == (scheme.index.size, scheme.index.size)

        # Should be approximately the inverse of the basis matrix
        # Build basis matrix at grid
        grid_points = scheme.grid()
        basis_matrix = scheme.evaluate_bases(grid_points)

        # Check that B^{-1} @ B = I
        product = basis_inv @ basis_matrix
        np.testing.assert_allclose(product, np.eye(scheme.index.size), atol=1e-10)

    def test_evaluate_bases_single_point(self):
        """Test evaluate_bases with a single point."""
        scheme = self._make_scheme()
        scheme.construct()

        # Single point as 1D array
        point = np.array([0.5, 0.5])
        bases = scheme.evaluate_bases(point)

        # Should return 1D array
        assert isinstance(bases, np.ndarray)
        assert bases.ndim == 1
        assert bases.shape == (scheme.index.size,)
