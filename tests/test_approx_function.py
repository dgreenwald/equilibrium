"""Tests for the Function class."""

import numpy as np
import pytest

from equilibrium.approx import (
    ChebyshevBasis1d,
    ChebyshevLobattoGrid1d,
    Function,
    HierarchicalHatBasis1d,
    HierarchicalHatBasisInterior1d,
    Scheme,
    SmolyakLevels,
    TensorProductLevels,
    UniformGrid1d,
    UniformGridWithBoundary1d,
    UniformHatBasisInterior1d,
)


class TestFunction:
    def _make_scheme_2d(self):
        """Create a simple 2D scheme for testing."""
        grid = UniformGridWithBoundary1d()
        basis = HierarchicalHatBasis1d()
        levels = SmolyakLevels(level=3)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=3,
            max_levels=(3, 3),
            auto_construct=False,
        )
        scheme.construct()
        return scheme

    def test_initialization(self):
        """Test that a function can be initialized correctly."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        assert func.scheme is scheme
        assert np.array_equal(func.lb, lb)
        assert np.array_equal(func.ub, ub)
        assert func.coefficients is None
        assert func.get_n_points() == scheme.index.size

    def test_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        scheme = self._make_scheme_2d()

        # Wrong dimension
        with pytest.raises(ValueError, match="lb must have shape"):
            Function(scheme, np.array([0.0]), np.array([1.0, 1.0]))

        # lb >= ub
        with pytest.raises(ValueError, match="lb must be < ub"):
            Function(scheme, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    def test_scheme_not_constructed(self):
        """Test that error is raised if scheme not constructed."""
        grid = UniformGridWithBoundary1d()
        basis = HierarchicalHatBasis1d()
        levels = SmolyakLevels(level=2)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            max_levels=(2, 2),
            auto_construct=False,
        )
        # Don't call construct()

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        with pytest.raises(RuntimeError, match="scheme.construct()"):
            Function(scheme, lb, ub, auto_construct=False)

    def test_scheme_auto_constructs_when_requested(self):
        grid = UniformGridWithBoundary1d()
        basis = HierarchicalHatBasis1d()
        levels = SmolyakLevels(level=2)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=2,
            max_levels=(2, 2),
            auto_construct=False,
        )
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)
        assert func.get_n_points() == scheme.index.size

    def test_get_grid_points(self):
        """Test getting grid points in user coordinates."""
        scheme = self._make_scheme_2d()
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        func = Function(scheme, lb, ub)

        grid_points = func.get_grid_points()
        assert grid_points.shape == (scheme.index.size, 2)

        # All points should be within user bounds
        assert np.all(grid_points[:, 0] >= lb[0])
        assert np.all(grid_points[:, 0] <= ub[0])
        assert np.all(grid_points[:, 1] >= lb[1])
        assert np.all(grid_points[:, 1] <= ub[1])

    def test_fit_scalar_function(self):
        """Test fitting a scalar function."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Simple test function: f(x,y) = x + y
        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]

        func.fit(values)
        assert func.coefficients is not None
        assert func.coefficients.ndim == 1
        assert func.coefficients.shape == (scheme.index.size,)

    def test_fit_vector_function(self):
        """Test fitting a vector-valued function."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Vector function: [x+y, x*y]
        grid_points = func.get_grid_points()
        values = np.column_stack([
            grid_points[:, 0] + grid_points[:, 1],
            grid_points[:, 0] * grid_points[:, 1],
        ])

        func.fit(values)
        assert func.coefficients is not None
        assert func.coefficients.ndim == 2
        assert func.coefficients.shape == (scheme.index.size, 2)

    def test_fit_wrong_size(self):
        """Test that fitting with wrong-sized values raises error."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Wrong number of points
        wrong_values = np.ones(5)
        with pytest.raises(ValueError, match="values must have length"):
            func.fit(wrong_values)

    def test_evaluate_before_fit(self):
        """Test that evaluation before fit raises error."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        with pytest.raises(RuntimeError, match="Must call fit"):
            func.evaluate(np.array([0.5, 0.5]))

    def test_evaluate_single_point_scalar(self):
        """Test evaluating scalar function at a single point."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Fit to f(x,y) = x + y
        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]
        func.fit(values)

        # Evaluate at a single point
        point = np.array([0.5, 0.3])
        result = func.evaluate(point)

        # Should be a scalar
        assert isinstance(result, (float, np.floating))

        # Check accuracy (piecewise linear should be exact)
        expected = 0.5 + 0.3
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_evaluate_multiple_points_scalar(self):
        """Test evaluating scalar function at multiple points."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Fit to f(x,y) = x + y
        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]
        func.fit(values)

        # Evaluate at multiple points
        points = np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.1]])
        result = func.evaluate(points)

        assert result.shape == (3,)
        expected = points[:, 0] + points[:, 1]
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_evaluate_vector_function(self):
        """Test evaluating vector-valued function."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Fit to [x+y, x*y]
        grid_points = func.get_grid_points()
        values = np.column_stack([
            grid_points[:, 0] + grid_points[:, 1],
            grid_points[:, 0] * grid_points[:, 1],
        ])
        func.fit(values)

        # Evaluate at single point
        point = np.array([0.5, 0.5])
        result = func.evaluate(point)

        assert result.shape == (2,)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-10)  # 0.5 + 0.5
        np.testing.assert_allclose(result[1], 0.25, atol=1e-10)  # 0.5 * 0.5

        # Evaluate at multiple points
        points = np.array([[0.5, 0.5], [0.2, 0.8]])
        result = func.evaluate(points)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 1], 0.25, atol=1e-10)
        np.testing.assert_allclose(result[1, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 1], 0.16, atol=1e-10)

    def test_evaluate_with_basis(self):
        """Test evaluate_with_basis method."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Fit
        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]
        func.fit(values)

        # Evaluate with basis
        point = np.array([0.5, 0.3])
        result, basis = func.evaluate_with_basis(point)

        # Check result
        expected = 0.5 + 0.3
        np.testing.assert_allclose(result, expected, atol=1e-10)

        # Check basis
        assert basis.shape == (scheme.index.size,)

        # Verify result = basis @ coefficients
        np.testing.assert_allclose(result, np.dot(basis, func.coefficients), atol=1e-10)

    def test_evaluate_with_gradient_scalar(self):
        """Gradients should match analytic derivatives for scalar functions."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        grid_points = func.get_grid_points()
        values = 2.0 * grid_points[:, 0] + 3.0 * grid_points[:, 1]
        func.fit(values)

        points = np.array([[0.2, 0.4], [0.6, 0.1]])
        vals, grads = func.evaluate_with_gradient(points)

        expected_values = 2.0 * points[:, 0] + 3.0 * points[:, 1]
        expected_grads = np.tile(np.array([2.0, 3.0]), (points.shape[0], 1))

        np.testing.assert_allclose(vals, expected_values, atol=1e-10)
        np.testing.assert_allclose(grads, expected_grads, atol=1e-10)

    def test_evaluate_with_gradient_vector_function(self):
        """Vector-valued gradients should have shape (n_eval, dim, n_outputs)."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        grid_points = func.get_grid_points()
        values = np.column_stack(
            [
                grid_points[:, 0] + grid_points[:, 1],
                grid_points[:, 0] * grid_points[:, 1],
            ]
        )
        func.fit(values)

        points = np.array([[0.3, 0.7], [0.2, 0.4]])
        vals, grads, basis, basis_grad = func.evaluate_with_gradient_and_basis(points)

        expected_vals = np.column_stack(
            [
                points[:, 0] + points[:, 1],
                points[:, 0] * points[:, 1],
            ]
        )
        expected_grads = np.empty((points.shape[0], 2, 2))
        expected_grads[:, 0, 0] = 1.0  # d/dx of x+y
        expected_grads[:, 1, 0] = 1.0  # d/dy of x+y
        expected_grads[:, 0, 1] = points[:, 1]  # d/dx of xy
        expected_grads[:, 1, 1] = points[:, 0]  # d/dy of xy

        assert basis.shape == (points.shape[0], scheme.index.size)
        assert basis_grad.shape == (points.shape[0], scheme.dimension, scheme.index.size)

        np.testing.assert_allclose(vals, expected_vals, atol=1e-10)
        np.testing.assert_allclose(grads, expected_grads, atol=1e-10)

    def test_single_point_gradient_shapes(self):
        """Single point gradient evaluation should return flattened arrays."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]
        func.fit(values)

        point = np.array([0.5, 0.5])
        val, grad, basis, basis_grad = func.evaluate_with_gradient_and_basis(point)

        assert isinstance(val, float)
        assert grad.shape == (scheme.dimension,)
        assert basis.shape == (scheme.index.size,)
        assert basis_grad.shape == (scheme.dimension, scheme.index.size)

    def test_tensor_uniform_vs_hierarchical_hat_basis(self):
        """Uniform and hierarchical hats should agree on tensor grids."""
        n_dim = 2
        n_points_1d = 7  # hierarchical basis prefers 2^L - 1 points
        grid = UniformGrid1d()
        levels = TensorProductLevels(n_points=n_points_1d)
        max_level = levels.level
        max_levels = tuple(max_level for _ in range(n_dim))
        max_total_level = int(sum(max_levels))

        scheme_hier = Scheme(
            grid=grid,
            basis=HierarchicalHatBasisInterior1d(),
            levels=levels,
            max_total_level=max_total_level,
            max_levels=max_levels,
            auto_construct=False,
        )
        scheme_hier.construct()

        scheme_uniform = Scheme(
            grid=grid,
            basis=UniformHatBasisInterior1d(),
            levels=levels,
            max_total_level=max_total_level,
            max_levels=max_levels,
            auto_construct=False,
        )
        scheme_uniform.construct()

        lb = np.zeros(n_dim)
        ub = np.ones(n_dim)
        func_hier = Function(scheme_hier, lb, ub)
        func_uniform = Function(scheme_uniform, lb, ub)

        grid_points = func_hier.get_grid_points()
        assert np.allclose(grid_points, func_uniform.get_grid_points())
        values = np.sin(np.pi * grid_points[:, 0]) + grid_points[:, 0] * grid_points[:, 1]
        func_hier.fit(values)
        func_uniform.fit(values)

        vals_hier = func_hier.evaluate(grid_points)
        vals_uniform = func_uniform.evaluate(grid_points)

        np.testing.assert_allclose(vals_hier, values, atol=1e-10)
        np.testing.assert_allclose(vals_uniform, values, atol=1e-10)
        np.testing.assert_allclose(vals_hier, vals_uniform, atol=1e-10)

    def test_coordinate_transformation(self):
        """Test that coordinate transformation works correctly."""
        scheme = self._make_scheme_2d()
        # Use non-trivial bounds
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        func = Function(scheme, lb, ub)

        # Fit to f(x,y) = x + y (linear function)
        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]
        func.fit(values)

        # Evaluate at points in user coordinates
        points = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        result = func.evaluate(points)

        expected = points[:, 0] + points[:, 1]
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_chebyshev_polynomial_approximation(self):
        """Test approximating a polynomial with Chebyshev basis."""
        grid = ChebyshevLobattoGrid1d()
        basis = ChebyshevBasis1d()
        levels = SmolyakLevels(level=4)
        scheme = Scheme(
            grid=grid,
            basis=basis,
            levels=levels,
            max_total_level=4,
            max_levels=(4, 4),
            auto_construct=False,
        )
        scheme.construct()

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        # Test polynomial: f(x,y) = x^2 + xy + y^2
        grid_points = func.get_grid_points()
        values = (
            grid_points[:, 0] ** 2
            + grid_points[:, 0] * grid_points[:, 1]
            + grid_points[:, 1] ** 2
        )
        func.fit(values)

        # Evaluate at test points
        test_points = np.array([
            [0.5, 0.5],
            [-0.5, 0.3],
            [0.8, -0.2],
        ])
        result = func.evaluate(test_points)
        expected = (
            test_points[:, 0] ** 2
            + test_points[:, 0] * test_points[:, 1]
            + test_points[:, 1] ** 2
        )

        # Chebyshev should approximate polynomials very accurately
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_repr(self):
        """Test string representation."""
        scheme = self._make_scheme_2d()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        func = Function(scheme, lb, ub)

        repr_str = repr(func)
        assert "Function" in repr_str
        assert "dimension=2" in repr_str
        assert "fitted=False" in repr_str

        # After fitting
        grid_points = func.get_grid_points()
        values = grid_points[:, 0] + grid_points[:, 1]
        func.fit(values)

        repr_str = repr(func)
        assert "fitted=True" in repr_str
