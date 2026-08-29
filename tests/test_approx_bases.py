"""Tests for 1D basis function implementations."""

import numpy as np
import pytest

from equilibrium.approx import (
    Basis1d,
    ChebyshevBasis1d,
    ChebyshevLobattoGrid1d,
    HierarchicalHatBasis1d,
    HierarchicalHatBasisInterior1d,
    ModifiedHierarchicalHatBasis1d,
    ModifiedUniformHatBasis1d,
    UniformHatBasis1d,
    UniformHatBasisInterior1d,
)


class TestChebyshevBasis1d:
    """Tests for ChebyshevBasis1d."""

    def test_is_basis1d(self):
        basis = ChebyshevBasis1d()
        assert isinstance(basis, Basis1d)

    def test_properties(self):
        basis = ChebyshevBasis1d()
        assert basis.basis_type == "chebyshev"

    def test_evaluate_shape(self):
        basis = ChebyshevBasis1d()
        x = np.array([0.0, 0.5, 1.0])
        result = basis.evaluate(x, 5)
        assert result.shape == (3, 5)

    def test_evaluate_T0(self):
        """T_0(x) = 1 for all x."""
        basis = ChebyshevBasis1d()
        x = np.linspace(-1, 1, 10)
        result = basis.evaluate(x, 5)
        np.testing.assert_array_almost_equal(result[:, 0], np.ones(10))

    def test_evaluate_T1(self):
        """T_1(x) = x."""
        basis = ChebyshevBasis1d()
        x = np.linspace(-1, 1, 10)
        result = basis.evaluate(x, 5)
        np.testing.assert_array_almost_equal(result[:, 1], x)

    def test_evaluate_T2(self):
        """T_2(x) = 2x^2 - 1."""
        basis = ChebyshevBasis1d()
        x = np.linspace(-1, 1, 10)
        result = basis.evaluate(x, 5)
        expected = 2 * x**2 - 1
        np.testing.assert_array_almost_equal(result[:, 2], expected)

    def test_evaluate_at_endpoints(self):
        """T_k(1) = 1 and T_k(-1) = (-1)^k."""
        basis = ChebyshevBasis1d()
        n_basis = 6

        result_1 = basis.evaluate(np.array([1.0]), n_basis)
        np.testing.assert_array_almost_equal(result_1[0], np.ones(n_basis))

        result_m1 = basis.evaluate(np.array([-1.0]), n_basis)
        expected = np.array([(-1) ** k for k in range(n_basis)])
        np.testing.assert_array_almost_equal(result_m1[0], expected)

    def test_basis_matrix_invertible(self):
        """Evaluating at grid points should give invertible matrix."""
        basis = ChebyshevBasis1d()
        grid_obj = ChebyshevLobattoGrid1d()
        n_basis = 5
        grid = grid_obj.make_grid(n_basis)
        B = basis.evaluate(grid, n_basis)

        # Should be invertible (non-zero determinant)
        det = np.linalg.det(B)
        assert abs(det) > 1e-10

    def test_invalid_n_basis(self):
        basis = ChebyshevBasis1d()
        with pytest.raises(ValueError):
            basis.evaluate(np.array([0.0]), 0)

    def test_repr(self):
        basis = ChebyshevBasis1d()
        assert repr(basis) == "ChebyshevBasis1d()"

    def test_gradients_match_finite_difference(self):
        basis = ChebyshevBasis1d()
        x = np.array([0.3])
        n_basis = 4
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)


class TestHierarchicalHatBasisInterior1d:
    """Tests for HierarchicalHatBasisInterior1d (without boundary hats)."""

    def test_is_basis1d(self):
        basis = HierarchicalHatBasisInterior1d()
        assert isinstance(basis, Basis1d)

    def test_properties(self):
        basis = HierarchicalHatBasisInterior1d()
        assert basis.basis_type == "hierarchical_hat_interior"

    def test_evaluate_shape(self):
        basis = HierarchicalHatBasisInterior1d()
        x = np.array([0.25, 0.5, 0.75])
        result = basis.evaluate(x, 3)
        assert result.shape == (3, 3)

    def test_constant_column(self):
        basis = HierarchicalHatBasisInterior1d()
        x = np.linspace(0.1, 0.9, 10)
        result = basis.evaluate(x, 3)
        np.testing.assert_array_almost_equal(result[:, 0], np.ones(10))

    def test_hat_peaks_at_center(self):
        basis = HierarchicalHatBasisInterior1d()
        n_basis = 4
        centers, _ = basis._centers_and_widths(n_basis)

        for idx, center in enumerate(centers, start=1):
            result = basis.evaluate(np.array([center]), n_basis)
            assert result[0, idx] == pytest.approx(1.0)

    def test_invalid_n_basis(self):
        basis = HierarchicalHatBasisInterior1d()
        with pytest.raises(ValueError):
            basis.evaluate(np.array([0.5]), 0)

    def test_repr(self):
        basis = HierarchicalHatBasisInterior1d()
        assert repr(basis) == "HierarchicalHatBasisInterior1d()"

    def test_gradients_match_finite_difference(self):
        basis = HierarchicalHatBasisInterior1d()
        x = np.array([0.3])
        n_basis = 5
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)


class TestHierarchicalHatBasis1d:
    """Tests for boundary-aware hierarchical hats."""

    def test_is_basis1d(self):
        basis = HierarchicalHatBasis1d()
        assert isinstance(basis, Basis1d)

    def test_boundary_hats(self):
        basis = HierarchicalHatBasis1d()
        x = np.array([0.0, 1.0])
        result = basis.evaluate(x, 3)
        assert result[0, 1] == pytest.approx(1.0)
        assert result[1, 2] == pytest.approx(1.0)

    def test_gradients_match_finite_difference(self):
        basis = HierarchicalHatBasis1d()
        x = np.array([0.3])
        n_basis = 5
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)


class TestModifiedHierarchicalHatBasis1d:
    """Tests for ModifiedHierarchicalHatBasis1d (edge extrapolation)."""

    def test_is_basis1d(self):
        basis = ModifiedHierarchicalHatBasis1d()
        assert isinstance(basis, Basis1d)

    def test_properties(self):
        basis = ModifiedHierarchicalHatBasis1d()
        assert basis.basis_type == "modified_hierarchical_hat"

    def test_edge_extrapolation(self):
        basis = ModifiedHierarchicalHatBasis1d()
        n_basis = 3
        x = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
        result = basis.evaluate(x, n_basis)

        left_hat = result[:, 1]
        right_hat = result[:, 2]

        assert left_hat[0] == pytest.approx(2.0)
        assert left_hat[1] == pytest.approx(1.5)
        assert left_hat[2] == pytest.approx(1.0)
        assert left_hat[3] == pytest.approx(0.0)

        assert right_hat[3] == pytest.approx(0.0)
        assert right_hat[4] == pytest.approx(1.0)
        assert right_hat[5] == pytest.approx(1.5)
        assert right_hat[6] == pytest.approx(2.0)

    def test_gradients_match_finite_difference(self):
        basis = ModifiedHierarchicalHatBasis1d()
        x = np.array([0.3])
        n_basis = 5
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)


class TestUniformHatBasisInterior1d:
    """Tests for UniformHatBasisInterior1d."""

    def test_is_basis1d(self):
        basis = UniformHatBasisInterior1d()
        assert isinstance(basis, Basis1d)

    def test_properties(self):
        basis = UniformHatBasisInterior1d()
        assert basis.basis_type == "uniform_hat_interior"

    def test_evaluate_shape(self):
        basis = UniformHatBasisInterior1d()
        x = np.array([0.25, 0.5, 0.75])
        result = basis.evaluate(x, 4)
        assert result.shape == (3, 4)

    def test_hat_centers_uniform(self):
        basis = UniformHatBasisInterior1d()
        n_basis = 5
        x = np.array([0.2, 0.4, 0.6, 0.8])
        result = basis.evaluate(x, n_basis)
        assert np.argmax(result[:, 1]) == 0

    def test_gradients_match_finite_difference(self):
        basis = UniformHatBasisInterior1d()
        x = np.array([0.3])
        n_basis = 6
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)


class TestUniformHatBasis1d:
    """Tests for boundary-aware uniform hats."""

    def test_is_basis1d(self):
        basis = UniformHatBasis1d()
        assert isinstance(basis, Basis1d)

    def test_nodal_hats(self):
        """Test that hats are 1 at their grid points and 0 at adjacent points.

        For n_basis points, we have:
        - Column 0: constant function (1 everywhere)
        - Columns 1 to n_basis-1: hats centered at grid points 1/(n-1), ..., 1

        At x=0: only the constant (column 0) is 1, all hats are 0
        At x=1: constant is 1, and the hat centered at 1 (last column) is also 1
        """
        basis = UniformHatBasis1d()
        n_basis = 4
        # Grid points for n_basis=4: [0, 1/3, 2/3, 1]
        x = np.array([0.0, 1.0 / 3, 2.0 / 3, 1.0])
        result = basis.evaluate(x, n_basis)

        # Constant column (column 0) is 1 everywhere
        np.testing.assert_allclose(result[:, 0], 1.0)

        # Each hat should be 1 at its grid point and 0 at others
        # Hat at 1/3 (column 1) should be 1 at x=1/3 (row 1)
        assert result[1, 1] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(0.0)  # x=0
        assert result[2, 1] == pytest.approx(0.0)  # x=2/3
        assert result[3, 1] == pytest.approx(0.0)  # x=1

        # Hat at 2/3 (column 2) should be 1 at x=2/3 (row 2)
        assert result[2, 2] == pytest.approx(1.0)

        # Hat at 1 (column 3) should be 1 at x=1 (row 3)
        assert result[3, 3] == pytest.approx(1.0)

    def test_gradients_match_finite_difference(self):
        basis = UniformHatBasis1d()
        x = np.array([0.3])
        n_basis = 6
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)


class TestModifiedUniformHatBasis1d:
    """Tests for ModifiedUniformHatBasis1d (edge extrapolation)."""

    def test_is_basis1d(self):
        basis = ModifiedUniformHatBasis1d()
        assert isinstance(basis, Basis1d)

    def test_properties(self):
        basis = ModifiedUniformHatBasis1d()
        assert basis.basis_type == "modified_uniform_hat"

    def test_edge_extrapolation(self):
        basis = ModifiedUniformHatBasis1d()
        n_basis = 4
        x = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
        result = basis.evaluate(x, n_basis)

        left_hat = result[:, 1]
        right_hat = result[:, 3]

        assert left_hat[0] == pytest.approx(2.0)
        assert left_hat[1] == pytest.approx(1.5)
        assert left_hat[2] == pytest.approx(1.0)
        assert left_hat[3] == pytest.approx(0.0)

        assert right_hat[3] == pytest.approx(0.0)
        assert right_hat[4] == pytest.approx(1.0)
        assert right_hat[5] == pytest.approx(1.5)
        assert right_hat[6] == pytest.approx(2.0)

    def test_gradients_match_finite_difference(self):
        basis = ModifiedUniformHatBasis1d()
        x = np.array([0.3])
        n_basis = 6
        _, grads = basis.evaluate_with_gradients(x, n_basis)

        eps = 1e-6
        plus = basis.evaluate(x + eps, n_basis)
        minus = basis.evaluate(x - eps, n_basis)
        fd = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd, atol=1e-6, rtol=1e-4)




class TestBasis1dInterface:
    """Tests for the Basis1d interface shared by all implementations."""

    @pytest.fixture(
        params=[
            ChebyshevBasis1d,
            HierarchicalHatBasis1d,
            HierarchicalHatBasisInterior1d,
            ModifiedHierarchicalHatBasis1d,
            ModifiedUniformHatBasis1d,
            UniformHatBasis1d,
            UniformHatBasisInterior1d,
        ]
    )
    def basis(self, request):
        return request.param()

    def test_has_required_methods(self, basis):
        """All bases should implement required methods."""
        assert hasattr(basis, "evaluate")
        assert hasattr(basis, "basis_type")

    def test_evaluate_returns_2d(self, basis):
        """evaluate should return 2D array."""
        x = np.array([0.5])  # Test point in domain
        result = basis.evaluate(x, 3)
        assert result.ndim == 2
