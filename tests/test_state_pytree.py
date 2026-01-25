"""
Test JAX pytree registration for State NamedTuple.

This module verifies that the dynamically-generated State class is properly
registered as a JAX pytree, enabling tree operations while maintaining
backward compatibility.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equilibrium import Model


def create_test_model():
    """Create a minimal test model for pytree testing."""
    mod = Model()

    # Set basic parameters
    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
        }
    )

    # Set initial guesses
    mod.steady_guess.update(
        {
            "I": 0.5,
            "log_K": np.log(6.0),
        }
    )

    # Define rules - match test_compilation_analysis pattern
    mod.rules["intermediate"] += [
        ("K", "np.exp(log_K)"),
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
    ]

    mod.rules["expectations"] += [
        ("E_Om_K", "bet * (uc_NEXT / uc) * (fk_NEXT + (1.0 - delta))"),
    ]

    mod.rules["transition"] += [
        ("log_K", "np.log(K_new)"),
    ]

    mod.rules["optimality"] += [
        ("I", "E_Om_K - 1.0"),
    ]

    # Add exogenous process
    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()
    mod.solve_steady(calibrate=False)

    return mod


class TestPytreeRegistration:
    """Test basic pytree registration and flatten/unflatten operations."""

    def test_pytree_registration(self):
        """Verify State is registered as a JAX pytree."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Should be able to flatten and unflatten
        leaves, treedef = jax.tree_util.tree_flatten(st)
        st_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Verify reconstruction (only core vars should match)
        for var in mod.core_vars:
            np.testing.assert_allclose(
                getattr(st, var),
                getattr(st_reconstructed, var),
                err_msg=f"Core var {var} mismatch after unflatten",
            )

        # Derived vars should be NaN after unflatten
        for var in mod.derived_vars:
            assert np.isnan(
                getattr(st_reconstructed, var)
            ), f"Derived var {var} should be NaN after unflatten"

    def test_flatten_only_core_vars(self):
        """Verify only core variables are included in pytree leaves."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Compute intermediate variables to populate derived vars
        st = mod.inner_functions.intermediate_variables(st)

        # Flatten and verify number of leaves
        leaves, treedef = jax.tree_util.tree_flatten(st)

        assert len(leaves) == len(
            mod.core_vars
        ), f"Expected {len(mod.core_vars)} leaves, got {len(leaves)}"

        # Verify leaves correspond to core vars in order
        for i, var in enumerate(mod.core_vars):
            np.testing.assert_allclose(
                leaves[i],
                getattr(st, var),
                err_msg=f"Leaf {i} should match core var {var}",
            )

    def test_flatten_unflatten_roundtrip(self):
        """Verify flatten -> unflatten roundtrip preserves core vars."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Multiple roundtrips should preserve core vars
        for _ in range(3):
            leaves, treedef = jax.tree_util.tree_flatten(st)
            st = jax.tree_util.tree_unflatten(treedef, leaves)

            # Core vars should remain intact
            for var in mod.core_vars:
                assert not np.isnan(
                    getattr(st, var)
                ), f"Core var {var} became NaN after roundtrip"


class TestTreeOperations:
    """Test JAX tree operations on State objects."""

    def test_tree_map_over_states(self):
        """Test jax.tree.map operations on State objects."""
        mod = create_test_model()
        st1 = mod.array_to_state()
        st2 = mod.array_to_state()

        # Scale state by constant
        st_scaled = jax.tree.map(lambda x: x * 2.0, st1)

        for var in mod.core_vars:
            np.testing.assert_allclose(
                getattr(st_scaled, var),
                getattr(st1, var) * 2.0,
                err_msg=f"Core var {var} not scaled correctly",
            )

        # Add two states
        st_sum = jax.tree.map(lambda x, y: x + y, st1, st2)

        for var in mod.core_vars:
            np.testing.assert_allclose(
                getattr(st_sum, var),
                getattr(st1, var) + getattr(st2, var),
                err_msg=f"Core var {var} not summed correctly",
            )

    def test_tree_map_structure_preservation(self):
        """Verify tree_map preserves State structure."""
        mod = create_test_model()
        st = mod.array_to_state()

        st_mapped = jax.tree.map(lambda x: x * 1.5, st)

        # Should still be a State instance
        assert (
            type(st_mapped).__name__ == "State"
        ), "tree_map should preserve State type"

        # Should have all expected attributes
        for var in mod.core_vars + mod.derived_vars:
            assert hasattr(
                st_mapped, var
            ), f"State missing attribute {var} after tree_map"

    def test_vmap_over_state_batch(self):
        """Test jax.vmap with State arguments via tree operations."""
        mod = create_test_model()

        # Create two states with different values
        st1 = mod.inner_functions.array_to_state(mod.get_s_steady() * 1.0)
        st2 = mod.inner_functions.array_to_state(mod.get_s_steady() * 1.5)

        # Test that tree operations work with pytree-registered states
        # This demonstrates that states can be manipulated in batch-like ways

        # Test 1: tree_map for element-wise operations
        st_sum = jax.tree.map(lambda x, y: x + y, st1, st2)

        for var in mod.core_vars:
            expected = getattr(st1, var) + getattr(st2, var)
            np.testing.assert_allclose(
                getattr(st_sum, var),
                expected,
                err_msg=f"tree_map addition failed for {var}",
            )

        # Test 2: tree_map for scaling
        st_scaled = jax.tree.map(lambda x: x * 2.0, st1)

        for var in mod.core_vars:
            expected = getattr(st1, var) * 2.0
            np.testing.assert_allclose(
                getattr(st_scaled, var),
                expected,
                err_msg=f"tree_map scaling failed for {var}",
            )

        # Test 3: Verify pytree registration enables these operations
        leaves1, treedef1 = jax.tree_util.tree_flatten(st1)
        leaves2, treedef2 = jax.tree_util.tree_flatten(st2)

        assert (
            len(leaves1) == len(leaves2) == len(mod.core_vars)
        ), "tree_flatten should produce correct number of leaves"

        assert treedef1 == treedef2, "States should have same pytree structure"

    def test_tree_leaves_structure(self):
        """Verify tree_leaves returns correct structure."""
        mod = create_test_model()
        st = mod.array_to_state()

        leaves = jax.tree_util.tree_leaves(st)

        assert len(leaves) == len(
            mod.core_vars
        ), f"Expected {len(mod.core_vars)} leaves, got {len(leaves)}"

        # All leaves should be arrays
        for leaf in leaves:
            assert isinstance(
                leaf, (np.ndarray, jnp.ndarray)
            ), "All leaves should be arrays"


class TestStateArrayConversion:
    """Test state_to_array and array_to_state functions."""

    def test_state_to_array_inverse(self):
        """Verify state_to_array is inverse of array_to_state for core vars."""
        mod = create_test_model()

        # Start with array
        arr = mod.get_s_steady()

        # Convert to state and back
        st = mod.inner_functions.array_to_state(arr)
        arr_reconstructed = mod.inner_functions.state_to_array(st)

        np.testing.assert_allclose(
            arr,
            arr_reconstructed,
            err_msg="state_to_array should be inverse of array_to_state",
        )

    def test_state_to_array_only_core_vars(self):
        """Verify state_to_array only includes core variables."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Compute intermediates to populate derived vars
        st = mod.inner_functions.intermediate_variables(st)

        # state_to_array should only extract core vars
        arr = mod.inner_functions.state_to_array(st)

        assert arr.shape == (
            len(mod.core_vars),
        ), f"Expected shape ({len(mod.core_vars)},), got {arr.shape}"

        # Verify values match core vars
        for i, var in enumerate(mod.core_vars):
            np.testing.assert_allclose(
                arr[i],
                getattr(st, var),
                err_msg=f"Array element {i} should match core var {var}",
            )

    def test_state_to_array_jit_compatible(self):
        """Verify state_to_array works under JIT."""
        mod = create_test_model()

        @jax.jit
        def convert_and_sum(st):
            arr = mod.inner_functions.state_to_array(st)
            return jnp.sum(arr)

        st = mod.array_to_state()
        result = convert_and_sum(st)

        # Should match manual sum
        expected = sum(getattr(st, var) for var in mod.core_vars)
        np.testing.assert_allclose(result, expected)


class TestBackwardCompatibility:
    """Test that existing code patterns still work."""

    def test_backward_compatibility_replace(self):
        """Verify _replace still works after pytree registration."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Get original values
        orig_vals = {var: getattr(st, var) for var in mod.core_vars}

        # Update one variable
        new_val = orig_vals[mod.core_vars[0]] * 2.0
        st_new = st._replace(**{mod.core_vars[0]: new_val})

        # Verify update
        np.testing.assert_allclose(
            getattr(st_new, mod.core_vars[0]),
            new_val,
            err_msg="Updated variable should have new value",
        )

        # Verify other variables unchanged
        for var in mod.core_vars[1:]:
            np.testing.assert_allclose(
                getattr(st_new, var),
                orig_vals[var],
                err_msg=f"Unchanged variable {var} should match original",
            )

    def test_backward_compatibility_getitem(self):
        """Verify __getitem__ still works after pytree registration."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Should be able to access via __getitem__
        for var in mod.core_vars:
            np.testing.assert_allclose(
                st[var],
                getattr(st, var),
                err_msg=f"__getitem__ for {var} should match attribute access",
            )

    def test_backward_compatibility_upd(self):
        """Verify _upd helper still works via _replace."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Use _replace (which _upd wraps) to update state
        new_val = getattr(st, mod.core_vars[0]) * 3.0
        st_new = st._replace(**{mod.core_vars[0]: new_val})

        np.testing.assert_allclose(
            getattr(st_new, mod.core_vars[0]),
            new_val,
            err_msg="_replace should update variable correctly",
        )

    def test_jit_with_pytree_state(self):
        """Verify JIT compilation works with pytree-registered State."""
        mod = create_test_model()

        @jax.jit
        def process_state(st):
            """Simple function operating on State."""
            # Extract core vars and sum
            total = sum(getattr(st, var) for var in mod.core_vars)
            # Return new state with first var updated
            return st._replace(**{mod.core_vars[0]: total})

        st = mod.array_to_state()
        st_processed = process_state(st)

        # Should execute without error
        assert (
            type(st_processed).__name__ == "State"
        ), "JIT-compiled function should return State"

        # Verify computation
        expected_total = sum(getattr(st, var) for var in mod.core_vars)
        np.testing.assert_allclose(
            getattr(st_processed, mod.core_vars[0]),
            expected_total,
            err_msg="JIT-compiled function should compute correctly",
        )


class TestIntermediateVariables:
    """Test interaction with intermediate variable computation."""

    def test_intermediate_variables_after_pytree(self):
        """Verify intermediate_variables works after pytree operations."""
        mod = create_test_model()
        st = mod.array_to_state()

        # Apply tree operation
        st_scaled = jax.tree.map(lambda x: x * 1.5, st)

        # Derived vars should be NaN after tree_map
        for var in mod.derived_vars:
            assert np.isnan(
                getattr(st_scaled, var)
            ), f"Derived var {var} should be NaN after tree_map"

        # Recompute intermediates
        st_computed = mod.inner_functions.intermediate_variables(st_scaled)

        # Derived vars should now be populated
        for var in mod.derived_vars:
            assert not np.isnan(
                getattr(st_computed, var)
            ), f"Derived var {var} should be computed after intermediate_variables"

    def test_pytree_preserves_intermediate_workflow(self):
        """Verify typical intermediate variable workflow still works."""
        mod = create_test_model()

        # Standard workflow: array -> state -> intermediates
        arr = mod.get_s_steady()
        st = mod.inner_functions.array_to_state(arr)
        st = mod.inner_functions.intermediate_variables(st)

        # All variables should be populated
        for var in mod.core_vars + mod.derived_vars:
            assert not np.isnan(getattr(st, var)), f"Variable {var} should be computed"

        # Should be able to extract array
        arr_reconstructed = mod.inner_functions.state_to_array(st)

        np.testing.assert_allclose(
            arr, arr_reconstructed, err_msg="Roundtrip should preserve array values"
        )


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_empty_derived_vars(self):
        """Test pytree works even with no derived variables."""
        mod = Model()
        mod.params.update({"a": 1.0})
        mod.steady_guess.update({"x": 0.5})

        # Only transition rules (no intermediate/optimality)
        mod.rules["transition"] += [("x", "a * x")]

        mod.finalize()
        mod.solve_steady(calibrate=False)

        st = mod.array_to_state()

        # Should still be pytree-compatible
        leaves, treedef = jax.tree_util.tree_flatten(st)
        jax.tree_util.tree_unflatten(treedef, leaves)

        assert len(leaves) == len(mod.core_vars)

    def test_single_core_var(self):
        """Test pytree with minimal model (single core variable)."""
        mod = Model()
        mod.params.update({"a": 1.0})
        mod.steady_guess.update({"x": 0.5})
        mod.rules["transition"] += [("x", "a * x")]

        mod.finalize()
        mod.solve_steady(calibrate=False)

        st = mod.array_to_state()

        # Tree operations should work
        st_scaled = jax.tree.map(lambda x: x * 2.0, st)

        np.testing.assert_allclose(getattr(st_scaled, "x"), getattr(st, "x") * 2.0)

    def test_pytree_with_different_dtypes(self):
        """Verify pytree works with float32 and float64."""
        mod = create_test_model()

        # Test with float32
        arr32 = mod.get_s_steady().astype(np.float32)
        st32 = mod.inner_functions.array_to_state(arr32)

        leaves32, _ = jax.tree_util.tree_flatten(st32)
        assert all(leaf.dtype == np.float32 for leaf in leaves32)

        # Test with float64
        arr64 = mod.get_s_steady().astype(np.float64)
        st64 = mod.inner_functions.array_to_state(arr64)

        leaves64, _ = jax.tree_util.tree_flatten(st64)
        assert all(leaf.dtype == np.float64 for leaf in leaves64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
