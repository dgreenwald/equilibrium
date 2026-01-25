#!/usr/bin/env python3
"""Test the joint derivative implementation for correctness."""

import jax
import jax.numpy as jnp
from equilibrium.utils.jax_function_bundle import FunctionBundle


def test_multi_derivative_indexing():
    """Test that multi-derivative indexing works correctly."""

    def test_func(x, y, z):
        """Simple test function."""
        return jnp.sum(x**2) + jnp.sum(y**3) + jnp.sum(z**4)

    # Test with contiguous argnums [0, 1, 2]
    print("Testing with contiguous argnums [0, 1, 2]...")
    bundle = FunctionBundle(test_func, argnums=[0, 1, 2])

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    z = jnp.array([5.0, 6.0])

    # Get individual derivatives the old way
    jac_0_old = bundle.jacobian_fwd_jit[0](x, y, z)
    jac_1_old = bundle.jacobian_fwd_jit[1](x, y, z)
    jac_2_old = bundle.jacobian_fwd_jit[2](x, y, z)

    # Get all derivatives at once the new way
    jac_tuple = bundle.jacobian_fwd_multi()(x, y, z)

    # Check indexing - THIS IS THE CRITICAL TEST
    # If we use jac_tuple[argnum] where argnum is the JAX argument number,
    # it should match the old individual derivatives
    jac_0_new = jac_tuple[0]  # argnum=0
    jac_1_new = jac_tuple[1]  # argnum=1
    jac_2_new = jac_tuple[2]  # argnum=2

    print(f"Derivative w.r.t. x (arg 0) match: {jnp.allclose(jac_0_old, jac_0_new)}")
    print(f"Derivative w.r.t. y (arg 1) match: {jnp.allclose(jac_1_old, jac_1_new)}")
    print(f"Derivative w.r.t. z (arg 2) match: {jnp.allclose(jac_2_old, jac_2_new)}")

    assert jnp.allclose(jac_0_old, jac_0_new), "Derivative w.r.t. arg 0 doesn't match!"
    assert jnp.allclose(jac_1_old, jac_1_new), "Derivative w.r.t. arg 1 doesn't match!"
    assert jnp.allclose(jac_2_old, jac_2_new), "Derivative w.r.t. arg 2 doesn't match!"

    print("✓ Contiguous argnums test passed!\n")

    # Test with non-contiguous argnums [0, 2] (skipping 1)
    print("Testing with non-contiguous argnums [0, 2]...")

    def test_func2(x, static_param, z):
        """Function where middle arg is static."""
        return jnp.sum(x**2) + static_param * jnp.sum(z**4)

    bundle2 = FunctionBundle(test_func2, argnums=[0, 2], static_argnums=(1,))

    x = jnp.array([1.0, 2.0])
    static_val = 2.0
    z = jnp.array([5.0, 6.0])

    # Get individual derivatives the old way
    jac_0_old = bundle2.jacobian_fwd_jit[0](x, static_val, z)
    jac_2_old = bundle2.jacobian_fwd_jit[2](x, static_val, z)

    # Get all derivatives at once
    jac_tuple = bundle2.jacobian_fwd_multi()(x, static_val, z)

    # CRITICAL: When argnums=[0, 2], jac_tuple has 2 elements:
    # jac_tuple[0] corresponds to derivative w.r.t. arg 0
    # jac_tuple[1] corresponds to derivative w.r.t. arg 2 (NOT arg 1!)
    #
    # So if we index with argnum directly like jac_tuple[argnum],
    # we'd get jac_tuple[2] which is out of bounds!

    print(f"Length of jac_tuple: {len(jac_tuple)}")
    print(f"Expected: 2 (derivatives for args 0 and 2)")

    # Correct indexing: by position in tuple, not by argnum
    jac_0_new = jac_tuple[0]  # First element = derivative w.r.t. first argnum (0)
    jac_2_new = jac_tuple[1]  # Second element = derivative w.r.t. second argnum (2)

    print(f"Derivative w.r.t. x (arg 0) match: {jnp.allclose(jac_0_old, jac_0_new)}")
    print(f"Derivative w.r.t. z (arg 2) match: {jnp.allclose(jac_2_old, jac_2_new)}")

    assert jnp.allclose(jac_0_old, jac_0_new), "Derivative w.r.t. arg 0 doesn't match!"
    assert jnp.allclose(jac_2_old, jac_2_new), "Derivative w.r.t. arg 2 doesn't match!"

    print("✓ Non-contiguous argnums test passed!\n")

    # Test what the Model.d() method does
    print("Testing Model.d() pattern (potential bug)...")
    print("If Model.d() does jac_tuple[argnum] where argnum=2:")
    print("  With argnums=[0, 2], jac_tuple has length 2")
    print("  Indexing jac_tuple[2] would cause IndexError!")
    print("  Should instead find index of argnum in argnums list: index=1")

    # Simulate the bug
    try:
        wrong_index = jac_tuple[2]  # This is what Model.d() might do
        print("  ERROR: No IndexError raised - might work if argnums are contiguous!")
    except IndexError:
        print("  ✓ Confirmed: Direct indexing by argnum fails with non-contiguous argnums")

    print("\n" + "="*60)
    print("CONCLUSION:")
    print("The Model.d() implementation has a potential bug:")
    print("  jac_tuple[argnum] should be jac_tuple[argnums.index(argnum)]")
    print("This bug doesn't manifest if argnums are always [0,1,2,...,n]")
    print("but would fail with non-contiguous argnums like [0,2,5]")
    print("="*60)


if __name__ == "__main__":
    test_multi_derivative_indexing()
