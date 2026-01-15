#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for FunctionBundle class in jax_function_bundle.py

This test was moved from the __main__ section of jax_function_bundle.py
"""

import jax
import jax.numpy as jnp
import numpy as onp

from equilibrium.utils.jax_function_bundle import FunctionBundle


def test_function_bundle_basic():
    """Test basic FunctionBundle functionality."""

    # f(params, x, loss_type=...) -> scalar loss
    def f(params, x, *, loss_type):
        w, b = params
        y = jnp.dot(x, w) + b
        if loss_type == "mse":
            return jnp.mean((y - 1.0) ** 2)
        else:
            return jnp.mean(jnp.abs(y - 1.0))

    bundle = FunctionBundle(
        f,
        argnums=0,  # differentiate wrt params
        has_aux=False,
        static_argnames=("loss_type",),
    )

    key = jax.random.key(0)
    w = jax.random.normal(key, (4,))
    b = jnp.array(0.0)
    params = (w, b)
    x = jnp.array(onp.random.randn(32, 4))
    loss_type = "mse"

    # Single compiled call that returns (value, grad)
    value, grad = bundle.value_and_grad_jit[0](params, x, loss_type=loss_type)

    # If you need just the value
    val_only = bundle.f_jit(params, x, loss_type=loss_type)

    # Jacobian/Hessian wrt params
    J = bundle.jacobian_jit[0](params, x, loss_type=loss_type)
    H = bundle.hessian_jit[0](params, x, loss_type=loss_type)

    # Basic checks
    assert isinstance(value, (float, jnp.ndarray))
    assert jnp.allclose(value, val_only)
    assert grad is not None
    assert J is not None
    assert H is not None

    print(f"Value: {value}")
    print(f"Grad shapes: {jax.tree_util.tree_map(lambda a: a.shape, grad)}")
    print("test_function_bundle_basic passed!")


def test_function_bundle_multiple_argnums():
    """Test FunctionBundle with multiple argnums."""

    def f(x, y, z):
        return jnp.sum(x**2) + jnp.sum(y**2) + jnp.sum(z**2)

    # Create bundle with multiple argnums
    bundle = FunctionBundle(
        f,
        argnums=[0, 1, 2],
        has_aux=False,
    )

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0])
    z = jnp.array([6.0])

    # Test that all jacobians are available
    jac_x = bundle.jacobian_jit[0](x, y, z)
    jac_y = bundle.jacobian_jit[1](x, y, z)
    jac_z = bundle.jacobian_jit[2](x, y, z)

    assert jac_x.shape == (3,)
    assert jac_y.shape == (2,)
    assert jac_z.shape == (1,)

    print("test_function_bundle_multiple_argnums passed!")


def test_function_bundle_forward_and_reverse():
    """Test both forward and reverse mode differentiation."""

    def f(x):
        return jnp.sum(x**3)

    bundle = FunctionBundle(f, argnums=0, has_aux=False)

    x = jnp.array([1.0, 2.0, 3.0])

    # Both forward and reverse should give same result for this function
    jac_fwd = bundle.jacobian_fwd_jit[0](x)
    jac_rev = bundle.jacobian_rev_jit[0](x)

    assert jnp.allclose(jac_fwd, jac_rev)

    print("test_function_bundle_forward_and_reverse passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running FunctionBundle Tests")
    print("=" * 60)

    test_function_bundle_basic()
    test_function_bundle_multiple_argnums()
    test_function_bundle_forward_and_reverse()

    print("\n" + "=" * 60)
    print("All FunctionBundle tests passed!")
    print("=" * 60)
