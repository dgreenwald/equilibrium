#!/usr/bin/env python3
"""Test the DerivativeResult class."""

import jax.numpy as jnp
import pytest

from equilibrium.model.model import DerivativeResult


def test_derivative_result_indexing():
    """Test indexing by name and argnum."""
    jac_tuple = (
        jnp.array([[1.0, 2.0]]),
        jnp.array([[3.0, 4.0]]),
        jnp.array([[5.0, 6.0]]),
    )
    argnums_list = [0, 1, 2]
    var_to_argnum = {"u": 0, "x": 1, "z": 2}

    result = DerivativeResult(jac_tuple, argnums_list, var_to_argnum)

    # Test string indexing
    assert jnp.allclose(result["u"], jac_tuple[0])
    assert jnp.allclose(result["x"], jac_tuple[1])
    assert jnp.allclose(result["z"], jac_tuple[2])

    # Test int indexing
    assert jnp.allclose(result[0], jac_tuple[0])
    assert jnp.allclose(result[1], jac_tuple[1])
    assert jnp.allclose(result[2], jac_tuple[2])


def test_derivative_result_hstack():
    """Test hstack functionality."""
    jac_tuple = (
        jnp.array([[1.0], [2.0]]),
        jnp.array([[3.0], [4.0]]),
    )
    argnums_list = [0, 1]
    var_to_argnum = {"u": 0, "x": 1}

    result = DerivativeResult(jac_tuple, argnums_list, var_to_argnum)

    # Test full hstack
    expected_full = jnp.hstack(jac_tuple)
    assert jnp.allclose(result.as_hstack(), expected_full)

    # Test selective hstack
    expected_ux = jnp.hstack([jac_tuple[0], jac_tuple[1]])
    assert jnp.allclose(result.as_hstack(["u", "x"]), expected_ux)


def test_derivative_result_noncontiguous():
    """Test with non-contiguous argnums."""
    jac_tuple = (
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
    )
    argnums_list = [0, 2]  # Non-contiguous
    var_to_argnum = {"u": 0, "z": 2}

    result = DerivativeResult(jac_tuple, argnums_list, var_to_argnum)

    # Should correctly map argnum to tuple index
    assert jnp.allclose(result["u"], jac_tuple[0])  # argnum 0 -> idx 0
    assert jnp.allclose(result["z"], jac_tuple[1])  # argnum 2 -> idx 1
    assert jnp.allclose(result[0], jac_tuple[0])
    assert jnp.allclose(result[2], jac_tuple[1])  # This is the key test


def test_derivative_result_as_tuple():
    """Test as_tuple() method."""
    jac_tuple = (
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
    )
    argnums_list = [0, 1]
    var_to_argnum = {"u": 0, "x": 1}

    result = DerivativeResult(jac_tuple, argnums_list, var_to_argnum)

    # Test tuple return
    returned_tuple = result.as_tuple()
    assert len(returned_tuple) == 2
    assert jnp.allclose(returned_tuple[0], jac_tuple[0])
    assert jnp.allclose(returned_tuple[1], jac_tuple[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
