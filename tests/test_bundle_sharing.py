#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if function bundles are being shared correctly between models.
"""

import os

# Enable JAX compilation logging BEFORE importing jax
os.environ["JAX_LOG_COMPILES"] = "1"

import jax
import numpy as np

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


def set_model(flags=None, params=None, steady_guess=None, **kwargs):

    mod = Model(flags=flags, params=params, steady_guess=steady_guess, **kwargs)

    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
        }
    )

    mod.steady_guess.update(
        {
            "I": 0.5,
            "log_K": np.log(6.0),
        }
    )

    mod.rules["intermediate"] += [
        ("K_new", "I + (1.0 - delta) * K"),
        ("Z", "Z_bar + Z_til"),
        ("fk", "alp * Z * (K ** (alp - 1.0))"),
        ("y", "Z * (K ** alp)"),
        ("c", "y - I"),
        ("uc", "c ** (-gam)"),
        ("K", "np.exp(log_K)"),
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

    mod.rules["calibration"] += [
        ("bet", "K - 6.0"),
    ]

    mod.add_exog("Z_til", pers=0.95, vol=0.1)

    mod.finalize()

    return mod


def test_function_bundle_sharing():
    """Check if function bundles are shared between models."""

    print("\n" + "=" * 70)
    print("FUNCTION BUNDLE SHARING TEST")
    print("=" * 70)

    # Create first model
    mod = set_model()
    mod.solve_steady(calibrate=True)
    mod.linearize()

    # Create second model with different params
    params_new = {"bet": mod.params["bet"] + 0.01}
    mod_new = mod.update_copy(params=params_new)

    # Check if function bundles are shared
    print("\nChecking if _shared_function_bundles are the same object:")
    same_bundles = mod._shared_function_bundles is mod_new._shared_function_bundles
    print(
        f"  mod._shared_function_bundles is mod_new._shared_function_bundles: {same_bundles}"
    )

    print("\nChecking if inner_functions are the same object:")
    same_inner = mod.inner_functions is mod_new.inner_functions
    print(f"  mod.inner_functions is mod_new.inner_functions: {same_inner}")

    print("\nChecking individual function bundles:")
    for key in ["expectations", "transition", "optimality", "intermediates"]:
        if (
            key in mod._shared_function_bundles
            and key in mod_new._shared_function_bundles
        ):
            bundle1 = mod._shared_function_bundles[key]["bundle"]
            bundle2 = mod_new._shared_function_bundles[key]["bundle"]
            is_same = bundle1 is bundle2
            print(f"  {key}: {is_same}")

            if is_same:
                # Check if the jitted functions are the same
                for argnum in range(min(3, len(bundle1.jacobian_fwd_jit))):
                    j1 = bundle1.jacobian_fwd_jit.get(argnum)
                    j2 = bundle2.jacobian_fwd_jit.get(argnum)
                    if j1 is not None and j2 is not None:
                        print(f"    jacobian_fwd_jit[{argnum}] same: {j1 is j2}")

    # Check steady state models
    print("\nChecking steady state models:")
    print(
        f"  mod.mod_steady is mod_new.mod_steady: {mod.mod_steady is mod_new.mod_steady}"
    )
    if hasattr(mod, "mod_steady_cal") and hasattr(mod_new, "mod_steady_cal"):
        print(
            f"  mod.mod_steady_cal is mod_new.mod_steady_cal: {mod.mod_steady_cal is mod_new.mod_steady_cal}"
        )

    # Check if steady state models share function bundles
    if hasattr(mod, "mod_steady") and hasattr(mod_new, "mod_steady"):
        print("\nChecking if steady state models share function bundles:")
        same_steady_bundles = (
            mod.mod_steady._shared_function_bundles
            is mod_new.mod_steady._shared_function_bundles
        )
        print(
            f"  mod.mod_steady._shared_function_bundles is mod_new.mod_steady._shared_function_bundles: {same_steady_bundles}"
        )

        same_steady_inner = (
            mod.mod_steady.inner_functions is mod_new.mod_steady.inner_functions
        )
        print(
            f"  mod.mod_steady.inner_functions is mod_new.mod_steady.inner_functions: {same_steady_inner}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_function_bundle_sharing()
