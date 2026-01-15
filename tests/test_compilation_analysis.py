#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed analysis of JAX compilations at each step of model operations.
"""

import os
import sys

# Enable JAX compilation logging BEFORE importing jax
os.environ["JAX_LOG_COMPILES"] = "1"

import jax
import numpy as np

from equilibrium import Model

jax.config.update("jax_enable_x64", True)


class CompilationCounter:
    """Context manager to count JAX compilations by intercepting stderr."""

    def __init__(self):
        self.count = 0
        self.original_stderr_write = None

    def __enter__(self):
        # Intercept stderr writes to count compilation messages
        self.original_stderr_write = sys.stderr.write
        self.count = 0

        def counting_write(text):
            # Count "Compiling" messages from JAX
            if "Compiling" in text:
                self.count += 1
            return self.original_stderr_write(text)

        sys.stderr.write = counting_write
        return self

    def __exit__(self, *args):
        sys.stderr.write = self.original_stderr_write


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


def test_step_by_step_compilation():
    """Analyze compilations at each step."""

    print("\n" + "=" * 70)
    print("STEP-BY-STEP COMPILATION ANALYSIS")
    print("=" * 70)

    # Step 1: Model creation
    print("\n1. Creating model...")
    with CompilationCounter() as counter:
        mod = set_model()
    print(f"   Compilations: {counter.count}")

    # Step 2: Solve steady state with calibration
    print("\n2. Solving steady state (with calibration)...")
    with CompilationCounter() as counter:
        mod.solve_steady(calibrate=True)
    print(f"   Compilations: {counter.count}")

    # Step 3: Linearize
    print("\n3. Linearizing model...")
    with CompilationCounter() as counter:
        mod.linearize()
    print(f"   Compilations: {counter.count}")

    # Step 4: Update model with new parameters
    print("\n4. Creating model copy with updated parameters...")
    params_new = {"bet": mod.params["bet"] + 0.01}
    with CompilationCounter() as counter:
        mod_new = mod.update_copy(params=params_new)
    print(f"   Compilations: {counter.count}")

    # Step 5: Solve steady state without calibration
    print("\n5. Solving steady state (without calibration) on new model...")
    with CompilationCounter() as counter:
        mod_new.solve_steady(calibrate=False)
    print(f"   Compilations: {counter.count}")

    # Step 6: Linearize new model
    print("\n6. Linearizing new model...")
    with CompilationCounter() as counter:
        mod_new.linearize()
    print(f"   Compilations: {counter.count}")

    # Step 7: Simulate linear
    print("\n7. Simulating linear transition...")
    Nt = 20
    s_steady = mod.get_s_steady()
    s_steady_new = mod_new.get_s_steady()
    s_hat_init = s_steady - s_steady_new

    with CompilationCounter() as counter:
        _ = mod_new.simulate_linear(Nt, s_init=s_hat_init)
    print(f"   Compilations: {counter.count}")

    # Step 8: Deterministic solve
    print("\n8. Solving deterministic path...")
    from equilibrium.solvers import deterministic

    N_ux = mod_new.N["u"] + mod_new.N["x"]

    z_trans = np.zeros((Nt + 1, mod_new.N["z"])) + mod_new.steady_components["z"]
    ux_init = s_steady[:N_ux]

    with CompilationCounter() as counter:
        _UX = deterministic.solve(mod_new, z_trans, ux_init, guess_method="linear")
    print(f"   Compilations: {counter.count}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_step_by_step_compilation()
