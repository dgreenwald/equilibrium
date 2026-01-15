#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of the updated plot_model_irfs functionality.
This script shows how to use the new features.
"""

import tempfile

import jax
import numpy as np

from equilibrium import Model
from equilibrium.plot import plot_model_irfs

jax.config.update("jax_enable_x64", True)


def create_simple_model(label="simple_model", add_g_shock=False):
    """Create a simple RBC-style model for demonstration."""
    mod = Model(label=label)

    mod.params.update(
        {
            "alp": 0.6,
            "bet": 0.95,
            "delta": 0.1,
            "gam": 2.0,
            "Z_bar": 0.5,
        }
    )

    if add_g_shock:
        mod.params["G_bar"] = 0.2

    mod.steady_guess.update(
        {
            "I": 0.5,
            "log_K": np.log(6.0),
        }
    )

    if add_g_shock:
        mod.rules["intermediate"] += [
            ("K_new", "I + (1.0 - delta) * K"),
            ("Z", "Z_bar + Z_til"),
            ("G", "G_bar + G_til"),
            ("fk", "alp * Z * (K ** (alp - 1.0))"),
            ("y", "Z * (K ** alp)"),
            ("c", "y - I - G"),
            ("uc", "c ** (-gam)"),
            ("K", "np.exp(log_K)"),
        ]
    else:
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

    if add_g_shock:
        mod.add_exog("G_til", pers=0.9, vol=0.05)

    mod.finalize()
    return mod


def demonstrate_new_features():
    """Demonstrate the new plot_model_irfs features."""

    print("=" * 70)
    print("Demonstrating new plot_model_irfs functionality")
    print("=" * 70)

    # Create two models: one with Z_til only, one with both Z_til and G_til
    print("\n1. Creating models...")
    mod1 = create_simple_model("Model1_ZOnly", add_g_shock=False)
    mod1.solve_steady(calibrate=True)
    mod1.linearize()
    mod1.compute_linear_irfs(30)
    print(f"   Model 1 shocks: {mod1.exog_list}")

    mod2 = create_simple_model("Model2_ZandG", add_g_shock=True)
    mod2.solve_steady(calibrate=True)
    mod2.linearize()
    mod2.compute_linear_irfs(30)
    print(f"   Model 2 shocks: {mod2.exog_list}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Example 1: Plot all shocks (default behavior)
        print("\n2. Example 1: Default behavior - plot all shocks from all models")
        paths = plot_model_irfs(
            [mod1, mod2],
            include_list=["I", "log_K", "c"],
            plot_dir=tmpdir,
            model_names=["Baseline", "With G"],
            prefix="demo1_all_shocks",
        )
        print(f"   Created {len(paths)} plot files for all shocks")

        # Example 2: Plot specific shock using backward compatible API
        print("\n3. Example 2: Backward compatible - single shock parameter")
        paths = plot_model_irfs(
            [mod1, mod2],
            shock="Z_til",  # Old API
            include_list=["I", "log_K"],
            plot_dir=tmpdir,
            model_names=["Baseline", "With G"],
            prefix="demo2_single_shock",
        )
        print(f"   Created {len(paths)} plot files for Z_til shock")

        # Example 3: Plot specific shocks using new API
        print("\n4. Example 3: New API - multiple shocks")
        paths = plot_model_irfs(
            [mod1, mod2],
            shocks=["Z_til", "G_til"],  # New API
            include_list=["I", "log_K", "c"],
            plot_dir=tmpdir,
            model_names=["Baseline", "With G"],
            prefix="demo3_explicit_shocks",
        )
        print("   Created {} plot files for specified shocks".format(len(paths)))
        print("   Note: Model1 doesn't have G_til, so NaN values are used")

        # Example 4: Single shock with missing data
        print("\n5. Example 4: Single shock with missing data in some models")
        paths = plot_model_irfs(
            [mod1, mod2],
            shock="G_til",
            include_list=["c"],
            plot_dir=tmpdir,
            model_names=["Baseline", "With G"],
            prefix="demo4_missing_shock",
        )
        print("   Created {} plot files".format(len(paths)))
        print("   Model1 IRFs will show NaN (missing shock)")

    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  - plot_model_irfs supports both single and multiple shocks")
    print("  - Use 'shock' parameter for a single shock")
    print("  - Use 'shocks' parameter for multiple shocks")
    print("  - Defaults to all shocks from all models if neither specified")
    print("  - Gracefully handles missing shocks (NaN values)")


if __name__ == "__main__":
    demonstrate_new_features()
