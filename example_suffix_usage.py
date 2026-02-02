#!/usr/bin/env python3
"""
Example demonstrating the suffix parameter in Model.add_block.
"""

from equilibrium.model import Model, ModelBlock

# Create a model
model = Model()

# Define a simple production block
production_block = ModelBlock(
    rules={
        "intermediate": [
            ("K", "np.exp(log_K)"),
            ("Y", "A * K ** alpha"),
        ],
        "transition": [("log_K", "np.log(K_NEXT)")],
    },
    params={"alpha": 0.33},
    steady_guess={"log_K": 2.0},
)

# Add the same block twice with different suffixes
model.add_block(production_block, suffix="_sector1")
model.add_block(production_block, suffix="_sector2")

print("=" * 60)
print("Example 1: Multiple sectors with suffix")
print("=" * 60)
print("\nIntermediate rules:")
for var, expr in model.rules["intermediate"].items():
    print(f"  {var:20s} = {expr}")

print("\nTransition rules:")
for var, expr in model.rules["transition"].items():
    print(f"  {var:20s} = {expr}")

print("\nParams (shared, not suffixed):")
for param, val in model.params.items():
    print(f"  {param:20s} = {val}")

print("\nSteady guess:")
for var, val in model.steady_guess.items():
    print(f"  {var:20s} = {val}")

# Example 2: Combining suffix with rename
print("\n" + "=" * 60)
print("Example 2: Combining suffix with rename")
print("=" * 60)

model2 = Model()

# Create a block with placeholders
agent_block = ModelBlock(
    rules={
        "intermediate": [
            ("c_AGENT", "income_AGENT * mpc"),
            ("util_AGENT", "np.log(c_AGENT)"),
        ]
    },
    params={"mpc": 0.8},
)

# Use both suffix and rename
# First suffix is applied: c_AGENT -> c_AGENT_household
# Then rename is applied: c_AGENT_household -> c_worker_household
model2.add_block(
    agent_block,
    suffix="_household",
    rename={"AGENT": "worker"},
)

print("\nRules after suffix + rename:")
for var, expr in model2.rules["intermediate"].items():
    print(f"  {var:25s} = {expr}")

# Example 3: Special _NEXT handling
print("\n" + "=" * 60)
print("Example 3: Special handling of _NEXT")
print("=" * 60)

model3 = Model()

savings_block = ModelBlock(
    rules={
        "intermediate": [("K", "10.0")],
        "expectations": [("E_return", "r_NEXT * K_NEXT")],
    }
)

model3.add_block(savings_block, suffix="_firm")

print("\nExpectation rules (_NEXT preserved):")
for var, expr in model3.rules["expectations"].items():
    print(f"  {var:20s} = {expr}")
print("\nNote: K_NEXT becomes K_firm_NEXT (suffix inserted before _NEXT)")
