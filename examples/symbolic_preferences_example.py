"""
Example usage of symbolic preference block generator.

This example demonstrates how to use the preference_block() function
to automatically generate marginal utilities using SymPy symbolic differentiation.
"""

from equilibrium import Model
from equilibrium.blocks import preference_block

# Example 1: Basic CRRA preferences
print("=" * 60)
print("Example 1: Basic CRRA preferences")
print("=" * 60)

block = preference_block(util_type="crra")

print("\nGenerated intermediate rules:")
for name, expr in block.rules["intermediate"].items():
    print(f"  {name} = {expr}")

# Example 2: Unit EIS (log utility)
print("\n" + "=" * 60)
print("Example 2: Unit EIS (log utility)")
print("=" * 60)

block = preference_block(util_type="unit_eis")

print("\nGenerated intermediate rules:")
for name, expr in block.rules["intermediate"].items():
    print(f"  {name} = {expr}")

# Example 3: GHH preferences with labor
print("\n" + "=" * 60)
print("Example 3: GHH preferences with labor")
print("=" * 60)

block = preference_block(util_type="ghh", labor=True)

print("\nGenerated intermediate rules:")
for name, expr in block.rules["intermediate"].items():
    print(f"  {name} = {expr}")

# Example 4: Cobb-Douglas housing aggregation
print("\n" + "=" * 60)
print("Example 4: Cobb-Douglas housing aggregation")
print("=" * 60)

block = preference_block(housing=True, housing_spec="cobb_douglas")

print("\nGenerated intermediate rules:")
for name, expr in block.rules["intermediate"].items():
    print(f"  {name} = {expr}")

# Example 5: CES housing aggregation
print("\n" + "=" * 60)
print("Example 5: CES housing aggregation")
print("=" * 60)

block = preference_block(housing=True, housing_spec="ces")

print("\nGenerated intermediate rules:")
for name, expr in block.rules["intermediate"].items():
    print(f"  {name} = {expr}")

# Example 6: Using in a model with multiple agents
print("\n" + "=" * 60)
print("Example 6: Multi-agent model")
print("=" * 60)

model = Model()

# Add borrower preferences
borrower_block = preference_block()
model.add_block(borrower_block, rename={"AGENT": "borrower", "ATYPE": "b"})

# Add lender preferences
lender_block = preference_block()
model.add_block(lender_block, rename={"AGENT": "lender", "ATYPE": "l"})

print("\nBorrower marginal utility:")
print(f"  uc_borrower = {model.rules['intermediate']['uc_borrower']}")

print("\nLender marginal utility:")
print(f"  uc_lender = {model.rules['intermediate']['uc_lender']}")

# Example 7: Complete specification with housing and labor
print("\n" + "=" * 60)
print("Example 7: Complete specification with housing and labor")
print("=" * 60)

block = preference_block(
    agent="household",
    atype="hh",
    housing=True,
    labor=True,
    housing_spec="cobb_douglas",
    util_type="crra",
    nominal=True,
)

print("\nGenerated intermediate rules:")
for name, expr in block.rules["intermediate"].items():
    print(f"  {name} = {expr}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
