# Performance Analysis: Joint Derivative Branch

## Benchmark Results

Running `/home/dan/research/frm/danare/main.py` on both branches:

| Metric | main | joint_derivative | Change |
|--------|------|------------------|--------|
| **Runtime** | 13.63s | 18.62s | **+4.99s (+36.6%)** ⚠️ |
| **Compilations** | 132 | 120 | **-12 (-9.1%)** ✅ |

## Analysis

### What Changed

The `joint_derivative` branch modifies `Model.d()` to compute derivatives for all variables at once using `jax.jacfwd(f, argnums=(0,1,2,..))` instead of individual derivatives.

**Benefit:** One compilation for all derivatives instead of N compilations
**Cost:** Computes all N derivatives even when only 1 is needed

### Why It's Slower

The deterministic solver calls `d()` in patterns like:

```python
# deterministic.py:120-124
tuple(
    mod.d("transition", var, u_lag, x_lag, z_lag, params)
    for var in ["u", "x"]
)
```

**Old behavior (main branch):**
- `d("transition", "u", args)` → compute 1 derivative for u
- `d("transition", "x", args)` → compile & compute 1 derivative for x
- Total: 2 compilations, 2 derivatives computed

**New behavior (joint_derivative branch):**
- `d("transition", "u", args)` → compute ALL derivatives [u, x, z, E, params], return u
- `d("transition", "x", args)` → compute ALL derivatives [u, x, z, E, params] again, return x
- Total: 1 compilation (reused), but 2×N derivatives computed (where N ≈ 5)

**Result:** Saves compilation time but wastes 10× more computation time.

### Where It Works Well

The `compute_derivatives()` method benefits from this approach:

```python
# OLD: Loop calling d() for each variable
for var in arg_list:
    self.derivatives[key][var] = self.d(key, var, *these_args)

# NEW: One call computing all derivatives
jac_tuple = bundle.jacobian_fwd_multi(argnums)(*these_args)
for idx, var in enumerate(included_vars):
    self.derivatives[key][var] = jac_tuple[idx]
```

This is a genuine optimization:
- **Old:** N compilations, N derivative computations
- **New:** 1 compilation, 1 multi-derivative computation

## Trade-off Analysis

### Compilation Cost vs Computation Cost

The trade-off depends on:

1. **How many times derivatives are computed?**
   - Few times: Compilation cost dominates → joint derivative wins
   - Many times: Computation cost dominates → current result

2. **Derivative computation time vs compilation time:**
   - If compilation takes 1s and derivative takes 0.01s:
     - Saving 12 compilations: saves ~12s
     - Wasting 100 derivative computations: costs ~1s
     - Net: +11s benefit
   - If compilation takes 0.1s and derivative takes 0.1s:
     - Saving 12 compilations: saves ~1.2s
     - Wasting 100 derivative computations: costs ~10s
     - Net: -8.8s cost (what we see)

The results suggest that in this workload, extra derivative computation costs more than compilation savings.

### Typical Workload Breakdown

Looking at `main.py`:
```python
mod.solve_steady(calibrate=True)  # Uses compute_derivatives() - benefits
mod.linearize()                    # Uses compute_derivatives() - benefits
mod.compute_linear_irfs()          # ?
deterministic.solve_sequence()     # Uses d() in loop - hurts performance
```

The deterministic solver appears to dominate runtime, and it uses the inefficient `d()` pattern.

## Recommendations

### Immediate Fix (Least Invasive)

Revert `Model.d()` to the old implementation:

```python
def d(self, name, wrt, *std_args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]
    return bundle.jacobian_fwd_jit[argnum](*std_args)  # OLD CODE
```

Keep the multi-derivative optimization in `compute_derivatives()` and `d_wrt_multi()`.

**Expected result:**
- Compilation: ~125 (between 120 and 132) - still better than main
- Runtime: ~13.5s (close to main) - regression eliminated

### Medium-term Fix (Better)

Add a new helper method to `d()` for computing multiple derivatives efficiently:

```python
def d_tuple(self, name, wrt_list, *std_args):
    """Compute multiple derivatives efficiently, return as tuple."""
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]
    argnums = tuple(var_to_argnum[wrt] for wrt in wrt_list)
    jac_tuple = bundle.jacobian_fwd_multi(argnums)(*std_args)
    return jac_tuple  # Return tuple, not hstack
```

Then update deterministic solver:
```python
# OLD
tuple(mod.d("transition", var, args) for var in ["u", "x"])

# NEW
mod.d_tuple("transition", ["u", "x"], *args)
```

**Expected result:**
- Compilation: ~115 (even better)
- Runtime: ~12s (faster than main)

### Long-term Fix (Best)

Profile the actual bottlenecks:
1. Is deterministic solving actually the bottleneck?
2. What's the compilation vs computation time ratio?
3. Can we cache derivative computations at call level?

## Correctness Issues

### Indexing Bug in `d()`

Current code (line 980):
```python
argnum = bundle_info["var_to_argnum"][wrt]
jac_tuple = bundle.jacobian_fwd_multi()(*std_args)
return jac_tuple[argnum]  # BUG: argnum might not equal tuple index
```

If `argnums = [0, 2, 5]`, then:
- `jac_tuple` has 3 elements
- `jac_tuple[0]` = derivative w.r.t. arg 0 ✓
- `jac_tuple[2]` = out of bounds! Should be `jac_tuple[1]` for arg 2

**Why it works now:** The codebase always uses contiguous argnums `[0,1,2,...]` so `argnum == tuple_index`.

**Fix:**
```python
argnum_idx = bundle._argnums_list.index(argnum)
return jac_tuple[argnum_idx]
```

Or revert to old code (avoids issue entirely).

## Conclusion

The joint derivative approach is sound in principle but needs selective application:

✅ **Use for:** `compute_derivatives()`, `d_wrt_multi()` - computing many derivatives at once
❌ **Don't use for:** `d()` - computing one derivative at a time

**Current status:** Applied everywhere → 9% fewer compilations but 37% slower runtime
**Recommended:** Apply selectively → best of both worlds
