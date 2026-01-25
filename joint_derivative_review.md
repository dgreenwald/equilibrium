# Joint Derivative Implementation Review

## Summary

The `joint_derivative` branch implements a new optimization where derivatives are computed jointly using `jax.jacfwd(f, argnums=(0,1,2))` instead of separately. This reduces compilations but introduces a performance regression.

**Results:**
- ✅ Compilations: -12 (-9.1%) - reduced from 132 to 120
- ❌ Runtime: +4.99s (+36.6%) - increased from 13.63s to 18.62s

## Code Changes

### 1. `FunctionBundle` class (`jax_function_bundle.py`)

Added two new methods:

```python
def jacobian_fwd_multi(self, argnums: Optional[Tuple[int, ...]] = None) -> Callable:
    """Get forward-mode jacobian over multiple argnums, returning tuple of Jacobians."""
    if argnums is None:
        argnums = tuple(self._argnums_list)  # Default: all argnums
    argnums = tuple(argnums)
    if argnums not in self.jacobian_fwd_multi_jit:
        self.jacobian_fwd_multi_jit[argnums] = self._jit(
            jax.jacfwd(self.f, argnums=argnums)
        )
    return self.jacobian_fwd_multi_jit[argnums]

def jacobian_rev_multi(self, argnums: Optional[Tuple[int, ...]] = None) -> Callable:
    """Similar but with jacrev."""
```

**Correctness:** ✅ These implementations are correct and properly cache compiled functions.

### 2. `Model.d()` method (`model.py:976-980`)

**OLD CODE:**
```python
def d(self, name, wrt, *std_args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]
    return bundle.jacobian_fwd_jit[argnum](*std_args)  # Single derivative
```

**NEW CODE:**
```python
def d(self, name, wrt, *std_args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]
    jac_tuple = bundle.jacobian_fwd_multi()(*std_args)  # ALL derivatives!
    return jac_tuple[argnum]  # Return only one
```

**Issues:**

1. **Performance Regression:** Computes derivatives for ALL variables but only returns one. If called multiple times with same function/args but different `wrt`, it recomputes all derivatives each time.

2. **Potential Indexing Bug:** Uses `jac_tuple[argnum]` where `argnum` is the JAX argument number (e.g., 0, 1, 2). This assumes `argnums` are contiguous `[0,1,2,...]`. If `argnums` were `[0, 2, 5]`, then `jac_tuple` has 3 elements, and `jac_tuple[2]` would actually be the derivative w.r.t. arg 5, not arg 2!

   **Current Safety:** The codebase always uses `argnums = list(range(len(var_list)))` (model.py:935), so argnums are always contiguous. But the code is fragile.

### 3. `Model.d_wrt_multi()` method (`model.py:982-990`)

**OLD CODE:**
```python
def d_wrt_multi(self, name, wrt_list, args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]
    return jnp.hstack(
        [bundle.jacobian_fwd_jit[var_to_argnum[wrt]](*args) for wrt in wrt_list]
    )
```

**NEW CODE:**
```python
def d_wrt_multi(self, name, wrt_list, args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]
    argnums = tuple(var_to_argnum[wrt] for wrt in wrt_list)
    jac_tuple = bundle.jacobian_fwd_multi(argnums)(*args)  # Only requested derivatives
    return jnp.hstack(jac_tuple)
```

**Correctness:** ✅ This is a good improvement - compiles once for the specific argnums requested.

### 4. `Model.compute_derivatives()` method (`model.py:2188-2202`)

**OLD CODE:**
```python
for key, arg_list in self.arg_lists.items():
    self.derivatives[key] = {}
    these_args = arg_dict[key]
    for var in arg_list:
        if var == "params" and not include_params:
            continue
        self.derivatives[key][var] = self.d(key, var, *these_args)
```

**NEW CODE:**
```python
for key, arg_list in self.arg_lists.items():
    self.derivatives[key] = {}
    these_args = arg_dict[key]
    bundle_info = self._shared_function_bundles[key]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]
    included_vars = [
        var for var in arg_list
        if not (var == "params" and not include_params)
    ]
    argnums = tuple(var_to_argnum[var] for var in included_vars)
    jac_tuple = bundle.jacobian_fwd_multi(argnums)(*these_args)
    for idx, var in enumerate(included_vars):
        self.derivatives[key][var] = jac_tuple[idx]
```

**Correctness:** ✅ This is excellent - computes all needed derivatives in one call, reduces compilations significantly.

## Performance Issue: Root Cause

The deterministic solver has patterns like this (`deterministic.py:120-124`):

```python
tuple(
    mod.d("transition", var, u_lag, x_lag, z_lag, params)
    for var in ["u", "x"]
)
```

With the new `d()` implementation:
- First call: `d("transition", "u", args)` → computes derivatives for [u, x, z, E, params], returns u derivative
- Second call: `d("transition", "x", args)` → computes derivatives for [u, x, z, E, params] again, returns x derivative

**Waste:** If there are 5 variables, we compute 5×2=10 derivatives but only use 2.

The old implementation would compute each derivative once (2 total).

## Recommendations

### Option 1: Revert `d()` to single-derivative (RECOMMENDED)

Keep the old behavior for `d()` and only use multi-derivative in `compute_derivatives()`:

```python
def d(self, name, wrt, *std_args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]
    return bundle.jacobian_fwd_jit[argnum](*std_args)  # OLD CODE - single derivative
```

**Pros:**
- Keeps compilation reduction in `compute_derivatives()` (the main benefit)
- Avoids performance regression in `d()`
- Simple and safe

**Cons:**
- Slightly more compilations for individual `d()` calls (but this is current behavior)

### Option 2: Update deterministic solver to use `d_wrt_multi()`

Change patterns like:
```python
tuple(mod.d("transition", var, args) for var in ["u", "x"])
```

To:
```python
mod.d_wrt_multi("transition", ["u", "x"], args)  # But returns hstack, not tuple
```

**Pros:**
- Maximum compilation reduction
- Best performance when derivatives are computed together

**Cons:**
- `d_wrt_multi()` returns hstack (single matrix) not tuple - would need API change
- More invasive code changes
- Harder to maintain

### Option 3: Add call-level caching to `d()`

Cache the result of `jacobian_fwd_multi()(*args)` within each call to `d()`:

```python
def d(self, name, wrt, *std_args):
    # Cache key: (name, args)
    # If cached, reuse; otherwise compute all and cache
```

**Pros:**
- Transparent optimization

**Cons:**
- Complex caching with JAX arrays (need hash/equality)
- Memory overhead
- Cache invalidation issues

## Fix for Indexing Bug

Regardless of which option, fix the indexing bug in `d()`:

```python
def d(self, name, wrt, *std_args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]
    jac_tuple = bundle.jacobian_fwd_multi()(*std_args)

    # FIX: Map argnum to index in tuple
    argnum_idx = bundle._argnums_list.index(argnum)
    return jac_tuple[argnum_idx]  # Correct indexing
```

Or if reverting to single-derivative (Option 1), this bug goes away entirely.

## Conclusion

The joint derivative approach is conceptually sound and works well for `compute_derivatives()`. However, the current implementation of `d()` causes a 36% performance regression by recomputing all derivatives on every call.

**Recommended fix:** Revert `d()` to use single derivatives, keep the multi-derivative optimization only in `compute_derivatives()` and `d_wrt_multi()`. This maintains the compilation reduction where it matters most while avoiding the performance penalty.
