# Recommended Fix for Joint Derivative Branch

## Option 1: Immediate Fix (Revert Model.d() - RECOMMENDED)

This is the simplest fix that keeps the benefits while eliminating the regression.

### Changes to `src/equilibrium/model/model.py`

**Revert Model.d() to original implementation:**

```python
def d(self, name, wrt, *std_args):
    """Compute derivative of function 'name' with respect to variable 'wrt'."""
    std_args = standardize_args(*args)
    for i, a in enumerate(std_args):
        assert isinstance(a, jax.Array), f"Arg {i} is not jax.Array: {type(a)}"

    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]

    # REVERTED: Use single derivative instead of multi
    return bundle.jacobian_fwd_jit[argnum](*std_args)
```

**Keep d_wrt_multi() as-is** (already optimized):
```python
def d_wrt_multi(self, name, wrt_list, args):
    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]
    argnums = tuple(var_to_argnum[wrt] for wrt in wrt_list)
    jac_tuple = bundle.jacobian_fwd_multi(argnums)(*args)
    return jnp.hstack(jac_tuple)
```

**Keep compute_derivatives() as-is** (this is where the real benefit is):
```python
def compute_derivatives(self, arg_dict, include_params=False):
    for key, arg_list in self.arg_lists.items():
        self.derivatives[key] = {}
        these_args = arg_dict[key]
        bundle_info = self._shared_function_bundles[key]
        bundle = bundle_info["bundle"]
        var_to_argnum = bundle_info["var_to_argnum"]
        included_vars = [
            var
            for var in arg_list
            if not (var == "params" and not include_params)
        ]
        argnums = tuple(var_to_argnum[var] for var in included_vars)
        jac_tuple = bundle.jacobian_fwd_multi(argnums)(*these_args)
        for idx, var in enumerate(included_vars):
            self.derivatives[key][var] = jac_tuple[idx]
    return None
```

### Expected Results

- **Compilations:** ~125 (still better than main's 132)
  - Saves compilations in `compute_derivatives()` and `d_wrt_multi()`
  - Individual `d()` calls compile separately (current behavior)

- **Runtime:** ~13.5s (close to main's 13.63s)
  - Eliminates wasted derivative computation in `d()`
  - Keeps efficiency gains in `compute_derivatives()`

### Testing

```bash
# Run the same profiling test
python profile_branches.py

# Expected output should show:
# - Fewer compilations than main
# - Similar or better runtime than main
```

---

## Option 2: Medium-term Fix (Add d_tuple() helper)

This provides better optimization but requires more code changes.

### New method in Model class

```python
def d_tuple(self, name, wrt_list, *std_args):
    """
    Compute multiple derivatives efficiently, return as tuple.

    Similar to d_wrt_multi() but returns tuple instead of hstack.
    Useful when derivatives need to be kept separate (e.g., for matrix construction).

    Args:
        name: Function name (e.g., 'transition', 'optimality')
        wrt_list: List of variable names to differentiate w.r.t.
        *std_args: Arguments to pass to the function

    Returns:
        Tuple of Jacobian matrices, one per variable in wrt_list

    Example:
        d_u, d_x = mod.d_tuple("transition", ["u", "x"], u, x, z, params)
    """
    std_args = standardize_args(*std_args)
    for i, a in enumerate(std_args):
        assert isinstance(a, jax.Array), f"Arg {i} is not jax.Array: {type(a)}"

    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]

    argnums = tuple(var_to_argnum[wrt] for wrt in wrt_list)
    jac_tuple = bundle.jacobian_fwd_multi(argnums)(*std_args)

    return jac_tuple
```

### Update deterministic.py to use d_tuple()

**OLD CODE:**
```python
# Line 120-124
tuple(
    mod.d("transition", var, u_lag, x_lag, z_lag, params)
    for var in ["u", "x"]
)
```

**NEW CODE:**
```python
mod.d_tuple("transition", ["u", "x"], u_lag, x_lag, z_lag, params)
```

**OLD CODE:**
```python
# Line 174-178
tuple(
    mod.d("transition", var, u_lag, x_lag, z_lag, params)
    for var in ["u", "x"]
)
```

**NEW CODE:**
```python
mod.d_tuple("transition", ["u", "x"], u_lag, x_lag, z_lag, params)
```

**OLD CODE:**
```python
# Line 186-193
tuple(
    mod.d("optimality", var, u_t, x_t, z_t, E, params)
    + d_opt_d_E @ mod.d("expectations", var, ...)
    for var in ["u", "x"]
)
```

**NEW CODE:**
```python
d_opt_u, d_opt_x = mod.d_tuple("optimality", ["u", "x"], u_t, x_t, z_t, E, params)
d_exp_u, d_exp_x = mod.d_tuple("expectations", ["u", "x"], ...)

tuple([
    d_opt_u + d_opt_d_E @ d_exp_u,
    d_opt_x + d_opt_d_E @ d_exp_x,
])
```

### Expected Results

- **Compilations:** ~115 (even better than Option 1)
- **Runtime:** ~12s (faster than main)

### Testing

```bash
# After making changes
pytest tests/test_deterministic.py -v
python /home/dan/research/frm/danare/main.py
```

---

## Option 3: Fix indexing bug (if keeping current approach)

If you want to keep the current `Model.d()` implementation but fix the indexing bug:

```python
def d(self, name, wrt, *std_args):
    std_args = standardize_args(*args)
    for i, a in enumerate(std_args):
        assert isinstance(a, jax.Array), f"Arg {i} is not jax.Array: {type(a)}"

    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]
    jac_tuple = bundle.jacobian_fwd_multi()(*std_args)

    # FIX: Convert argnum to tuple index
    argnum_idx = bundle._argnums_list.index(argnum)
    return jac_tuple[argnum_idx]
```

**Note:** This fixes correctness but doesn't fix the performance regression.

---

## Comparison

| Option | Compilations | Runtime | Code Changes | Risk |
|--------|-------------|---------|--------------|------|
| 1. Revert d() | ~125 | ~13.5s | Minimal (1 function) | Low |
| 2. Add d_tuple() | ~115 | ~12s | Medium (new method + deterministic.py) | Medium |
| 3. Fix indexing only | 120 | 18.6s | Minimal (1 line) | Low |

**Recommendation:** Start with Option 1 (immediate fix), then consider Option 2 if profiling shows it's worth the effort.
