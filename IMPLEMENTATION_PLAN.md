# Implementation Plan: Joint Derivative with DerivativeResult Container

## Overview

Implement a system where joint derivatives are computed once and stored in a container object that allows efficient slicing. Users are responsible for refreshing derivatives when arguments change.

## Design Principles

1. **Compute once, slice many:** Use `d_all()` to compute all derivatives jointly, then access individual derivatives by name/argnum
2. **No automatic staleness checking:** User responsible for calling `d_all()` at appropriate refresh points
3. **Backward compatible:** Keep `d()` for single derivatives (lazy compilation)
4. **All machinery in FunctionBundle:** Minimize Model class complexity

## Current Status Analysis

### Usage Pattern 1: Linearization (already optimized ✅)
- Location: `LinearModel.linearize()` in `src/equilibrium/model/linear.py`
- Pattern: Call `model.steady_state_derivatives()` ONCE, then access `model.derivatives` dict many times
- Status: Already uses joint derivative optimization via `compute_derivatives()`

### Usage Pattern 2: Deterministic Solver (needs optimization ❌)
- Location: `deterministic.py` lines 115-236
- Pattern: Within each time-step, calls `d()` multiple times with SAME arguments:
  ```python
  # Lines 173-179
  -np.hstack(tuple(
      mod.d("transition", var, u_lag, x_lag, z_lag, params)
      for var in ["u", "x"]
  ))

  # Lines 185-203
  np.hstack(tuple(
      mod.d("optimality", var, u_t, x_t, z_t, E, params)
      + d_opt_d_E @ mod.d("expectations", var, ...)
      for var in ["u", "x"]
  ))

  # Lines 216-232
  np.hstack(tuple(
      d_opt_d_E @ mod.d("expectations", var + "_new", ...)
      for var in ["u", "x"]
  ))
  ```
- Issue: Each `d()` call computes ALL derivatives but only returns one
- Solution: Use `d_all()` to compute once, then slice

## Implementation Steps

### Step 1: Add `DerivativeResult` class

**Location:** `src/equilibrium/model/model.py` (top of file, after imports)

```python
class DerivativeResult:
    """
    Container for joint derivative computation with convenient access.

    Allows indexing by variable name (str) or argnum (int).
    User is responsible for ensuring derivatives match current arguments.

    Attributes
    ----------
    _jac_tuple : tuple of jax.Array
        Tuple of Jacobian matrices from jax.jacfwd
    _argnum_to_idx : dict
        Maps JAX argument number to tuple index
    _var_to_argnum : dict
        Maps variable name to JAX argument number

    Examples
    --------
    >>> derivs = mod.d_all("transition", u, x, z, params)
    >>> d_u = derivs["u"]  # Access by variable name
    >>> d_x = derivs["x"]
    >>> # Or access by argnum:
    >>> d_0 = derivs[0]
    >>> # Get as hstack:
    >>> d_ux = derivs.as_hstack(["u", "x"])
    """

    def __init__(self, jac_tuple, argnums_list, var_to_argnum):
        """
        Initialize derivative result container.

        Parameters
        ----------
        jac_tuple : tuple of jax.Array
            Tuple of Jacobian matrices
        argnums_list : list of int
            List of argument numbers (e.g., [0, 1, 2, 3])
        var_to_argnum : dict
            Maps variable name to argument number
        """
        self._jac_tuple = jac_tuple
        self._argnum_to_idx = {argnum: idx for idx, argnum in enumerate(argnums_list)}
        self._var_to_argnum = var_to_argnum

    def __getitem__(self, key):
        """
        Access derivative by variable name (str) or argnum (int).

        Parameters
        ----------
        key : str or int
            Variable name (e.g., "u", "x") or JAX argument number

        Returns
        -------
        jax.Array
            Jacobian matrix for the specified variable/argument
        """
        if isinstance(key, str):
            argnum = self._var_to_argnum[key]
        else:
            argnum = key
        idx = self._argnum_to_idx[argnum]
        return self._jac_tuple[idx]

    def as_tuple(self):
        """
        Return raw tuple of Jacobians.

        Returns
        -------
        tuple of jax.Array
            Raw Jacobian tuple
        """
        return self._jac_tuple

    def as_hstack(self, vars=None):
        """
        Return as horizontally stacked array.

        Parameters
        ----------
        vars : list of str, optional
            Variable names to include. If None, includes all in order.

        Returns
        -------
        jax.Array
            Horizontally stacked Jacobian matrix

        Examples
        --------
        >>> derivs.as_hstack(["u", "x"])  # Only u and x
        >>> derivs.as_hstack()  # All variables
        """
        if vars is None:
            return jnp.hstack(self._jac_tuple)
        return jnp.hstack([self[v] for v in vars])
```

### Step 2: Add method to `FunctionBundle`

**Location:** `src/equilibrium/utils/jax_function_bundle.py`

Add this method to the `FunctionBundle` class (after `jacobian_rev_multi`):

```python
def compute_all_derivatives(self, *args):
    """
    Compute joint derivative for all argnums.

    This is a convenience wrapper around jacobian_fwd_multi() that
    returns both the jacobian tuple and the argnums list, making it
    easy to construct a DerivativeResult container.

    Parameters
    ----------
    *args : tuple
        Arguments to pass to the differentiated function

    Returns
    -------
    jac_tuple : tuple of jax.Array
        Tuple of Jacobian matrices
    argnums_list : list of int
        List of argument numbers that were differentiated

    Examples
    --------
    >>> bundle = FunctionBundle(my_func, argnums=[0, 1, 2])
    >>> jac_tuple, argnums = bundle.compute_all_derivatives(x, y, z)
    >>> # jac_tuple has 3 elements (derivatives w.r.t. args 0, 1, 2)
    """
    jac_tuple = self.jacobian_fwd_multi()(*args)
    return jac_tuple, self._argnums_list
```

### Step 3: Add `d_all()` method to `Model`

**Location:** `src/equilibrium/model/model.py`

Add after the existing `d_wrt_multi()` method (around line 991):

```python
def d_all(self, name, *args):
    """
    Compute all derivatives for function 'name', return sliceable result.

    Computes derivatives w.r.t. all registered variables once, then allows
    efficient access to individual derivatives by variable name or argnum.

    **Important:** User is responsible for refreshing when arguments change.
    This method does not cache or check for stale arguments.

    Parameters
    ----------
    name : str
        Function name ('transition', 'optimality', 'expectations', etc.)
    *args : tuple
        Arguments to pass to the function (will be standardized)

    Returns
    -------
    DerivativeResult
        Container object supporting indexing by variable name or argnum

    Examples
    --------
    Compute all derivatives once, access multiple times:

    >>> derivs = mod.d_all("transition", u, x, z, params)
    >>> d_u = derivs["u"]  # Access by variable name
    >>> d_x = derivs["x"]
    >>> # Or use in construction:
    >>> L_t = -np.hstack((derivs["u"], derivs["x"]))
    >>> # Or use helper:
    >>> L_t = -derivs.as_hstack(["u", "x"])

    Notes
    -----
    - This computes derivatives for ALL variables registered with the function
    - Use this when you need multiple derivatives with the same arguments
    - Use `d()` for single derivatives (lazy compilation)
    - The result object does NOT store arguments - user must refresh manually
    """
    std_args = standardize_args(*args)
    for i, a in enumerate(std_args):
        assert isinstance(a, jax.Array), f"Arg {i} is not jax.Array: {type(a)}"

    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    var_to_argnum = bundle_info["var_to_argnum"]

    jac_tuple, argnums_list = bundle.compute_all_derivatives(*std_args)

    return DerivativeResult(jac_tuple, argnums_list, var_to_argnum)
```

### Step 4: Revert `Model.d()` to original implementation

**Location:** `src/equilibrium/model/model.py` lines 968-980

Replace the current implementation with the original:

```python
def d(self, name, wrt, *args):
    """
    Compute derivative of function 'name' with respect to variable 'wrt'.

    Uses lazy compilation - only compiles the specific derivative requested.
    For computing multiple derivatives with same arguments, use d_all() instead.

    Parameters
    ----------
    name : str
        Function name ('transition', 'optimality', etc.)
    wrt : str
        Variable name to differentiate with respect to ('u', 'x', 'z', etc.)
    *args : tuple
        Arguments to pass to the function

    Returns
    -------
    jax.Array
        Jacobian matrix (derivative of function w.r.t. wrt variable)
    """
    std_args = standardize_args(*args)
    for i, a in enumerate(std_args):
        assert isinstance(a, jax.Array), f"Arg {i} is not jax.Array: {type(a)}"

    bundle_info = self._shared_function_bundles[name]
    bundle = bundle_info["bundle"]
    argnum = bundle_info["var_to_argnum"][wrt]

    # REVERTED: Use single derivative (lazy compilation)
    return bundle.jacobian_fwd_jit[argnum](*std_args)
```

### Step 5: Update deterministic solver

**Location:** `src/equilibrium/solvers/deterministic.py`

Update the patterns that compute multiple derivatives with same arguments.

#### Change 1: Lines 120-125 (first occurrence)

**BEFORE:**
```python
-np.hstack(
    tuple(
        # mod.jacobians['transition'][var](u_lag, x_lag, z_lag, params)
        mod.d("transition", var, u_lag, x_lag, z_lag, params)
        for var in ["u", "x"]
    )
)
```

**AFTER:**
```python
trans_derivs = mod.d_all("transition", u_lag, x_lag, z_lag, params)
-trans_derivs.as_hstack(["u", "x"])
```

#### Change 2: Lines 174-179 (second occurrence)

Same pattern - replace with:
```python
trans_derivs = mod.d_all("transition", u_lag, x_lag, z_lag, params)
-trans_derivs.as_hstack(["u", "x"])
```

#### Change 3: Lines 168, 186-202 (optimality + expectations)

**BEFORE:**
```python
d_opt_d_E = mod.d("optimality", "E", u_t, x_t, z_t, E, params)

C_t = np.vstack(
    (
        np.hstack(
            tuple(
                mod.d("optimality", var, u_t, x_t, z_t, E, params)
                + d_opt_d_E @
                mod.d(
                    "expectations",
                    var,
                    u_t,
                    x_t,
                    z_t,
                    u_next,
                    x_next,
                    z_next,
                    params,
                )
                for var in ["u", "x"]
            )
        ),
        ...
    )
)
```

**AFTER:**
```python
d_opt_d_E = mod.d("optimality", "E", u_t, x_t, z_t, E, params)
opt_derivs = mod.d_all("optimality", u_t, x_t, z_t, E, params)
exp_derivs = mod.d_all("expectations", u_t, x_t, z_t, u_next, x_next, z_next, params)

C_t = np.vstack(
    (
        np.hstack(
            tuple(
                opt_derivs[var] + d_opt_d_E @ exp_derivs[var]
                for var in ["u", "x"]
            )
        ),
        ...
    )
)
```

Or even cleaner:
```python
d_opt_d_E = mod.d("optimality", "E", u_t, x_t, z_t, E, params)
opt_derivs = mod.d_all("optimality", u_t, x_t, z_t, E, params)
exp_derivs = mod.d_all("expectations", u_t, x_t, z_t, u_next, x_next, z_next, params)

C_t = np.vstack(
    (
        np.hstack([
            opt_derivs["u"] + d_opt_d_E @ exp_derivs["u"],
            opt_derivs["x"] + d_opt_d_E @ exp_derivs["x"],
        ]),
        ...
    )
)
```

#### Change 4: Lines 217-232 (expectations _new variables)

**BEFORE:**
```python
F_t = np.vstack(
    (
        np.hstack(
            tuple(
                d_opt_d_E @
                mod.d(
                    "expectations",
                    var + "_new",
                    u_t,
                    x_t,
                    z_t,
                    u_next,
                    x_next,
                    z_next,
                    params,
                )
                for var in ["u", "x"]
            )
        ),
        np.zeros((mod.N["x"], N_ux)),
    )
)
```

**AFTER:**
```python
# Reuse exp_derivs from above if in same scope,
# or compute again if needed
exp_derivs = mod.d_all("expectations", u_t, x_t, z_t, u_next, x_next, z_next, params)

F_t = np.vstack(
    (
        np.hstack([
            d_opt_d_E @ exp_derivs["u_new"],
            d_opt_d_E @ exp_derivs["x_new"],
        ]),
        np.zeros((mod.N["x"], N_ux)),
    )
)
```

**Note:** Since `exp_derivs` is computed once above (line ~189), we can reuse it for both `var` and `var + "_new"` access, further improving efficiency.

### Step 6: Update imports

**Location:** `src/equilibrium/model/model.py`

Ensure `jax.numpy` is imported (should already be there):
```python
import jax.numpy as jnp
```

## Testing Plan

### 1. Unit tests for `DerivativeResult`

Create `tests/test_derivative_result.py`:

```python
import jax.numpy as jnp
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
```

### 2. Integration test for `Model.d_all()`

Add to existing test file (e.g., `tests/test_model.py`):

```python
def test_d_all_equivalence():
    """Test that d_all() gives same results as individual d() calls."""
    from equilibrium import Model

    # Create simple model
    model = Model()
    model.params.update({'alpha': 0.6, 'beta': 0.95})
    model.steady_guess.update({'k': 6.0, 'c': 1.0})

    model.rules['intermediate'] += [
        ('y', 'k ** alpha'),
    ]
    model.rules['transition'] += [
        ('k', 'k_new'),
    ]
    model.rules['optimality'] += [
        ('c', 'beta * c - 1.0'),
    ]

    model.finalize()
    model.solve_steady(calibrate=False)

    # Get test arguments
    u = model.steady_components['u']
    x = model.steady_components['x']
    z = model.steady_components['z']
    params = model.steady_params

    # Compare d_all() with individual d() calls
    derivs_all = model.d_all("transition", u, x, z, params)

    d_u_single = model.d("transition", "u", u, x, z, params)
    d_x_single = model.d("transition", "x", u, x, z, params)

    assert jnp.allclose(derivs_all["u"], d_u_single)
    assert jnp.allclose(derivs_all["x"], d_x_single)
```

### 3. Performance regression test

Run existing profiling script:

```bash
python profile_branches.py
```

Expected results after all changes:
- Compilations: ~115 (better than both main and current joint_derivative)
- Runtime: ~11-12s (faster than main's 13.63s)

### 4. Existing test suite

Ensure all existing tests pass:

```bash
pytest tests/
```

## Expected Performance Improvements

| Metric | main | joint_derivative (current) | joint_derivative (after fix) |
|--------|------|---------------------------|------------------------------|
| **Compilations** | 132 | 120 | ~120 (same) |
| **Runtime** | 13.63s | 18.62s | ~11-12s |
| **Status** | Baseline | ❌ Regression | ✅ Improvement |

**Why compilations stay at ~120?**

Analysis shows that of the 132 total compilations on main:
- ~14 are model function compilations (transition, optimality, etc.)
- ~18 are individual derivative compilations (main branch only)
- ~100 are JAX primitive operations (concatenate, matmul, add, etc.)

The joint_derivative branch already uses joint derivatives, saving ~12 compilations (132 → 120).

**After our fix:**
- Compilations: ~120 (no change - already using joint derivatives)
- Runtime: ~11-12s (**BIG improvement** - eliminates redundant computation)

The real benefit is **eliminating 400+ redundant derivative evaluations** in the deterministic solver, not reducing compilation count.

## Staleness Management Strategy

**User responsibility - no automatic checking:**

1. **Linearization context:**
   - Call `model.steady_state_derivatives()` once
   - Access `model.derivatives` dict many times
   - Already handled correctly

2. **Deterministic solver context:**
   - Call `d_all()` once per time-step (when args change)
   - Reuse result object multiple times within time-step
   - Arguments change each iteration, so derivatives are fresh

3. **General principle:**
   - Call `d_all()` when arguments change
   - Reuse result object while arguments stay constant
   - No magic - user must understand when to refresh

## Backward Compatibility

✅ **Fully backward compatible:**

- `d()` still works for single derivatives
- `d_wrt_multi()` unchanged
- `compute_derivatives()` unchanged
- `model.derivatives` dict unchanged
- New `d_all()` is purely additive API

## Implementation Checklist

- [ ] Step 1: Add `DerivativeResult` class to `model.py`
- [ ] Step 2: Add `compute_all_derivatives()` to `FunctionBundle`
- [ ] Step 3: Add `d_all()` to `Model`
- [ ] Step 4: Revert `d()` to original
- [ ] Step 5: Update `deterministic.py` patterns
- [ ] Step 6: Verify imports
- [ ] Test 1: Unit tests for `DerivativeResult`
- [ ] Test 2: Integration test for `d_all()`
- [ ] Test 3: Run profiling comparison
- [ ] Test 4: Run full test suite
- [ ] Documentation: Update docstrings
- [ ] Documentation: Add example to README if appropriate

## Questions to Resolve

None - design is ready for implementation.

## Notes

- Implementation should be done on the `joint_derivative` branch (not main)
- The `jacobian_fwd_multi()` infrastructure is already in place and correct
- Main work is adding the container class and updating deterministic solver
- Expected total time: 2-3 hours including testing
