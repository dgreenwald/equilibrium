# Performance Analysis of Equilibrium Toolbox

**Date**: 2026-01-23
**Workflow Analyzed**: `/home/dan/research/frm/danare/main.py`

## Summary

Running the FRM workflow takes **62.5 seconds** total:
- **JAX compilation**: 38.3s (61%)
- **Linearization**: 37.3s (two calls)
- **Steady-state solve**: 14.0s
- **Deterministic solve**: 17.6s

There are **137 JAX compilations** happening, which is the dominant bottleneck.

---

## Key Bottlenecks Identified

### 1. Excessive JAX Primitive Compilations

The compilation logs show many small operations being compiled separately:
```
Compiling convert_element_type...
Compiling broadcast_in_dim...
Compiling concatenate...
```

**Root cause**: The `array_to_state()` function in the generated code creates NamedTuples by indexing into arrays with individual scalar extractions:
```python
return State(
    R_new=x[0],
    ph=x[1],
    mu_ltv_exact=x[2],
    ...  # 200+ fields
)
```

Each index operation becomes a separate JAX primitive that gets traced and compiled individually.

### 2. Large State NamedTuple (~200 fields)

The baseline model has a State with ~140 intermediate variables. This causes:
- Slow `_replace()` operations for state updates
- Large trace graphs during JIT compilation
- Memory overhead from tracking unused intermediates

### 3. FunctionBundle Creates Unused Derivatives

`FunctionBundle.__post_init__()` eagerly creates:
- `grad_jit`
- `value_and_grad_jit`
- `jacobian_fwd_jit`
- `jacobian_rev_jit`
- `hessian_jit`

But only `jacobian_fwd_jit` is used in the typical workflow. Each unused derivative adds compilation overhead.

### 4. No JAX Compilation Cache Persistence

JAX supports persistent compilation caching, but it's not enabled. Every session recompiles from scratch.

### 5. Linearization Recomputes Jacobians

`steady_state_derivatives()` computes 25 Jacobians (5 function bundles × ~5 argnums each). This happens twice in the workflow (once per `linearize()` call), but bundle sharing prevents recompilation the second time.

---

## Recommendations

### High-Impact Changes

#### 1. Enable JAX Persistent Compilation Cache

Add to your environment or at model initialization:
```python
import jax
jax.config.update("jax_compilation_cache_dir", "/path/to/cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
```

This could eliminate most of the 38s compilation time on subsequent runs.

**Implementation location**: `src/equilibrium/settings.py` or model initialization

#### 2. Lazy Derivative Creation in FunctionBundle

Change from eager creation to lazy (compute on first access):
```python
@property
def hessian_jit(self):
    if self._hessian_jit is None:
        self._hessian_jit = {argnum: self._jit(jax.hessian(self.f, argnums=argnum))
                            for argnum in self._argnums_list}
    return self._hessian_jit
```

**Implementation location**: `src/equilibrium/utils/jax_function_bundle.py`

#### 3. Vectorize State Construction

Instead of scalar indexing in `array_to_state()`, use `jnp.split` or slicing:
```python
def array_to_state(x):
    slices = jnp.split(x, indices)  # Single JAX operation
    return State(*slices, *nans)
```

**Implementation location**: `src/equilibrium/templates/functions.py.jinja`

#### 4. Consider Pytree Flatten/Unflatten

Register State as a JAX pytree to enable more efficient operations:
```python
jax.tree_util.register_pytree_node(
    State,
    lambda s: (tuple(s), None),
    lambda _, children: State(*children)
)
```

**Implementation location**: Generated code in `functions.py.jinja`

### Medium-Impact Changes

#### 5. Prune Intermediate Variables

Many intermediate variables are only needed for debugging/output. Consider a "lean" mode that only computes variables needed for optimization.

**Implementation approach**: Add a `lean_mode` flag to `Model.finalize()` that excludes non-essential intermediates from the State.

#### 6. Use Reverse-Mode Jacobians Strategically

For functions with many outputs and few inputs (like `intermediate_variables` with ~140 outputs, ~70 inputs), forward-mode is expensive. Consider:
- Using `jacrev` instead of `jacfwd` where appropriate
- Chunking large Jacobian computations

**Implementation location**: `src/equilibrium/model/model.py` in the `d()` method

#### 7. Profile-Guided JIT Boundaries

The `intermediate_variables` function is called 106 times. Consider:
- Making it static during Newton iterations (compute once, reuse)
- Fusing it with the objective function

**Implementation location**: `src/equilibrium/model/model.py` in steady-state solver

### Lower-Impact but Good Practice

#### 8. Add Compilation Monitoring

Add a context manager or hook to track compilations:
```python
from equilibrium.utils import compilation_counter
with compilation_counter() as cc:
    model.solve_steady(...)
print(f"Compilations: {cc.count}")
```

**Note**: This already exists in `tests/utils.py` as `CompilationCounter`

#### 9. Warm-up Pattern Documentation

Document a warm-up workflow for users who run many similar models:
```python
# First model triggers all compilations
model1 = create_model(...)
model1.solve_steady()
model1.linearize()

# Subsequent models reuse compiled functions
model2 = model1.update_copy(params={...})  # Zero additional compilations
```

**Implementation**: Add to `CLAUDE.md` and user documentation

---

## Expected Gains

| Optimization | Estimated Savings | Complexity |
|-------------|-------------------|------------|
| JAX persistent cache | 35-38s on warm runs | Low |
| Lazy derivative creation | 5-10s on first run | Medium |
| Vectorized state construction | 3-5s | Medium |
| Intermediate pruning | 2-5s | High |

With all optimizations, a warm run could complete in **5-10 seconds** instead of 62 seconds.

---

## Profiling Commands

### Run with JAX compilation logging
```bash
JAX_LOG_COMPILES=1 python main.py 2>&1 | grep "Compiling"
```

### Count total compilations
```bash
JAX_LOG_COMPILES=1 python main.py 2>&1 | grep -c "Compiling"
```

### Profile with cProfile
```bash
python -m cProfile -s cumtime main.py 2>&1 | head -50
```

### Use existing compilation counter
```python
from tests.utils import CompilationCounter

with CompilationCounter() as counter:
    model.linearize()
print(f"Compilations: {counter.count}")
```

---

## Implementation Priority

1. **JAX persistent cache** - Lowest effort, highest impact for repeated runs
2. **Lazy derivative creation** - Medium effort, good impact for first runs
3. **Vectorized state construction** - Medium effort, helps with trace size
4. **Documentation of warm-up pattern** - Low effort, helps users

---

## Related Files

- `src/equilibrium/utils/jax_function_bundle.py` - FunctionBundle implementation
- `src/equilibrium/templates/functions.py.jinja` - Code generation template
- `src/equilibrium/model/model.py` - Main model class
- `src/equilibrium/settings.py` - Configuration management
- `tests/test_compilation_analysis.py` - Compilation tracking tests
