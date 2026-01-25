# Evaluation of Remaining Optimization Opportunities

**Date**: 2026-01-24
**Context**: After joint derivative optimization implementation

## Executive Summary

With the joint derivative optimization complete (16.7% speedup achieved), the remaining optimization opportunities fall into three categories:

1. **Recommended** (3 items): Medium effort, measurable benefits
2. **Research needed** (2 items): Potential benefits unclear, needs profiling
3. **Not recommended** (1 item): Completed or not worthwhile

---

## Recommended Optimizations

### 1. Vectorized State Construction ⭐ (HIGH VALUE)

**Status**: Not implemented
**Effort**: Medium
**Expected impact**: 5-10% runtime improvement, 20-50 fewer compilations

**Current situation**:
```python
# Generated in array_to_state()
return State(
    R_new=x[0],
    ph=x[1],
    mu_ltv_exact=x[2],
    # ... 200+ individual scalar extractions
)
```

Each `x[i]` becomes a separate JAX primitive operation (`dynamic_slice`), leading to ~100-200 primitive compilations for state construction alone.

**Proposed solution**:
```python
def array_to_state(x):
    # Split array once instead of 200+ scalar extractions
    core_slices = jnp.split(x, core_split_indices)

    # Fill derived variables with NaN (will be recomputed)
    derived_nans = tuple(jnp.nan for _ in range(n_derived))

    return State(*core_slices, *derived_nans)
```

**Implementation steps**:
1. Modify `src/equilibrium/templates/functions.py.jinja`
2. Generate split indices during code generation
3. Update State construction to use `jnp.split()`
4. Verify with existing tests (should pass unchanged)

**Why this helps**:
- Single `jnp.split()` call → 1 compilation instead of 200+
- Smaller trace graphs during JIT
- Faster state construction at runtime
- No API changes, purely internal optimization

**Risks**: Low
- Template change is straightforward
- Pytree registration already handles this pattern
- Can test incrementally

**Recommendation**: ✅ **IMPLEMENT** - Good ROI, well-understood solution

---

### 2. Document Warm-up Pattern for Users ⭐ (HIGH VALUE)

**Status**: Partially documented (in CLAUDE.md only)
**Effort**: Low (1-2 hours)
**Expected impact**: User education, faster batch workflows

**Current situation**:
Users may not know that `update_copy()` shares function bundles, leading to 0 additional compilations for subsequent models.

**Pattern to document**:
```python
# First model: triggers all compilations (~125 compilations)
base_model = create_model(baseline_params)
base_model.solve_steady(calibrate=True)
base_model.linearize()

# Subsequent models: reuse compiled functions (0 new compilations!)
for param_set in parameter_variations:
    model = base_model.update_copy(params=param_set)
    model.solve_steady(calibrate=False)  # Already compiled
    model.linearize()                    # Already compiled
    # Process results...
```

**Where to document**:
1. Add section to `README.md` under "Performance Tips"
2. Add example to user guide (if exists)
3. Add to `examples/` directory with timing comparisons

**Benefits**:
- Users with batch workflows get 10-50x speedup on subsequent models
- Raises awareness of function bundle sharing
- Encourages good patterns (single base model + variations)

**Recommendation**: ✅ **IMPLEMENT** - Low effort, helps users significantly

---

### 3. Expose CompilationCounter as Public API ⭐

**Status**: Exists in `tests/utils.py`, not public
**Effort**: Low (2-3 hours)
**Expected impact**: Better debugging, user awareness

**Current situation**:
`CompilationCounter` is internal testing utility. Users have no easy way to track compilation overhead.

**Proposed**:
```python
# In src/equilibrium/utils/monitoring.py
from equilibrium.utils import CompilationCounter

with CompilationCounter() as counter:
    model.solve_steady()
    model.linearize()

print(f"Total compilations: {counter.count}")
# Output: Total compilations: 125
```

**Implementation steps**:
1. Move `CompilationCounter` from `tests/utils.py` to `src/equilibrium/utils/monitoring.py`
2. Add to public API in `src/equilibrium/utils/__init__.py`
3. Document in README and docstrings
4. Add example usage

**Benefits**:
- Users can profile their own workflows
- Helps identify compilation bottlenecks
- Educational value (understand JAX compilation)
- Debugging tool for performance issues

**Recommendation**: ✅ **IMPLEMENT** - Low effort, good developer experience

---

## Research Needed (Profile First)

### 4. Strategic Reverse-Mode Derivatives

**Status**: Not implemented
**Effort**: Medium
**Expected impact**: Unknown (needs profiling)

**Hypothesis**:
Functions with many outputs and few inputs (e.g., `intermediate_variables` with ~140 outputs, ~70 inputs) might be faster with reverse-mode derivatives (`jacrev`) instead of forward-mode (`jacfwd`).

**Theory**:
- Forward-mode: O(n_inputs) for full Jacobian
- Reverse-mode: O(n_outputs) for full Jacobian
- If n_outputs >> n_inputs: reverse-mode wins
- If n_inputs >> n_outputs: forward-mode wins

**Current situation**:
All derivatives use forward-mode (`jacfwd`) regardless of function shape.

**Why this might not help**:
1. Most model functions have similar input/output counts
2. JAX optimizes well automatically
3. Added complexity in choosing mode
4. `intermediate_variables` Jacobian may not be performance-critical

**Action needed**:
1. Profile Jacobian computation time for each function
2. Measure n_inputs vs n_outputs for each
3. Test reverse-mode on high-output functions
4. Measure actual speedup (if any)

**Recommendation**: 🔬 **RESEARCH** - Profile first, only implement if clear gains (>5% speedup)

---

### 5. Profile-Guided JIT Optimization

**Status**: Not implemented
**Effort**: High
**Expected impact**: Unknown (needs profiling)

**Hypothesis**:
`intermediate_variables()` is called many times. Could we:
- Fuse it with other functions?
- Cache it during Newton iterations?
- Make it static for steady state?

**Why this is uncertain**:
1. JAX already fuses operations well
2. Function call overhead may be negligible
3. Caching adds complexity (staleness issues)
4. May not be on critical path

**Action needed**:
1. Profile with `cProfile` to find actual bottlenecks
2. Measure function call overhead
3. Identify if `intermediate_variables` is even significant
4. Design specific optimization based on findings

**Recommendation**: 🔬 **RESEARCH** - Only pursue if profiling shows it's a bottleneck

---

## Not Recommended

### 6. Lean Mode (Prune Intermediate Variables)

**Status**: Not implemented
**Effort**: High
**Expected impact**: 10-20% potential, but high risk
**Recommendation**: ❌ **NOT RECOMMENDED** at this time

**Why not now**:

1. **Complexity**: Requires full dependency graph analysis
   - Which intermediates are essential for optimization?
   - Which are only needed for output/debugging?
   - Breaking user code that expects certain variables

2. **Maintenance burden**: Two code paths
   - Normal mode vs lean mode
   - Testing both modes
   - Documentation complexity

3. **User confusion**: When to use lean mode?
   - Power users only?
   - Breaks expectation of variable availability
   - Hard to debug when missing variables

4. **Uncertain benefits**:
   - Pytree registration already excludes derived vars from tree operations
   - State construction is being optimized (vectorization)
   - May not actually speed things up much

**Alternative approach**:
Instead of a mode flag, consider:
- Better documentation of which variables are essential
- Guide users to define only needed intermediates
- Lazy evaluation of intermediates (compute on access)

**Recommendation**: Defer until concrete use case emerges where this is clearly needed

---

## Priority Ranking

| Optimization | Effort | Impact | Risk | Priority |
|-------------|--------|--------|------|----------|
| 1. Vectorized state construction | Medium | High (5-10%) | Low | **P0** |
| 2. Document warm-up pattern | Low | Medium (user ed.) | None | **P0** |
| 3. Expose CompilationCounter | Low | Medium (dev UX) | None | **P1** |
| 4. Reverse-mode derivatives | Medium | Unknown | Low | **P2** (research first) |
| 5. Profile-guided JIT | High | Unknown | Medium | **P2** (research first) |
| 6. Lean mode | High | Medium | High | **P3** (defer) |

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. ✅ Vectorized state construction
   - Modify template
   - Test with existing suite
   - Measure improvement

2. ✅ Document warm-up pattern
   - Add to README
   - Create example script
   - Show timing comparison

3. ✅ Expose CompilationCounter
   - Move to public API
   - Document usage
   - Add examples

### Phase 2: Research (2-4 weeks)
1. Profile actual bottlenecks
   - Use cProfile on representative workflows
   - Measure Jacobian computation time by function
   - Identify top 5 time sinks

2. Evaluate reverse-mode derivatives
   - Test on functions with n_outputs >> n_inputs
   - Measure speedup (if any)
   - Implement only if >5% gain

3. Evaluate JIT optimization opportunities
   - Measure function call overhead
   - Test fusion hypotheses
   - Implement only if clear benefit

### Phase 3: Advanced (if needed)
- Consider lean mode if specific use case emerges
- Explore custom JIT boundaries based on profiling
- Investigate JAX-specific optimizations

---

## Success Metrics

**Current baseline** (joint_derivative branch):
- Runtime: 11.74s
- Compilations: 125
- Speedup vs main: 16.7%

**Target after Phase 1**:
- Runtime: ~10.5-11s (10% improvement)
- Compilations: ~100-110 (vectorized state)
- Better user experience (documentation, monitoring)

**Target after Phase 2** (if research pans out):
- Runtime: ~9-10s (20% total improvement vs main)
- Compilations: Minimal further reduction (already optimized)
- Data-driven optimizations based on profiling

---

## Conclusion

The joint derivative optimization was a major success (16.7% speedup). The remaining opportunities are:

**Clear wins** (should implement):
1. Vectorized state construction - well-understood, measurable benefit
2. Documentation improvements - low effort, helps users
3. Better tooling (CompilationCounter) - improves developer experience

**Speculative** (research first):
1. Reverse-mode derivatives - theory sounds good, needs validation
2. JIT optimization - only if profiling shows it's worthwhile

**Not worth it now**:
1. Lean mode - too complex, uncertain benefit, pytree already helps

Focus on the clear wins first, then use profiling to guide further optimization.
