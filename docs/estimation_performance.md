# Estimation Module Performance Optimizations

## Summary

This document describes performance optimizations applied to the
`equilibrium.estimation` module, focusing on functions called many
times during a Random-Walk Metropolis-Hastings (RWMC) estimation run.

All changes preserve the existing public API and numerical results.
The full test suite (71 estimation tests, 597 total) continues to pass.

---

## Critical Path During MCMC

Each MCMC draw evaluates:

```
posterior(x)
├── check_bounds(x)
├── log_like(x)                          ← dominant cost
│   ├── model.update_copy(params)
│   ├── model.solve_steady()
│   ├── model.linearize()
│   └── log_likelihood(model, data)
│       ├── build_state_space(model)
│       └── log_likelihood_ssm(ssm, data)
│           └── StateSpaceEstimates.kalman_filter()   ← main bottleneck
│               └── (Nt iterations of matrix ops + log-pdf)
└── prior.logpdf(x)
```

A typical run evaluates `posterior()` tens of thousands of times
(Nsim × stride × Nblock), making the Kalman filter the dominant cost.

---

## Optimizations Applied

### 1. Kalman Filter — `state_space.py` (Highest Impact)

**Custom multivariate-normal log-pdf** (`_mvn_logpdf`):  
Replaced `scipy.stats.multivariate_normal.logpdf()` — which validates
inputs, allocates temporaries, and refactorises on every call — with a
direct Cholesky implementation:

```python
L = cholesky(F, lower=True)
z = solve_triangular(L, err, lower=True)
logpdf = -0.5 * (n * log(2π) + 2·Σlog(diag(L)) + z·z)
```

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| `mvn.logpdf` (n=5, per call) | 73 µs | 26 µs | **2.8×** |

**Fast path for fully-observed data**:  
When no observations are missing (the common case), the filter skips
per-step boolean indexing on `Z`, `H`, and the error vector, and
instead uses the full matrices directly.

**Solver upgrade for F⁻¹ computation**:  
Replaced `nm.rsolve(Z.T, F)` (general `np.linalg.solve`) with
`scipy.linalg.solve(F, Z, assume_a="pos")`, which exploits the
positive-definite structure of the innovation covariance F via
a Cholesky-based solve internally.

**Local variable caching**:  
SSM matrices (`A`, `Z`, `H`, `RQR`) and result arrays are cached as
local variables before the time loop, avoiding repeated Python attribute
lookups (`.self.ssm.A`, etc.) on each of the Nt iterations.

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Full Kalman filter (Nt=200, Nx=8, Ny=4) | 38.1 ms | 15.5 ms | **2.45×** |

Since the Kalman filter dominates the cost of each posterior evaluation,
this translates almost directly to a **~2× end-to-end speedup** for an
MCMC run with a fixed model solving/linearization cost.

### 2. Numerical Helpers — `numerical.py`

**`robust_cholesky`**: Switched from `np.linalg.eig` (general
eigendecomposition) to `scipy.linalg.eigh` (symmetric-aware), which is
numerically more stable and avoids complex arithmetic.  Also replaced
`vecs @ diag(sqrt(vals))` with the broadcasting form
`vecs * sqrt(vals)`, eliminating a temporary diagonal matrix.

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| `robust_cholesky` (n=10) | 35.5 µs | 33.4 µs | 1.1× |

**`hessian`**: Replaced `itertools.product(range(n), repeat=2)` with
nested `for` loops and pre-computed `1/(4·eps²)` outside the loop.
Removed the `itertools` import.  The algorithmic complexity is unchanged
but the per-iteration overhead is slightly lower.

**`rsolve`**: Changed from `np.linalg.solve` to `scipy.linalg.solve`,
which uses optimised LAPACK routines directly and has lower call
overhead.

### 3. MCMC Inner Loop — `mcmc.py`

**Scalar math operations**: Replaced `np.exp()` / `np.log()` with
`math.exp()` / `math.log()` in `adapt_jump_scale` and
`metropolis_step` where the arguments are Python scalars.  NumPy
array-function dispatch has significant overhead for scalar inputs.

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| `adapt_jump_scale` | 0.46 µs | 0.18 µs | **2.5×** |

**Local variable caching in `RWMC.sample`**:  
Cached `self.blocks`, `self.C_list`, `self.Nblock`, `self.stride`,
`self.draws`, `self.post_sim`, and `self.jump_scale` as local variables
before the main sampling loop.  This avoids the overhead of Python
dictionary lookups on `self.__dict__` at every step.

**Operator use**: Replaced `np.dot(A, b)` with `A @ b` throughout the
hot loop for consistency and slight call overhead reduction.

### 4. Prior — `prior.py`

**Cached non-flat indices**: Added a `_non_flat_indices` list that is
built incrementally during `add()`.  The `logpdf()` method iterates
over these cached integer indices directly instead of using a list
comprehension with a `zip` + `if dist is not None` filter on every call.
Uses direct float accumulation instead of building a list and calling
`np.sum()`.

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| `Prior.logpdf` (5 components) | 175 µs | 171 µs | 1.0× |

The speedup is marginal here because `scipy.stats` distribution
`logpdf` calls dominate the cost.

---

## Aggregate Impact

For a typical MCMC estimation run, the posterior evaluation is called
O(Nsim × stride × Nblock) times.  The Kalman filter is the dominant
cost within each posterior evaluation.

| Scenario | Estimated Speedup |
|----------|-------------------|
| Kalman filter per evaluation | **2.5×** |
| Full MCMC run (model solve/linearize is fast) | **~2×** |
| Full MCMC run (model solve/linearize dominates) | **~1.2–1.5×** |

The actual speedup for a complete estimation depends on the relative
cost of model solving/linearization vs. the Kalman filter.  For small
to medium models where the Kalman filter is the bottleneck, the
improvements are most pronounced.

---

## Files Changed

| File | Changes |
|------|---------|
| `state_space.py` | Custom `_mvn_logpdf`, fast-path Kalman filter, `sla.solve` with `assume_a`, local caching |
| `numerical.py` | `scipy.linalg.eigh`, broadcasting in `robust_cholesky`, `sla.solve` in `rsolve`, optimized `hessian` |
| `mcmc.py` | `math.exp`/`math.log` for scalars, local variable caching in `RWMC.sample` |
| `prior.py` | Cached `_non_flat_indices` for `logpdf` |
