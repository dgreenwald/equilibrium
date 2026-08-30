# Nonlinear Global Solutions in Equilibrium

## Background

The original `econ-project` package (`~/numerical/econ-project`) solves nonlinear DSGE models via **collocation**: it approximates policy functions over the state space using the sparse-grid function approximation tools in `sparse_approx`, then iterates until the Euler-equation residuals vanish at every collocation point. The C++ implementation in `colloc/code/` supports multiple solution algorithms (Newton on the stacked coefficient system, time iteration grid-point-by-grid-point, fixed-point iteration) and relies on `sparse_approx` for Smolyak sparse grids with Chebyshev or hat bases.

`funcapprox` (`~/dev/funcapprox`) is a modern, pure-Python rewrite of `sparse_approx`. It provides the same core functionality — sparse and tensor-product grids, Chebyshev polynomial and hierarchical hat bases, Smolyak-level indexing, collocation fitting, and function evaluation — but in a clean, typed, NumPy-based package (~1,500 lines of core code). It has no C++ dependency.

This document proposes bringing global nonlinear solution methods into the `equilibrium` package by:

1. Integrating (or depending on) `funcapprox` for function approximation.
2. Implementing collocation-based projection solvers in `equilibrium.solvers`.

---

## Dependency vs. Internalization

### Option A: Add `funcapprox` as a dependency

| Pros | Cons |
|------|------|
| Single source of truth — bug fixes flow both ways | Adds an external dependency on an unreleased package |
| Lighter `equilibrium` tree | Requires installing from git or a local path (`pip install -e ~/dev/funcapprox`) |
| `funcapprox` can be used independently for non-econ approximation tasks | Version coordination needed across two repos |

### Option B: Vendor / internalize funcapprox code (recommended)

| Pros | Cons |
|------|------|
| Self-contained: no install friction, no version drift | Code divergence if funcapprox evolves separately |
| Can adapt to JAX arrays (`jax.numpy`) in-place | Slightly larger equilibrium tree |
| Natural home: function approximation is integral to the solver | Must manually port upstream fixes |

### Recommendation

**Internalize the core modules** as a new subpackage `equilibrium.approx`. The reasons:

1. **JAX compatibility.** The solver needs `jax.numpy`-based evaluation for auto-differentiation of Euler residuals with respect to coefficients. `funcapprox` currently uses pure NumPy. Internalizing lets us swap `numpy` → `jax.numpy` in the hot paths (basis evaluation, coordinate transforms) without forking the upstream package. Grid construction and index logic can remain plain NumPy since they run once at setup time.

2. **Tight coupling.** The collocation solver will call `evaluate_bases()` inside `jax.jit`-compiled residual functions. This requires the basis code to be traceable by JAX, which is easiest to guarantee when it lives in-tree.

3. **Simplicity.** Since `funcapprox` is unreleased and has the same author, there is no maintenance burden from a third-party API. The core module is small (~1,500 lines excluding benchmarks/tests), so duplication cost is low.

4. **Optional re-export.** The `equilibrium.approx` subpackage can be designed to be usable standalone for non-solver approximation tasks, preserving the `funcapprox` API for anyone who wants it.

If `funcapprox` is eventually published to PyPI, this decision can be revisited — the internalized code could be replaced by a thin adapter layer over the external package.

---

## Architecture Overview

```
src/equilibrium/
├── approx/                    ← NEW: internalized from funcapprox
│   ├── __init__.py            # Re-export public API
│   ├── jax_eval.py            # Stateless Chebyshev JAX evaluation
│   ├── py.typed               # Typed-package marker
│   ├── UPSTREAM.md            # Source provenance and port history
│   ├── bases/
│   │   ├── base.py            # Basis1d ABC
│   │   ├── chebyshev.py       # ChebyshevBasis1d
│   │   └── hat.py             # Hat basis family
│   ├── grids/
│   │   ├── base.py            # Grid1d ABC
│   │   ├── chebyshev.py       # ChebyshevLobattoGrid1d
│   │   └── uniform.py         # UniformGrid1d, UniformGridWithBoundary1d
│   ├── levels/
│   │   ├── base.py            # Levels ABC
│   │   ├── smolyak.py         # Smolyak level families
│   │   └── tensor.py          # TensorProductLevels
│   ├── core/
│   │   ├── index.py           # Index, IndexBlock
│   │   ├── scheme.py          # Scheme (sparse grid construction)
│   │   └── function.py        # Function (fit / evaluate wrapper)
│   └── presets.py             # make_smolyak_chebyshev, etc.
│
├── solvers/
│   ├── collocation.py         ← NEW: projection solver
│   ├── colloc_spec.py         ← NEW: solver configuration
│   └── ...                    # existing modules unchanged
│
└── ...
```

The `benchmark` subpackage from `funcapprox` (test functions, plotting utilities, diagnostics) would **not** be ported — it is a development/research aid, not runtime code.

### JAX adaptation strategy

The completed Phase 1 port follows a **split setup/evaluation** pattern:

- **Setup-time code** (grid construction, index building, `Scheme.construct()`, basis-inverse precomputation) stays on NumPy. These run once and produce static arrays.
- **Compatibility evaluation** (`Basis1d.evaluate()`, `Scheme.evaluate_bases()`, and `Function.evaluate()`) remains NumPy-based.
- **Solver evaluation** uses the stateless `make_jax_data()`, `evaluate_bases_jax()`, and `evaluate_jax()` API. Coefficients are explicit traced arguments, while immutable scheme data is carried by a JAX PyTree.

The JAX path currently supports Chebyshev schemes. Hat bases remain available through the NumPy API and are rejected clearly by the JAX adapter. See [the Phase 1 implementation plan](nonlinear-phase-1.md) and [function approximation guide](function-approximation.md).

---

## Collocation Solver Design

### Conceptual algorithm

Given an equilibrium model with:
- **States** $s_t = (x_t, z_t)$: endogenous states $x$ and exogenous AR(1) processes $z$
- **Controls** $u_t$: policy (decision) variables
- **Expectations** $E_t$: forward-looking terms evaluated via numerical quadrature

The collocation method approximates the policy function $u(s)$ as:

$$u(s) \approx \sum_{j=1}^{N} c_j \, \phi_j(s)$$

where $\{\phi_j\}$ are basis functions on a sparse grid over the state space, and $\{c_j\}$ are coefficients to be determined.

The system of residual equations is:

$$R_i = \text{optimality}\bigl(u(s_i),\, s_i,\, E_i\bigr) = 0, \qquad i = 1, \ldots, N$$

where at each collocation point $s_i$:

1. Evaluate $u_i = u(s_i)$ from the current approximation.
2. Compute the state transition: $x_{i}' = \text{transition}(u_i, s_i)$.
3. For each quadrature node $\epsilon_k$ with weight $w_k$:
   - Compute next-period exogenous state: $z'_k = \Phi z_i + \sigma \epsilon_k$.
   - Form next-period state: $s'_k = (x_{i}', z'_k)$.
   - Evaluate next-period policy: $u'_k = u(s'_k)$.
   - Accumulate expectations: $E_i \mathrel{+}= w_k \cdot g(u_i, s_i, u'_k, s'_k)$.
4. Evaluate the optimality residual: $R_i = f(u_i, s_i, E_i)$.

These $N \times n_u$ residual equations are solved for the $N \times n_u$ coefficients $\{c_j\}$.

### Solution algorithms

Following the original `colloc` design, three algorithms would be supported:

#### 1. Time iteration (default, most robust)

At each collocation point, solve for $u_i$ given the current policy approximation for $u'$. This is a small $(n_u \times n_u)$ root-finding problem per grid point, iterated until the policy function converges. This maps directly to `colloc/code/time_iter.cpp`.

```
for each outer iteration:
    for each collocation point s_i:
        solve R_i(u_i; u_old) = 0  for u_i    # inner Newton, n_u × n_u
    fit new coefficients c from {u_i} values
    check convergence: ||c_new - c_old|| < tol
```

**Advantages:** Very robust; inner problems are small; naturally parallelizable (each grid point is independent).

**JAX integration:** The inner Newton loop per grid point can be `vmap`-ed across all collocation points. The model's compiled `transition`, `expectations`, and `optimality` functions (available as `FunctionBundle` instances on the `Model`) provide both residuals and Jacobians via JAX autodiff.

#### 2. Newton on stacked system (fastest near solution)

Solve the full $N \cdot n_u$ stacked residual system simultaneously for all coefficients. The Jacobian with respect to coefficients $c$ is assembled analytically:

$$\frac{\partial R_i}{\partial c_j} = \frac{\partial f}{\partial u}\,\phi_j(s_i) + \frac{\partial f}{\partial E}\sum_k w_k \Bigl[\frac{\partial g}{\partial u'}\,\phi_j(s'_k) + \ldots\Bigr]$$

This is the stacked analog of the Klein system, but nonlinear. Maps to `colloc/code/newton_search.cpp`.

**Advantages:** Quadratic convergence when close to the solution.

**Disadvantages:** Requires assembling and factoring a large $(N \cdot n_u) \times (N \cdot n_u)$ Jacobian. Can be fragile far from the solution.

#### 3. Hybrid (recommended default workflow)

Use time iteration to get close, then switch to Newton for final refinement. This mirrors the `use_final_algo` pattern in the original `colloc`.

### Quadrature for expectations

The existing `Model.exog_list` identifies the exogenous AR(1) processes. The collocation solver needs Gauss-Hermite quadrature nodes and weights to integrate over next-period shock innovations $\epsilon \sim N(0, I)$. Shock scaling must have a single authoritative source (for example, an innovation impact matrix); the quadrature routine must not apply volatility a second time.

The one-dimensional rule should follow the established `ghquad_norm()` convention in `~/numerical/py_tools/numerical/core.py`:

```python
x, w = np.polynomial.hermite.hermgauss(n_quad)
w = w / np.sum(w)
x = mu + np.sqrt(2.0) * sig * x
```

`numpy.polynomial.hermite.hermgauss()` integrates against $e^{-x^2}$. Multiplying its nodes by $\sqrt{2}\,\sigma$, shifting by $\mu$, and normalizing its weights to sum to one yields nodes and probability weights for $N(\mu, \sigma^2)$. Thus the standard-normal rule uses `mu=0.0` and `sig=1.0`, and directly approximates

$$
\mathbb{E}[h(\epsilon)] \approx \sum_k w_k h(x_k),
\qquad \sum_k w_k = 1.
$$

For independent shocks, a tensor-product rule uses Cartesian products of the one-dimensional nodes and products of their normalized weights, which therefore also sum to one. For larger shock dimensions, add a separately validated Smolyak quadrature construction; sparse-grid interpolation indices alone do not define the necessary quadrature weights.

### Proposed API

```python
from equilibrium.solvers.collocation import solve_collocation, CollocationResult
from equilibrium.approx import make_smolyak_chebyshev

# 1. Define and finalize model (existing workflow)
mod = Model(label="rbc")
# ... add rules, params, exog ...
mod.finalize()
mod.solve_steady()

# 2. Solve nonlinear global policy function
result = solve_collocation(
    mod,
    # Approximation configuration
    approx_type="smolyak_chebyshev",  # or pass an equilibrium.approx.Function
    max_level=3,                       # Smolyak level
    domain=None,                       # auto from steady state ± 2σ (from linear solution)

    # Quadrature for expectations
    n_quad=5,                          # per-dimension quadrature nodes

    # Algorithm
    algorithm="hybrid",                # "time_iteration", "newton", "hybrid"
    tol=1e-8,
    max_iter=1000,

    # Initial guess
    init="linear",                     # "linear" (from linearization), "steady", or array
)

# 3. Use result
result.policy_functions     # dict of Function objects keyed by control var name
result.coefficients         # raw coefficient array (n_basis × n_controls)
result.converged            # bool
result.euler_errors         # residuals evaluated on a test grid

# Evaluate policy at arbitrary state
u = result.evaluate(x=np.array([...]), z=np.array([...]))

# Simulate
sim = result.simulate(T=10000, seed=42)

# IRFs using nonlinear policy
irfs = result.compute_irfs(T=40, shock_name="Z_til", shock_size=1.0, n_sims=1000)
```

### Convenience method on Model

```python
class Model:
    def solve_nonlinear(self, **kwargs) -> CollocationResult:
        """Solve for global nonlinear policy functions via collocation."""
        from ..solvers.collocation import solve_collocation
        return solve_collocation(self, **kwargs)
```

---

## Result container

```python
@dataclass
class CollocationResult:
    """Result of a collocation-based nonlinear global solve."""

    # Policy function approximations
    policy_functions: dict[str, Function]   # var_name -> Function
    coefficients: np.ndarray                # (n_basis, n_controls)

    # Convergence info
    converged: bool
    n_iterations: int
    final_residual: float
    algorithm: str

    # Grid info
    collocation_points: np.ndarray          # (n_points, n_states)
    domain_lb: np.ndarray
    domain_ub: np.ndarray

    # Model reference
    model_label: str
    var_names: list[str]                    # control variable names
    state_names: list[str]                  # state variable names

    # Methods
    def evaluate(self, x, z) -> np.ndarray: ...
    def simulate(self, T, seed=None) -> PathResult: ...
    def compute_irfs(self, T, shock_name, ...) -> IrfResult: ...
    def euler_errors(self, n_test=1000, seed=None) -> np.ndarray: ...
    def save(self, filepath) -> Path: ...
```

---

## Implementation Plan

### Phase 1: Port `funcapprox` into `equilibrium.approx` (complete)

1. Copy core source files from `funcapprox/src/funcapprox/` into `src/equilibrium/approx/`, excluding `benchmark/` and retaining `py.typed` plus upstream provenance.
2. Update internal imports (`from funcapprox.` → `from equilibrium.approx.`).
3. Add a stateless, Chebyshev-first JAX evaluation path with explicit coefficients while preserving the NumPy compatibility API.
4. Re-export the public API from `equilibrium.approx.__init__`.
5. Port relevant tests from `funcapprox/tests/` into `tests/test_approx*.py`.

Implementation details and validation results are recorded in
[`docs/nonlinear-phase-1.md`](nonlinear-phase-1.md).

### Phase 2: Quadrature infrastructure (complete)

1. Add `src/equilibrium/solvers/quadrature.py` with:
   - A normalized Gauss-Hermite nodes/weights generator matching `ghquad_norm()` (`hermgauss`, weights summing to one, nodes scaled by `sqrt(2) * sig` and shifted by `mu`).
   - Tensor-product rules for a small number of independent shocks.
   - A separately validated sparse quadrature rule for higher-dimensional shock models.
   - Utility to compute next-period exogenous states given quadrature nodes.

The completed implementation provides immutable NumPy containers, array-only
JAX data, normalized one-dimensional and tensor rules, a signed-weight Smolyak
rule, explicit `PERS_*`/`VOL_*` model scaling, and differentiable single or
batched JAX transitions. API decisions, sparse-level semantics, and validation
are recorded in [`docs/nonlinear-phase-2.md`](nonlinear-phase-2.md).

### Phase 3: Collocation solver core

1. Create `src/equilibrium/solvers/colloc_spec.py`:
   - `CollocationSpec` dataclass for solver configuration.
   - Domain auto-detection from linear solution (steady state ± multiple of unconditional standard deviation).
2. Create `src/equilibrium/solvers/collocation.py`:
   - `solve_collocation(mod, ...)` main entry point.
   - `_solve_time_iteration(...)` inner algorithm.
   - `_solve_newton_stacked(...)` inner algorithm.
   - Residual function builder that wires `Model.fcn()` calls with basis evaluation and quadrature.
3. Add `CollocationResult` to `solvers/results.py`.
4. Add `Model.solve_nonlinear()` convenience method.
5. Export in `solvers/__init__.py` and `equilibrium/__init__.py`.

### Phase 4: Simulation and IRFs

1. Add `CollocationResult.simulate()` — forward simulation using the nonlinear policy function with stochastic shocks.
2. Add `CollocationResult.compute_irfs()` — generalized IRFs via Monte Carlo (simulate with and without a shock impulse, average the difference).
3. Add `CollocationResult.euler_errors()` — evaluate Euler residuals on a separate random test grid to assess solution accuracy.

### Phase 5: Testing and validation

1. Solve the RBC model nonlinearly and compare against:
   - Linear perturbation solution (should match in the small-shock limit).
   - Deterministic path solver (should match for zero-volatility paths).
2. Convergence tests: verify Newton convergence rate.
3. Quadrature tests: verify that weights sum to one and that the rule reproduces the requested normal mean and variance, following the existing `ghquad_norm()` tests.
4. Accuracy tests: Euler errors on random test grid should be < 1e-4 for standard calibrations.

---

## Mapping from econ-project to equilibrium

The table below maps concepts from the old C++ `colloc` to the new Python/JAX implementation:

| econ-project (`colloc`) | equilibrium | Notes |
|---|---|---|
| `sparse_approx::Grid` | `equilibrium.approx.Scheme` | Smolyak grid + basis construction |
| `sparse_approx::Function` | `equilibrium.approx.Function` | Collocation fitting and evaluation |
| `sparse_approx::Basis1d` | `equilibrium.approx.Basis1d` | 1D basis ABC |
| `colloc::BaseParams` | `CollocationSpec` + `Model.params` | Grid/algo configuration split from model params |
| `colloc::StateSpace` | Inline in solver (uses `Model.fcn()`) | JAX-compiled model equations replace manual state management |
| `colloc::Model.pol_fcns` | `CollocationResult.policy_functions` | Per-control `Function` objects |
| `colloc::Model.expect_fcns` | Not needed (computed inline) | Expectations integrated directly via quadrature |
| `colloc::solveTimeIter` | `_solve_time_iteration()` | Grid-point-by-grid-point Newton, `vmap`-able |
| `colloc::solveNewton` | `_solve_newton_stacked()` | Full coefficient Newton |
| `colloc::linearizeKlein` | `Model.linearize()` (existing) | Already implemented |
| `ADFunction` (autodiff) | `FunctionBundle` (JAX autodiff) | JAX `jacfwd`/`jacrev` replaces CppAD |
| `newtonSolve` (C++ Newton) | `equilibrium.solvers.newton.root` | Already implemented |

---

## Open questions

1. **Domain specification.** The original `colloc` supports adaptive domains that rescale based on the steady state and unconditional standard deviations. Should we auto-detect bounds from the linear solution, require user specification, or support both? *(Recommendation: auto-detect with user override.)*

2. **JAX traceability of basis evaluation.** The hat basis functions use `np.maximum` and conditional logic that may need adjustment for JAX tracing. Chebyshev bases (pure recurrences) are naturally JIT-friendly. Should we prioritize Chebyshev-only for the initial implementation? *(Recommendation: yes, Chebyshev first; hat bases can follow.)*

3. **Expectations approximation.** The original `colloc` optionally pre-approximates expectation terms as separate `Function` objects (the `expect_fcns` in `colloc::Model`) for efficiency. This adds complexity. Should we include this in the initial implementation? *(Recommendation: defer; direct quadrature is simpler and with JAX JIT should be fast enough for models with ≤ 3-4 shocks.)*

4. **Parallelism.** Time iteration is embarrassingly parallel across grid points. Beyond `jax.vmap`, should we support `jax.pmap` for multi-GPU or distributed execution? *(Recommendation: defer; `vmap` + `jit` should handle typical model sizes.)*
