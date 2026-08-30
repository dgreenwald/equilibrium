# Nonlinear Solutions Phase 3: Collocation Solver Core

## Purpose

Phase 3 implements the first complete global nonlinear policy solve. It joins
the Chebyshev approximation path from Phase 1 to the Gaussian quadrature and
exogenous-process path from Phase 2, then solves the model's optimality
conditions at every collocation node.

This phase establishes:

- a validated `CollocationSpec` configuration object;
- automatic or explicit state-space domains;
- one pure JAX residual path shared by all algorithms;
- time iteration with batched pointwise Newton solves;
- dense Newton on the stacked coefficient system;
- a hybrid time-iteration/Newton workflow;
- a `CollocationResult` policy container and evaluation API;
- `solve_collocation()` and `Model.solve_nonlinear()` entry points.

Phase 3 does **not** implement stochastic simulation, nonlinear impulse
responses, generalized impulse responses, off-grid Euler-error sampling,
adaptive domains, expectation-function approximation, hat-basis JAX support,
or multi-device parallelism. Those remain later work.

## Status

In progress. Work package 1 was completed on 2026-08-29. It added the validated
`CollocationSpec`, model lifecycle and dimension checks, ordered exogenous
process resolution, conditional linearization with the resolved process, and
explicit, approximation-derived, or stationary-covariance domains. The
implementation follows the decisions in this document without material API or
algorithm deviations. Public exports remain deferred to work package 5.

Work package 1 added 57 focused tests. Validation completed with all 238
collocation-spec and quadrature tests passing and the full suite at 1,051
passing tests. Ruff and Black pass for the new source and tests. MyPy was not
installed in the active environment, so its optional check was not run.

## Inputs already provided by Phases 1 and 2

Phase 3 must build on the existing public boundaries instead of duplicating
their logic.

From `equilibrium.approx`:

- `make_smolyak_chebyshev()` and `make_tensor_chebyshev()` construct a grid and
  precompute its collocation basis inverse with NumPy;
- `Function.get_grid_points()` returns rows shaped `(n_points, n_states)`;
- `Function.fit()` maps nodal values to coefficients;
- `make_jax_data()` creates immutable JAX approximation data;
- `evaluate_jax(data, coefficients, points)` evaluates scalar or vector-valued
  Chebyshev policies and is differentiable with respect to coefficients and
  points.

From `equilibrium.solvers.quadrature`:

- `QuadratureRule` and `ExogenousProcess` are validated immutable NumPy setup
  objects;
- their `as_jax()` methods provide arrays for the hot path;
- `tensor_gauss_hermite()`, `smolyak_gauss_hermite()`, and
  `deterministic_quadrature()` construct standardized innovation rules;
- `exogenous_process_from_model()` resolves the default
  `PERS_<name>`/`VOL_<name>` convention;
- `next_exogenous_states_jax()` computes single or batched conditional
  transitions.

The collocation solver must not reconstruct basis metadata or quadrature rules
inside JIT, read `VOL_*` independently, or use the mutable coefficients stored
on `Function` in a differentiated residual.

## Core decisions

### State, control, and coefficient ordering

The state vector has one fixed order throughout the implementation:

```text
s = [x, z]
```

where `x` follows `model.var_lists["x"]` and `z` follows
`model.exog_list`. Controls follow `model.var_lists["u"]`. Collocation points
are rows, policy values are rows, and policy coefficients have shape:

```text
(n_basis, n_controls)
```

The approximation is one vector-valued `Function`, not one duplicated
`Function` per control. This is the representation already supported by Phase
1 and makes the coefficient block explicit to JAX.

When a coefficient array or residual is flattened for stacked Newton, C order
is authoritative: point/basis index is the outer index and control index is the
inner index. Reshaping the flat vector to `(n_basis, n_controls)` must therefore
recover the public coefficient layout exactly.

Models with no controls are invalid collocation problems. Models with no
dynamic state (`n_x + n_z == 0`) should also be rejected with guidance to use
the steady-state solver. Zero endogenous-state, zero exogenous-state, and
zero-expectation blocks remain valid when the total state dimension is
positive.

### Model lifecycle

`solve_collocation()` requires a finalized model with a successfully solved
steady state. It must not call `solve_steady()` implicitly because calibration,
display, and failure choices belong to the caller.

Linearization is required when either:

- the domain is automatic; or
- initialization is `"linear"`.

In those cases the solver calls:

```python
model.linearize(
    Phi=process.persistence,
    impact_matrix=process.innovation_impact,
)
```

even if the model was previously linearized. Re-linearizing ensures the local
policy and state dynamics use the same process convention as nonlinear
quadrature, rather than the identity impact matrix installed by
`Model.finalize()`. An explicit domain combined with steady-state or explicit
coefficient initialization does not require linearization.

The parameter vector passed to every model function is a frozen JAX copy of
`model.steady_components["params"]`. The solver must not mutate model
parameters, steady-state arrays, rules, or equation bundles during a solve.

### Domain construction

The caller may supply explicit lower and upper bounds. Explicit bounds must be
finite arrays of shape `(n_states,)` and must satisfy `lower < upper` in every
coordinate. They are used without rescaling. If `spec.domain` is absent but a
custom `Function` is supplied, that function's bounds are authoritative and
are validated under the same contract. Automatic domain construction is used
only when neither source provides bounds.

When bounds are omitted, construct them from the linearized joint state law in
deviations from steady state:

$$
\begin{bmatrix}x_{t+1}\\z_{t+1}\end{bmatrix}
=
\underbrace{\begin{bmatrix}H_x&H_z\\0&\Phi\end{bmatrix}}_{A}
\begin{bmatrix}x_t\\z_t\end{bmatrix}
+
\underbrace{\begin{bmatrix}0\\L\end{bmatrix}}_{B}\epsilon_{t+1},
\qquad \epsilon_{t+1}\sim N(0,I).
$$

Use `scipy.linalg.solve_discrete_lyapunov(A, B @ B.T)` to obtain the stationary
covariance. Before solving, require a finite `A` and spectral radius strictly
below `1 - 1e-10`. Validate that the returned covariance is finite and has no
materially negative diagonal entry; clip only roundoff-sized negative diagonal
values to zero.

The center is:

```text
center = concatenate([steady_components["x"], steady_components["z"]])
```

and the default half-width in coordinate `j` is:

```text
max(
    domain_stddevs * sqrt(covariance[j, j]),
    domain_min_half_width * max(1.0, abs(center[j])),
)
```

with defaults `domain_stddevs=3.0` and
`domain_min_half_width=1e-4`. The scale-aware floor keeps deterministic or
unshocked coordinates from producing a degenerate Chebyshev interval. Invalid
or unstable linear dynamics produce a setup error asking for explicit bounds;
the solver must not silently invent a covariance.

The domain is a collocation region, not a state constraint. Gaussian
quadrature can legitimately produce next states outside a finite domain, and
Phase 1 deliberately evaluates those states by polynomial extrapolation. The
solver records final extrapolation diagnostics and never clips a next state.
`extrapolation="error"` is available for callers who require strict bounds;
the default is `"allow"`.

### Approximation construction

The initial implementation supports two JAX-compatible choices:

- `"smolyak_chebyshev"` using `max_levels` and `max_total_level`;
- `"tensor_chebyshev"` using `tensor_points`.

The default is a Smolyak Chebyshev grid with `max_levels=3` in every state
dimension and `max_total_level=3`. The constructor's existing semantics remain
authoritative; Phase 3 must not introduce a second sparse-index convention.

Alternatively, the caller may pass an already constructed `Function`. Its
dimension and, when `spec.domain` is explicit, bounds must match the resolved
state dimension and domain; `make_jax_data()` must accept it. With no explicit
domain, the custom function supplies the bounds and bypasses automatic domain
construction. Passing a custom function is the extension point for anisotropic
or manually constructed Chebyshev schemes. Configuration fields that would
rebuild an approximation are ignored only when they retain their defaults;
contradictory custom-function and builder settings should be rejected rather
than silently discarded.

The basis inverse is converted once to a JAX array. Refitting nodal values in
time iteration is then the matrix product:

$$
C_{new}=B^{-1}U_{nodes},
$$

so no NumPy/device round trip is required inside the iteration loop.

### Quadrature construction

The default rule is tensor Gauss-Hermite with degree 5 per structural
innovation and the existing `max_nodes=100_000` allocation guard. The caller
may select Smolyak quadrature with `quadrature_kind="smolyak"` and an explicit
`quadrature_level`, or pass a `QuadratureRule` directly.

For zero innovation dimensions, use `deterministic_quadrature()`. Otherwise,
the rule dimension must equal
`process.innovation_impact.shape[1]`. Rules contain standardized innovations;
only `ExogenousProcess.innovation_impact` applies volatility or correlation.

There is no automatic switch from tensor to Smolyak quadrature. An automatic
heuristic would make accuracy change implicitly with model dimension; the
existing node guard instead produces a clear request for an explicit sparse
choice.

### Direct expectations

Phase 3 directly integrates the model's expectation integrand at every
collocation point and quadrature node. It does not fit separate approximation
functions for expectation variables. This keeps one source of approximation
error and one coefficient system; expectation-function approximation can be
added later only if profiling justifies it.

### Numerical precision

Collocation solves use the repository's configured JAX floating-point mode.
They require 64-bit floating point because global Newton systems and
high-degree Chebyshev bases are not reliably conditioned in float32. Setup
must fail clearly when `jax_enable_x64` is disabled rather than silently
weakening the requested tolerances.

## Proposed public API

### `CollocationSpec`

Add `src/equilibrium/solvers/colloc_spec.py` with an immutable configuration
container:

```python
@dataclass(frozen=True)
class CollocationSpec:
    approximation: Literal[
        "smolyak_chebyshev", "tensor_chebyshev"
    ] = "smolyak_chebyshev"
    max_levels: int | tuple[int, ...] = 3
    max_total_level: int = 3
    tensor_points: int | tuple[int, ...] = 5

    domain: tuple[ArrayLike, ArrayLike] | None = None
    domain_stddevs: float = 3.0
    domain_min_half_width: float = 1e-4

    quadrature_kind: Literal["tensor", "smolyak"] = "tensor"
    quadrature_degree: int | tuple[int, ...] = 5
    quadrature_level: int = 2
    quadrature_max_nodes: int | None = 100_000

    algorithm: Literal["time_iteration", "newton", "hybrid"] = "hybrid"
    initialization: Literal["linear", "steady"] = "linear"
    tolerance: float = 1e-8
    inner_tolerance: float = 1e-10
    max_time_iterations: int = 500
    max_inner_iterations: int = 30
    max_newton_iterations: int = 50
    max_backtracks: int = 12
    damping: float = 1.0
    hybrid_switch_tolerance: float = 1e-4
    hybrid_max_time_iterations: int = 50
    max_newton_unknowns: int | None = 2_000

    extrapolation: Literal["allow", "error"] = "allow"
    verbose: bool = False
```

Validation occurs before expensive grid construction or JAX compilation:

- tolerances and domain multipliers are finite and strictly positive;
- iteration limits and quadrature levels/degrees are valid integers;
- `0 < damping <= 1`;
- per-dimension tuples have the resolved dimension;
- algorithm, approximation, quadrature, initialization, and extrapolation
  values are recognized;
- explicit bounds satisfy the domain contract;
- `hybrid_switch_tolerance >= tolerance`;
- a finite `max_newton_unknowns` is positive.

Arrays accepted through `domain` are defensively copied and marked read-only,
following the Phase 2 container convention.

### Main entry point

```python
def solve_collocation(
    model: Model,
    spec: CollocationSpec | None = None,
    *,
    approximation: Function | None = None,
    quadrature_rule: QuadratureRule | None = None,
    process: ExogenousProcess | None = None,
    initial_coefficients: ArrayLike | None = None,
) -> CollocationResult: ...
```

`None` for `spec` means `CollocationSpec()`. Explicit process and rule objects
override their corresponding builders and are validated against the model and
one another. A custom process must use the same exogenous names in the same
order as `model.exog_list`. `initial_coefficients` overrides
`spec.initialization` and must have exact shape `(n_basis, n_controls)` with
finite entries.

Invalid inputs, model lifecycle violations, unstable automatic-domain systems,
unsupported dtypes, and allocation-limit violations raise descriptive
exceptions. Numerical failure after a valid solve begins returns a
`CollocationResult` with `converged=False` and a `failure_reason`; it does not
discard the best finite coefficient iterate.

The convenience method is:

```python
def Model.solve_nonlinear(
    self,
    spec: CollocationSpec | None = None,
    **kwargs,
) -> CollocationResult:
    from ..solvers.collocation import solve_collocation

    return solve_collocation(self, spec, **kwargs)
```

### Result and iteration records

Add the following non-path result types to `solvers/results.py`:

```python
@dataclass(frozen=True)
class CollocationIteration:
    stage: Literal["time_iteration", "newton"]
    iteration: int
    residual_norm: float
    coefficient_change: float | None
    step_size: float | None
    failed_nodes: int = 0


@dataclass
class CollocationResult:
    policy_function: Function
    coefficients: np.ndarray
    spec: CollocationSpec

    converged: bool
    failure_reason: str | None
    n_iterations: int
    time_iterations: int
    newton_iterations: int
    final_residual: float
    final_coefficient_change: float | None
    history: tuple[CollocationIteration, ...]

    collocation_points: np.ndarray
    domain_lb: np.ndarray
    domain_ub: np.ndarray
    max_extrapolation: float
    extrapolated_next_states: int

    model_label: str
    control_names: tuple[str, ...]
    state_names: tuple[str, ...]

    def evaluate_states(self, states: ArrayLike) -> np.ndarray: ...
    def evaluate(self, x: ArrayLike, z: ArrayLike) -> np.ndarray: ...
```

The result's arrays are defensive, read-only NumPy copies. The stored
`policy_function.coefficients` is initialized from the same coefficient values,
and result construction validates that its domain and basis count agree.
`coefficients` remains authoritative: result evaluation calls `evaluate_jax()`
with those explicit coefficients, so later reassignment of the mutable
compatibility `Function.coefficients` attribute cannot change the result.
`evaluate_states()` accepts `(n_states,)` or `(n_eval, n_states)` and preserves
the corresponding unbatched/batched return shape. `evaluate(x, z)` performs
the same operation after validating and concatenating the two named state
blocks.

Simulation, IRFs, Euler-error sampling, and persistence are intentionally not
methods on the Phase 3 result. Adding them before their data and reproducibility
contracts are designed would blur the Phase 3/4 boundary.

## Residual system

### Pure point residual for time iteration

For current control `u_i`, state `s_i = [x_i, z_i]`, and frozen old policy
coefficients `C_old`, define:

1. `x_next = model.transition(u_i, x_i, z_i, params)`;
2. `z_next[k] = Phi @ z_i + L @ epsilon[k]`;
3. `s_next[k] = concatenate([x_next, z_next[k]])`;
4. `u_next[k] = evaluate_jax(approx_data, C_old, s_next[k])`;
5. evaluate
   `model.expectations(u_i, x_i, z_i, u_next[k], x_next, z_next[k], params)`
   at all nodes;
6. contract the quadrature axis with `weights` to obtain `E_i`;
7. return `model.optimality(u_i, x_i, z_i, E_i, params)`.

The shape contract is:

```text
u_i                 (n_controls,)
x_i                 (n_x,)
z_i                 (n_z,)
x_next              (n_x,)
z_next              (n_quad, n_z)
s_next              (n_quad, n_states)
u_next              (n_quad, n_controls)
expectation values  (n_quad, n_expectations)
E_i                 (n_expectations,)
residual             (n_controls,)
```

The transition is evaluated once per current point, outside the quadrature
axis, because `x_{t+1}` is predetermined by current `u`, `x`, and `z` under the
existing model function signature. Automatic differentiation through the
point residual must still include the effect of current `u` on `x_next`, the
next-policy evaluation, and the expectation integrand.

### Batched coefficient residual

The stacked residual uses the same economic operations, vectorized across all
collocation nodes:

```text
states                  (n_points, n_states)
current policy          (n_points, n_controls)
next endogenous states  (n_points, n_x)
next exogenous states   (n_points, n_quad, n_z)
next full states         (n_points, n_quad, n_states)
next policy              (n_points, n_quad, n_controls)
expectation integrands   (n_points, n_quad, n_expectations)
integrated expectations  (n_points, n_expectations)
residuals                (n_points, n_controls)
```

Flatten `(n_points, n_controls)` only at the outer boundary needed by Newton.
The implementation should flatten the point/quadrature axes once for policy
and expectation evaluation, use `jax.vmap` for model functions, and reshape
back. Avoid Python loops over points or nodes in the residual hot path.

Create one private residual builder in `collocation.py` that returns JITted
callables for:

- the batched coefficient residual;
- the point residual used by time iteration;
- the final next-state extrapolation diagnostic.

All algorithms must call these shared functions. A separate formula for each
algorithm would invite timing, scaling, and state-order drift.

### Residual norm and extrapolation diagnostic

Solver convergence uses the maximum absolute optimality residual:

$$
\lVert R\rVert_\infty=\max_{i,j}|R_{i,j}|.
$$

For each final next state, measure the normalized domain violation:

$$
v_j(s')=\max\left(
\frac{lb_j-s'_j}{ub_j-lb_j},
\frac{s'_j-ub_j}{ub_j-lb_j},
0
\right).
$$

Store the maximum `v_j` and the number of next-state rows with any violation
larger than `1e-12`. Under `extrapolation="error"`, an accepted iterate with a
violation stops the solve with `converged=False` and
`failure_reason="next_state_outside_domain"`. Trial line-search points may be
evaluated by extrapolation, but cannot be accepted in strict mode.

## Initialization

Initialization always produces nodal policy values first, then applies the
precomputed basis inverse.

For `"steady"` initialization:

```text
u_initial(s_i) = u_steady
```

for every grid point.

For `"linear"` initialization, use deviations in the same state order:

$$
u(s_i)=u_{ss}+G_x(x_i-x_{ss})+G_z(z_i-z_{ss}).
$$

`z_ss` is taken from `steady_components["z"]`, not hard-coded, even though it
is currently zero. Validate `G_x` and `G_z` shapes and finiteness after
linearization.

Explicit initial coefficients bypass nodal initialization and are copied
directly. Regardless of source, evaluate the initial stacked residual once and
return immediately as converged if it already satisfies the tolerance. Also
validate here that the batched residual has exact shape
`(n_points, n_controls)`, making the square-system assumption explicit before
either Newton algorithm begins.

## Solution algorithms

### Time iteration

At outer iteration `m`, hold `C_old` fixed wherever the next-period policy is
evaluated. For every collocation point, independently solve:

$$
R_i(u_i;C_{old})=0.
$$

Initialize each point solve from `evaluate_jax(..., C_old, grid_points)`. This
uses the previous policy at the current nodes and avoids carrying a second,
potentially inconsistent nodal state between outer iterations.

Implement a collocation-local batched Newton kernel rather than applying
`vmap` to `solvers.newton.root`, whose Python control flow is not JIT/vmap
compatible. The kernel uses:

- `jax.vmap(jax.jacfwd(point_residual))` for the `(n_points, n_u, n_u)`
  Jacobian batch;
- `jnp.linalg.solve` for each point's Newton direction;
- independent per-point half-step backtracking;
- active/converged masks so completed points remain fixed;
- a maximum of `max_inner_iterations` and `max_backtracks`;
- finite residual, Jacobian, step, and iterate checks.

The merit function for line search is the squared Euclidean norm of each
point's residual. A trial must strictly reduce that point's merit. Singular
linear systems, nonfinite values, failed line searches, or remaining residuals
above `inner_tolerance` mark those node indices as failed. Do not replace a
failed solve with a pseudoinverse silently.

After all nodes solve:

1. fit `C_fit = basis_inverse @ U_nodes`;
2. apply outer damping,
   `C_new = C_old + damping * (C_fit - C_old)`;
3. evaluate the full equilibrium residual using `C_new` on both current and
   next policies;
4. compute scaled coefficient change
   `max(abs(C_new - C_old)) / (1 + max(abs(C_new)))`;
5. append one `CollocationIteration` record.

Time iteration converges only when both the full residual norm and scaled
coefficient change are at most `tolerance`. If any pointwise solve fails, stop
with the best previous finite coefficients and include the failed-node count in
the history and failure reason. If the maximum outer count is reached, return
the latest finite iterate as nonconverged.

### Dense stacked Newton

For coefficient vector `c = C.ravel()`, define:

```python
def flat_residual(c):
    C = c.reshape(n_basis, n_controls)
    return batched_residual(C).ravel()
```

Use `jax.jacfwd(flat_residual)` to form the exact dense Jacobian. Forward mode
is the initial choice because the system is square and it matches the
coefficient-direction formulation; benchmark `jacrev` later rather than
changing mode heuristically.

Before compilation, reject a solve when `n_basis * n_controls` exceeds
`max_newton_unknowns`, unless the caller explicitly sets that guard to `None`.
The default of 2,000 limits accidental dense Jacobian allocations. The error
must report the unknown count and suggest time iteration or a smaller grid.

Use a global Newton step from `jnp.linalg.solve` and half-step backtracking on
the squared Euclidean norm of the full residual. Convergence is still judged by
the infinity norm. Record the accepted step length in history. A singular
Jacobian, nonfinite value, exhausted line search, strict-domain violation, or
iteration limit returns the best accepted finite iterate and a specific
failure reason.

The existing generic `solvers.newton.root` may share small utility functions,
but Phase 3 should not force its result shape or logging behavior onto the
collocation result. The stacked implementation needs coefficient reshaping,
the allocation guard, extrapolation checks, and structured history.

### Hybrid

Hybrid is the default and has two deterministic stages:

1. run time iteration until the scaled coefficient change is at most
   `hybrid_switch_tolerance`, the full solve tolerance is reached, or
   `hybrid_max_time_iterations` is reached;
2. unless already converged, pass the latest coefficients to stacked Newton.

Reaching the hybrid time-iteration cap is a switch condition, not a failure,
provided the latest iterate is finite and its pointwise solves succeeded. A
pointwise Newton failure remains terminal because it does not provide a
trustworthy warm start. The stacked allocation guard is validated before the
hybrid solve begins so a long warm-up cannot end in a predictable configuration
error.

Hybrid history uses stage-local iteration numbers and the result reports both
stage counts plus their sum. The final `algorithm` remains available through
the stored spec; no undocumented fallback from stacked Newton back to time
iteration occurs.

## Internal setup object and file layout

Use a private immutable setup container to keep normalized arrays and dimensions
together without expanding the public API:

```python
@dataclass(frozen=True)
class _CollocationSetup:
    approximation: Function
    approximation_data: JaxApproximationData
    basis_inverse: jax.Array
    points: jax.Array
    params: jax.Array
    process: JaxExogenousProcess
    rule: JaxQuadratureRule
    domain_lb: jax.Array
    domain_ub: jax.Array
    n_controls: int
    n_x: int
    n_z: int
```

Expected file changes are:

```text
src/equilibrium/
├── __init__.py                       # Export Phase 3 public API
├── model/model.py                    # Add solve_nonlinear convenience method
└── solvers/
    ├── __init__.py                   # Export Phase 3 public API
    ├── colloc_spec.py                # NEW: configuration and domain resolution
    ├── collocation.py                # NEW: setup, residuals, algorithms
    └── results.py                    # Add result and iteration records

tests/
├── test_colloc_spec.py               # NEW: validation and auto-domain tests
├── test_collocation_residual.py      # NEW: equation wiring and JAX tests
└── test_collocation.py               # NEW: algorithms and public API

docs/
├── nonlinear-solutions.md            # Mark Phase 3 complete after implementation
└── nonlinear-phase-3.md              # This plan and status record
```

Avoid introducing a new runtime dependency: SciPy, NumPy, and JAX are already
part of the project.

## Detailed work plan

### Work package 1: Specification, lifecycle checks, and domain resolution

1. Add `CollocationSpec` with the fields, defaults, and validation above.
2. Add normalization helpers for scalar/per-dimension grid and quadrature
   settings.
3. Validate finalized/steady-solved model state and derive dimensions and name
   order from `model.N` and `model.var_lists`.
4. Resolve or validate `ExogenousProcess` before any linearization.
5. Implement explicit-domain validation.
6. Implement joint linear dynamics, stability checks, stationary covariance,
   scale-aware minimum widths, and automatic bounds.
7. Test that automatic linearization receives the resolved `Phi` and `L`, and
   that no linearization occurs when explicit bounds and non-linear
   initialization make it unnecessary.

Exit criterion: every solve reaches approximation construction with one valid,
nondegenerate domain and one documented model/process convention.

### Work package 2: Approximation, quadrature, initialization, and residuals

1. Build or validate the Chebyshev approximation and convert its static data
   and basis inverse once to JAX.
2. Build or validate quadrature, including deterministic zero-innovation
   behavior and innovation-dimension matching.
3. Implement steady, linear, and explicit-coefficient initialization.
4. Implement the shared point and batched residual paths with exact shapes and
   timing described above.
5. JIT the residual functions with arrays as dynamic arguments and the model's
   equation callables closed over as static Python objects.
6. Implement normalized next-state extrapolation diagnostics without clipping.
7. Test eager/JIT parity and differentiation with respect to current controls,
   policy coefficients, and states.

Exit criterion: a coefficient matrix produces the same finite economic
residual in point, batched, eager, and JITted forms.

### Work package 3: Time iteration

1. Implement the masked batched pointwise Newton kernel with autodiff
   Jacobians and independent line searches.
2. Implement nodal refitting through the JAX basis inverse and outer damping.
3. Add the two-part convergence check, finite checks, failed-node reporting,
   strict extrapolation behavior, and iteration history.
4. Test single- and multi-control systems, convergence from steady and linear
   initializations, damping, already-converged inputs, and every failure path.
5. Confirm the hot iteration path contains no NumPy conversion or Python loop
   over grid/quadrature rows.

Exit criterion: time iteration solves a known nonlinear policy problem and
returns reproducible diagnostics for success and failure.

### Work package 4: Stacked Newton and hybrid solve

1. Implement the flat coefficient residual and exact dense JAX Jacobian.
2. Enforce the unknown-count allocation guard before tracing.
3. Implement global Newton backtracking, finite/singularity checks, strict
   extrapolation behavior, and history.
4. Implement the exact hybrid switch rules and stage accounting.
5. Test stacked convergence from a near solution, allocation rejection,
   singular Jacobians, backtracking, hybrid switching by tolerance and by cap,
   and direct convergence during the warm-up stage.
6. Compare final time-iteration, Newton, and hybrid policies on the same small
   model to tolerance.

Exit criterion: all three algorithms consume the shared residual and agree on
the solved policy while exposing predictable resource and failure behavior.

### Work package 5: Result container and public integration

1. Add `CollocationIteration` and `CollocationResult` to `solvers/results.py`
   without inheriting from `PathResult`.
2. Implement defensive result construction and single/batched policy
   evaluation.
3. Add `solve_collocation()` orchestration and the initial-residual fast path.
4. Add `Model.solve_nonlinear()` using a local import to avoid model/solver
   import cycles.
5. Export `CollocationSpec`, `CollocationIteration`, `CollocationResult`, and
   `solve_collocation` from `equilibrium.solvers` and the root `equilibrium`
   namespace.
6. Add API tests for direct, convenience, solver-package, and root imports.

Exit criterion: users can configure, solve, inspect, and evaluate a nonlinear
policy without importing private helpers.

### Work package 6: End-to-end validation and documentation

1. Add a compact one-state stochastic growth fixture based on the repository's
   existing RBC test model; keep expensive grid sizes out of routine CI.
2. Solve it with time iteration and hybrid, assert convergence, shapes,
   float64 data, finite coefficients, and residual tolerance.
3. Compare the nonlinear policy derivative at steady state with `G_x` and
   `G_z` on a calibration where the local comparison is meaningful.
4. Add a deterministic analytic model with a known policy to separate solver
   correctness from RBC calibration complexity.
5. Document a minimal Phase 3 solve and evaluation example in
   `nonlinear-solutions.md` and remove or update the superseded sketch there.
6. Record implementation deviations and final validation totals in this
   document.
7. Run targeted tests, the full suite, Ruff, Black, and MyPy when available.

Exit criterion: the public nonlinear solve passes analytic, model-integrated,
JAX, regression, lint, and formatting checks with no Phase 4 functionality
mixed into the implementation.

## Test matrix

### Specification and domain tests

- every invalid enum, tolerance, limit, damping value, and tuple length;
- explicit bound shape, finiteness, ordering, defensive copy, and immutability;
- state/control ordering from a real finalized model;
- steady-state lifecycle failure;
- automatic domain for scalar and multivariate joint dynamics;
- correlated and rectangular innovation impacts;
- zero-variance scale-aware width;
- unstable, nonfinite, and invalid stationary covariance failures;
- resolved process matrices passed to linearization;
- no unnecessary linearization with explicit domain and steady initialization.

### Setup and initialization tests

- Smolyak and tensor Chebyshev builders;
- anisotropic level/point tuples;
- custom compatible and incompatible `Function` objects;
- tensor, Smolyak, custom, and deterministic quadrature rules;
- quadrature/process innovation-dimension mismatch;
- steady nodal initialization;
- exact `u_ss + G_x dx + G_z dz` linear initialization;
- explicit coefficient shape and finiteness;
- x64 requirement.

### Residual tests

- hand-computed scalar deterministic residual;
- stochastic conditional expectation with nonunit volatility;
- multi-control and multi-expectation shapes;
- `n_x == 0`, `n_z == 0`, and `n_expectations == 0` where meaningful;
- eager/JIT and point/batch agreement;
- quadrature contraction on the node axis;
- transition evaluated with current rather than next exogenous state;
- gradients through transition, next-policy evaluation, expectations, and
  optimality;
- coefficient flatten/reshape ordering;
- no double volatility scaling;
- extrapolation magnitude and row count.

### Time-iteration tests

- known scalar policy convergence;
- multiple independent point roots solved in one batch;
- inner tolerance and maximum-iteration behavior;
- per-node line-search masks;
- damping and coefficient-change calculation;
- full-residual and coefficient-change joint convergence;
- singular, nonfinite, line-search, and failed-node outcomes;
- strict versus allowed extrapolation;
- deterministic repeatability.

### Stacked Newton and hybrid tests

- exact autodiff Jacobian versus finite differences on a tiny system;
- dense Newton convergence and accepted step history;
- initial solution fast path;
- unknown-count guard before Jacobian tracing;
- singular, nonfinite, line-search, and iteration-limit outcomes;
- hybrid tolerance switch, iteration-cap switch, and warm-up convergence;
- final agreement among algorithms on a shared small model.

### Result and integration tests

- result array shapes, immutability, metadata, and failure reason;
- `evaluate_states()` and `evaluate()` for one point and a batch;
- zero-length x or z blocks;
- direct solver and `Model.solve_nonlinear()` equivalence;
- public imports from `equilibrium.solvers` and `equilibrium`;
- stochastic-growth/RBC convergence on a small CI grid;
- local derivative comparison with the linear solution;
- no regression in approximation, quadrature, steady, deterministic, or linear
  solver tests.

## Validation commands

Run each work package's new tests as it is implemented:

```bash
pytest tests/test_colloc_spec.py
pytest tests/test_collocation_residual.py
pytest tests/test_collocation.py
```

Then run the nonlinear dependency slice:

```bash
pytest \
  tests/test_approx_*.py \
  tests/test_quadrature.py \
  tests/test_colloc_spec.py \
  tests/test_collocation_residual.py \
  tests/test_collocation.py
```

Static checks for changed code:

```bash
ruff check \
  src/equilibrium/solvers/colloc_spec.py \
  src/equilibrium/solvers/collocation.py \
  src/equilibrium/solvers/results.py \
  src/equilibrium/model/model.py \
  tests/test_colloc_spec.py \
  tests/test_collocation_residual.py \
  tests/test_collocation.py

black --check \
  src/equilibrium/solvers/colloc_spec.py \
  src/equilibrium/solvers/collocation.py \
  src/equilibrium/solvers/results.py \
  src/equilibrium/model/model.py \
  tests/test_colloc_spec.py \
  tests/test_collocation_residual.py \
  tests/test_collocation.py

mypy \
  src/equilibrium/solvers/colloc_spec.py \
  src/equilibrium/solvers/collocation.py
```

Finally run repository-wide regression checks:

```bash
pytest
ruff check src/equilibrium tests
black --check .
```

## Acceptance criteria

Phase 3 is complete when:

- a single state/control/coefficient ordering is enforced across setup,
  residuals, algorithms, results, and docs;
- explicit domains are validated and automatic domains use the stationary
  covariance of the correctly scaled joint linear state process;
- deterministic coordinates receive a scale-aware nonzero width;
- the solver consumes Phase 1 JAX approximation data and Phase 2 JAX
  quadrature/process data without rebuilding them in the hot path;
- the point and stacked residuals implement the documented model timing and
  agree numerically;
- expectation integration applies innovation scaling exactly once;
- time iteration uses batched autodiff Newton solves and reports failed nodes;
- stacked Newton has an exact JAX Jacobian, backtracking, and a pre-trace dense
  allocation guard;
- hybrid switching and iteration accounting are deterministic;
- all algorithms retain the best finite iterate on numerical failure;
- next states are never clipped and extrapolation is measured explicitly;
- `CollocationResult` evaluates vector-valued policies for single and batched
  states;
- direct, model convenience, solver-package, and root imports work;
- analytic and small stochastic-growth models converge to their asserted
  residual tolerances;
- no simulation, nonlinear IRF, or random Euler-error API is added early;
- all targeted and full-suite tests pass;
- Ruff and Black pass, and MyPy passes when available;
- no new runtime dependency is introduced.

## Risks and mitigations

### Automatic bounds use inconsistent shock scaling

Mitigation: resolve the Phase 2 `ExogenousProcess` first, pass its exact `Phi`
and `L` to linearization, and use those same matrices in the joint covariance
and nonlinear transition tests.

### Gaussian next states leave the collocation domain

Mitigation: treat bounds as an approximation region, never clip states, expose
normalized extrapolation diagnostics, and provide strict failure as an opt-in.
Phase 4 accuracy diagnostics can use these values when choosing a larger
domain.

### Sparse grids create ill-conditioned fits

Mitigation: reuse the constructed scheme's basis inverse, require float64,
check all iterates for finiteness, test multiple levels, and report numerical
failure without overwriting the last finite policy.

### Time iteration appears converged while Euler residuals remain large

Mitigation: require both scaled coefficient change and the full equilibrium
residual to meet tolerance. The pointwise inner residual alone is not an outer
convergence test.

### Pointwise Newton fails at a subset of nodes

Mitigation: track convergence and line search per node, return the failed-node
count, stop before fitting contaminated values, and preserve the previous
finite coefficient iterate.

### Dense Newton allocates an impractical Jacobian

Mitigation: compute the scalar unknown count before JAX tracing and enforce the
explicit `max_newton_unknowns` guard. Time iteration remains available without
the dense coefficient Jacobian.

### JAX recompiles excessively

Mitigation: normalize all setup objects once, keep array shapes fixed for a
solve, JIT the shared residuals once, and pass coefficients and current policy
values as dynamic arrays rather than closing over changing iterates.

### Algorithm implementations drift economically

Mitigation: time iteration and stacked Newton share one point/batch residual
builder, with parity tests covering timing, ordering, quadrature contraction,
and shock scaling.

### Result policy and raw coefficients diverge

Mitigation: construct both from one defensive coefficient copy, make the
read-only result coefficient array authoritative for evaluation, and test that
`evaluate()` agrees with direct `evaluate_jax()` even if the compatibility
`Function.coefficients` attribute is later reassigned.

## Handoff to Phase 4

Phase 4 should receive a converged `CollocationResult` that can evaluate
controls at arbitrary `[x, z]` states, plus the stable process convention used
during the solve. Phase 4 can then add forward simulation, common-random-number
generalized IRFs, and off-grid Euler-error diagnostics without changing the
Phase 3 coefficient or state-order contracts.

The essential handoff is:

```python
result = solve_collocation(model, CollocationSpec(...))
u_t = result.evaluate(x_t, z_t)
```

Simulation must continue to use
`z_next = Phi @ z + L @ epsilon` with the same standardized-innovation
convention. It should not reinterpret `VOL_*`, and it should use the Phase 3
extrapolation diagnostics to inform—not silently alter—domain behavior.
