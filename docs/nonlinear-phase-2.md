# Nonlinear Solutions Phase 2: Quadrature Infrastructure

## Purpose

Phase 2 adds validated Gaussian quadrature infrastructure for evaluating the
expectation terms used by the nonlinear collocation solver. It establishes:

- normalized one-dimensional Gauss-Hermite rules;
- tensor-product rules for independent Gaussian innovations;
- a separately constructed and validated Smolyak rule for higher dimensions;
- an explicit model-to-innovation scaling convention;
- immutable NumPy rule objects and JAX-ready rule data;
- JAX-compatible exogenous-state transition utilities.

This phase does **not** implement collocation residuals, time iteration, stacked
Newton, endogenous-state transitions, simulation, or nonlinear IRFs.

## Current repository conventions

Equilibrium currently contains three related but distinct shock conventions:

1. `Model.exog_list` fixes the order and names of exogenous AR(1) states.
2. `PERS_<name>` and `VOL_<name>` are documented as persistence and innovation
   standard deviation. The estimation state-space builder uses
   `diag(VOL**2)` as the innovation covariance.
3. `linear_mod.impact_matrix` maps explicitly supplied innovations into linear
   IRFs and deterministic exogenous paths. `Model.finalize()` currently
   initializes this matrix to identity, so it does not automatically contain
   `VOL_*` scaling.

Phase 2 must not silently combine `VOL_*` and `linear_mod.impact_matrix`; doing
so could apply volatility twice. It must also avoid changing existing linear
IRF or deterministic-path behavior as a side effect of nonlinear solver work.

## Decisions

### Innovation convention

Quadrature nodes represent standardized structural innovations

$$
\epsilon_{t+1} \sim N(0, I).
$$

The exogenous law of motion is

$$
z_{t+1} = \Phi z_t + L\epsilon_{t+1},
$$

where `Phi` is the persistence matrix and `L` is the innovation-impact matrix.
The unconditional innovation covariance is therefore `L @ L.T`.

For model-derived nonlinear expectations, the default resolver uses:

- `Phi = diag(PERS_<name>)`;
- `L = diag(VOL_<name>)`.

This follows the documented meaning of `VOL_*` and the existing estimation
convention. A caller may pass explicit `Phi` and `L` matrices, including a
rectangular `L` for correlated or lower-dimensional structural innovations.
The resolver will **not** automatically use or multiply by
`linear_mod.impact_matrix`. A caller that intentionally wants that matrix must
pass it explicitly as `innovation_impact`.

This decision is local to stochastic integration. Phase 2 does not alter
`Model.finalize()`, `LinearModel`, `DetSpec`, or the interpretation of explicit
IRF shock sizes.

### Timing convention

For a current state `z`, each quadrature node produces a next-period state:

```text
conditional mean:  Phi @ z
node innovation:   L @ epsilon_k
next state:        Phi @ z + L @ epsilon_k
```

No `per + 1` indexing is involved; that convention belongs to construction of
whole deterministic paths. Phase 2 computes one conditional transition.

### NumPy/JAX boundary

Quadrature rule construction is a setup-time NumPy operation. Nodes and weights
are then converted once to immutable JAX arrays. The Phase 3 residual will
consume the JAX representation directly and must not rebuild rules inside JIT.

The exogenous transition kernel uses JAX because it runs for every collocation
point and quadrature node. It accepts matrices and rule nodes explicitly, making
it compatible with `jax.jit`, `jax.vmap`, and automatic differentiation.

### Zero-innovation models

A model with no stochastic innovations uses a deterministic rule containing
one node with shape `(1, 0)` and weight `[1.0]`. This lets expectation code keep
one uniform weighted-sum path rather than adding a special branch deep inside
the solver.

## Target files

```text
src/equilibrium/solvers/
├── quadrature.py              # NEW: rules, model resolver, JAX transition
└── __init__.py                # Export stable Phase 2 API

tests/
└── test_quadrature.py         # NEW: moments, shapes, sparse rule, JAX tests

docs/
├── nonlinear-solutions.md     # Mark Phase 2 complete after implementation
└── nonlinear-phase-2.md       # This implementation plan and status record
```

No new runtime dependency is required: NumPy, SciPy, and JAX are already
dependencies of Equilibrium. The core rule construction should use NumPy and
the standard library; SciPy may be used in tests for reference moments but is
not necessary for the one-dimensional rule itself.

## Proposed API

### Rule containers

```python
@dataclass(frozen=True)
class QuadratureRule:
    nodes: np.ndarray             # (n_nodes, dimension)
    weights: np.ndarray           # (n_nodes,)
    kind: str                     # "deterministic", "tensor", or "smolyak"
    orders: tuple[int, ...] | None # Per-dimension tensor orders; None for sparse
    level: int | None = None      # Smolyak level, otherwise None

    @property
    def dimension(self) -> int: ...

    @property
    def n_nodes(self) -> int: ...

    def integrate(self, values: np.ndarray, axis: int = 0) -> np.ndarray: ...

    def as_jax(self) -> JaxQuadratureRule: ...


class JaxQuadratureRule(NamedTuple):
    nodes: jax.Array
    weights: jax.Array
```

`QuadratureRule.__post_init__()` should validate and defensively copy its
arrays. Required invariants are:

- nodes are finite and two-dimensional;
- weights are finite and one-dimensional;
- node and weight counts agree;
- at least one node exists;
- weights sum to one within a documented tolerance;
- tensor and deterministic weights are nonnegative;
- sparse rules may contain negative combination weights;
- tensor `orders` has one entry per dimension, deterministic `orders` is empty,
  and Smolyak `orders` is `None` because it combines several 1D orders.

The NumPy arrays should be marked read-only after validation to make the frozen
container meaningful. `integrate()` applies the weight vector along an explicit
node axis and validates that the selected axis has length `n_nodes`.

`JaxQuadratureRule` contains arrays only and can naturally pass through JAX
transformations; it does not need custom static metadata.

### Constructors

```python
def deterministic_quadrature() -> QuadratureRule: ...


def gauss_hermite_normal(
    degree: int,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> QuadratureRule: ...


def tensor_gauss_hermite(
    degrees: int | Sequence[int],
    *,
    dimension: int | None = None,
    mu: float | Sequence[float] = 0.0,
    sigma: float | Sequence[float] = 1.0,
    max_nodes: int | None = 100_000,
) -> QuadratureRule: ...


def smolyak_gauss_hermite(
    dimension: int,
    level: int,
    *,
    mu: float | Sequence[float] = 0.0,
    sigma: float | Sequence[float] = 1.0,
    merge_tolerance: float = 1e-14,
    weight_tolerance: float = 1e-15,
    max_nodes: int | None = 1_000_000,
) -> QuadratureRule: ...
```

All constructors return nodes in rows. This differs from the old C++ helper's
`(dimension, n_nodes)` layout but matches the row-oriented state batches used by
Equilibrium and JAX.

`gauss_hermite_normal()` always returns nodes shaped `(degree, 1)`. Tensor and
Smolyak constructors accept scalar `mu`/`sigma` values, broadcast them across
dimensions, and also accept per-dimension sequences. `sigma` must be finite and
strictly positive. Correlation is not encoded here; it is introduced by the
explicit impact matrix `L`.

### Exogenous process data and transition

```python
@dataclass(frozen=True)
class ExogenousProcess:
    names: tuple[str, ...]
    persistence: np.ndarray       # (n_exog, n_exog)
    innovation_impact: np.ndarray # (n_exog, n_innovations)

    def as_jax(self) -> JaxExogenousProcess: ...


class JaxExogenousProcess(NamedTuple):
    persistence: jax.Array
    innovation_impact: jax.Array


def exogenous_process_from_model(
    model,
    *,
    persistence: np.ndarray | None = None,
    innovation_impact: np.ndarray | None = None,
) -> ExogenousProcess: ...


def next_exogenous_states_jax(
    process: JaxExogenousProcess,
    current_z: jax.Array,
    innovation_nodes: jax.Array,
) -> jax.Array: ...
```

Shape behavior for `next_exogenous_states_jax()` is fixed:

- `current_z.shape == (n_exog,)` returns `(n_nodes, n_exog)`;
- `current_z.shape == (n_batch, n_exog)` returns
  `(n_batch, n_nodes, n_exog)`;
- `innovation_nodes.shape == (n_nodes, n_innovations)`;
- `innovation_impact.shape == (n_exog, n_innovations)`.

The function performs no random draws and applies no additional volatility.
It should preserve float64 under Equilibrium's normal JAX configuration.

## Normalized one-dimensional Gauss-Hermite rule

The implementation follows `ghquad_norm()` in
`~/numerical/py_tools/numerical/core.py` exactly:

```python
x, w = np.polynomial.hermite.hermgauss(degree)
w = w / np.sum(w)
x = mu + np.sqrt(2.0) * sigma * x
```

`hermgauss()` integrates against `exp(-x**2)`. The `sqrt(2) * sigma` node
scaling and normalized probability weights transform it into a rule for
`N(mu, sigma**2)`:

$$
E[h(X)] \approx \sum_{k=1}^{n} w_k h(x_k),
\qquad \sum_k w_k = 1.
$$

Do not use `hermegauss()` and do not apply an additional `sqrt(2*pi)` factor.
For degree `n`, tests should verify exactness for normal polynomial moments
through degree `2n - 1`, allowing for floating-point tolerance.

## Tensor-product construction

For dimensions `d` with one-dimensional rules `(x_j, w_j)`, construct:

$$
\epsilon_{k_1,\ldots,k_d}
  = (x_{1,k_1},\ldots,x_{d,k_d}),
\qquad
w_{k_1,\ldots,k_d} = \prod_{j=1}^{d} w_{j,k_j}.
$$

Implementation requirements:

1. Normalize scalar/sequence degrees and distribution parameters.
2. Use `np.meshgrid(..., indexing="ij")` for deterministic ordering.
3. Flatten node coordinates into `(prod(degrees), dimension)`.
4. Form weight products with the same ordering.
5. Check the final weight sum; only renormalize roundoff-level drift.
6. Return the deterministic one-node rule when `dimension == 0`.
7. Reject configurations whose node count exceeds a conservative optional
   `max_nodes` guard, so an accidental high-dimensional tensor rule fails
   before allocating a huge array.

The default `max_nodes` should be documented and overridable. It is a safety
limit, not a numerical restriction.

## Smolyak Gauss-Hermite construction

Sparse quadrature must be implemented independently of
`equilibrium.approx.Index`: interpolation admissibility and quadrature
combination weights are different concepts.

Use a conventional non-nested Smolyak combination of normalized
one-dimensional Gauss-Hermite rules. To match the original C++
`sparseGHQuad()` call through `gqn_order`, define one-dimensional levels
`i >= 1` with linear growth

$$
m(i) = i,
$$

and, for user level `L >= 0` and dimension `d`, set `q = L + d`. Construct

$$
A(q,d) =
\sum_{q-d+1 \le |\mathbf{i}| \le q}
(-1)^{q-|\mathbf{i}|}
{d-1 \choose q-|\mathbf{i}|}
\bigotimes_{j=1}^{d} Q_{i_j},
$$

where every `i_j >= 1` and `Q_i` is the normalized degree-`m(i)` rule.
Under this convention, `level=0` reduces to the single origin node.

The old C++ `nwspgr` interface used a one-based sparse level `K`; the new
zero-based level is `L = K - 1`. Thus the old default `quad_level=2` corresponds
to `level=1` here. Record this mapping in the public docstring and low-level
regression tests so future solver configuration does not introduce an
off-by-one change in accuracy or node count.

Because Gauss-Hermite rules are not generally nested, construction must:

1. enumerate admissible positive level vectors directly;
2. build each tensor component with its signed combination coefficient;
3. concatenate component nodes and weights;
4. merge coincident nodes deterministically within `merge_tolerance`;
5. sum their signed weights;
6. drop only weights below `weight_tolerance`;
7. sort nodes lexicographically for reproducible output;
8. verify that the resulting weights sum to one.

Negative weights are expected and must not be clipped. Final normalization may
correct roundoff-level drift only; it must not hide an incorrect combination
formula. If the pre-normalization sum differs materially from one, raise an
error.

The first implementation should favor clarity over extreme performance.
Sparse rules are constructed once, and their correctness is more important
than setup speed.

## Model resolver behavior

`exogenous_process_from_model()` should:

1. Preserve `model.exog_list` ordering exactly.
2. For default persistence, require every `PERS_<name>` and construct a diagonal
   matrix.
3. For default innovation impact, require every `VOL_<name>` and construct a
   diagonal matrix.
4. Accept an explicit full persistence matrix.
5. Accept an explicit rectangular innovation-impact matrix.
6. Validate finite values and compatible dimensions.
7. Reject duplicate exogenous names.
8. Support zero exogenous variables with `(0, 0)` matrices.
9. Avoid requiring `model.linearize()` or reading `linear_mod`.
10. Never mutate model parameters or solver state.

This resolver centralizes stochastic-integration semantics without changing the
existing model API. A future general shock-process specification can replace it
behind the same `ExogenousProcess` contract.

## Detailed work plan

### Work package 1: Define containers and invariants

1. Add `quadrature.py` with `QuadratureRule`, `JaxQuadratureRule`,
   `ExogenousProcess`, and `JaxExogenousProcess`.
2. Implement shape, finiteness, weight-sum, mutability, and metadata checks.
3. Implement `integrate()` and NumPy-to-JAX conversion methods.
4. Add deterministic zero-dimensional quadrature.
5. Test invalid construction paths and read-only NumPy arrays.

Exit criterion: rule and process data have one documented, immutable shape
contract in both NumPy and JAX.

### Work package 2: Implement normalized one-dimensional rules

1. Implement `gauss_hermite_normal()` using `hermgauss()`.
2. Match `ghquad_norm()` node scaling and weight normalization.
3. Validate degree, mean, and standard deviation inputs.
4. Test weight sums, symmetry, shifted means, variances, and polynomial
   exactness through `2n - 1`.
5. Compare representative rules directly with `ghquad_norm()` when the local
   reference project is available; keep independent expected-moment tests for
   CI.

Exit criterion: the one-dimensional rule reproduces the requested normal
distribution and the established reference convention.

### Work package 3: Implement tensor-product rules

1. Normalize scalar and per-dimension degrees, means, and standard deviations.
2. Implement deterministic Cartesian-product ordering.
3. Add the allocation guard.
4. Test node counts, ordering, marginal moments, cross moments, anisotropic
   degrees, and nonstandard independent normals.
5. Test the zero-dimensional deterministic case.

Exit criterion: tensor rules integrate constants, marginal means/variances, and
supported mixed polynomial moments to expected tolerance.

### Work package 4: Implement and validate sparse rules

1. Implement positive level-vector enumeration and Smolyak coefficients.
2. Construct signed tensor components using the Phase 2 one-dimensional rule.
3. Merge duplicate nodes and remove negligible weights deterministically.
4. Test levels 0 through at least 3 in dimensions 1 through 4.
5. Verify constants, zero odd moments, covariance, fourth moments, and selected
   mixed moments at levels with sufficient exactness.
6. Compare node counts and smooth-function accuracy with tensor rules; sparse
   rules should use fewer nodes in representative dimensions, without asserting
   a universal speed or accuracy advantage.
7. Add regression fixtures for exact low-level nodes and weights in one and two
   dimensions.

Exit criterion: the sparse rule has a verified combination formula, normalized
signed weights, deterministic output, and documented level semantics.

### Work package 5: Implement model scaling and JAX transitions

1. Implement `exogenous_process_from_model()` with the `PERS_*`/`VOL_*`
   convention and explicit overrides.
2. Implement `next_exogenous_states_jax()` for single and batched current
   states.
3. Verify eager/JIT parity and `vmap` compatibility.
4. Test diagonal volatility, correlated explicit impacts, rectangular impacts,
   non-diagonal persistence, and zero-dimensional states.
5. Test conditional means and covariances implied by quadrature nodes.
6. Test gradients with respect to current states, persistence, and impact
   matrices.
7. Confirm no double scaling and no model mutation.

Exit criterion: model-derived and explicitly specified processes produce the
correct conditional state distribution through a pure JAX kernel.

### Work package 6: Export, document, and run regression checks

1. Export the stable Phase 2 API from `equilibrium.solvers`; do not add all
   names to the root `equilibrium` namespace yet.
2. Add a quadrature section to `docs/function-approximation.md` or a focused
   quadrature guide with tensor, sparse, and model-derived examples.
3. Mark Phase 2 complete in `docs/nonlinear-solutions.md` and record any API
   deviations in this file.
4. Confirm no new dependency was introduced.
5. Run targeted tests, the full test suite, Ruff, Black, and MyPy when
   available.

Exit criterion: Phase 3 can import stable rule/process objects and consume their
JAX arrays without depending on private implementation details.

## Test matrix

Create `tests/test_quadrature.py` with deterministic, seeded tests covering:

### Container tests

- valid scalar/vector integration;
- arbitrary integration axis;
- mismatched node and weight counts;
- nonfinite inputs;
- invalid weight sums;
- negative tensor weights rejected;
- negative sparse weights retained;
- NumPy arrays cannot be mutated;
- JAX conversion preserves values and float64 dtype.

### One-dimensional tests

- degrees 1, 2, 3, 5, and 10;
- weights sum to one and are positive;
- standard-normal symmetry;
- nonzero mean and nonunit standard deviation;
- moments through the supported polynomial degree;
- parity with `ghquad_norm()` for representative inputs.

### Tensor tests

- dimensions 0 through 4;
- isotropic and anisotropic degrees;
- node count equals the degree product;
- weights equal Cartesian products in the documented order;
- independent covariance and mixed moments;
- allocation guard failures.

### Sparse tests

- dimensions 1 through 4 and levels 0 through 3;
- constant integration and normalized signed weights;
- deterministic node ordering and duplicate merging;
- symmetry and odd moments;
- covariance and selected fourth/mixed moments when the level supports them;
- low-level regression fixtures;
- representative node-count comparison with tensor rules.

### Process and JAX tests

- `PERS_*` and `VOL_*` resolution in `exog_list` order;
- missing, duplicate, nonfinite, and wrongly shaped process data;
- explicit full persistence and rectangular impact matrices;
- single/batched current-state shapes;
- eager/JIT/vmap parity;
- conditional moments;
- derivatives with respect to state and matrices;
- zero-exogenous-state behavior;
- no dependence on `linear_mod.impact_matrix` unless explicitly supplied.

## Validation commands

```bash
pytest tests/test_quadrature.py
pytest tests/test_approx_*.py tests/test_quadrature.py
ruff check src/equilibrium/solvers/quadrature.py tests/test_quadrature.py
black --check src/equilibrium/solvers/quadrature.py tests/test_quadrature.py
mypy src/equilibrium/solvers/quadrature.py
```

After targeted checks:

```bash
pytest
ruff check src/equilibrium tests
```

## Acceptance criteria

Phase 2 is complete when:

- normalized one-dimensional rules match `ghquad_norm()`;
- every rule uses nodes shaped `(n_nodes, dimension)` and weights shaped
  `(n_nodes,)`;
- tensor rules correctly integrate supported independent-normal moments;
- sparse rules use a documented Smolyak formula, retain legitimate negative
  weights, and pass exactness/regression tests;
- zero-dimensional stochastic integration reduces to one deterministic node;
- the default model resolver uses `PERS_*` and `VOL_*` exactly once;
- explicit persistence and innovation-impact overrides are supported;
- no Phase 2 code silently reads or composes `linear_mod.impact_matrix`;
- next-period state construction works eagerly and under JIT/vmap;
- JAX transition gradients are correct;
- NumPy/JAX rule data agree and preserve float64;
- no new runtime dependency is added;
- all new tests and the full Equilibrium suite pass;
- Ruff and Black pass, and MyPy passes when available.

## Risks and mitigations

### Shock scaling remains ambiguous across solver families

Mitigation: document the nonlinear expectation convention explicitly, keep its
resolver separate, require explicit overrides for alternative impact matrices,
and do not mutate existing linear/deterministic behavior.

### Volatility is applied twice

Mitigation: quadrature constructors generate standardized innovations for
solver use; only `ExogenousProcess.innovation_impact` scales them. Tests should
use nonunit volatility and assert the resulting covariance exactly.

### Tensor node counts explode

Mitigation: compute the requested count before allocation and enforce an
overridable `max_nodes` guard. Solver configuration should prefer sparse rules
as dimension grows.

### Sparse combination weights are implemented incorrectly

Mitigation: keep interpolation indices out of the implementation, state the
formula and level convention in docstrings, test low-level fixtures, and verify
normal moments independently of the implementation.

### Duplicate floating-point nodes do not merge reliably

Mitigation: generate rules deterministically, use a documented tolerance-based
key, sum weights before dropping negligible entries, and test symmetry and
repeatability across runs.

### Negative sparse weights are mistaken for invalid probabilities

Mitigation: encode rule kind in the container, permit signed weights only for
Smolyak rules, and test that signed weights are preserved rather than clipped.

### JAX recompiles for every rule

Mitigation: construct and convert rules once, keep node-array shapes stable for
a given solver configuration, and pass arrays through the residual rather than
recreating Python rule objects inside JIT.

## Handoff to Phase 3

Phase 3 should receive:

```python
rule = tensor_gauss_hermite(n_quad, dimension=n_innovations)
jax_rule = rule.as_jax()

process = exogenous_process_from_model(model).as_jax()
z_next = next_exogenous_states_jax(process, z_current, jax_rule.nodes)
```

The residual can then evaluate next-period policies at every row of `z_next`,
compute expectation integrands, and contract the quadrature axis with
`jax_rule.weights`. It should not read `VOL_*`, construct grids, or reinterpret
shock scaling itself.
