# Nonlinear Solutions Phase 1: Function Approximation Infrastructure

## Purpose

Phase 1 introduces an in-tree `equilibrium.approx` package derived from
`funcapprox` and establishes the JAX-compatible evaluation primitive needed by
the later collocation solver. This phase does **not** implement quadrature,
collocation residuals, nonlinear solution algorithms, simulation, or nonlinear
IRFs.

The implementation should preserve the existing NumPy behavior of
`funcapprox` while adding a small, stateless JAX evaluation layer. The solver
hot path must take coefficients as an explicit argument so JAX can differentiate
residuals with respect to them; it must not depend on mutating a `Function`
object inside traced code.

## Source baseline and provenance

The initial port is based on the local repository:

- Source: `~/dev/funcapprox`
- Source commit: `4efed5bb24c78c9196f69f99ead7b9744ec63977`
- Source package version: `0.1.0`
- Approximate runtime source size: 2,300 lines excluding `benchmark/`

The source has a license metadata discrepancy: `funcapprox/pyproject.toml`
declares MIT, while `funcapprox/LICENSE` contains the GPL version 3 text. The
author confirmed on 2026-08-28 that GPL governs the port, consistent with
Equilibrium's GPL-3.0-only license. Do not describe the port as MIT-licensed
unless that decision is explicitly changed.

Add `src/equilibrium/approx/UPSTREAM.md` containing:

- upstream repository and commit;
- upstream version;
- confirmed license;
- port date;
- files included and excluded;
- material changes made for Equilibrium;
- instructions for comparing or synchronizing a future upstream revision.

Ensure `UPSTREAM.md` and `py.typed` are included in wheels and source
distributions. Retain author attribution in the provenance file rather than
adding repetitive notices to every source file.

## Scope

### Included

- Grid construction and validation.
- Smolyak and tensor-product level/index construction.
- Chebyshev and hat basis families.
- Sparse basis matrix construction and collocation fitting.
- Coordinate transformation between user and canonical grid domains.
- Scalar- and vector-valued `Function` fitting and NumPy evaluation.
- Existing gradient-evaluation behavior in the NumPy API.
- Existing approximation presets, with names adapted to the new namespace.
- A stateless JAX kernel for Chebyshev basis and policy evaluation.
- JIT, batching, and automatic differentiation tests for the JAX kernel.
- Ported unit tests for all included NumPy functionality.

### Explicitly deferred

- `funcapprox.benchmark`, benchmark functions, plotting, and diagnostics.
- JAX kernels for the hat basis families.
- JAX-based grid construction or fitting.
- A general runtime-selectable NumPy/JAX backend abstraction.
- Adaptive grids, domain expansion, and out-of-domain policy decisions.
- Sparse quadrature; interpolation indices must not be reused as quadrature
  weights.
- Collocation specifications, solver result types, and changes to `Model`.

Hat bases remain fully supported by the NumPy API. Requesting JAX evaluation for
a non-Chebyshev scheme should raise a clear `NotImplementedError` during Phase
1, rather than falling back to NumPy inside traced code.

## Target package layout

```text
src/equilibrium/approx/
├── __init__.py
├── py.typed
├── UPSTREAM.md
├── presets.py
├── jax_eval.py                 # NEW: stateless JAX evaluation API
├── bases/
│   ├── __init__.py
│   ├── base.py
│   ├── chebyshev.py
│   └── hat.py
├── grids/
│   ├── __init__.py
│   ├── base.py
│   ├── chebyshev.py
│   └── uniform.py
├── levels/
│   ├── __init__.py
│   ├── base.py
│   ├── smolyak.py
│   └── tensor.py
└── core/
    ├── __init__.py
    ├── index.py
    ├── scheme.py
    └── function.py
```

Use `levels/tensor.py` as in the actual upstream tree; do not merge
`TensorProductLevels` into `smolyak.py`. Exclude the entire upstream
`benchmark/` directory and its tests.

## Public API

### Preserved NumPy API

`equilibrium.approx` should re-export the included upstream API:

- `Grid1d`, `ChebyshevLobattoGrid1d`, `UniformGrid1d`, and
  `UniformGridWithBoundary1d`;
- `Basis1d`, `ChebyshevBasis1d`, and all existing hat basis classes;
- `Levels`, `SmolyakLevels`, `SmolyakInteriorLevels`, and
  `TensorProductLevels`;
- `Index`, `IndexBlock`, `Scheme`, and `Function`;
- all existing tensor and Smolyak preset builders;
- `make_funcapprox`, `create_approximation`, and
  `VALID_FUNCAPPROX_NAMES`.

Do not re-export a `benchmark` name. Update internal imports from `funcapprox`
to relative `equilibrium.approx` imports, avoiding circular imports through the
top-level package where possible.

The port should initially preserve upstream method names, argument conventions,
shape behavior, exceptions, and return types. In particular:

- `Function.fit(values)` remains a NumPy setup-time operation and continues to
  set `Function.coefficients`.
- `Function.evaluate(points)` remains the compatibility-oriented NumPy method.
- Single-point scalar evaluation may continue returning a Python scalar.
- Vector-valued coefficients retain shape `(n_basis, n_outputs)`.
- `Scheme.construct()` continues to precompute the collocation grid and basis
  inverse with NumPy.

### New JAX API

Add a separate functional API in `equilibrium.approx.jax_eval`. The exact names
may be refined during implementation, but the intended boundary is:

```python
class JaxApproximationData(NamedTuple):
    basis_indices: jax.Array       # (n_basis, dimension), integer
    lower_bounds: jax.Array        # (dimension,)
    upper_bounds: jax.Array        # (dimension,)
    canonical_lower: jax.Array     # (dimension,)
    canonical_upper: jax.Array     # (dimension,)
    n_basis_1d: int                # static maximum basis count
    dimension: int                 # static state dimension


def make_jax_data(function: Function) -> JaxApproximationData: ...


def evaluate_bases_jax(
    data: JaxApproximationData,
    points: jax.Array,
) -> jax.Array: ...


def evaluate_jax(
    data: JaxApproximationData,
    coefficients: jax.Array,
    points: jax.Array,
) -> jax.Array: ...
```

The key contract is that `coefficients` is explicit and may be a JAX tracer.
`evaluate_jax` must not read or modify `Function.coefficients`. This permits the
Phase 3 residual to have the pure signature
`residual(coefficients, model_data) -> residuals`.

`make_jax_data()` is a setup-time adapter. It should:

1. Validate that the scheme is constructed.
2. Validate that every basis dimension uses `ChebyshevBasis1d`.
3. Reshape the flat upstream `basis_ix` array to
   `(n_basis, dimension)`.
4. convert required numeric arrays to JAX arrays once;
5. retain shape-defining integers as static Python data;
6. raise `NotImplementedError` for hat or mixed-basis schemes.

If a `NamedTuple` containing Python integers causes unwanted dynamic tracing,
replace it with a small frozen dataclass registered as a JAX PyTree, placing
shape metadata in auxiliary/static fields. Do not make the full mutable
`Function`, `Scheme`, or `Index` object a PyTree.

## NumPy/JAX boundary

### NumPy setup path

The following remain NumPy-only because they run outside iterative solver hot
paths:

- level and index enumeration;
- grid construction;
- basis-index construction;
- collocation basis matrix construction;
- basis inversion;
- fitting coefficients from nodal values;
- input validation and preset dispatch;
- optional analytical gradient helpers used by the standalone NumPy API.

### JAX evaluation path

The JAX kernel should implement only operations needed repeatedly by a solver:

1. Normalize user points to the canonical Chebyshev interval.
2. Evaluate all one-dimensional Chebyshev polynomials needed in each dimension.
3. Gather the indexed polynomial for every sparse basis term.
4. Multiply across dimensions to form the sparse basis matrix.
5. Multiply the basis matrix by scalar- or vector-valued coefficients.

For batched points with shape `(n_eval, dimension)`, an efficient layout is:

```text
one-dimensional values: (n_eval, dimension, n_basis_1d)
basis index table:        (n_sparse_basis, dimension)
sparse basis matrix:      (n_eval, n_sparse_basis)
coefficients:             (n_sparse_basis,) or
                          (n_sparse_basis, n_outputs)
result:                   (n_eval,) or (n_eval, n_outputs)
```

Use the Chebyshev recurrence

```text
T_0(x) = 1
T_1(x) = x
T_n(x) = 2 x T_{n-1}(x) - T_{n-2}(x)
```

implemented with JAX array updates and a static loop or `jax.lax.fori_loop`.
Avoid Python mutation of traced arrays. Gather basis values with JAX indexing or
`take_along_axis`, then reduce with `jnp.prod` across the dimension axis.

Support both point shapes:

- `(dimension,)` for a single point;
- `(n_eval, dimension)` for a batch.

Preserve a predictable JAX shape contract: single-point vector policies return
`(n_outputs,)`, and batched vector policies return `(n_eval, n_outputs)`. Do not
convert traced scalar results to Python `float`.

Gradients in the JAX path should come from `jax.jacfwd`, `jax.jacrev`, or
`jax.grad`; do not port the manual gradient formulas merely for the collocation
solver. The existing NumPy gradient API remains available independently.

## Detailed work plan

### Work package 1: Freeze the upstream baseline

1. Confirm the upstream tree is clean or record any uncommitted changes that
   must be included.
2. Resolve the license discrepancy.
3. Record the source commit and a file manifest in `UPSTREAM.md`.
4. Run the upstream non-benchmark tests in the upstream environment and record
   the baseline result. If a baseline test fails, document it rather than
   changing behavior silently during the port.

Exit criterion: provenance, license, source commit, and baseline test status are
unambiguous.

Status: **complete on 2026-08-28**. The upstream worktree was clean, GPLv3 was
confirmed, provenance and source hashes were recorded in
`src/equilibrium/approx/UPSTREAM.md`, and all 146 non-benchmark upstream tests
passed under Python 3.13.5 and pytest 8.4.1.

### Work package 2: Port the NumPy package without behavioral changes

1. Copy `bases/`, `grids/`, `levels/`, `core/`, `presets.py`, and `py.typed`.
2. Create the adapted package `__init__.py` without the `benchmark` import.
3. Rewrite imports to the new namespace, preferring relative imports.
4. Apply Black's 88-column formatting only after the mechanical port is
   complete, so semantic differences remain reviewable.
5. Update Hatch build configuration if necessary to guarantee inclusion of
   `py.typed` and `UPSTREAM.md`.
6. Do not export approximation names from `equilibrium.__init__` during Phase
   1; users should import from `equilibrium.approx`. This avoids expanding the
   root namespace before the API stabilizes.

Exit criterion: the package imports without `funcapprox` installed and exposes
the intended NumPy API.

Status: **complete on 2026-08-28**. The frozen runtime tree was copied into
`equilibrium.approx`, imports were rewritten to the in-tree namespace, the
benchmark export was removed, Black formatting and Ruff checks passed, and an
API smoke test confirmed that no external `funcapprox` import is required.
`py.typed` and `UPSTREAM.md` were added explicitly to the wheel include list.

### Work package 3: Port and organize upstream tests

Port tests into module-aligned files:

```text
tests/test_approx_bases.py
tests/test_approx_function.py
tests/test_approx_grids.py
tests/test_approx_index.py
tests/test_approx_levels.py
tests/test_approx_presets.py
tests/test_approx_scheme.py
```

Change imports only; preserve assertions and tolerances initially. Exclude
`test_benchmark.py` and `test_functions.py`, which cover the deferred benchmark
package. Keep tests for all hat families even though their JAX kernels are
deferred, because they remain part of the NumPy public API.

Add one packaging/import test that confirms:

- `equilibrium.approx` imports;
- no `funcapprox` module is required;
- no benchmark symbols are exported;
- the public `__all__` contains the documented API.

Exit criterion: all ported NumPy tests pass in Equilibrium with no changes to
expected numerical results.

Status: **complete on 2026-08-28**. All seven non-benchmark upstream test
modules were ported with namespace and formatting changes only. Their 146 tests
pass unchanged, and three Equilibrium-specific package tests verify the exact
public API, benchmark exclusion, and absence of external `funcapprox` imports
for a total of 149 passing approximation tests.

### Work package 4: Implement the stateless Chebyshev JAX kernel

1. Add `jax_eval.py` with coordinate transformation, one-dimensional
   Chebyshev recurrence, sparse basis assembly, and coefficient evaluation.
2. Add `make_jax_data()` to convert a constructed Chebyshev `Function` into
   immutable solver data.
3. Keep coefficients out of `JaxApproximationData`; pass them explicitly to
   every evaluation.
4. Validate point dimension and coefficient leading dimension at the eager API
   boundary. Avoid data-dependent Python validation inside JIT.
5. Raise a targeted `NotImplementedError` for hat and mixed schemes.
6. Confirm both scalar- and vector-valued coefficient arrays work.
7. Document that Phase 1 evaluation extrapolates according to the Chebyshev
   polynomial outside the supplied domain; later solver code must detect or
   control out-of-domain states explicitly.

Exit criterion: Chebyshev evaluation is pure, JIT-compatible, batch-compatible,
and differentiable with respect to both points and coefficients.

Status: **complete on 2026-08-28**. `jax_eval.py` now provides an immutable
PyTree adapter, stateless sparse Chebyshev basis assembly, and scalar/vector
evaluation with explicit coefficients. Eager and JIT evaluation match the
NumPy implementation; automatic differentiation works for points and
coefficients; invalid shapes fail clearly; and hat schemes are rejected at the
adapter boundary. Focused contract tests were added ahead of the broader work
package 5 parity matrix.

### Work package 5: JAX parity and transformation tests

Add `tests/test_approx_jax.py` covering:

1. NumPy/JAX basis-matrix parity for one-, two-, and three-dimensional
   Chebyshev schemes.
2. Tensor-product and Smolyak schemes.
3. Single and batched points.
4. Scalar- and vector-valued coefficients.
5. Non-canonical user domains, including asymmetric bounds.
6. `jax.jit(evaluate_jax)` execution.
7. `jax.vmap` over single-point evaluation.
8. `jax.jacfwd` or `jax.jacrev` with respect to points, checked against the
   NumPy analytical gradient or centered finite differences away from domain
   boundaries.
9. Differentiation with respect to coefficients, checked against the basis
   matrix.
10. A regression test showing that changing explicit coefficients changes the
    result without mutating the source `Function`.
11. Clear rejection of hat and mixed-basis schemes.
12. Float64 results under Equilibrium's normal JAX configuration.

Use deterministic points and tight tolerances appropriate for 64-bit mode. Do
not rely on randomized test inputs unless they are seeded.

Exit criterion: every required JAX transformation succeeds, and NumPy/JAX
values agree to the selected tolerance.

Status: **complete on 2026-08-28**. The JAX matrix now covers one through three
dimensions, tensor-product and Smolyak schemes, single and batched points,
scalar and vector outputs, asymmetric domains, JIT, `vmap`, point and
coefficient derivatives, explicit-coefficient immutability, float64 behavior,
and rejection of hat and mixed-basis schemes. All 19 JAX-specific tests and all
168 approximation tests pass.

### Work package 6: Documentation and integration cleanup

1. Add a short approximation section to the project README or a focused
   `docs/function-approximation.md` showing NumPy fitting and JAX evaluation.
2. Update `AGENTS.md` to mention `approx/` in the package structure.
3. Update `docs/nonlinear-solutions.md` if final names differ from this plan.
4. Confirm no `funcapprox` dependency was added to `pyproject.toml` or
   requirements files.
5. Check the built wheel contents for `py.typed` and `UPSTREAM.md`.

Exit criterion: a contributor can discover the package, reproduce its source
provenance, and understand which portions are JAX-compatible.

Status: **complete on 2026-08-28**. The focused function-approximation guide
documents NumPy fitting, stateless JAX evaluation, supported shapes, and current
boundaries. README and contributor guidance reference the new package, the
nonlinear roadmap reflects the implemented split API, and no external
`funcapprox` dependency is present. Wheel and source-distribution inspection
confirmed that `py.typed`, `UPSTREAM.md`, and `jax_eval.py` are shipped. The
documentation example passed, Ruff passed, and the full repository suite passed
all 813 tests.

MyPy was not installed in the active environment, so the targeted MyPy command
could not be executed. This is an environment-level outstanding check rather
than a known type-check failure.

## Validation commands

Run targeted checks while developing:

```bash
pytest tests/test_approx_bases.py \
       tests/test_approx_function.py \
       tests/test_approx_grids.py \
       tests/test_approx_index.py \
       tests/test_approx_levels.py \
       tests/test_approx_presets.py \
       tests/test_approx_scheme.py \
       tests/test_approx_jax.py

ruff check src/equilibrium/approx tests/test_approx_*.py
black --check src/equilibrium/approx tests/test_approx_*.py
mypy src/equilibrium/approx
```

Then run repository-level regression checks:

```bash
pytest
ruff check src/equilibrium tests
```

Finally build and inspect distributions:

```bash
python -m build
python -m zipfile -l dist/equilibrium-*.whl
```

The build check may require adding `build` to the development extras or using an
environment where it is already installed. Do not add a runtime dependency for
distribution inspection.

## Acceptance criteria

Phase 1 is complete when all of the following hold:

- `equilibrium.approx` works without the external `funcapprox` package.
- The upstream source commit and confirmed license are documented.
- All included upstream NumPy tests pass after namespace-only adaptation.
- All NumPy approximation presets retain their existing behavior.
- `benchmark/` is not shipped or exposed.
- A Chebyshev approximation can be evaluated from explicit coefficients with
  `jax.jit` and `jax.vmap`.
- JAX evaluation is differentiable with respect to points and coefficients.
- NumPy and JAX evaluation agree for scalar and vector outputs on tensor and
  Smolyak schemes.
- Hat schemes continue to work through NumPy and fail clearly if passed to the
  JAX adapter.
- The JAX hot path performs no NumPy conversion, Python scalar conversion, or
  mutation of `Function.coefficients`.
- `py.typed` and provenance metadata appear in the built wheel.
- The full Equilibrium test suite has no regressions.
- Ruff passes for the package and tests, and MyPy passes for the new package.

## Risks and mitigations

### Upstream license ambiguity

Mitigation: resolve before copying, then record the answer and exact source
commit in the repository.

### Mechanical port and JAX changes become entangled

Mitigation: land or review the namespace-only port separately from
`jax_eval.py`. Preserve upstream tests before introducing new behavior.

### Static metadata causes JAX recompilation or tracer errors

Mitigation: keep dimension and basis counts static, numeric tables immutable,
and coefficients explicit. Use a registered PyTree only if the simple
`NamedTuple` design does not separate static and dynamic fields adequately.

### NumPy compatibility regresses when adding JAX

Mitigation: do not replace NumPy throughout the existing classes. Add a
separate JAX kernel and compare it against the preserved NumPy implementation.

### Sparse basis assembly is slow

Mitigation: reshape the basis index table once during setup, vectorize gathers
and products, and avoid Python loops over sparse terms in the traced function.
Benchmark only after correctness and transformation tests pass.

### Hat-basis support expands the initial JAX scope

Mitigation: retain hat bases in the standalone NumPy API, but reject them at the
JAX adapter boundary until a later phase supplies dedicated kernels and
nondifferentiability conventions.

## Phase 1 handoff to later phases

Phase 2 should consume only the public approximation interfaces established
here. Phase 3 should build residuals around the stateless call:

```python
controls = evaluate_jax(approx_data, coefficients, states)
```

It should not depend on `Function.fit()`, mutate a policy object, or reach into
private `Scheme` or `Index` fields. If the collocation design cannot be
implemented through this boundary, revise the public JAX data contract before
adding solver-specific workarounds.
