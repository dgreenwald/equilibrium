# Function approximation

`equilibrium.approx` provides tensor-product and Smolyak sparse-grid function
approximation. It includes Chebyshev polynomial and hat bases, NumPy fitting and
evaluation, and a stateless Chebyshev evaluation path for JAX-compiled solvers.

The package is self-contained; it does not require the external `funcapprox`
project. Its upstream source and adaptation history are recorded in
`src/equilibrium/approx/UPSTREAM.md`.

## NumPy workflow

Use a preset builder to construct a grid and approximation domain, evaluate the
target at the grid points, and fit its coefficients:

```python
import numpy as np

from equilibrium.approx import make_smolyak_chebyshev

approx = make_smolyak_chebyshev(
    dimension=2,
    max_levels=(3, 3),
    max_total_level=3,
    lb=np.array([-2.0, 0.5]),
    ub=np.array([2.0, 4.0]),
)

grid = approx.get_grid_points()
values = np.column_stack(
    (
        grid[:, 0] ** 2 + grid[:, 1],
        np.exp(0.1 * grid[:, 0]) * grid[:, 1],
    )
)
approx.fit(values)

# One point -> shape (2,); a batch would return (n_points, 2).
value = approx.evaluate(np.array([0.25, 2.0]))
```

`Function.fit()` and the grid/index construction routines intentionally use
NumPy. They run once during setup and may store fitted coefficients on the
mutable `Function` wrapper. Hat basis families are supported through this
NumPy API.

Available preset families include:

- `make_smolyak_chebyshev`
- `make_smolyak_hierarchical_hat`
- `make_smolyak_modified_hat`
- `make_tensor_chebyshev`
- `make_tensor_uniform_hat`
- `make_tensor_modified_hat`

`make_funcapprox(name, ...)` and `create_approximation(...)` provide named
dispatch when configuration selects a family dynamically.

## Stateless JAX evaluation

Repeated solver evaluation uses a separate functional interface:

```python
import jax
import jax.numpy as jnp

from equilibrium.approx import evaluate_jax, make_jax_data

data = make_jax_data(approx)
coefficients = jnp.asarray(approx.coefficients)
points = jnp.array([[0.25, 2.0], [1.0, 3.5]])

evaluate_compiled = jax.jit(evaluate_jax)
values = evaluate_compiled(data, coefficients, points)

# Coefficients are explicit, so solver residuals can differentiate through them.
coefficient_jacobian = jax.jacfwd(evaluate_jax, argnums=1)(
    data, coefficients, points
)
```

`make_jax_data()` converts the immutable bounds and sparse basis-index table
once. `evaluate_jax(data, coefficients, points)` then performs coordinate
transformation, Chebyshev recurrence, sparse basis assembly, and coefficient
multiplication entirely with JAX operations.

The explicit coefficient argument is important: JIT-compiled code and
automatic differentiation do not read or mutate `Function.coefficients`.
`JaxApproximationData` is an immutable JAX PyTree whose dimensions and basis
counts are static compilation metadata.

The JAX path supports:

- scalar coefficients shaped `(n_basis,)`;
- vector coefficients shaped `(n_basis, n_outputs)`;
- one point shaped `(dimension,)`;
- batched points shaped `(n_eval, dimension)`;
- `jax.jit`, `jax.vmap`, `jax.jacfwd`, and `jax.jacrev`.

## Gaussian quadrature

`equilibrium.solvers` provides setup-time NumPy quadrature rules for nonlinear
expectations. All rules store nodes in rows with shape
`(n_nodes, dimension)` and normalized weights with shape `(n_nodes,)`:

```python
from equilibrium.solvers import (
    gauss_hermite_normal,
    smolyak_gauss_hermite,
    tensor_gauss_hermite,
)

one_dimensional = gauss_hermite_normal(5)
tensor = tensor_gauss_hermite(3, dimension=2)
sparse = smolyak_gauss_hermite(dimension=4, level=2)

# Values may have additional axes; select the quadrature-node axis explicitly.
expectation = tensor.integrate(values, axis=0)
```

`gauss_hermite_normal(n, mu=..., sigma=...)` integrates against a normal
distribution and is exact for univariate polynomials through degree
`2*n - 1`. Tensor rules accept isotropic or per-dimension degrees and are best
suited to a small number of independent innovations. Their default 100,000-node
allocation guard can be changed with `max_nodes`.

The Smolyak constructor uses non-nested Gauss-Hermite rules with linear growth:
one-dimensional level `i` has degree `i`. Its public level is zero-based;
`level=0` is the mean node, and the legacy C++ `nwspgr` level is
`K = level + 1`. Sparse combination weights may be negative and must not be
treated as probabilities or clipped. The default allocation guard applies to
raw component nodes before coincident nodes are merged.

Rule construction stays outside JIT. Convert a completed rule once and pass its
array-only representation into compiled code:

```python
import jax
import jax.numpy as jnp

from equilibrium.solvers import (
    exogenous_process_from_model,
    next_exogenous_states_jax,
)

jax_rule = tensor.as_jax()
process = exogenous_process_from_model(model).as_jax()
z_current = jnp.zeros(len(model.exog_list))

z_next = jax.jit(next_exogenous_states_jax)(
    process, z_current, jax_rule.nodes
)
expected_value = jnp.tensordot(jax_rule.weights, integrand(z_next), axes=1)
```

Quadrature nodes used with a model represent standardized structural
innovations. By default, `exogenous_process_from_model()` constructs
`Phi = diag(PERS_<name>)` and `L = diag(VOL_<name>)`, then applies
`z_next = Phi @ z + L @ epsilon`. Explicit full `Phi` and rectangular `L`
matrices are supported. The resolver never reads or composes
`linear_mod.impact_matrix`, preventing accidental double application of shock
volatility. A model without exogenous innovations uses
`deterministic_quadrature()`, whose single node has shape `(1, 0)`.

## Current boundaries

- JAX evaluation currently supports Chebyshev schemes only. Passing a hat or
  mixed-basis scheme to `make_jax_data()` raises `NotImplementedError`.
- Points outside the declared domain are evaluated by polynomial
  extrapolation. Callers—particularly nonlinear solvers—must detect or control
  out-of-domain states explicitly.
- Grid construction, basis inversion, and fitting are not JIT-compiled.
- The first JIT call includes compilation cost. JAX is intended for repeated or
  batched evaluations and differentiation, not isolated single-point calls.

## Development checks

The approximation tests are isolated under `tests/test_approx_*.py`:

```bash
pytest tests/test_approx_*.py
pytest tests/test_quadrature.py
ruff check src/equilibrium/approx tests/test_approx_*.py
```

The JAX tests compare tensor and Smolyak schemes across one to three dimensions
against the preserved NumPy implementation, including gradients, JIT, `vmap`,
shape behavior, float64 operation, and immutability.
