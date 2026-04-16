# Plan: Port MCMC Estimation into Equilibrium

## New package layout

```
src/equilibrium/estimation/
    __init__.py          # public API re-exports
    prior.py             # port of py_tools/bayesian/prior.py
    mcmc.py              # port of py_tools/bayesian/mcmc.py (MonteCarlo + RWMC only)
    state_space.py       # port of py_tools/time_series/state_space.py
    numerical.py         # vendored helpers: hessian, bound_transform, robust_cholesky, rsolve
    likelihood.py        # build StateSpaceModel from a linearized equilibrium.Model
    estimate.py          # high-level entry point, EstimParam types, estimate(...)
    io.py                # save/load estimation outputs under settings.paths.save_dir
```

Public surface (added to `equilibrium/__init__.py`):
`Prior`, `RWMC`, `EstimParam`, `estimate`, `EstimationResult`, `load_estimation`.

## Step 1 — Vendor minimal numerical helpers (`numerical.py`)

Inline the four functions we need (~60 lines total), no `py_tools` dependency:

- `hessian(f, x, eps)` — central-difference Hessian.
- `bound_transform(vals, lb, ub, to_bdd)` — logit/logistic + one-sided exp transforms.
- `robust_cholesky(A, min_eig)` — eigen-decomp with clipped eigenvalues.
- `rsolve(b, A)` — right-side solve used by the Kalman filter.

## Step 2 — Port `prior.py` verbatim

Copy `py_tools/bayesian/prior.py` to `estimation/prior.py` as-is. No dependencies beyond NumPy + SciPy (already required). Keep `get_prior`, `BETA`/`GAMMA`/etc. constants, and `Prior` class unchanged.

## Step 3 — Port `mcmc.py`, RWMC only

Copy `MonteCarlo` base + `RWMC`; **omit** `SMC`, `_load_parallel_tools`, `importance_sample`'s MPI branch (keep serial).

Changes during the copy:

- Imports: `from .prior import Prior`, `from . import numerical as nm`, drop `py_tools` / `mpi4py` / `io` imports.
- Replace `save_file`/`load_file` (npy/pickle) with `.npz`-based I/O wired through `equilibrium.io.save_results` / `load_results`. One file per chain, config as JSON sidecar (see Step 7). Drop `pkl_list` entirely; `names`/`bounds`/etc. move into the JSON config.
- `save_chain` / `load_chain` / `save_all` / `load_all` / `save_list` / `load_list` get replaced by a single `save_chain(chain_no)` that writes `chain{N}.npz` with {`draws`, `post_sim`, `acc_rate`, `jump_scale`} and a `save_metadata()` that writes `mode.npz` / `hessian.npz` / `config.json`. Simpler than the original.
- `MonteCarlo.__init__`: replace `out_dir`/`suffix` with `model_label`/`estimation_label`; compute output directory from `settings.paths.save_dir / "estimation" / model_label / estimation_label`.
- Keep everything else (acceptance adaptation, covariance recomputation, blocks, mode-finding, Hessian) identical — that's the hot path and the user has a track record with it.

## Step 4 — Port `state_space.py`

Copy `StateSpaceModel` + `StateSpaceEstimates` to `estimation/state_space.py`. Replace:

- `from py_tools import numerical as nm, stats as st` → `from . import numerical as nm`.
- `st.draw_norm(P)` → inline one-liner: `np.linalg.cholesky(P) @ np.random.randn(P.shape[0])` (used once, in `draw_states`).

Keep every method (`kalman_filter`, `disturbance_smoother`, `state_smoother`, `shock_smoother`, decompositions, `draw_states`) so smoothing is available later.

## Step 5 — Estimation-level observable selection, bridge to `StateSpaceModel`

Observables are **not** a separate model rule category. Instead, the model defines ordinary variables in the existing rule system (`u`, `x`, `z`, `intermediate`), and the estimation layer selects which of those variables are observed in a given dataset.

This keeps the model reusable across multiple datasets and measurement systems:
- the model defines what quantities exist,
- the estimation specifies which of those quantities are observed,
- transformed measurement objects (annualized rates, spreads, growth rates, etc.) should be defined as ordinary model variables, typically in `intermediate`.

Example user code:
```python
model.rules["intermediate"] += [
    ("fed_funds_ann", "400 * np.log(R)"),
    ("gdp_growth", "log_Y - log_Y_lag"),
]

observables = ["gdp_growth", "pi", "fed_funds_ann"]
```

Growth rates and differences still work by expanding the state — e.g., add `log_Y_lag` as an `x` variable with transition `log_Y_lag_new = log_Y`. No special lag mechanism is needed.

### 5a. Accept `observables=[...]` at the estimation / likelihood layer

- Do **not** add `"observable"` to `RULE_KEYS` or `UNIQUE_RULE_KEYS`.
- Do **not** add a new codegen/function-bundle path for observables.
- Add an estimation-level selector:
  - `build_state_space(model, observables, meas_err=None)`
  - `log_likelihood(model, data, *, observables, meas_err=None, fixed_init=None)`
- Require each observable name to already exist after `finalize()`, typically in one of:
  - `u`
  - `x`
  - `z`
  - `intermediate`
  - `read_expectations`

### 5b. Assemble `Z` and `b` from existing model variables

Construct the measurement system directly from the existing linearized model plus steady-state values:

- For observables in `u`, `x`, or `z`, use a basis row selecting that variable from the state vector `[u, x, z]`.
- For observables in `intermediate`, use the corresponding row of the existing intermediate Jacobian:
```python
Z_i = np.hstack((J_inter_u[i], J_inter_x[i], J_inter_z[i]))
```
- For observables in `read_expectations`, use the corresponding row of the existing linear mapping
  from `[u, x, z]` to `read_expectations` values:
```python
Z_i = linear_model.L[i]
```
- Set:
```python
b_i = steady_dict[observable_name]
```

This keeps measurement selection outside the model codegen path and avoids duplicated symbolic definitions like `("inflation_obs", "pi")`.

### 5c. `build_state_space(model, observables, meas_err=None) -> StateSpaceModel`

In `estimation/likelihood.py`:
- **A** ← `linear_model.A_s`
- **R** ← `linear_model.B_s`
- **Q** ← diagonal from `params["VOL_<shock>"] ** 2` over `model.exog_list`
- **Z, b** ← assembled from the selected observables (Step 5b)
- **H** ← optional `meas_err` kwarg: `dict[obs_name, stdev]` → diagonal `H`, or full matrix; default zero

### 5d. Split likelihood evaluation into a model-facing wrapper and an SSM-facing core

Use two layers:

```python
def log_likelihood_ssm(
    ssm: StateSpaceModel,
    data: np.ndarray,
    *,
    fixed_init: list[int] | None = None,
) -> float
```

and

```python
def log_likelihood(
    model: Model,
    data: np.ndarray,
    *,
    observables: list[str],
    meas_err: dict[str, float] | np.ndarray | None = None,
    fixed_init: list[int] | None = None,
) -> float
```

`log_likelihood(...)` should build the `StateSpaceModel` and then delegate to `log_likelihood_ssm(...)`.

Rationale:
- `log_likelihood_ssm(...)` avoids repeated state-space assembly when the caller already has `ssm`,
- it cleanly separates "build measurement system" from "run Kalman filter",
- it avoids recomputing derivatives inside the filtering step itself.

On any failure (bad steady state, non-PSD covariance, etc.) return `-1e10`. The `data` columns must match the order of the passed `observables`.

## Step 6 — `EstimParam` types + high-level `estimate()` (`estimate.py`)

Why `EstimParam` exists: the MCMC sampler proposes a raw vector `x`. Something has to turn that vector into parameter changes on the `Model`, then re-solve/linearize, then call `log_likelihood`. The existing calibration code in `solvers/calibration.py` already has this pattern — `RegimeParam`, `ModelParam`, `ShockParam` know how to read/write their value on a `Model`. I'd mirror that rather than reuse, because estimation parameters additionally carry a prior, bounds, and optionally a transform:

```python
@dataclass
class EstimParam:
    name: str                       # model param name (e.g. "bet", "VOL_log_Z")
    prior: str | None               # "beta", "gamma", "norm", etc.
    mean: float                     # prior mean
    sd: float                       # prior sd
    lb: float = -np.inf             # hard bound for bound_transform / bounds check
    ub: float = np.inf
    initial: float | None = None    # starting value; defaults to model.params[name]
```

Helpers convert a list of `EstimParam` into the `Prior`, `lb`, `ub`, `names`, `x0` arrays RWMC expects.

Top-level function:

```python
def estimate(
    model: Model,
    params_to_estimate: list[EstimParam],
    data: np.ndarray,                    # (Nt, Ny); columns match `observables`
    *,
    observables: list[str],
    estimation_label: str,
    Nsim: int = 10_000,
    n_chains: int = 1,
    meas_err: dict[str, float] | None = None,
    fixed_init: list[int] | None = None,
    find_mode: bool = True,
    compute_hessian: bool = True,
    sample_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    CH_inv: np.ndarray | None = None,    # skip Hessian, use this proposal
) -> EstimationResult
```

Internals:

1. Validate & snapshot `model.params`.
2. Build the closure `log_like(x)`:
   - Copy the model (`update_copy` already shares function bundles → no recompile).
   - Write `x[i] → model.params[param.name]` for each `EstimParam`.
   - Call `model.solve_steady()` then `linear_model = model.linearize()`.
   - Return `log_likelihood(model, data, observables=observables, meas_err=...)`.
   - Wrap in `try/except` → `-1e10` so bad parameter draws don't crash the chain.
3. Instantiate `RWMC(log_like=log_like, prior=prior, lb=lb, ub=ub, names=names, model_label=model.label, estimation_label=estimation_label)`.
4. Optionally `find_mode(x0)` → `compute_hessian()`; otherwise use `CH_inv`.
5. For each chain, `initialize()` + `sample(Nsim, n_save=..., n_print=..., chain_no=i)`.
6. Return `EstimationResult` (dataclass holding mode, Hessian, chain list, metadata).

## Step 7 — IO layout (`io.py`)

Under `settings.paths.save_dir`:

```
estimation/<model_label>/<estimation_label>/
    config.json          # param names, priors, bounds, data shape, seed, date
    mode.npz             # x_mode, post_mode
    hessian.npz          # H, H_inv, CH_inv
    chain0.npz           # draws, post_sim, acc_rate, jump_scale
    chain1.npz
    ...
    log_chain0.txt       # sampler logs (unchanged from existing log_chain file)
```

API:

```python
def estimation_dir(model_label, estimation_label) -> Path
def save_estimation(result, overwrite=False) -> Path
def load_estimation(model_label, estimation_label) -> EstimationResult
```

Hooks into existing `resolve_output_path` where reasonable but mostly stays self-contained (it's a directory rather than a single file).

**Chain resumption** is *not* in this step — deferred per instruction. The format is compatible with future resumption (each chain is self-describing), and a `TODO` note will mark the place to revisit.

## Step 8 — Wire into package

- Add to `src/equilibrium/__init__.py`: `Prior`, `RWMC`, `EstimParam`, `estimate`, `load_estimation`, `EstimationResult`.
- Add a smoke test `tests/test_estimation.py`:
  - Build the RBC model from the scaffolding,
  - Simulate 100 periods of fake data from known params,
  - Estimate 2 parameters with short chain (`Nsim=500`) and 1 chain,
  - Assert the chain produces a finite posterior mean near the truth and writes files to the expected directory.
- No other changes to existing tests.

## What is deliberately NOT done in this plan

- SMC sampler (deferred).
- MPI/parallel path (deferred with SMC).
- Chain resumption (deferred per #6).
- Rewriting the Kalman filter in JAX (current loop-based numpy version works; re-JAX-ifying is a later optimization).
- Touching the existing `calibrate()` pipeline. Estimation and calibration stay separate.
- Anything that makes `estimate()` infer transformed observables automatically — user must define any such quantities as ordinary model variables first, then select them via `observables=[...]`.

## Rough LOC / risk estimate

| File | Approx lines | Risk |
|------|----|------|
| `numerical.py` | 90 (vendored) | Low |
| `prior.py` | 190 (verbatim) | None |
| `mcmc.py` | ~700 (ported, IO simplified) | Medium — IO rewrite is where bugs land |
| `state_space.py` | ~550 (verbatim except 2 imports) | Low |
| `likelihood.py` | ~120 (new, includes measurement selection + Kalman wrapper) | Low |
| `estimate.py` | ~200 (new) | Medium |
| `io.py` | ~120 (new) | Low |
| Model changes for measurement selection support | ~20 across `model.py`/`linear.py` | Low |
| tests | ~100 | — |

## Open questions flagged for follow-up

- **Initial-state covariance**: defaulting to the unconditional cov from `solve_discrete_lyapunov`. If models often have unit roots / non-stationary components, `fixed_init` will be needed routinely — exposed as a passthrough kwarg for now.
- **JIT compatibility**: ported Kalman filter is plain NumPy (loops over time). For a typical DSGE estimation (~200 quarters, ~8 observables) it will run fine but won't use JAX. If chains are slow, that is the first optimization target.
