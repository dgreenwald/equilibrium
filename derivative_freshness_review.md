# Review: Derivative Freshness in deterministic.py

## Summary
✅ **All derivatives are fresh.** No staleness issues found.

## Analysis

### Function Call Pattern

`compute_time_period()` is called in a loop:
```python
for tt in range(Nt):
    f_t, L_t, C_t, F_t = compute_time_period(
        tt, Nt, UX, Z, ux_init, mod, params, N_ux, compute_grad, terminal_condition
    )
```

**Key insight:** Each call to `compute_time_period()` uses different slices of `UX` and `Z` based on `tt`, so arguments change every call.

### Usage 1: Terminal Period (tt == Nt-1, terminal_condition == "stable")

**Location:** Line 116
```python
trans_derivs = mod.d_all("transition", u_lag, x_lag, z_lag, params)
```

**Arguments:**
- `u_lag, x_lag = UX[tt-1, :]` → `UX[Nt-2, :]`
- `z_lag = Z[tt-1, :]` → `Z[Nt-2, :]`
- `params` (constant)

**Freshness check:**
- ✅ Called once per `compute_time_period()` invocation
- ✅ Arguments are slices from current iteration's `UX` and `Z`
- ✅ Between Newton iterations, `UX` changes → arguments change → derivatives fresh
- ✅ DerivativeResult stored in local variable `trans_derivs`, used immediately, goes out of scope

**Used for:**
- Line 121: `trans_derivs.as_hstack(["u", "x"])`

### Usage 2: Middle Periods (else branch, 0 < tt < Nt-1)

**Location:** Line 166
```python
trans_derivs = mod.d_all("transition", u_lag, x_lag, z_lag, params)
```

**Arguments:**
- `u_lag, x_lag = UX[tt-1, :]` → different for each `tt`
- `z_lag = Z[tt-1, :]` → different for each `tt`
- `params` (constant)

**Freshness check:**
- ✅ Called once per `compute_time_period()` invocation
- ✅ Arguments change every call (different `tt`)
- ✅ Between Newton iterations, `UX[tt-1]` changes → derivatives fresh
- ✅ Local scope, no reuse across calls

**Used for:**
- Line 171: `trans_derivs.as_hstack(["u", "x"])`

### Usage 3: Optimality Derivatives (else branch, 0 < tt < Nt-1)

**Location:** Line 175
```python
opt_derivs = mod.d_all("optimality", u_t, x_t, z_t, E, params)
```

**Arguments:**
- `u_t, x_t = UX[tt, :]` → different for each `tt`
- `z_t = Z[tt, :]` → different for each `tt`
- `E = mod.fcn("expectations", ...)` → computed fresh on line 152
- `params` (constant)

**Freshness check:**
- ✅ `E` is computed fresh each call (line 152)
- ✅ `u_t, x_t, z_t` change every call
- ✅ Between Newton iterations, values change → derivatives fresh
- ✅ Local scope

**Used for:**
- Line 181: `opt_derivs["u"] + d_opt_d_E @ exp_derivs["u"]`
- Line 182: `opt_derivs["x"] + d_opt_d_E @ exp_derivs["x"]`

### Usage 4: Expectations Derivatives (else branch, 0 < tt < Nt-1)

**Location:** Line 176
```python
exp_derivs = mod.d_all("expectations", u_t, x_t, z_t, u_next, x_next, z_next, params)
```

**Arguments:**
- `u_t, x_t, z_t` → current period (changes each `tt`)
- `u_next, x_next = UX[tt+1, :]` → next period (changes each `tt`)
- `z_next = Z[tt+1, :]` → next period (changes each `tt`)
- `params` (constant)

**Freshness check:**
- ✅ All arguments change with `tt`
- ✅ Between Newton iterations, `UX` changes → derivatives fresh
- ✅ Local scope
- ✅ **Correctly used for both current and next period variables** (see note below)

**Used for:**
- Line 181: `exp_derivs["u"]` (derivative w.r.t. current period u)
- Line 182: `exp_derivs["x"]` (derivative w.r.t. current period x)
- Line 196: `exp_derivs["u_new"]` (derivative w.r.t. next period u)
- Line 197: `exp_derivs["x_new"]` (derivative w.r.t. next period x)

**Important:** The `expectations` function signature is:
```python
expectations(u, x, z, u_new, x_new, z_new, params)
```

So a single `d_all()` call computes derivatives w.r.t. **all 7 arguments**, including both current-period (u, x, z) and next-period (u_new, x_new, z_new) variables. This is correct usage - we compute once and slice both current and next period derivatives from the same result.

## Iteration Structure

### Outer Loop: Newton Iterations
The deterministic solver iteratively updates `UX` until convergence:
```
Iteration 1: UX_1 → compute all derivatives with UX_1
Iteration 2: UX_2 → compute all derivatives with UX_2  (fresh!)
...
```

### Inner Loop: Time Periods
For each Newton iteration, loop over time periods:
```
tt=0:    derivatives at time 0  (if needed)
tt=1:    derivatives at time 1  (with UX[0], UX[1], UX[2])
tt=2:    derivatives at time 2  (with UX[1], UX[2], UX[3])  (fresh!)
...
```

Each `d_all()` call uses slices from different rows of `UX`, so arguments are always fresh.

## Potential Issues Checked

### ❌ Could derivatives be stale across time periods?
**No.** Each time period uses different slices of `UX` (different `tt`), so arguments differ.

### ❌ Could derivatives be stale across Newton iterations?
**No.** `UX` is updated between iterations, so all arguments change.

### ❌ Could we reuse a DerivativeResult object inappropriately?
**No.** All DerivativeResult objects are local variables that go out of scope after use.

### ❌ Are we computing the same derivatives multiple times unnecessarily?
**No.** Within each branch:
- `trans_derivs`: computed once, used once (as_hstack)
- `opt_derivs`: computed once, used twice (["u"], ["x"])
- `exp_derivs`: computed once, used four times (["u"], ["x"], ["u_new"], ["x_new"])

This is exactly the intended usage pattern - compute once, slice many times.

### ✅ Are we correctly using exp_derivs for both current and next period?
**Yes.** The `expectations` function takes both current (u, x, z) and next (u_new, x_new, z_new) as arguments, so a single derivative computation gives us derivatives w.r.t. all of them. This is correct and efficient.

## Conclusion

✅ **All derivative computations are fresh and correct.**

- Each `d_all()` call occurs exactly once per branch
- Arguments change appropriately between calls (different time periods, different Newton iterations)
- DerivativeResult objects are used locally and not reused across contexts
- The expectations derivatives correctly capture both current and next period in a single computation

**No changes needed.** The implementation is correct and efficient.
