"""High-level estimation entry points built on RWMC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..model import Model
from .likelihood import log_likelihood
from .mcmc import RWMC
from .prior import Prior


@dataclass
class EstimParam:
    """Specification for one estimated model parameter."""

    name: str
    prior: str | None
    mean: float
    sd: float
    lb: float = -np.inf
    ub: float = np.inf
    initial: float | None = None

    def resolved_initial(self, model: Model) -> float:
        """Return the starting value, defaulting to the model's current parameter."""
        if self.initial is not None:
            return float(self.initial)
        if self.name not in model.params:
            raise KeyError(f"Parameter '{self.name}' is not present in model.params.")
        return float(model.params[self.name])

    def validate(self, model: Model) -> None:
        """Validate bounds and existence against a reference model."""
        if not self.name:
            raise ValueError("EstimParam name must be non-empty.")
        if self.lb > self.ub:
            raise ValueError(
                f"EstimParam bounds are inverted for '{self.name}': {self.lb} > {self.ub}."
            )
        initial = self.resolved_initial(model)
        if not (self.lb <= initial <= self.ub):
            raise ValueError(
                f"Initial value {initial} for '{self.name}' is outside bounds "
                f"({self.lb}, {self.ub})."
            )


@dataclass
class EstimationResult:
    """Container for estimation outputs available at Step 6."""

    model_label: str
    estimation_label: str
    observables: list[str]
    estim_params: list[EstimParam]
    param_names: list[str]
    x0: np.ndarray
    mode: np.ndarray | None
    post_mode: float | None
    H: np.ndarray | None
    H_inv: np.ndarray | None
    CH_inv: np.ndarray | None
    chains: list[RWMC] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_model_ready(model: Model) -> None:
    if getattr(model, "var_lists", None) is None:
        raise RuntimeError("Model must be finalized before estimation.")


def _validate_observables(model: Model, observables: list[str]) -> None:
    if not observables:
        raise ValueError("estimate() requires at least one observable.")

    valid_categories = ["u", "x", "z", "intermediate", "read_expectations"]
    valid_names = {
        name
        for category in valid_categories
        for name in model.var_lists.get(category, [])
    }

    missing = [name for name in observables if name not in valid_names]
    if missing:
        raise ValueError(
            "Unknown observables: "
            + ", ".join(missing)
            + ". Observables must be existing model variables from "
            "'u', 'x', 'z', 'intermediate', or 'read_expectations'."
        )


def _build_prior_and_arrays(model: Model, params_to_estimate: list[EstimParam]):
    prior = Prior()
    names = []
    lb = []
    ub = []
    x0 = []

    for param in params_to_estimate:
        param.validate(model)
        initial = param.resolved_initial(model)
        prior.add(param.prior, mean=param.mean, sd=param.sd, name=param.name)
        names.append(param.name)
        lb.append(float(param.lb))
        ub.append(float(param.ub))
        x0.append(initial)

    return (
        prior,
        names,
        np.asarray(lb, dtype=float),
        np.asarray(ub, dtype=float),
        np.asarray(x0, dtype=float),
    )


def _split_sample_kwargs(
    sample_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    init_keys = {
        "jump_scale",
        "jump_mult",
        "stride",
        "C",
        "C_list",
        "blocks",
        "bool_blocks",
        "n_blocks",
        "adapt_sens",
        "adapt_range",
        "adapt_target",
    }
    init_kwargs = {k: v for k, v in sample_kwargs.items() if k in init_keys}
    run_kwargs = {k: v for k, v in sample_kwargs.items() if k not in init_keys}
    return init_kwargs, run_kwargs


def estimate(
    model: Model,
    params_to_estimate: list[EstimParam],
    data: np.ndarray,
    *,
    observables: list[str],
    estimation_label: str,
    Nsim: int = 10_000,
    n_chains: int = 1,
    meas_err: dict[str, float] | np.ndarray | None = None,
    fixed_init: list[int] | None = None,
    find_mode: bool = True,
    compute_hessian: bool = True,
    sample_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    CH_inv: np.ndarray | None = None,
) -> EstimationResult:
    """Estimate model parameters by RWMC on the posterior."""
    _validate_model_ready(model)
    _validate_observables(model, observables)

    y = np.asarray(data, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"estimate() data must be 2-D, got shape {y.shape}.")
    if y.shape[1] != len(observables):
        raise ValueError(
            f"estimate() data must have {len(observables)} columns to match observables, "
            f"got {y.shape[1]}."
        )
    if not params_to_estimate:
        raise ValueError("estimate() requires at least one EstimParam.")
    if n_chains < 1:
        raise ValueError("n_chains must be at least 1.")

    if mode_kwargs is None:
        mode_kwargs = {}
    if sample_kwargs is None:
        sample_kwargs = {}

    init_kwargs, run_kwargs = _split_sample_kwargs(sample_kwargs)

    prior, names, lb, ub, x0 = _build_prior_and_arrays(model, params_to_estimate)

    def log_like(x):
        try:
            param_updates = {
                param.name: float(x[idx])
                for idx, param in enumerate(params_to_estimate)
            }
            this_model = model.update_copy(params=param_updates)
            this_model.solve_steady(calibrate=False, display=False)
            this_model.linearize()
            return log_likelihood(
                this_model,
                y,
                observables=observables,
                meas_err=meas_err,
                fixed_init=fixed_init,
            )
        except Exception:
            return -1e10

    master = RWMC(
        log_like=log_like,
        prior=prior,
        lb=lb,
        ub=ub,
        names=names,
        model_label=model.label,
        estimation_label=estimation_label,
    )

    if find_mode:
        master.find_mode(x0, **mode_kwargs)
        if compute_hessian:
            master.compute_hessian()
        if CH_inv is not None:
            master.set_CH_inv(np.asarray(CH_inv, dtype=float))
    else:
        master.x_mode = x0.copy()
        master.post_mode = master.posterior(x0)
        if compute_hessian:
            raise ValueError("compute_hessian=True requires find_mode=True.")
        if CH_inv is None:
            raise ValueError("When find_mode=False, estimate() requires CH_inv.")
        master.set_CH_inv(np.asarray(CH_inv, dtype=float))

    if master.CH_inv is None:
        raise ValueError(
            "estimate() requires a proposal covariance factor. "
            "Use find_mode/compute_hessian or provide CH_inv."
        )

    if master.out_dir is not None:
        master.save_metadata()

    chains = []
    for chain_no in range(n_chains):
        chain = RWMC(
            log_like=log_like,
            prior=prior,
            lb=lb,
            ub=ub,
            names=names,
            model_label=model.label,
            estimation_label=estimation_label,
        )
        chain.x_mode = (
            None if master.x_mode is None else np.array(master.x_mode, copy=True)
        )
        chain.post_mode = master.post_mode
        chain.H = None if master.H is None else np.array(master.H, copy=True)
        chain.H_inv = (
            None if master.H_inv is None else np.array(master.H_inv, copy=True)
        )
        chain.CH_inv = np.array(master.CH_inv, copy=True)

        chain.initialize(x0=x0.copy(), **init_kwargs)
        chain.sample(Nsim=Nsim, chain_no=chain_no, **run_kwargs)
        if chain.out_dir is not None:
            chain.save_chain(chain_no=chain_no)
        chains.append(chain)

    metadata = {
        "n_chains": n_chains,
        "Nsim": Nsim,
        "meas_err": meas_err,
        "fixed_init": fixed_init,
    }

    result = EstimationResult(
        model_label=model.label,
        estimation_label=estimation_label,
        observables=list(observables),
        estim_params=[EstimParam(**param.__dict__) for param in params_to_estimate],
        param_names=names,
        x0=x0,
        mode=None if master.x_mode is None else np.array(master.x_mode, copy=True),
        post_mode=master.post_mode,
        H=None if master.H is None else np.array(master.H, copy=True),
        H_inv=None if master.H_inv is None else np.array(master.H_inv, copy=True),
        CH_inv=None if master.CH_inv is None else np.array(master.CH_inv, copy=True),
        chains=chains,
        metadata=metadata,
    )
    from .io import save_estimation

    save_estimation(result, overwrite=True)
    return result
