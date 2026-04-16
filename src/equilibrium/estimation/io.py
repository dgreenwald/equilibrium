"""Directory-level save/load helpers for estimation results."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..settings import get_settings
from .estimate import EstimationResult, EstimParam
from .mcmc import RWMC, _load_scalar


def estimation_dir(model_label, estimation_label) -> Path:
    """Return the canonical directory for one estimation run."""
    settings = get_settings()
    return (
        Path(settings.paths.save_dir)
        / "estimation"
        / (model_label if model_label is not None else "_default")
        / estimation_label
    )


def _jsonable_meas_err(meas_err):
    if meas_err is None or isinstance(meas_err, dict):
        return meas_err
    return np.asarray(meas_err, dtype=float).tolist()


def _restore_meas_err(meas_err):
    if meas_err is None or isinstance(meas_err, dict):
        return meas_err
    return np.asarray(meas_err, dtype=float)


def save_estimation(result, overwrite=False) -> Path:
    """Save estimation config and arrays under the canonical estimation directory."""
    out_dir = estimation_dir(result.model_label, result.estimation_label)
    if out_dir.exists() and not overwrite and any(out_dir.iterdir()):
        raise FileExistsError(f"Estimation directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for chain_no, chain in enumerate(result.chains):
        chain.save_chain(chain_no=chain_no)

    if result.mode is not None:
        np.savez(
            out_dir / "mode.npz",
            x_mode=result.mode,
            post_mode=np.atleast_1d(result.post_mode),
        )

    hess_arrays = {}
    if result.H is not None:
        hess_arrays["H"] = result.H
    if result.H_inv is not None:
        hess_arrays["H_inv"] = result.H_inv
    if result.CH_inv is not None:
        hess_arrays["CH_inv"] = result.CH_inv
    if hess_arrays:
        np.savez(out_dir / "hessian.npz", **hess_arrays)

    metadata = dict(result.metadata)
    metadata["meas_err"] = _jsonable_meas_err(metadata.get("meas_err"))

    config = {
        "model_label": result.model_label,
        "estimation_label": result.estimation_label,
        "observables": list(result.observables),
        "param_names": list(result.param_names),
        "estim_params": [asdict(param) for param in result.estim_params],
        "x0": result.x0.tolist(),
        "metadata": metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    return out_dir


def load_estimation(model_label, estimation_label) -> EstimationResult:
    """Load a saved estimation directory into an EstimationResult."""
    out_dir = estimation_dir(model_label, estimation_label)
    config = json.loads((out_dir / "config.json").read_text())

    estim_params = [EstimParam(**item) for item in config["estim_params"]]
    metadata = dict(config.get("metadata", {}))
    metadata["meas_err"] = _restore_meas_err(metadata.get("meas_err"))

    mode = None
    post_mode = None
    mode_path = out_dir / "mode.npz"
    if mode_path.exists():
        data = np.load(mode_path)
        mode = data["x_mode"]
        post_mode = _load_scalar(data["post_mode"], "post_mode")

    H = None
    H_inv = None
    CH_inv = None
    hessian_path = out_dir / "hessian.npz"
    if hessian_path.exists():
        data = np.load(hessian_path)
        H = data["H"] if "H" in data else None
        H_inv = data["H_inv"] if "H_inv" in data else None
        CH_inv = data["CH_inv"] if "CH_inv" in data else None

    n_chains = int(metadata.get("n_chains", 0))
    chains = []
    for chain_no in range(n_chains):
        chain = RWMC(
            Nx=len(config["param_names"]),
            names=config["param_names"],
            model_label=model_label,
            estimation_label=estimation_label,
        )
        chain.load_chain(chain_no=chain_no)
        chain.x_mode = None if mode is None else np.array(mode, copy=True)
        chain.post_mode = post_mode
        chain.H = None if H is None else np.array(H, copy=True)
        chain.H_inv = None if H_inv is None else np.array(H_inv, copy=True)
        chain.CH_inv = None if CH_inv is None else np.array(CH_inv, copy=True)
        chains.append(chain)

    return EstimationResult(
        model_label=config["model_label"],
        estimation_label=config["estimation_label"],
        observables=list(config["observables"]),
        estim_params=estim_params,
        param_names=list(config["param_names"]),
        x0=np.asarray(config["x0"], dtype=float),
        mode=None if mode is None else np.array(mode, copy=True),
        post_mode=post_mode,
        H=None if H is None else np.array(H, copy=True),
        H_inv=None if H_inv is None else np.array(H_inv, copy=True),
        CH_inv=None if CH_inv is None else np.array(CH_inv, copy=True),
        chains=chains,
        metadata=metadata,
    )
