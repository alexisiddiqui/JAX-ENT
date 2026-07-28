"""Recover convergence labels for histories written before format version 2."""

from __future__ import annotations

import json
from pathlib import Path

from jaxent.src.opt.base import OptimizationHistory
from jaxent.src.opt.track import create_convergence_thresholds


def _sidecar_path(hdf5_path: str | Path) -> Path:
    path = Path(hdf5_path)
    stem = path.name
    for suffix in ("_results.hdf5", ".hdf5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return path.parent / f"{stem}_config.json"


def recover_convergence_thresholds_from_sidecar(
    hdf5_path: str | Path, history: OptimizationHistory
) -> tuple[float, ...]:
    sidecar = _sidecar_path(hdf5_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"No sidecar config found at {sidecar} to recover convergence thresholds"
        )
    data = json.loads(sidecar.read_text())
    opt_config = data.get("opt_config")
    if not isinstance(opt_config, dict):
        raise ValueError(f"{sidecar} has no 'opt_config' section")
    convergence = opt_config.get("convergence_rates", opt_config.get("convergence"))
    if convergence is None:
        raise ValueError(f"{sidecar}['opt_config'] has no convergence ladder")
    learning_rate = opt_config.get("learning_rate")
    if learning_rate is None:
        raise ValueError(f"{sidecar}['opt_config'] has no 'learning_rate'")

    ladder = create_convergence_thresholds(convergence, learning_rate)
    n_needed = len(history.convergence_states)
    if len(ladder) < n_needed:
        raise ValueError(
            f"{sidecar} ladder has {len(ladder)} entries, shorter than the "
            f"{n_needed} convergence_states recorded in {hdf5_path}"
        )
    return tuple(float(value) for value in ladder[:n_needed])
