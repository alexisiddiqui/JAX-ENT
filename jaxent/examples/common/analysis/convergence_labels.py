"""Shared helpers for consuming labeled convergence snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from jaxent.src.opt.base import OptimizationHistory, OptimizationState


@dataclass(frozen=True)
class LabeledConvergenceState:
    rank: int
    threshold: float
    state: OptimizationState


def iter_labeled_convergence_states(
    history: OptimizationHistory,
) -> list[LabeledConvergenceState]:
    return [
        LabeledConvergenceState(rank=index, threshold=threshold, state=state)
        for index, (threshold, state) in enumerate(history.iter_labeled_convergence_states())
    ]


def convergence_rows_from_history(
    history: OptimizationHistory, base_fields: dict
) -> list[dict]:
    return [
        {
            **base_fields,
            "convergence_rank": labeled.rank,
            "convergence_threshold": labeled.threshold,
        }
        for labeled in iter_labeled_convergence_states(history)
    ]


def write_convergence_thresholds_sidecar(
    run_dir: str | Path, history: OptimizationHistory
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "convergence_thresholds.txt").open("w") as file:
        for threshold in history.convergence_thresholds:
            file.write(f"{threshold}\n")


def find_best_labeled_state(
    history: OptimizationHistory,
) -> tuple[float | None, OptimizationState | None]:
    labeled = iter_labeled_convergence_states(history)
    if not labeled:
        return None, None
    best = min(labeled, key=lambda item: item.state.losses.val_losses[0])
    return best.threshold, best.state


def find_state_nearest_threshold(
    history: OptimizationHistory, target_threshold: float
) -> tuple[float | None, OptimizationState | None]:
    labeled = iter_labeled_convergence_states(history)
    if not labeled:
        return None, None
    nearest = min(
        labeled,
        key=lambda item: abs(math.log(item.threshold) - math.log(target_threshold)),
    )
    return nearest.threshold, nearest.state
