"""Shared post-fit diagnostics and figures for joint ``diag(D)`` fits.

The functions in this module deliberately accept fitted weights and already persisted
tables.  They never participate in a fit, gate, or cell-selection decision.  This keeps
the same implementation usable by the frozen-target and future block-coordinate target
runners without exposing NMR/state information to their optimization graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from jaxent.src.analysis.state_population import FULL_STATE_SUPPORT


def per_state_weight_diagnostics(
    weights: Sequence[float],
    states: Sequence[str],
    *,
    cluster_labels: Sequence[object] | None = None,
    support: Sequence[str] = FULL_STATE_SUPPORT,
    mass_floor: float = 1e-9,
) -> dict[str, float | int | str]:
    """Return per-state mass/ESS and the dominant frame's cluster.

    State ESS is computed from the unnormalised within-state weights:
    ``(sum(w_s)**2) / sum(w_s**2)``.  Empty or numerically empty states return
    zero ESS.  ``dominant_cluster`` is the raw cluster label attached to the
    globally highest-weight frame; it is intentionally a post-fit audit field.
    """

    weights = np.asarray(weights, dtype=np.float64)
    states = np.asarray(states)
    if weights.ndim != 1 or states.ndim != 1 or weights.size != states.size:
        raise ValueError("weights and states must be one-dimensional arrays of equal length")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("weights must be finite and non-negative")
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels)
        if cluster_labels.ndim != 1 or cluster_labels.size != weights.size:
            raise ValueError("cluster_labels must have one entry per frame")

    diagnostics: dict[str, float | int | str] = {}
    for state in support:
        selected = weights[states == state]
        mass = float(selected.sum())
        sumsq = float(np.square(selected).sum())
        ess = mass * mass / sumsq if mass >= mass_floor and sumsq > 0.0 else 0.0
        diagnostics[f"ess_{state}"] = ess
        diagnostics[f"mass_{state}"] = mass

    dominant_index = int(np.argmax(weights)) if weights.size else -1
    diagnostics["dominant_cluster"] = (
        cluster_labels[dominant_index].item()
        if cluster_labels is not None and dominant_index >= 0
        else ""
    )
    diagnostics["dominant_weight"] = (
        float(weights[dominant_index]) if dominant_index >= 0 else 0.0
    )
    return diagnostics


def load_frame_cluster_labels(
    cluster_csv: str | Path,
    ensemble_name: str,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    """Load raw cluster labels in the same trajectory order as state revelation."""

    frame_table = pd.read_csv(cluster_csv)
    ensemble = frame_table[frame_table["ensemble_name"] == ensemble_name]
    if ensemble.empty:
        raise ValueError(f"ensemble {ensemble_name!r} not present in {cluster_csv}")
    ensemble = ensemble.sort_values("global_frame_index")
    indices = ensemble["global_frame_index"].to_numpy(dtype=int)
    labels = ensemble["cluster_label"].to_numpy(dtype=int)
    if not np.array_equal(indices, np.arange(indices[0], indices[0] + labels.size)):
        raise ValueError(f"ensemble {ensemble_name!r} cluster frames are not contiguous")
    if expected_frames is not None and labels.size != expected_frames:
        raise ValueError(
            f"ensemble {ensemble_name!r} has {labels.size} cluster labels, expected {expected_frames}"
        )
    return labels


def _load_tables(input_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    input_dir = Path(input_dir)
    aggregate = pd.read_csv(input_dir / "joint_diag_d_fit.csv")
    replicate_path = input_dir / "joint_diag_d_fit_replicates.csv"
    replicate = pd.read_csv(replicate_path) if replicate_path.exists() else None
    return aggregate, replicate


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _annotated_heatmap(
    ax,
    data: pd.DataFrame,
    value: str,
    title: str,
    cmap: str = "viridis",
    value_fmt: str | None = None,
):
    """Draw an η×γ heatmap and border cells by persisted gate-pass fraction.

    Cell text defaults to the gate-pass fraction (percent).  When ``value_fmt``
    is given, the panel ``value`` is annotated instead (formatted by that spec),
    so a value heatmap can show values rather than percentages; the green/red
    gate-pass border is unchanged either way.
    """

    import matplotlib.patches as patches

    gammas = sorted(data["gamma"].unique())
    etas = sorted(data["eta"].unique())
    grid = data.pivot(index="eta", columns="gamma", values=value).reindex(
        index=etas, columns=gammas
    )
    image = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(gammas)), [f"{v:g}" for v in gammas])
    ax.set_yticks(range(len(etas)), [f"{v:g}" for v in etas])
    ax.set_xlabel("γ")
    ax.set_ylabel("η")
    ax.set_title(title)
    for row_index, eta in enumerate(etas):
        for col_index, gamma in enumerate(gammas):
            cell = data[(data.eta == eta) & (data.gamma == gamma)]
            if cell.empty:
                continue
            gate = float(cell["mean_gate_passed"].iloc[0])
            edge = "tab:green" if gate >= 1.0 - 1e-12 else "tab:red"
            ax.add_patch(
                patches.Rectangle(
                    (col_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=edge,
                    linewidth=1.8,
                )
            )
            if value_fmt is not None:
                label = format(float(cell[value].iloc[0]), value_fmt)
            else:
                label = f"{gate:.0%}"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                path_effects=[],
            )
    return image


def _plot_heatmap_family(
    aggregate: pd.DataFrame,
    input_dir: Path,
    arm: str,
    metric: str,
    filename: str,
    title: str,
    cmap: str,
    value_fmt: str | None = None,
) -> Path:
    plt = _import_pyplot()
    subset = aggregate[aggregate["arm"] == arm]
    ensembles = list(subset["ensemble"].drop_duplicates())
    figure, axes = plt.subplots(1, len(ensembles), figsize=(6.5 * len(ensembles), 5.2), squeeze=False)
    for axis, ensemble in zip(axes[0], ensembles):
        image = _annotated_heatmap(
            axis,
            subset[subset.ensemble == ensemble],
            metric,
            f"{ensemble}: {title}",
            cmap,
            value_fmt=value_fmt,
        )
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(f"{arm} — {title}")
    figure.tight_layout()
    path = input_dir / filename
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_gate_ratio(aggregate: pd.DataFrame, replicate: pd.DataFrame | None, input_dir: Path, arm: str) -> Path:
    plt = _import_pyplot()
    source = replicate if replicate is not None else aggregate
    subset = source[source["arm"] == arm].copy()
    subset["gate_ratio"] = subset["val_mse"] / subset["mean_gate_reference_mse"]
    ensembles = list(subset.ensemble.drop_duplicates())
    figure, axes = plt.subplots(1, len(ensembles), figsize=(7 * len(ensembles), 4.8), squeeze=False)
    for axis, ensemble in zip(axes[0], ensembles):
        current = subset[subset.ensemble == ensemble]
        for gamma, gamma_rows in current.groupby("gamma", sort=True):
            grouped = gamma_rows.groupby("eta")["gate_ratio"]
            means = grouped.mean()
            errors = grouped.std(ddof=0) if replicate is not None else means * 0.0
            axis.errorbar(
                means.index,
                means.values,
                yerr=errors.values,
                marker="o",
                capsize=3,
                label=f"γ={gamma:g}",
            )
        axis.axhline(1.05, color="tab:red", linestyle="--", label="1.05 gate")
        axis.set_title(ensemble)
        axis.set_xlabel("η")
        axis.set_ylabel("held-out val MSE / γ=0 reference")
        axis.set_xscale("symlog", linthresh=1e-3)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.suptitle(f"{arm} — mean-gate ratio versus η")
    figure.tight_layout()
    path = input_dir / f"{arm}_gate_ratio_vs_eta.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_per_state_ess(aggregate: pd.DataFrame, input_dir: Path, arm: str) -> Path:
    plt = _import_pyplot()
    states = ("PUF3", "unfolded", "PUF2-like")
    subset = aggregate[aggregate["arm"] == arm]
    ensembles = list(subset["ensemble"].drop_duplicates())
    figure, axes = plt.subplots(len(ensembles), len(states), figsize=(15, 5 * len(ensembles)), squeeze=False)
    for row_index, ensemble in enumerate(ensembles):
        current = subset[subset.ensemble == ensemble]
        for col_index, state in enumerate(states):
            axis = axes[row_index, col_index]
            column = f"ess_{state}"
            if column not in current:
                axis.text(0.5, 0.5, "no persisted per-state ESS", ha="center", va="center")
                axis.set_axis_off()
                continue
            image = _annotated_heatmap(
                axis, current, column, f"{ensemble}: {state}", "magma", value_fmt=".1f"
            )
            figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(f"{arm} — per-state ESS (post-fit audit)")
    figure.tight_layout()
    path = input_dir / f"{arm}_per_cluster_ess.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_joint_diag_d_fit(input_dir: str | Path, arm: str | None = None) -> list[Path]:
    """Generate all persisted-table figures for one arm or all arms."""

    input_dir = Path(input_dir)
    aggregate, replicate = _load_tables(input_dir)
    arms = [arm] if arm is not None else list(aggregate["arm"].drop_duplicates())
    unknown = set(arms) - set(aggregate["arm"].unique())
    if unknown:
        raise ValueError(f"unknown arm(s): {sorted(unknown)}")
    paths = []
    for selected_arm in arms:
        paths.extend(
            [
                _plot_heatmap_family(aggregate, input_dir, selected_arm, "ess", "{}_ess_heatmap.png".format(selected_arm), "ESS", "viridis", value_fmt=".1f"),
                _plot_heatmap_family(aggregate, input_dir, selected_arm, "recovery", "{}_recovery_heatmap.png".format(selected_arm), "recovery (%)", "cividis", value_fmt=".1f"),
                _plot_heatmap_family(aggregate, input_dir, selected_arm, "val_mse", "{}_val_mse_heatmap.png".format(selected_arm), "held-out val MSE", "plasma", value_fmt=".4f"),
                _plot_heatmap_family(aggregate, input_dir, selected_arm, "decoy", "{}_decoy_mass_heatmap.png".format(selected_arm), "decoy mass", "Reds", value_fmt=".3f"),
                _plot_gate_ratio(aggregate, replicate, input_dir, selected_arm),
                _plot_per_state_ess(aggregate, input_dir, selected_arm),
            ]
        )
    return paths
