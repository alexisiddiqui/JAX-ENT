#!/usr/bin/env python3
"""Generate the ISO Validation intro figures.

This script writes two static publication-style figures into ``data/_output``:

1. A 40/60 open/closed composition chart.
2. A mean + SD HDX uptake plot using the checked-in target data and ensemble
   prediction inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jaxent.src.models.HDX.BV.features import BV_input_features


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "_output"
TRAJ_DIR = SCRIPT_DIR / "_Bradshaw" / "Reproducibility_pack_v2" / "data" / "trajectories"
FEATURE_DIR = SCRIPT_DIR.parent / "fitting" / "jaxENT" / "_featurise"

TARGET_DFRAC = OUTPUT_DIR / "mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 9,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )



FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 9,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLORS = {
    "target": "#F28E2B",
    "open": "#7A3DB8",
    "closed": "#F1E2A1",
    "iso_bi": "#000000",
    "iso_tri": "#808080",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(FIGURE_STYLE)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def parse_timepoints(path: Path) -> np.ndarray:
    with path.open() as handle:
        header = handle.readline().lstrip("#").split()
    times: list[float] = []
    for token in header:
        try:
            times.append(float(token))
        except ValueError:
            break
    if not times:
        raise ValueError(f"No timepoints found in header of {path}")
    return np.asarray(times, dtype=float)


def load_target_uptake(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    times = parse_timepoints(path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D target data in {path}, found {data.shape}")
    if data.shape[1] != len(times):
        raise ValueError(
            f"Target data/timepoint mismatch in {path}: {data.shape[1]} columns vs {len(times)} times"
        )
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return times, mean, std


def load_state_uptake(
    features_path: Path, timepoints: np.ndarray, reference_kints: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    features = BV_input_features.load(features_path)
    heavy = np.asarray(features.heavy_contacts, dtype=float)
    acceptor = np.asarray(features.acceptor_contacts, dtype=float)

    # Average contacts first (across frames)
    if heavy.ndim == 2:
        heavy = heavy.mean(axis=1)
    if acceptor.ndim == 2:
        acceptor = acceptor.mean(axis=1)

    if reference_kints is not None:
        kints = reference_kints
    else:
        kints = np.asarray(features.k_ints, dtype=float)
    bc = 0.35
    bh = 2.0
    log_pf = bc * heavy + bh * acceptor
    pf = np.exp(log_pf)

    # Compute uptake: (n_times, n_residues)
    # timepoints: (n_times, 1), kints: (1, n_residues), pf: (1, n_residues)
    uptake = 1.0 - np.exp(-(kints[None, :] * timepoints[:, None]) / pf[None, :])

    if uptake.ndim != 2:
        raise ValueError(f"Expected uptake array with 2 dimensions, found {uptake.shape}")

    mean = uptake.mean(axis=1)
    std = uptake.std(axis=1)
    return mean, std


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")


def plot_composition() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 3.3), subplot_kw={"aspect": "equal"})
    sizes = [40, 60]
    labels = ["Open", "Closed"]
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=[COLORS["open"], COLORS["closed"]],
        startangle=90,
        counterclock=False,
        autopct="%1.0f%%",
        pctdistance=0.7,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
    ax.set_title("Iso-Validation target mixture", pad=10)
    ax.text(0, -1.18, "Open 40%  |  Closed 60%", ha="center", va="top")
    ax.set_axis_off()
    save_figure(fig, "iso_validation_open_closed_composition")
    plt.close(fig)


def plot_hdx_summary() -> None:
    times, target_mean, target_std = load_target_uptake(TARGET_DFRAC)

    state_inputs = [
        ("Open", TRAJ_DIR / "open_features" / "features.npz", COLORS["open"], "-", None),
        ("Closed", TRAJ_DIR / "closed_features" / "features.npz", COLORS["closed"], "--", None),
        ("Ground Truth-Only", FEATURE_DIR / "features_iso_bi.npz", COLORS["iso_bi"], "-.", None),
        ("Ground Truth+Alternate", FEATURE_DIR / "features_iso_tri.npz", COLORS["iso_tri"], ":", None),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.fill_between(
        times,
        np.clip(target_mean - target_std, 0, 1),
        np.clip(target_mean + target_std, 0, 1),
        color=COLORS["target"],
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        times,
        target_mean,
        color=COLORS["target"],
        marker="o",
        linewidth=2.0,
        label="Experimental target",
    )

    # Load reference k_ints from Ground Truth-Only to ensure consistency
    ref_features = BV_input_features.load(FEATURE_DIR / "features_iso_bi.npz")
    ref_kints = np.asarray(ref_features.k_ints, dtype=float)

    for label, features_path, color, linestyle, marker in state_inputs:
        mean, std = load_state_uptake(features_path, times, reference_kints=ref_kints)
        ax.fill_between(
            times,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            times,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
            label=label,
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean uptake fraction")
    ax.set_title("ISO Validation HDX summary")
    ax.set_xlim(times.min() * 0.85, times.max() * 1.05)
    ax.set_ylim(0, 1.05)
    # ax.set_xscale("log")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "iso_validation_hdx_summary")
    plt.close(fig)


def main() -> None:
    apply_publication_style()
    ensure_exists(TARGET_DFRAC)
    ensure_exists(TRAJ_DIR / "open_features" / "features.npz")
    ensure_exists(TRAJ_DIR / "closed_features" / "features.npz")
    ensure_exists(FEATURE_DIR / "features_iso_bi.npz")
    ensure_exists(FEATURE_DIR / "features_iso_tri.npz")
    plot_composition()
    plot_hdx_summary()


if __name__ == "__main__":
    main()
