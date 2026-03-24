"""Analyse CaM pulldown fitting results.

Agnostic to the fitting scenario — works with any HDF5 outputs whose
run_name follows the convention:

    {scenario}_{target}_{split_type}_split{idx:03d}_maxent{strength}

e.g. SAXS_CaM+CDZ_random-stratified_split000_maxent0.01

Produces per-(scenario, target) plots:
  - Effective sample size (ESS)
  - Primary cluster weight distribution (APO-like/HOLO-like/Neither)
  - n_Ca_final vs n_CDZ_ligands_final heatmap
  - Ensemble-weighted SAXS curves vs experimental (SAXS scenario only)

Usage:
    python CaM_pulldown_analyse_results.py --results-dir _optimise_CaM_SAXS_KLD_20260324_120000
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from jaxent.src.utils.hdf import load_optimization_history_from_file
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters

from common import load_dat

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FRAME_ORDERING_PATH = (
    SCRIPT_DIR.parent
    / "ensemble_generation"
    / "neuralplexer"
    / "collected_structures"
    / "frame_ordering.csv"
)
SAXS_FEATURES_PATH = DATA_DIR / "_SAXS_features" / "SAXS_curve_input_features.npz"
EXPERIMENTAL_DATA = {
    "CaM+CDZ": DATA_DIR / "_CaM" / "raw_data" / "SASDNY3" / "experimental_data" / "SASDNY3.dat",
    "CaM-CDZ": DATA_DIR / "_CaM" / "raw_data" / "SASDNX3" / "experimental_data" / "SASDNX3.dat",
}

# Matches known split types to avoid ambiguity with "+" and "-" in target names
_RUN_NAME_PATTERN = re.compile(
    r"^([A-Za-z]+)_([\w+\-]+)_(random-stratified|stratified|random)"
    r"_split(\d+)_maxent([\d.eE+\-]+)"
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_kld(weights: np.ndarray) -> float:
    """KL divergence from uniform distribution."""
    n = len(weights)
    uniform = 1.0 / n
    w = np.clip(weights, 1e-10, None)
    return float(np.sum(w * np.log(w / uniform)))


def compute_ess(weights: np.ndarray) -> float:
    """Effective sample size."""
    return float(1.0 / np.sum(weights ** 2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_run_name(filename: str) -> dict | None:
    """Parse run_name from HDF5 filename, return dict or None if no match."""
    m = _RUN_NAME_PATTERN.search(filename)
    if m is None:
        return None
    return {
        "scenario": m.group(1),
        "target": m.group(2),
        "split_type": m.group(3),
        "split_index": int(m.group(4)),
        "maxent_strength": float(m.group(5)),
    }


def load_results(results_dir: Path) -> pd.DataFrame:
    """Walk results directory, load HDF5 histories, compute metrics.

    Returns DataFrame with columns:
        scenario, target, split_type, split_index, maxent_strength,
        kld, ess, best_val_loss, final_frame_weights, hdf_path
    """
    records = []
    for hdf_path in sorted(results_dir.rglob("*_results.hdf5")):
        if "_EMA" in hdf_path.name:
            continue

        params = parse_run_name(hdf_path.stem)
        if params is None:
            print(f"Warning: could not parse {hdf_path.name}")
            continue

        try:
            history = load_optimization_history_from_file(
                str(hdf_path),
                default_model_params_cls=SAXS_direct_Model_Parameters,
            )
        except Exception as e:
            print(f"Warning: could not load {hdf_path}: {e}")
            continue

        best = history.get_best_state()
        weights = np.abs(np.array(best.params.frame_weights))
        weights /= weights.sum()

        records.append({
            **params,
            "kld": compute_kld(weights),
            "ess": compute_ess(weights),
            "best_val_loss": float(best.losses.val_losses[0]),
            "final_frame_weights": weights,
            "hdf_path": str(hdf_path),
        })

    return pd.DataFrame(records)


def select_best_per_replicate(df: pd.DataFrame) -> pd.DataFrame:
    """For each (scenario, target, split_type, split_index), keep the row
    with the minimum validation loss (best maxent_strength)."""
    best = []
    for _, group in df.groupby(["scenario", "target", "split_type", "split_index"]):
        best.append(group.loc[group["best_val_loss"].idxmin()])
    return pd.DataFrame(best).reset_index(drop=True)


def load_frame_ordering() -> pd.DataFrame:
    """Load frame_ordering.csv and normalise Primary_cluster to string."""
    fo = pd.read_csv(FRAME_ORDERING_PATH)
    fo["Primary_cluster"] = fo["Primary_cluster"].astype(str)
    return fo


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def plot_ess(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of ESS per split type, one figure per (scenario, target)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for (scenario, target), group in df.groupby(["scenario", "target"]):
        split_types = sorted(group["split_type"].unique())
        means, ses = zip(*[
            _mean_se(group[group["split_type"] == st]["ess"].tolist())
            for st in split_types
        ])

        fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(split_types)), 4))
        x = np.arange(len(split_types))
        ax.bar(x, means, yerr=ses, capsize=5, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(split_types)
        ax.set_yscale("log")
        ax.set_ylabel("ESS")
        ax.set_title(f"Effective Sample Size — {scenario} | {target}")
        ax.grid(True, alpha=0.3, axis="y", which="both")
        plt.tight_layout()

        fname = f"ESS_{scenario}_{target}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_primary_cluster_distribution(
    df: pd.DataFrame, frame_ordering: pd.DataFrame, output_dir: Path
) -> None:
    """Bar chart of weighted Primary_cluster fractions per split type.

    Primary_cluster values: '0' (APO-like), '1' (HOLO-like), 'n' (Neither).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_labels = sorted(frame_ordering["Primary_cluster"].unique())
    label_map = {"0": "APO-like", "1": "HOLO-like", "n": "Neither"}
    colors = {"0": "#1f77b4", "1": "#d62728", "n": "#7f7f7f"}

    for (scenario, target), group in df.groupby(["scenario", "target"]):
        split_types = sorted(group["split_type"].unique())
        fig, axes = plt.subplots(
            1, len(split_types),
            figsize=(4 * len(split_types), 4),
            sharey=True,
        )
        if len(split_types) == 1:
            axes = [axes]

        fig.suptitle(f"Primary Cluster Distribution — {scenario} | {target}")

        for ax, split_type in zip(axes, split_types):
            group_st = group[group["split_type"] == split_type]
            fractions_per_rep = []
            for _, row in group_st.iterrows():
                w = row["final_frame_weights"]
                fracs = {
                    c: float(w[frame_ordering["Primary_cluster"] == c].sum())
                    for c in cluster_labels
                }
                fractions_per_rep.append(fracs)

            x = np.arange(len(cluster_labels))
            for i, c in enumerate(cluster_labels):
                vals = [r[c] for r in fractions_per_rep]
                mean, se = _mean_se(vals)
                ax.bar(
                    i, mean, yerr=se, capsize=5,
                    color=colors.get(c, "gray"), alpha=0.75,
                    label=label_map.get(c, c),
                )

            ax.set_xticks(x)
            ax.set_xticklabels([label_map.get(c, c) for c in cluster_labels], fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_title(split_type)
            ax.grid(True, alpha=0.3, axis="y")
            if ax is axes[0]:
                ax.set_ylabel("Weighted fraction")

        plt.tight_layout()
        fname = f"PrimaryCluster_{scenario}_{target}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_ca_cdz_heatmap(
    df: pd.DataFrame, frame_ordering: pd.DataFrame, output_dir: Path
) -> None:
    """Mean heatmap of weights over n_Ca_final (0-4) × n_CDZ_ligands_final (0-2).

    One figure per (scenario, target) with one panel per split type.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_ca_vals = sorted(frame_ordering["n_Ca_final"].unique())
    n_cdz_vals = sorted(frame_ordering["n_CDZ_ligands_final"].unique())

    for (scenario, target), group in df.groupby(["scenario", "target"]):
        split_types = sorted(group["split_type"].unique())
        fig, axes = plt.subplots(
            1, len(split_types),
            figsize=(4 * len(split_types), 4),
            sharey=True,
        )
        if len(split_types) == 1:
            axes = [axes]

        fig.suptitle(f"Ca²⁺ / CDZ occupancy — {scenario} | {target}")

        vmax = 0.0
        heatmaps = {}
        for split_type in split_types:
            group_st = group[group["split_type"] == split_type]
            maps = []
            for _, row in group_st.iterrows():
                w = row["final_frame_weights"]
                hm = np.zeros((len(n_ca_vals), len(n_cdz_vals)))
                for i, ca in enumerate(n_ca_vals):
                    for j, cdz in enumerate(n_cdz_vals):
                        mask = (
                            (frame_ordering["n_Ca_final"] == ca)
                            & (frame_ordering["n_CDZ_ligands_final"] == cdz)
                        )
                        hm[i, j] = float(w[mask].sum())
                maps.append(hm)
            mean_hm = np.mean(maps, axis=0)
            heatmaps[split_type] = mean_hm
            vmax = max(vmax, mean_hm.max())

        for ax, split_type in zip(axes, split_types):
            im = ax.imshow(
                heatmaps[split_type],
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=vmax,
                cmap="Blues",
            )
            ax.set_xticks(range(len(n_cdz_vals)))
            ax.set_xticklabels(n_cdz_vals)
            ax.set_yticks(range(len(n_ca_vals)))
            ax.set_yticklabels(n_ca_vals)
            ax.set_xlabel("n_CDZ_ligands_final")
            ax.set_title(split_type)
            if ax is axes[0]:
                ax.set_ylabel("n_Ca_final")
            fig.colorbar(im, ax=ax, label="Weighted fraction")

        plt.tight_layout()
        fname = f"CaCDZ_heatmap_{scenario}_{target}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_saxs_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Ensemble-weighted SAXS curves vs experimental (SAXS scenario only)."""
    saxs_df = df[df["scenario"] == "SAXS"]
    if saxs_df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_matrix = np.load(SAXS_FEATURES_PATH)["intensities"]  # (n_q, n_frames)

    for target, group in saxs_df.groupby("target"):
        exp_dat = EXPERIMENTAL_DATA.get(target)
        if exp_dat is None or not exp_dat.exists():
            print(f"Warning: experimental data not found for {target}, skipping SAXS curves")
            continue

        exp_q, exp_I, exp_err = load_dat(exp_dat)
        split_types = sorted(group["split_type"].unique())

        fig, axes = plt.subplots(
            1, len(split_types),
            figsize=(5 * len(split_types), 4),
            sharey=True,
        )
        if len(split_types) == 1:
            axes = [axes]

        fig.suptitle(f"Ensemble SAXS curves — SAXS | {target}")

        for ax, split_type in zip(axes, split_types):
            group_st = group[group["split_type"] == split_type]
            curves = []
            for _, row in group_st.iterrows():
                curves.append(curve_matrix @ row["final_frame_weights"])

            curves = np.array(curves)
            mean_curve = curves.mean(axis=0)
            sd_curve = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_curve)

            pearson_r, _ = stats.pearsonr(mean_curve, exp_I)
            spearman_r, _ = stats.spearmanr(mean_curve, exp_I)

            ax.semilogy(exp_q, exp_I, "k-", lw=2, label="Experimental")
            ax.fill_between(exp_q, exp_I - exp_err, exp_I + exp_err, color="gray", alpha=0.2)
            ax.semilogy(exp_q, mean_curve, "r--", lw=2, label="Ensemble mean")
            lower = np.maximum(mean_curve - sd_curve, 1e-15)
            ax.fill_between(exp_q, lower, mean_curve + sd_curve, color="red", alpha=0.2)

            ax.set_xlabel("q (Å⁻¹)")
            ax.set_title(f"{split_type}\nPearson r={pearson_r:.3f}  Spearman ρ={spearman_r:.3f}")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend()
            if ax is axes[0]:
                ax.set_ylabel("I(q)")

        plt.tight_layout()
        fname = f"SAXS_curves_SAXS_{target}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse CaM pulldown fitting results")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Root directory containing HDF5 result files")
    args = parser.parse_args()

    print("Loading results...")
    df = load_results(args.results_dir)

    if df.empty:
        print("No results found — check --results-dir path.")
        return

    print(f"Loaded {len(df)} runs")
    print(df[["scenario", "target", "split_type", "split_index",
               "maxent_strength", "ess", "best_val_loss"]].to_string(index=False))

    print("\nSelecting best model per replicate...")
    best_df = select_best_per_replicate(df)
    print(f"{len(best_df)} best models selected")

    print("\nLoading frame ordering...")
    frame_ordering = load_frame_ordering()
    print(f"  {len(frame_ordering)} frames, Primary_cluster values: "
          f"{sorted(frame_ordering['Primary_cluster'].unique())}")

    output_dir = args.results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPlotting ESS...")
    plot_ess(best_df, output_dir)

    print("Plotting primary cluster distribution...")
    plot_primary_cluster_distribution(best_df, frame_ordering, output_dir)

    print("Plotting Ca²⁺/CDZ heatmaps...")
    plot_ca_cdz_heatmap(best_df, frame_ordering, output_dir)

    print("Plotting SAXS curves...")
    plot_saxs_curves(best_df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
