"""
Split SAXS q-points into train/val sets using three strategies.

Loads an experimental SAXS .dat file (q, I(q), error) with 501 q-points,
splits across train/val using built-in DataSplitter whole-system strategies
(random, stratified, random-stratified),
saves .npz splits for downstream fitting, and produces publication figures.

Example:

python jaxent/examples/4_SAXS/data/split_data_SAXS_SASBDB.py \
--saxs-data jaxent/examples/4_SAXS/data/_CaM/raw_data/SASDNY3/experimental_data/SASDNY3.dat \
--output-dir jaxent/examples/4_SAXS/fitting/_datasplits_CaM+CDZ \
--name CaM+CDZ \
--n-splits 3 \
--train-size 0.5 \
--seed 42



python jaxent/examples/4_SAXS/data/split_data_SAXS_SASBDB.py \
--saxs-data jaxent/examples/4_SAXS/data/_CaM/raw_data/SASDNX3/experimental_data/SASDNX3.dat \
--output-dir jaxent/examples/4_SAXS/fitting/_datasplits_CaM-CDZ \
--name CaM-CDZ \
--n-splits 3 \
--train-size 0.5 \
--seed 42

"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from jaxent.src.custom_types.SAXS import SAXS_curve
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology import Partial_Topology


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAXS .dat file (handles SASBDB headers and column names).

    Returns:
        (q_values, intensities, errors) as numpy arrays of shape (n_q,)
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the first line that contains numeric values (q, I, sd)
    skiprows = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                # Try converting first part to float; if successful, this is likely data
                float(parts[0])
                skiprows = i
                break
            except ValueError:
                continue

    dat = np.loadtxt(path, skiprows=skiprows)
    return dat[:, 0], dat[:, 1], dat[:, 2]


# ============================================================================
# SPLITTING FUNCTIONS (built-in DataSplitter whole-system methods)
# ============================================================================

SPLIT_METHODS = {
    "random": "random_split",
    "stratified": "stratified_deterministic_split",
    "random-stratified": "stratified_random_split",
}


def _build_saxs_dataloader(
    q_values: np.ndarray,
    intensities: np.ndarray,
    errors: np.ndarray,
) -> ExpD_Dataloader:
    """Build a single-point whole-system SAXS dataloader for DataSplitter."""
    full_topology = Partial_Topology(
        chain="A",
        residues=[1],
        fragment_name="whole_construct",
        fragment_index=0,
    )
    saxs_curve = SAXS_curve(
        top=full_topology,
        intensities=intensities,
        q_values=q_values,
        errors=errors,
    )
    return ExpD_Dataloader(data=[saxs_curve])


def _split_with_datasplitter(
    q_values: np.ndarray,
    intensities: np.ndarray,
    errors: np.ndarray,
    split_type: str,
    train_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one split replicate via built-in DataSplitter and return q-indices."""
    dataloader = _build_saxs_dataloader(q_values, intensities, errors)
    splitter = DataSplitter(
        dataloader,
        random_seed=seed,
        train_size=train_size,
        ensemble=None,
        common_residues=None,
    )

    if split_type == "random":
        splitter.random_split()
    elif split_type == "stratified":
        splitter.stratified_deterministic_split()
    elif split_type == "random-stratified":
        splitter.stratified_random_split(n_strata=10)
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    train_idx = np.asarray(splitter.last_train_q_indices, dtype=np.int32)
    val_idx = np.asarray(splitter.last_val_q_indices, dtype=np.int32)

    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError(
            f"DataSplitter returned empty train/val indices for split type '{split_type}'"
        )

    return np.sort(train_idx), np.sort(val_idx)


# ============================================================================
# SPLIT SAVING
# ============================================================================

def save_splits(
    output_dir: Path,
    q: np.ndarray,
    intensities: np.ndarray,
    errors: np.ndarray,
    split_type: str,
    split_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Save a single train/val split to .npz files.

    Saves to:
        {output_dir}/{split_type}/split_{split_idx:03d}/train.npz
        {output_dir}/{split_type}/split_{split_idx:03d}/val.npz

    Each .npz contains: q_values, intensities, errors
    """
    split_dir = output_dir / split_type / f"split_{split_idx:03d}"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Train split
    np.savez(
        split_dir / "train.npz",
        q_values=q[train_idx],
        intensities=intensities[train_idx],
        errors=errors[train_idx],
    )

    # Val split
    np.savez(
        split_dir / "val.npz",
        q_values=q[val_idx],
        intensities=intensities[val_idx],
        errors=errors[val_idx],
    )


def save_full_dataset(
    output_dir: Path,
    q: np.ndarray,
    intensities: np.ndarray,
    errors: np.ndarray,
) -> None:
    """Save full dataset for reference."""
    np.savez(
        output_dir / "full_dataset.npz",
        q_values=q,
        intensities=intensities,
        errors=errors,
    )


# ============================================================================
# FIGURES
# ============================================================================

def plot_split_curves(
    output_dir: Path,
    split_type: str,
    q: np.ndarray,
    intensities: np.ndarray,
    errors: np.ndarray,
    splits_data: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Plot per-replicate train/val split curves.

    Creates a 1xN figure with one panel per replicate, showing the base curve
    with train/val points overlaid. Saves to split_type/split_visualization.png.

    Args:
        output_dir: Base output directory
        split_type: Name of split strategy
        q: Full q array
        intensities: Full intensities array
        errors: Full errors array
        splits_data: List of (train_idx, val_idx) tuples (one per replicate)
    """
    n_replicates = len(splits_data)

    rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, n_replicates, figsize=(5 * n_replicates, 4), sharey=True)
    if n_replicates == 1:
        axes = [axes]

    for i, (train_idx, val_idx) in enumerate(splits_data):
        ax = axes[i]

        # Base curve (light gray)
        ax.semilogy(q, intensities, color="lightgray", lw=0.8, zorder=1)

        # Train and val overlays
        ax.errorbar(
            q[train_idx], intensities[train_idx],
            yerr=errors[train_idx],
            fmt="o", ms=3, lw=0.8, color="#1f77b4",
            label=f"Train ({len(train_idx)})", zorder=3
        )
        ax.errorbar(
            q[val_idx], intensities[val_idx],
            yerr=errors[val_idx],
            fmt="o", ms=3, lw=0.8, color="#ff7f0e",
            label=f"Val ({len(val_idx)})", zorder=2
        )

        ax.set_title(f"{split_type} — replicate {i}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_xlabel("q (Å⁻¹)")
    axes[0].set_ylabel("I(q)")

    fig.tight_layout()
    output_path = output_dir / split_type / "split_visualization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_q_coverage(
    output_dir: Path,
    q: np.ndarray,
    split_types: list[str],
    splits_by_type: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Plot per-split-type q-point coverage across replicates.

    For each split type, computes the fraction of replicates where each q-point
    is in the train set. Saves to q_coverage.png.

    Args:
        output_dir: Base output directory
        q: Full q array
        split_types: List of split type names
        splits_by_type: Dict mapping split_type -> list of (train_idx, val_idx) tuples
    """
    rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, len(split_types), figsize=(5 * len(split_types), 4), sharey=True)
    if len(split_types) == 1:
        axes = [axes]

    for ax, split_type in zip(axes, split_types):
        splits = splits_by_type[split_type]
        n_q = len(q)
        n_replicates = len(splits)

        # Compute fraction of replicates where each q is in train
        train_frac = np.zeros(n_q)
        for train_idx, _ in splits:
            train_frac[train_idx] += 1
        train_frac /= n_replicates

        # Fill between
        ax.fill_between(q, train_frac, 1.0, color="#ff7f0e", alpha=0.6, label="Val")
        ax.fill_between(q, 0, train_frac, color="#1f77b4", alpha=0.6, label="Train")

        # Reference line
        ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)

        ax.set_ylim(0, 1)
        ax.set_title(split_type)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_xlabel("q (Å⁻¹)")
    axes[0].set_ylabel("Fraction in train set")

    fig.tight_layout()
    output_path = output_dir / "q_coverage.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split SAXS q-points into train/val sets using built-in DataSplitter strategies."
    )
    parser.add_argument(
        "--saxs-data",
        type=Path,
        required=True,
        help="Path to .dat file (q, I(q), error)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for splits and figures",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Label for plot titles (default: stem of saxs-data)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of replicates per split type",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.5,
        help="Fraction of q-points for train set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (replicate i uses seed+i)",
    )
    parser.add_argument(
        "--split-types",
        nargs="+",
        default=list(SPLIT_METHODS.keys()),
        choices=list(SPLIT_METHODS.keys()),
        help="Subset of split types to run",
    )
    parser.add_argument("--max_q", type=float, default=0.5, help="Maximum q-value to consider from the .dat files.",)

    args = parser.parse_args()

    # Load data
    q, intensities, errors = load_dat(args.saxs_data)
    n_q_full = len(q)

    # Apply max_q filtering
    mask = q <= args.max_q
    q = q[mask]
    intensities = intensities[mask]
    errors = errors[mask]
    n_q = len(q)

    print(f"Loaded {n_q_full} q-points; after max_q={args.max_q} filtering: {n_q} q-points")

    # Save full dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_full_dataset(args.output_dir, q, intensities, errors)
    print(f"Saved full dataset ({n_q} q-points) to {args.output_dir}/full_dataset.npz")

    # Generate splits
    splits_by_type = {}
    for split_type in args.split_types:
        splits_by_type[split_type] = []

        for split_idx in range(args.n_splits):
            seed = args.seed + split_idx

            train_idx, val_idx = _split_with_datasplitter(
                q_values=q,
                intensities=intensities,
                errors=errors,
                split_type=split_type,
                train_size=args.train_size,
                seed=seed,
            )

            # Save split
            save_splits(
                args.output_dir,
                q, intensities, errors,
                split_type, split_idx,
                train_idx, val_idx
            )
            splits_by_type[split_type].append((train_idx, val_idx))

        print(f"Saved {args.n_splits} {split_type} splits")

    # Create figures
    for split_type in args.split_types:
        plot_split_curves(
            args.output_dir,
            split_type,
            q, intensities, errors,
            splits_by_type[split_type]
        )
        print(f"Saved {split_type}/split_visualization.png")

    plot_q_coverage(args.output_dir, q, args.split_types, splits_by_type)
    print(f"Saved q_coverage.png")


if __name__ == "__main__":
    main()
