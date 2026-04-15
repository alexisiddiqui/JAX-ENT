"""
jaxent-ipca — multi-trajectory PCA visualisation CLI.

Usage example
-------------
jaxent-ipca \\
    --topology   wt.pdb       --topology   mut.pdb      --topology   mut.pdb \\
    --trajectory wt_r1.xtc    --trajectory mut_r1.xtc   --trajectory mut_r2.xtc \\
    --condition  WT            --condition  MUT           --condition  MUT \\
    --replicate  1             --replicate  1             --replicate  2 \\
    --output_dir ./ipca_out \\
    --name       my_analysis \\
    --atom_selection "name CA" \\
    --num_components 10 \\
    --chunk_size 100




    
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import MDAnalysis as mda
import numpy as np

from jaxent.src.analysis.PCA.core import calculate_multi_traj_pca
from jaxent.src.analysis.PCA.plots import (
    _build_condition_cmap_map,
    plot_combined_density,
    plot_combined_scatter,
    plot_condition_replicates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("jaxent.ipca")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="jaxent-ipca",
        description=(
            "Multi-trajectory incremental PCA with condition/replicate-aware "
            "publication-quality plots."
        ),
    )
    p.add_argument(
        "--topology",
        dest="topologies",
        action="append",
        required=True,
        metavar="FILE",
        help="Topology file (repeat once per trajectory).",
    )
    p.add_argument(
        "--trajectory",
        dest="trajectories",
        action="append",
        required=True,
        metavar="FILE",
        help="Trajectory file (repeat once per topology).",
    )
    p.add_argument(
        "--condition",
        dest="conditions",
        action="append",
        required=True,
        metavar="LABEL",
        help="Condition label, e.g. WT or MUT (repeat once per trajectory).",
    )
    p.add_argument(
        "--replicate",
        dest="replicates",
        type=int,
        action="append",
        required=True,
        metavar="INT",
        help="Replicate number (repeat once per trajectory).",
    )
    p.add_argument("--output_dir", required=True, help="Root output directory.")
    p.add_argument("--name", required=True, help="Analysis name used in filenames.")
    p.add_argument(
        "--atom_selection",
        default="name CA",
        help='MDAnalysis atom selection string (default: "name CA").',
    )
    p.add_argument(
        "--num_components",
        type=int,
        default=10,
        help="Number of PCA components (default: 10).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Frames per IncrementalPCA batch (default: 100).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace) -> None:
    """Hard-fail with an informative message on malformed inputs."""
    lists = [args.topologies, args.trajectories, args.conditions, args.replicates]
    lengths = [len(x) for x in lists]
    if len(set(lengths)) != 1:
        names = ["--topology", "--trajectory", "--condition", "--replicate"]
        counts = dict(zip(names, lengths))
        msg = (
            "All repeated arguments must be given the same number of times.\n"
            + "  "
            + "\n  ".join(f"{k}: {v}" for k, v in counts.items())
        )
        logger.error(msg)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)

    plots_dir = os.path.join(args.output_dir, "plots")
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load universes
    # ------------------------------------------------------------------
    logger.info("Loading %d trajectory/topology pair(s)…", len(args.topologies))
    universes: list[mda.Universe] = []
    for top, traj, cond, rep in zip(
        args.topologies, args.trajectories, args.conditions, args.replicates
    ):
        logger.info("  [%s rep%d] topology=%s  trajectory=%s", cond, rep, top, traj)
        universes.append(mda.Universe(top, traj))

    # ------------------------------------------------------------------
    # 2. Validate atom counts
    # ------------------------------------------------------------------
    ref_n_atoms: int | None = None
    for i, u in enumerate(universes):
        n = u.select_atoms(args.atom_selection).n_atoms
        if ref_n_atoms is None:
            ref_n_atoms = n
        elif n != ref_n_atoms:
            logger.error(
                "Atom count mismatch: topology 0 has %d atoms but topology %d has %d atoms "
                "(selection: '%s'). All topologies must produce the same atom count.",
                ref_n_atoms,
                i,
                n,
                args.atom_selection,
            )
            sys.exit(1)
    logger.info("Atom count OK: %d atoms per topology.", ref_n_atoms)

    # ------------------------------------------------------------------
    # 3. Joint PCA
    # ------------------------------------------------------------------
    pca_coords, explained_variance, metadata = calculate_multi_traj_pca(
        universes,
        atom_selection=args.atom_selection,
        n_components=args.num_components,
        chunk_size=args.chunk_size,
    )

    # Attach condition / replicate metadata alongside the core traj/frame indices
    conditions_arr = np.empty(len(metadata["traj_idx"]), dtype=object)
    replicates_arr = np.empty(len(metadata["traj_idx"]), dtype=np.int32)
    for i, (cond, rep) in enumerate(zip(args.conditions, args.replicates)):
        mask = metadata["traj_idx"] == i
        conditions_arr[mask] = cond
        replicates_arr[mask] = rep

    metadata["conditions"] = conditions_arr
    metadata["replicates"] = replicates_arr

    # ------------------------------------------------------------------
    # 4. Save data
    # ------------------------------------------------------------------
    coords_path = os.path.join(data_dir, "pca_coordinates.npz")
    np.savez(
        coords_path,
        coords=pca_coords,
        conditions=conditions_arr,
        replicates=replicates_arr,
        traj_indices=metadata["traj_idx"],
        frame_indices=metadata["frame_idx"],
    )
    logger.info("Saved PCA coordinates → %s", coords_path)

    variance_path = os.path.join(data_dir, "pca_variance.npz")
    np.savez(variance_path, explained_variance_ratio=explained_variance)
    logger.info(
        "Saved variance data → %s  (total explained: %.1f%%)",
        variance_path,
        explained_variance.sum() * 100,
    )

    # ------------------------------------------------------------------
    # 5. Build condition→colormap mapping
    # ------------------------------------------------------------------
    cmap_map = _build_condition_cmap_map(list(args.conditions))

    # ------------------------------------------------------------------
    # 6. Figure 1: combined scatter
    # ------------------------------------------------------------------
    scatter_path = os.path.join(plots_dir, "ipca_combined_scatter.png")
    plot_combined_scatter(
        pca_coords, metadata, cmap_map, scatter_path, explained_variance
    )

    # ------------------------------------------------------------------
    # 7. Figure 2: combined density
    # ------------------------------------------------------------------
    density_path = os.path.join(plots_dir, "ipca_combined_density.png")
    plot_combined_density(
        pca_coords, metadata, cmap_map, density_path, explained_variance
    )

    # ------------------------------------------------------------------
    # 8. Figure 3: per-condition replicate panels (one per condition)
    # ------------------------------------------------------------------
    unique_conditions = list(dict.fromkeys(args.conditions))
    for cond in unique_conditions:
        rep_path = os.path.join(plots_dir, f"ipca_{cond}_replicates.png")
        plot_condition_replicates(
            pca_coords, metadata, cond, rep_path, explained_variance
        )

    logger.info("jaxent-ipca complete — results in %s", args.output_dir)


if __name__ == "__main__":
    main()
