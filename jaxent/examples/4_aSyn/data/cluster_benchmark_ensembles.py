#!/usr/bin/env python3
"""cluster_benchmark_ensembles.py

Generates per-frame clustering outputs for all 5 new (md_source × time) combinations
needed for the ensemble-size benchmark. tris_MD 1.0us is the reference and already
has complete outputs in data/_cluster_inertia/.

Combos handled (all via MDAnalysis on the already-truncated trajectories):
  tris_MD  0.25us  → tris_all_combined_0.25us.xtc + project onto tris GMM
  tris_MD  0.5us   → tris_all_combined_0.5us.xtc  + project onto tris GMM
  control_MD 0.25us → control_all_combined_0.25us.xtc + project onto tris GMM
  control_MD 0.5us  → control_all_combined_0.5us.xtc  + project onto tris GMM
  control_MD 1.0us  → control_all_combined.xtc         + project onto tris GMM
                       (skipped by default if _cluster_inertia_control/ exists;
                        use --overwrite to regenerate)

Each output directory contains:
  shape_axes.npy           (n_frames, 2)  — [I1/I3, I2/I3] per frame
  ctail_rg.npy             (n_frames,)    — C-tail Rg in Angstroms
  cluster_labels.npy       (n_frames,)    — GMM component IDs (int)
  macro_cluster_labels.npy (n_frames,)    — strings: Rod / Wavy / Compact
  macro_cluster_map.json   — copy of reference mapping
  cluster_method.json      — metadata: source, time_us, n_frames, method

Usage (run from repo root or 4_aSyn/ directory):
    python data/cluster_benchmark_ensembles.py [--dry-run] [--overwrite] [--base-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import MDAnalysis as mda
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from inertia_moments_clustering import compute_ctail_rg, compute_inertia_ratios  # noqa: E402


def _build_reverse_map(cluster_map: dict) -> dict[int, str]:
    rev: dict[int, str] = {}
    for name, ids in cluster_map.items():
        for cid in ids:
            rev[int(cid)] = name
    return rev


def _save_clustering(
    outdir: Path,
    shape_axes: np.ndarray,
    cluster_labels: np.ndarray,
    ctail_rg: np.ndarray,
    macro_labels: np.ndarray,
    cluster_map: dict,
    metadata: dict,
    dry_run: bool,
    overwrite: bool,
) -> bool:
    if outdir.exists() and not overwrite:
        print(f"  [SKIP] {outdir} already exists (use --overwrite to regenerate)")
        return False
    print(f"  -> {outdir}  ({len(cluster_labels)} frames)")
    if dry_run:
        print("  [dry-run] skipping write")
        return True
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "shape_axes.npy", shape_axes)
    np.save(outdir / "cluster_labels.npy", cluster_labels)
    np.save(outdir / "ctail_rg.npy", ctail_rg)
    np.save(outdir / "macro_cluster_labels.npy", macro_labels)
    np.save(outdir / "coarse_cluster_labels.npy", macro_labels)
    with open(outdir / "macro_cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)
    with open(outdir / "coarse_cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)
    with open(outdir / "cluster_method.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return True


def _project_trajectory(
    top: Path,
    traj: Path,
    time_label: str,
    shape_sel: str,
    ctail_sel: str,
    gmm,
    rev_map: dict[int, str],
    ref_metadata: dict,
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    print(f"  Loading Universe: {traj.name} ...")
    u = mda.Universe(str(top), str(traj))
    x_ratio, y_ratio, *_ = compute_inertia_ratios(u, shape_sel)
    shape_axes = np.column_stack([x_ratio, y_ratio])
    ctail_rg = compute_ctail_rg(u, ctail_sel)
    cluster_labels = gmm.predict(shape_axes)
    macro_labels = np.array(
        [rev_map.get(int(lbl), "Unknown") for lbl in cluster_labels],
        dtype=object,
    )
    metadata = {
        **ref_metadata,
        "source": source,
        "time_us": float(time_label.replace("us", "")),
        "n_frames": int(len(cluster_labels)),
        "method": "projected_from_tris_gmm",
    }
    return shape_axes, cluster_labels, ctail_rg, macro_labels, metadata


def main(dry_run: bool, overwrite: bool, base_dir: Path) -> None:
    ref_dir = base_dir / "data/_cluster_inertia"
    tris_dir = base_dir / "data/_aSyn/tris_MD"
    ctrl_dir = base_dir / "data/_aSyn/control_MD"

    print(f"Loading reference clustering metadata from {ref_dir} ...")
    with open(ref_dir / "coarse_cluster_map.json") as f:
        macro_cluster_map: dict = json.load(f)
    with open(ref_dir / "cluster_method.json") as f:
        ref_metadata: dict = json.load(f)

    ctail_sel = ref_metadata.get("ctail_selection", "resid 115:135")
    shape_sel = ref_metadata["shape_selection"]
    gmm = joblib.load(ref_dir / "gmm_model.pkl")
    rev_map = _build_reverse_map(macro_cluster_map)

    # ── all combos: project truncated trajectories onto tris GMM ──────────
    all_combos = [
        ("tris_MD",    "0.25us", tris_dir / "tris_all_combined_0.25us.xtc",
         tris_dir / "md_mol_center_coil.pdb",
         base_dir / "data/_cluster_inertia_tris_0.25us"),
        ("tris_MD",    "0.5us",  tris_dir / "tris_all_combined_0.5us.xtc",
         tris_dir / "md_mol_center_coil.pdb",
         base_dir / "data/_cluster_inertia_tris_0.5us"),
        ("control_MD", "0.25us", ctrl_dir / "control_all_combined_0.25us.xtc",
         ctrl_dir / "md.gro.pdb",
         base_dir / "data/_cluster_inertia_control_0.25us"),
        ("control_MD", "0.5us",  ctrl_dir / "control_all_combined_0.5us.xtc",
         ctrl_dir / "md.gro.pdb",
         base_dir / "data/_cluster_inertia_control_0.5us"),
        ("control_MD", "1.0us",  ctrl_dir / "control_all_combined.xtc",
         ctrl_dir / "md.gro.pdb",
         base_dir / "data/_cluster_inertia_control"),
    ]

    for source, time_label, traj_path, top_path, out_dir in all_combos:
        print(f"\n=== {source} {time_label} (project onto tris GMM) ===")
        if not traj_path.exists():
            print(f"  WARNING: trajectory not found: {traj_path} — skipping.")
            continue
        sa, cl, cr, ml, meta = _project_trajectory(
            top_path, traj_path,
            time_label, shape_sel, ctail_sel, gmm, rev_map, ref_metadata,
            source=source,
        )
        _save_clustering(out_dir, sa, cl, cr, ml, macro_cluster_map, meta, dry_run, overwrite)

    print("\n=== All clustering complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate shape-space clustering for benchmark ensemble combos."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output directories (default: skip existing).",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Base 4_aSyn directory. Defaults to two levels above this script.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve() if args.base_dir else SCRIPT_DIR.parent
    main(dry_run=args.dry_run, overwrite=args.overwrite, base_dir=base)
