#!/usr/bin/env python3
"""Compute the Stage 0 data-reward/MaxEnt-cost budget without fitting."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.frame_averaging import (
    residue_uptake_legacy,
    weights_from_cluster_populations,
)
from jaxent.src.analysis.pf_variance import kl_to_uniform

LABELS = {"open": 0, "closed": 1, "intermediate": -1}


def load_generator_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_iso_targets", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load target generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage0_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.stage0_dir / "stage0_sweep_manifest.json").read_text())
    fitting_root = Path(__file__).resolve().parent
    example_root = fitting_root.parents[1]
    generator = load_generator_module(example_root / "data/generate_iso_targets.py")
    shipped_segments = (
        example_root / "data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"
    )

    uniform_predictions: dict[tuple[str, str], np.ndarray] = {}
    assignments_by_ensemble: dict[str, np.ndarray] = {}
    rows = []
    for cell in manifest["cells"]:
        ensemble = cell["ensemble"]
        layout = cell["layout"]
        key = (ensemble, layout)
        suffix = ensemble.removeprefix("iso_").upper()
        if ensemble not in assignments_by_ensemble:
            assignments_by_ensemble[ensemble] = pd.read_csv(
                example_root
                / f"data/_clustering_results/cluster_assignments_ISO_{suffix}.csv"
            )["cluster_assignment"].to_numpy(dtype=int)
        assignments = assignments_by_ensemble[ensemble]
        populations = {LABELS[name]: value for name, value in cell["populations"].items()}
        true_weights = weights_from_cluster_populations(assignments, populations)

        if key not in uniform_predictions:
            feature_path = fitting_root / f"_featurise/features_{ensemble}.npz"
            topology_path = fitting_root / f"_featurise/topology_{ensemble}.json"
            with np.load(feature_path) as features:
                log_pf = 0.35 * features["heavy_contacts"] + 2.0 * features["acceptor_contacts"]
                k_ints = np.asarray(features["k_ints"])
            uniform = np.full(len(assignments), 1.0 / len(assignments))
            uptake = residue_uptake_legacy(
                log_pf, k_ints, generator.TIMEPOINTS, uniform, tau=0.0
            )
            uptake, segments = generator.align_residue_layout(
                uptake, topology_path, shipped_segments
            )
            if layout == "width10":
                uptake, _ = generator.peptide_aggregate(uptake, segments, 10)
            uniform_predictions[key] = uptake.T

        target = np.loadtxt(Path(cell["root"]) / "target/target_dfrac.dat")
        mse_reward = float(np.mean(np.square(uniform_predictions[key] - target)))
        kl_cost = float(kl_to_uniform(true_weights))
        rows.append(
            {
                "ensemble": ensemble,
                "layout": layout,
                **{f"true_{name}": value for name, value in cell["populations"].items()},
                "kl_true_to_uniform": kl_cost,
                "mse_uniform_to_target": mse_reward,
                "critical_maxent_strength": mse_reward / kl_cost if kl_cost > 0 else np.inf,
            }
        )

    output = args.stage0_dir / "stage0_budget.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
