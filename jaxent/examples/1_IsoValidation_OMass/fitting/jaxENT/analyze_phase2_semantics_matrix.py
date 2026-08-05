#!/usr/bin/env python3
"""Analyze population recovery in the gated Phase-2 semantics matrix."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.frame_averaging import (
    residue_uptake_fast,
    residue_uptake_legacy,
    residue_uptake_slow2,
)
from jaxent.examples.common.analysis.stats import effective_sample_size
from jaxent.src.analysis.pf_variance import kl_to_uniform
from jaxent.src.utils.hdf import load_optimization_history_from_file

LABELS = {"open": 0, "closed": 1, "intermediate": -1}
SEMANTICS_FN = {
    "legacy": residue_uptake_legacy,
    "fast": residue_uptake_fast,
    "slow2": residue_uptake_slow2,
}


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("generate_iso_targets_phase2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()
    root = args.matrix_dir.resolve()
    manifest = json.loads((root / "phase2_matrix_manifest.json").read_text())
    fitting_root = Path(__file__).resolve().parent
    example_root = fitting_root.parents[1]
    generator = load_generator(example_root / "data/generate_iso_targets.py")
    shipped_segments = (
        example_root
        / "data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"
    )

    assignments = {}
    features = {}
    for ensemble in ("iso_bi", "iso_tri"):
        assignments[ensemble] = pd.read_csv(
            example_root
            / f"data/_clustering_results/cluster_assignments_{ensemble.upper()}.csv"
        )["cluster_assignment"].to_numpy(dtype=int)
        with np.load(fitting_root / f"_featurise/features_{ensemble}.npz") as data:
            features[ensemble] = (
                0.35 * data["heavy_contacts"] + 2.0 * data["acceptor_contacts"],
                np.asarray(data["k_ints"]),
            )

    def predict(weights, ensemble, layout, semantics):
        log_pf, k_ints = features[ensemble]
        kwargs = {}
        if semantics == "slow2":
            kwargs["assignments"] = assignments[ensemble]
        uptake = SEMANTICS_FN[semantics](
            log_pf, k_ints, generator.TIMEPOINTS, weights, tau=0.0, **kwargs
        )
        uptake, segments = generator.align_residue_layout(
            uptake,
            fitting_root / f"_featurise/topology_{ensemble}.json",
            shipped_segments,
        )
        if layout == "width10":
            uptake, _ = generator.peptide_aggregate(uptake, segments, 10)
        return uptake.T

    rows = []
    for cell in manifest["cells"]:
        ensemble = cell["ensemble"]
        labels = (0, 1) if ensemble == "iso_bi" else (0, -1, 1)
        names = ("open", "closed") if ensemble == "iso_bi" else (
            "open", "intermediate", "closed"
        )
        target = np.loadtxt(Path(cell["target_dir"]) / "target_dfrac.dat")
        for history_path in sorted(
            (Path(cell["fit_dir"]) / "random").glob("*_results.hdf5")
        ):
            history = load_optimization_history_from_file(str(history_path))
            state = history.get_best_state()
            weights = np.asarray(state.params.frame_weight_simplex, dtype=float)
            weights /= weights.sum()
            recovered = [float(weights[assignments[ensemble] == label].sum()) for label in labels]
            fitted = predict(
                weights, ensemble, cell["layout"], cell["fitter_semantics"]
            )
            row = {
                "ensemble": ensemble,
                "layout": cell["layout"],
                "maxent": cell["maxent"],
                "tilt_kl": cell["tilt_kl"],
                "target_semantics": cell["target_semantics"],
                "fitter_semantics": cell["fitter_semantics"],
                "diagonal": cell["target_semantics"] == cell["fitter_semantics"],
                "history_file": str(history_path),
                "best_step": int(state.step),
                "mse_fit": float(np.mean(np.square(fitted - target))),
                "ess_total": effective_sample_size(weights),
                "kl_from_prior": float(kl_to_uniform(weights)),
            }
            for name in names:
                row[f"true_{name}"] = cell["populations"][name]
            for name, value in zip(names, recovered, strict=True):
                row[f"recovered_{name}"] = value
                row[f"error_{name}"] = value - cell["populations"][name]
            row["abs_open_error"] = abs(row["error_open"])
            rows.append(row)

    if not rows:
        raise FileNotFoundError("no Phase-2 histories found")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "phase2_semantics_matrix.csv", index=False)
    summary = (
        frame.groupby(
            ["ensemble", "layout", "target_semantics", "fitter_semantics"],
            as_index=False,
        )
        .agg(
            mae_open=("abs_open_error", "mean"),
            bias_open=("error_open", "mean"),
            median_mse=("mse_fit", "median"),
            median_ess=("ess_total", "median"),
            n_fits=("history_file", "size"),
        )
    )
    summary.to_csv(root / "phase2_semantics_summary.csv", index=False)

    order = ["legacy", "fast", "slow2"]
    for (ensemble, layout), group in summary.groupby(["ensemble", "layout"]):
        matrix = group.pivot(
            index="target_semantics", columns="fitter_semantics", values="mae_open"
        ).reindex(index=order, columns=order)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix.to_numpy(), cmap="magma_r", vmin=0)
        for row_index in range(3):
            for column_index in range(3):
                ax.text(
                    column_index,
                    row_index,
                    f"{matrix.iloc[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        ax.set_xticks(range(3), order)
        ax.set_yticks(range(3), order)
        ax.set(xlabel="Fitter semantics", ylabel="Target semantics", title=f"{ensemble} {layout}: open-population MAE")
        fig.colorbar(image, ax=ax, label="absolute population error")
        fig.tight_layout()
        fig.savefig(root / f"phase2_matrix_{ensemble}_{layout}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
