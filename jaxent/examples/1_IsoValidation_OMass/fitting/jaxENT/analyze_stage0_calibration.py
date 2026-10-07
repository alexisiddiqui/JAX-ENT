#!/usr/bin/env python3
"""Analyze Stage 0 population recovery, ESS, KL, and recovery Jacobians."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.stats import effective_sample_size
from jaxent.examples.common.analysis.frame_averaging import (
    residue_uptake_legacy,
    weights_from_cluster_populations,
)
from jaxent.src.analysis.pf_variance import conditional_subset_effective_sample_size, kl_to_uniform
from jaxent.src.utils.hdf import load_optimization_history_from_file

LABEL_ORDER = {"iso_tri": (0, -1, 1), "iso_bi": (0, 1)}
NAME_ORDER = {"iso_tri": ("open", "intermediate", "closed"), "iso_bi": ("open", "closed")}
LABELS = {"open": 0, "closed": 1, "intermediate": -1}


def load_generator_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_iso_targets", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load target generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_probabilities(weights: np.ndarray, assignments: np.ndarray, ensemble: str) -> np.ndarray:
    return np.asarray([weights[assignments == label].sum() for label in LABEL_ORDER[ensemble]])


def history_states(path: Path):
    history = load_optimization_history_from_file(str(path))
    candidates = [(f"state_{index}", state) for index, state in enumerate(history.states)]
    candidates.extend(
        (f"convergence_{index}", state)
        for index, state in enumerate(history.convergence_states)
    )
    if history.best_state is not None:
        candidates.append(("best_state", history.best_state))
    if not candidates:
        raise ValueError(f"optimization history contains no fitted state: {path}")
    return history, candidates


def state_weights(state) -> np.ndarray:
    weights = np.asarray(state.params.frame_weight_simplex, dtype=float)
    return weights / weights.sum()


def add_jacobian(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["jacobian_condition"] = np.nan
    output["sloppy_direction"] = ""
    for (ensemble, layout, maxent), indices in output.groupby(
        ["ensemble", "layout", "maxent"]
    ).groups.items():
        group = output.loc[indices]
        if ensemble == "iso_tri":
            true = group[["true_open", "true_intermediate"]].to_numpy()
            recovered = group[["recovered_open", "recovered_intermediate"]].to_numpy()
            design = np.column_stack((np.ones(len(true)), true))
            jacobian = np.linalg.lstsq(design, recovered, rcond=None)[0][1:].T
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            condition = singular_values[0] / singular_values[-1] if singular_values[-1] else np.inf
            eigenvalues, eigenvectors = np.linalg.eigh(jacobian.T @ jacobian)
            sloppy = eigenvectors[:, np.argmin(eigenvalues)]
            output.loc[indices, "jacobian_open_open"] = jacobian[0, 0]
            output.loc[indices, "jacobian_open_intermediate"] = jacobian[0, 1]
            output.loc[indices, "jacobian_intermediate_open"] = jacobian[1, 0]
            output.loc[indices, "jacobian_intermediate_intermediate"] = jacobian[1, 1]
            output.loc[indices, "sloppy_direction"] = f"[{sloppy[0]:.6g},{sloppy[1]:.6g}]"
        else:
            slope = np.polyfit(group.true_open, group.recovered_open, 1)[0]
            condition = 1.0 if slope != 0 else np.inf
            output.loc[indices, "jacobian_open_open"] = slope
            output.loc[indices, "sloppy_direction"] = "[1]"
        output.loc[indices, "jacobian_condition"] = condition
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage0_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.stage0_dir / "stage0_sweep_manifest.json").read_text())
    example_root = Path(__file__).resolve().parents[2]
    fitting_root = Path(__file__).resolve().parent
    generator = load_generator_module(example_root / "data/generate_iso_targets.py")
    shipped_segments = (
        example_root / "data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"
    )
    assignment_dir = example_root / "data/_clustering_results"
    assignments = {
        ensemble: pd.read_csv(assignment_dir / f"cluster_assignments_ISO_{ensemble.removeprefix('iso_').upper()}.csv")["cluster_assignment"].to_numpy(dtype=int)
        for ensemble in LABEL_ORDER
    }
    feature_data = {}
    for ensemble in LABEL_ORDER:
        with np.load(fitting_root / f"_featurise/features_{ensemble}.npz") as features:
            feature_data[ensemble] = (
                0.35 * features["heavy_contacts"] + 2.0 * features["acceptor_contacts"],
                np.asarray(features["k_ints"]),
            )

    def prediction(weights: np.ndarray, ensemble: str, layout: str) -> np.ndarray:
        log_pf, k_ints = feature_data[ensemble]
        uptake = residue_uptake_legacy(
            log_pf, k_ints, generator.TIMEPOINTS, weights, tau=0.0
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
        root = Path(cell["root"])
        ensemble = cell["ensemble"]
        layout = cell["layout"]
        target = np.loadtxt(root / "target/target_dfrac.dat")
        true_weights = weights_from_cluster_populations(
            assignments[ensemble],
            {LABELS[name]: value for name, value in cell["populations"].items()},
        )
        uniform_weights = np.full(len(assignments[ensemble]), 1.0 / len(assignments[ensemble]))
        mse_true = float(np.mean(np.square(prediction(true_weights, ensemble, layout) - target)))
        mse_uniform = float(
            np.mean(np.square(prediction(uniform_weights, ensemble, layout) - target))
        )
        fit_dirs = cell.get("fit_dirs", {"1": str(root / "fit")})
        for maxent_text, fit_dir_text in fit_dirs.items():
            for history_path in sorted((Path(fit_dir_text) / "random").glob("*_results.hdf5")):
                history, candidates = history_states(history_path)
                evaluated = []
                for source, state in candidates:
                    candidate_weights = state_weights(state)
                    candidate_mse = float(
                        np.mean(
                            np.square(prediction(candidate_weights, ensemble, layout) - target)
                        )
                    )
                    evaluated.append((candidate_mse, source, state, candidate_weights))
                mse_fit, selected_source, selected_state, weights = min(
                    evaluated, key=lambda item: item[0]
                )
                recovered = cluster_probabilities(weights, assignments[ensemble], ensemble)
                names = NAME_ORDER[ensemble]
                open_mask = assignments[ensemble] == 0
                open_ess = float(conditional_subset_effective_sample_size(weights, open_mask))
                last_convergence_mse = np.nan
                if history.convergence_states:
                    last_weights = state_weights(history.convergence_states[-1])
                    last_convergence_mse = float(
                        np.mean(np.square(prediction(last_weights, ensemble, layout) - target))
                    )
                best_state_mse = np.nan
                if history.best_state is not None:
                    best_weights = state_weights(history.best_state)
                    best_state_mse = float(
                        np.mean(np.square(prediction(best_weights, ensemble, layout) - target))
                    )
                row = {
                    "ensemble": ensemble,
                    "layout": cell["layout"],
                    "maxent": float(maxent_text),
                    "history_file": str(history_path),
                    "selected_state": selected_source,
                    "selected_step": int(selected_state.step),
                    "ess_total": effective_sample_size(weights),
                    "ess_open": open_ess,
                    "ess_open_fraction": open_ess / int(open_mask.sum()),
                    "kl_from_prior": float(kl_to_uniform(weights)),
                    "mse_true": mse_true,
                    "mse_fit": mse_fit,
                    "mse_uniform": mse_uniform,
                    "mse_fit_minus_uniform": mse_fit - mse_uniform,
                    "mse_last_convergence": last_convergence_mse,
                    "mse_best_state": best_state_mse,
                }
                row.update({f"true_{name}": cell["populations"][name] for name in names})
                row.update({f"recovered_{name}": value for name, value in zip(names, recovered, strict=True)})
                rows.append(row)
    if not rows:
        raise FileNotFoundError("no optimization history files found under the Stage 0 directory")
    frame = add_jacobian(pd.DataFrame(rows))
    frame.to_csv(args.stage0_dir / "stage0_calibration.csv", index=False)
    for (ensemble, layout), group in frame.groupby(["ensemble", "layout"]):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        for maxent, strength_group in group.groupby("maxent"):
            label = f"{maxent:g}"
            axes[0].scatter(
                strength_group.kl_from_prior,
                strength_group.recovered_open,
                label=label,
                alpha=0.7,
            )
            axes[1].scatter(
                strength_group.true_open,
                strength_group.recovered_open,
                label=label,
                alpha=0.7,
            )
        axes[0].set(xlabel="KL(fitted weights || uniform prior)", ylabel="Recovered open population", title="Recovery vs KL from prior")
        axes[0].legend(title="MaxEnt")
        limits = [0, 1]
        axes[1].plot(limits, limits, "k--", linewidth=1)
        axes[1].set(xlabel="True open population", ylabel="Recovered open population", xlim=limits, ylim=limits, title="Open-state calibration")
        axes[1].legend(title="MaxEnt")
        summary = group.groupby("maxent", as_index=False).first().sort_values("maxent")
        axes[2].plot(summary.maxent, summary.jacobian_open_open, marker="o")
        axes[2].set_xscale("symlog", linthresh=1e-3)
        axes[2].set(
            xlabel="MaxEnt strength",
            ylabel="Open/open Jacobian",
            title="Recovery sensitivity vs regularisation",
        )
        fit_quality = group.mse_fit / group.mse_uniform
        axes[3].scatter(fit_quality, group.recovered_open - group.true_open, c=group.maxent)
        axes[3].axvline(1.0, color="k", linestyle="--", linewidth=1)
        axes[3].axhline(0.0, color="k", linestyle=":", linewidth=1)
        axes[3].set(
            xlabel="MSE(fit) / MSE(uniform)",
            ylabel="Recovered open - true open",
            title="Recovery error vs fit quality",
        )
        fig.tight_layout()
        fig.savefig(args.stage0_dir / f"stage0_calibration_{ensemble}_{layout}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
