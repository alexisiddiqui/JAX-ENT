"""Checkpoint 23: OpenMM vacuum energies versus the MD structural-W1 KDE target."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis.cluster_stratified_checkpoint22 import (
    FIT,
    TEST,
    relation,
    structural_clusters,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    density_targets,
    mass_metrics,
    scalar_scale,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import (
    load_backbone_dihedrals,
    pair_endpoints,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    OUTPUT as CP19_OUTPUT,
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint23_openmm_vacuum"
ENERGY_DIR = OUTPUT / "energies"
GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324
COMPONENT_MODELS = ("total", "bonded", "torsion", "nonbonded")


def load_energy_frames(system: str, config: dict) -> dict[str, np.ndarray]:
    by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in ("total", "bond", "angle", "torsion", "nonbonded", "other")
    }
    expected_keep = None
    for replica in (1, 2, 3):
        path = ENERGY_DIR / system / f"{system}_R{replica}.energies.npz"
        with np.load(path, allow_pickle=False) as archive:
            frames = np.asarray(archive["frame"], dtype=int)
            keep = (
                frames * config["analysis"]["frame_interval_ns"]
                > config["analysis"]["equilibration_ns"]
            )
            if expected_keep is None:
                expected_keep = int(np.sum(keep))
            if int(np.sum(keep)) != expected_keep:
                raise ValueError(
                    f"inconsistent retained energy frames for {system} R{replica}"
                )
            for name in by_name:
                values = np.asarray(archive[f"energy_{name}_kj_mol"], dtype=float)[keep]
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"nonfinite {name} energy for {system} R{replica}")
                by_name[name].append(values)
    joined = {name: np.concatenate(parts) for name, parts in by_name.items()}
    joined["bonded"] = (
        joined["bond"] + joined["angle"] + joined["torsion"] + joined["other"]
    )
    return joined


def pair_energy_difference(
    energy: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return energy[left] - energy[right]


def direct_boltzmann_log_ratio(
    delta_energy_kj_mol: np.ndarray, temperature_k: float
) -> np.ndarray:
    return -np.asarray(delta_energy_kj_mol, dtype=float) / (
        GAS_CONSTANT_KJ_MOL_K * temperature_k
    )


def evaluate_system(
    row: dict, config: dict, edges: np.ndarray
) -> tuple[list[dict], list[dict]]:
    data = system_data(row, config)
    system = data["system"]
    signed_targets, bandwidth = density_targets(data, FIT, PRIMARY_RANK)
    indices = {
        replica: sampled_indices(system, replica, len(values), PAIR_CAP)
        for replica, values in signed_targets.items()
    }
    energy = load_energy_frames(system, config)
    work = thermodynamic_pair_features(data["z"], data["pairs"])["work_scale"]
    deltas: dict[str, dict[int, np.ndarray]] = {name: {} for name in COMPONENT_MODELS}
    for replica, pair_frame in data["pairs"].items():
        left, right = pair_endpoints(pair_frame, indices[replica])
        for name in COMPONENT_MODELS:
            deltas[name][replica] = pair_energy_difference(energy[name], left, right)
    signed_y = {r: signed_targets[r][indices[r]] for r in (1, 2, 3)}
    magnitude_y = {r: np.abs(signed_y[r]) for r in (1, 2, 3)}
    predictions: dict[str, tuple[str, dict[int, np.ndarray]]] = {}
    fit_rows = []
    work_selected = {r: work[r][indices[r]] for r in (1, 2, 3)}
    work_alpha = scalar_scale(work_selected[FIT], magnitude_y[FIT], True)[0]
    predictions["work_scale"] = (
        "magnitude",
        {r: work_alpha * work_selected[r] for r in (1, 2, 3)},
    )
    fit_rows.append({"system_id": system, "model": "work_scale", "alpha": work_alpha})
    for name in COMPONENT_MODELS:
        alpha = scalar_scale(np.abs(deltas[name][FIT]), magnitude_y[FIT], True)[0]
        model_name = f"openmm_{name}_fitted_abs"
        predictions[model_name] = (
            "magnitude",
            {r: alpha * np.abs(deltas[name][r]) for r in (1, 2, 3)},
        )
        fit_rows.append(
            {
                "system_id": system,
                "model": model_name,
                "alpha_mol_per_kj": alpha,
                "effective_temperature_k": (
                    1.0 / (GAS_CONSTANT_KJ_MOL_K * alpha) if alpha > 0 else np.inf
                ),
            }
        )
    boltzmann = {
        r: direct_boltzmann_log_ratio(
            deltas["total"][r], config["protocol"]["temperature_k"]
        )
        for r in (1, 2, 3)
    }
    predictions["openmm_total_boltzmann_signed"] = ("signed", boltzmann)
    predictions["openmm_total_boltzmann_magnitude"] = (
        "magnitude",
        {r: np.abs(values) for r, values in boltzmann.items()},
    )

    angles, _ = load_backbone_dihedrals(row, config)
    labels, supported, _, _, cluster_meta = structural_clusters(
        row, config, data, angles
    )
    test_pairs = data["pairs"][TEST]
    left, right = pair_endpoints(test_pairs, indices[TEST])
    relations = {"all": np.ones(len(left), dtype=bool)} | relation(
        labels, supported, left, right
    )
    test_w1 = test_pairs.w1.to_numpy()[indices[TEST]]
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    result_rows = []
    for model, (target_kind, values) in predictions.items():
        target = signed_y if target_kind == "signed" else magnitude_y
        for relation_name, relation_mask in relations.items():
            for band in range(6):
                band_mask = (test_w1 >= edges[band]) & (
                    (test_w1 < edges[band + 1]) if band < 5 else True
                )
                mask = relation_mask & band_mask
                unique_frames = len(np.unique(np.r_[left[mask], right[mask]]))
                if mask.sum() < 30 or unique_frames < 20:
                    continue
                result_rows.append(
                    {
                        "system_id": system,
                        "model": model,
                        "target_kind": target_kind,
                        "relation": relation_name,
                        "band": f"q{band}",
                        "pairs": int(mask.sum()),
                        "unique_frames": unique_frames,
                        "bandwidth_angstrom": bandwidth,
                        "cluster_space": cluster_meta["selected_space"],
                        "mae": float(
                            np.mean(np.abs(target[TEST][mask] - values[TEST][mask]))
                        ),
                        **mass_metrics(
                            target[TEST][mask],
                            values[TEST][mask],
                            target[FIT],
                            settings["distribution_bins"],
                            settings["distribution_smoothing"],
                        ),
                    }
                )
    return result_rows, fit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()
    if not args.full:
        pilot = set(pd.read_parquet(CP19_OUTPUT / "pilot_systems.parquet").system_id)
        rows = [row for row in rows if row["system_id"] in pilot]
    with open(
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml"
    ) as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    result_rows, fit_rows = [], []
    for index, row in enumerate(rows, 1):
        result, fits = evaluate_system(row, config, edges)
        result_rows.extend(result)
        fit_rows.extend(fits)
        print(f"[{index}/{len(rows)}] {row['system_id']}", flush=True)
    suffix = "full_single_assignment" if args.full else "pilot"
    destination = OUTPUT / suffix
    destination.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    fits = pd.DataFrame(fit_rows)
    atomic_parquet(results, destination / "openmm_population_results.parquet")
    atomic_parquet(fits, destination / "openmm_population_fits.parquet")
    summary = results.groupby(["model", "relation", "band"], as_index=False).agg(
        recovery=("distribution_recovery", "median"),
        recovery_sd=("distribution_recovery", "std"),
        systems=("system_id", "nunique"),
    )
    atomic_parquet(summary, destination / "openmm_population_summary.parquet")
    models = list(results.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, relation_name in zip(axes, ("all", "between")):
        for model in models:
            selected = summary[
                (summary.model == model) & (summary.relation == relation_name)
            ].set_index("band")
            values = np.array(
                [selected.recovery.get(f"q{i}", np.nan) for i in range(6)]
            )
            sd = np.array([selected.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            ax.plot(range(6), 100 * values, marker="o", label=model)
            ax.fill_between(
                range(6),
                np.clip(100 * (values - sd), 0, 100),
                np.clip(100 * (values + sd), 0, 100),
                alpha=0.06,
            )
        ax.set_title("All pairs" if relation_name == "all" else "Between clusters")
        counts = (
            results[results.relation == relation_name]
            .groupby("band")
            .system_id.nunique()
        )
        ax.set_xticks(
            range(6),
            [
                f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å\nn={counts.get(f'q{i}', 0)}"
                for i in range(6)
            ],
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "openmm_recovery_global_w1.png", dpi=180)
    plt.close(fig)
    atomic_yaml(
        destination / "checkpoint23_report.yaml",
        {
            "checkpoint": 23,
            "systems": len(rows),
            "assignment": "A-fit/B-unused/C-test; A-only structural clusters",
            "temperature_k": config["protocol"]["temperature_k"],
            "status": "pilot_complete" if not args.full else "full_complete",
        },
    )


if __name__ == "__main__":
    main()
