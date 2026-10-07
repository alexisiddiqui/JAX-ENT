"""Checkpoint 24: PyRosetta scores versus the MD structural-W1 KDE target."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

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
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_common import score_term_group
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    OUTPUT as CP19_OUTPUT,
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint24_pyrosetta_energy"
ENERGY_DIR = OUTPUT / "energies"
CP23_PILOT = (
    HERE / "outputs/analysis/pairwise_geometry/checkpoint23_openmm_vacuum/pilot"
)
SCOREFUNCTIONS = ("ref2015", "ref2015_cart")
GROUPS = (
    "packing",
    "solvation",
    "electrostatic_hbond",
    "torsional_rotamer",
    "reference_disulfide",
    "cartesian_bonded",
    "other",
)


def load_score_frames(system: str, config: dict) -> dict[str, np.ndarray]:
    joined: dict[str, list[np.ndarray]] = {}
    expected_keys = None
    for replica in (1, 2, 3):
        path = ENERGY_DIR / system / f"{system}_R{replica}.energies.npz"
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files) - {"frame"}
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"score terms differ across replicas for {system}")
            frames = np.asarray(archive["frame"], dtype=int)
            keep = (
                frames * config["analysis"]["frame_interval_ns"]
                > config["analysis"]["equilibration_ns"]
            )
            for key in sorted(keys):
                values = np.asarray(archive[key], dtype=float)[keep]
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"nonfinite {key} scores for {system} R{replica}")
                joined.setdefault(key, []).append(values)
    result = {key: np.concatenate(parts) for key, parts in joined.items()}
    for scorefunction in SCOREFUNCTIONS:
        terms = {
            key.split("__", 1)[1]: values
            for key, values in result.items()
            if key.startswith(scorefunction + "__") and not key.endswith("__total")
        }
        for group in GROUPS:
            selected = [values for term, values in terms.items() if score_term_group(term) == group]
            if selected:
                result[f"{scorefunction}__group_{group}"] = np.sum(selected, axis=0)
    return result


def pair_difference(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return values[left] - values[right]


def finite_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return np.nan
    return float(pd.Series(target).corr(pd.Series(prediction), method="spearman"))


def evaluate_system(row: dict, config: dict, edges: np.ndarray) -> tuple[list[dict], list[dict]]:
    data = system_data(row, config)
    system = data["system"]
    signed_targets, bandwidth = density_targets(data, FIT, PRIMARY_RANK)
    indices = {
        replica: sampled_indices(system, replica, len(values), PAIR_CAP)
        for replica, values in signed_targets.items()
    }
    scores = load_score_frames(system, config)
    signed_y = {r: signed_targets[r][indices[r]] for r in (1, 2, 3)}
    magnitude_y = {r: np.abs(signed_y[r]) for r in (1, 2, 3)}
    predictions: dict[str, tuple[str, dict[int, np.ndarray]]] = {}
    fit_rows: list[dict] = []

    work = thermodynamic_pair_features(data["z"], data["pairs"])["work_scale"]
    work_selected = {r: work[r][indices[r]] for r in (1, 2, 3)}
    work_alpha = scalar_scale(work_selected[FIT], magnitude_y[FIT], True)[0]
    predictions["work_scale"] = (
        "magnitude",
        {r: work_alpha * work_selected[r] for r in (1, 2, 3)},
    )
    fit_rows.append({"system_id": system, "model": "work_scale", "alpha": work_alpha})

    score_features = [
        key for key in scores if key.endswith("__total") or "__group_" in key
    ]
    for feature in score_features:
        deltas = {}
        for replica, pair_frame in data["pairs"].items():
            left, right = pair_endpoints(pair_frame, indices[replica])
            deltas[replica] = pair_difference(scores[feature], left, right)
        model_base = "pyro_" + feature.replace("__", "_")
        alpha = scalar_scale(np.abs(deltas[FIT]), magnitude_y[FIT], True)[0]
        predictions[model_base + "_fitted_abs"] = (
            "magnitude",
            {r: alpha * np.abs(deltas[r]) for r in (1, 2, 3)},
        )
        fit_rows.append(
            {
                "system_id": system,
                "model": model_base + "_fitted_abs",
                "alpha_log_density_per_reu": alpha,
            }
        )
        if feature.endswith("__total"):
            predictions[model_base + "_raw_signed"] = (
                "signed",
                {r: -deltas[r] for r in (1, 2, 3)},
            )
            predictions[model_base + "_raw_magnitude"] = (
                "magnitude",
                {r: np.abs(deltas[r]) for r in (1, 2, 3)},
            )

    angles, _ = load_backbone_dihedrals(row, config)
    labels, supported, _, _, cluster_meta = structural_clusters(row, config, data, angles)
    test_pairs = data["pairs"][TEST]
    left, right = pair_endpoints(test_pairs, indices[TEST])
    relations = {"all": np.ones(len(left), dtype=bool)} | relation(labels, supported, left, right)
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
                        "mae": float(np.mean(np.abs(target[TEST][mask] - values[TEST][mask]))),
                        "spearman": finite_spearman(
                            target[TEST][mask], values[TEST][mask]
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


def append_openmm_comparators(results: pd.DataFrame) -> pd.DataFrame:
    path = CP23_PILOT / "openmm_population_results.parquet"
    if not path.exists():
        return results
    openmm = pd.read_parquet(path)
    keep = {
        "openmm_total_fitted_abs",
        "openmm_nonbonded_fitted_abs",
        "openmm_torsion_fitted_abs",
    }
    openmm = openmm[openmm.model.isin(keep)].copy()
    common = sorted(set(results.columns) & set(openmm.columns))
    return pd.concat([results, openmm[common]], ignore_index=True, sort=False)


def bootstrap_tail_differences(results: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("pyro_ref2015_total_fitted_abs", "work_scale"),
        ("pyro_ref2015_total_fitted_abs", "openmm_total_fitted_abs"),
        ("pyro_ref2015_cart_total_fitted_abs", "work_scale"),
        ("pyro_ref2015_cart_total_fitted_abs", "openmm_total_fitted_abs"),
    ]
    rng = np.random.default_rng(2401)
    rows = []
    for relation_name in ("all", "between"):
        for band in ("q4", "q5"):
            selected = results[(results.relation == relation_name) & (results.band == band)]
            pivot = selected.pivot_table(
                index="system_id", columns="model", values="distribution_recovery"
            )
            for model, baseline in comparisons:
                if model not in pivot or baseline not in pivot:
                    continue
                differences = (pivot[model] - pivot[baseline]).dropna().to_numpy()
                if not len(differences):
                    continue
                sampled = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
                rows.append(
                    {
                        "model": model,
                        "baseline": baseline,
                        "relation": relation_name,
                        "band": band,
                        "systems": len(differences),
                        "median_paired_difference": float(np.median(differences)),
                        "mean_paired_difference": float(np.mean(differences)),
                        "bootstrap_ci_low": float(np.quantile(sampled, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(sampled, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, summary: pd.DataFrame, fits: pd.DataFrame, edges: np.ndarray, destination) -> None:
    panels = [
        ("ref2015 components", "all", [m for m in summary.model.unique() if m.startswith("pyro_ref2015_group")]),
        ("ref2015_cart components", "all", [m for m in summary.model.unique() if m.startswith("pyro_ref2015_cart_group")]),
        ("Headline: all pairs", "all", ["work_scale", "openmm_total_fitted_abs", "openmm_torsion_fitted_abs", "pyro_ref2015_total_fitted_abs", "pyro_ref2015_cart_total_fitted_abs"]),
        ("Headline: between clusters", "between", ["work_scale", "openmm_total_fitted_abs", "openmm_torsion_fitted_abs", "pyro_ref2015_total_fitted_abs", "pyro_ref2015_cart_total_fitted_abs"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(19, 13), sharey=True)
    for ax, (title, relation_name, models) in zip(axes.flat, panels):
        for model in models:
            selected = summary[(summary.model == model) & (summary.relation == relation_name)].set_index("band")
            if selected.empty:
                continue
            values = np.array([selected.recovery.get(f"q{i}", np.nan) for i in range(6)])
            sd = np.array([selected.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            label = model.replace("pyro_", "").replace("_fitted_abs", "").replace("_group_", ": ")
            ax.plot(range(6), 100 * values, marker="o", label=label)
            ax.fill_between(range(6), np.clip(100 * (values - sd), 0, 100), np.clip(100 * (values + sd), 0, 100), alpha=0.07)
        ax.set_title(title)
        ax.set_xticks(range(6), [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1, 0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    fig.suptitle("PyRosetta score differences versus MD W1-kernel target density")
    fig.tight_layout()
    fig.savefig(destination / "pyrosetta_recovery_global_w1.png", dpi=180)
    plt.close(fig)

    alpha_column = "alpha_log_density_per_reu"
    selected_fits = fits.dropna(subset=[alpha_column])
    models = list(selected_fits.model.unique())
    raw_data = [
        selected_fits.loc[selected_fits.model == model, alpha_column].to_numpy()
        for model in models
    ]
    data = [values[values > 0] for values in raw_data]
    labels = []
    for model, values in zip(models, raw_data):
        label = model.replace("pyro_", "").replace("_fitted_abs", "")
        zero_count = int(np.sum(values == 0))
        labels.append(label + (f"\n({zero_count} zero)" if zero_count else ""))
    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.8), 7))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel(r"Fitted $\alpha$ (log-density / REU)")
    ax.set_title("PyRosetta population-metric scale coefficients")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "pyrosetta_alpha_distributions.png", dpi=180)
    plt.close(fig)


def evaluate_task(arguments):
    row, config, edges = arguments
    return row["system_id"], evaluate_system(row, config, edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()
    if not args.full:
        pilot = set(pd.read_parquet(CP19_OUTPUT / "pilot_systems.parquet").system_id)
        rows = [row for row in rows if row["system_id"] in pilot]
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml") as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    destination = OUTPUT / ("full_single_assignment" if args.full else "pilot")
    destination.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        pyro_results = pd.read_parquet(destination / "pyrosetta_population_results.parquet")
        fits = pd.read_parquet(destination / "pyrosetta_population_fits.parquet")
    else:
        result_rows, fit_rows = [], []
        tasks = [(row, config, edges) for row in rows]
        if args.workers > 1:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            evaluated = executor.map(evaluate_task, tasks)
        else:
            executor = None
            evaluated = map(evaluate_task, tasks)
        for index, (system_id, (result, system_fits)) in enumerate(evaluated, 1):
            result_rows.extend(result)
            fit_rows.extend(system_fits)
            print(f"[{index}/{len(rows)}] {system_id}", flush=True)
        if executor is not None:
            executor.shutdown()
        pyro_results = pd.DataFrame(result_rows)
        fits = pd.DataFrame(fit_rows)
    results = append_openmm_comparators(pyro_results) if not args.full else pyro_results
    summary = results.groupby(["model", "relation", "band"], as_index=False).agg(
        recovery=("distribution_recovery", "median"),
        recovery_sd=("distribution_recovery", "std"),
        mae=("mae", "median"),
        spearman=("spearman", "median"),
        systems=("system_id", "nunique"),
        pairs=("pairs", "sum"),
    )
    tail = bootstrap_tail_differences(results)
    atomic_parquet(pyro_results, destination / "pyrosetta_population_results.parquet")
    atomic_parquet(fits, destination / "pyrosetta_population_fits.parquet")
    atomic_parquet(summary, destination / "pyrosetta_population_summary.parquet")
    atomic_parquet(tail, destination / "pyrosetta_tail_comparisons.parquet")
    plot_results(results, summary, fits, edges, destination)
    atomic_yaml(
        destination / "checkpoint24_report.yaml",
        {
            "checkpoint": 24,
            "systems": len(rows),
            "scorefunctions": list(SCOREFUNCTIONS),
            "assignment": "A-fit/B-unused/C-test; A-only structural clusters",
            "target": "rank-10 MD structural-W1 KDE log-density difference",
            "raw_score_controls": "REU differences; not physical Boltzmann energies",
            "status": "pilot_complete" if not args.full else "full_complete",
        },
    )


if __name__ == "__main__":
    main()
