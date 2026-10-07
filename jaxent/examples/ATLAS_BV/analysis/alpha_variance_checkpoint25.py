"""Checkpoint 25: system-specific scale coefficients versus system variance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jaxent.examples.ATLAS_BV.analysis.cluster_stratified_checkpoint22 import FIT
from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_systems
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    density_targets,
    scalar_scale,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.openmm_energy_population_checkpoint23 import (
    ENERGY_DIR as OPENMM_ENERGY_DIR,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import pair_endpoints
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_population_checkpoint24 import (
    ENERGY_DIR as PYROSETTA_ENERGY_DIR,
    load_score_frames,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_frame_features,
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint25_alpha_variance"
OPENMM_TOTAL_ONLY_DIR = (
    HERE
    / "outputs/analysis/pairwise_geometry/checkpoint23_openmm_vacuum/energies_total_only"
)
MODELS = (
    "work_scale",
    "work_density_legacy_zq",
    "openmm_total",
    "pyro_ref2015_total",
    "pyro_ref2015_cart_total",
)
LABELS = {
    "work_scale": "Work Scale",
    "work_density_legacy_zq": "Work Density, legacy Zq",
    "openmm_total": "OpenMM total",
    "pyro_ref2015_total": "PyRosetta ref2015 total",
    "pyro_ref2015_cart_total": "PyRosetta ref2015_cart total",
}


def retained_frame_values(values: np.ndarray, config: dict) -> np.ndarray:
    frames = np.arange(len(values), dtype=int)
    keep = (
        frames * config["analysis"]["frame_interval_ns"]
        > config["analysis"]["equilibration_ns"]
    )
    return np.asarray(values, dtype=float)[keep]


def evaluate_system(row: dict, config: dict) -> list[dict]:
    data = system_data(row, config)
    system = data["system"]
    signed_targets, bandwidth = density_targets(data, FIT, PRIMARY_RANK)
    indices = sampled_indices(system, FIT, len(signed_targets[FIT]), PAIR_CAP)
    target = np.abs(signed_targets[FIT][indices])
    pair_frame = data["pairs"][FIT]
    left, right = pair_endpoints(pair_frame, indices)
    thermo_pairs = thermodynamic_pair_features(data["z"], data["pairs"])
    thermo_frames = thermodynamic_frame_features(data["z"])
    features: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
        "work_scale": (
            thermo_pairs["work_scale"][FIT][indices],
            np.asarray(thermo_frames["work_scale"], dtype=float),
        ),
        "work_density_legacy_zq": (
            thermo_pairs["work_density_legacy_zq"][FIT][indices],
            None,
        ),
    }
    openmm_root = (
        OPENMM_TOTAL_ONLY_DIR
        if (OPENMM_TOTAL_ONLY_DIR / system).exists()
        else OPENMM_ENERGY_DIR
    )
    if (openmm_root / system).exists():
        parts = []
        for replica in (1, 2, 3):
            path = openmm_root / system / f"{system}_R{replica}.energies.npz"
            with np.load(path, allow_pickle=False) as archive:
                frames = np.asarray(archive["frame"], dtype=int)
                keep = (
                    frames * config["analysis"]["frame_interval_ns"]
                    > config["analysis"]["equilibration_ns"]
                )
                parts.append(np.asarray(archive["energy_total_kj_mol"], dtype=float)[keep])
        openmm = np.concatenate(parts)
        features["openmm_total"] = (np.abs(openmm[left] - openmm[right]), openmm)
    if (PYROSETTA_ENERGY_DIR / system).exists():
        pyrosetta = load_score_frames(system, config)
        for scorefunction in ("ref2015", "ref2015_cart"):
            values = pyrosetta[f"{scorefunction}__total"]
            features[f"pyro_{scorefunction}_total"] = (
                np.abs(values[left] - values[right]),
                values,
            )
    rows = []
    for model, (feature, frame_score) in features.items():
        alpha = scalar_scale(feature, target, True)[0]
        rows.append(
            {
                "system_id": system,
                "model": model,
                "alpha": alpha,
                "pair_feature_variance": float(np.var(feature, ddof=1)),
                "pair_feature_sd": float(np.std(feature, ddof=1)),
                "pair_feature_mean": float(np.mean(feature)),
                "target_variance": float(np.var(target, ddof=1)),
                "target_sd": float(np.std(target, ddof=1)),
                "target_mean": float(np.mean(target)),
                "frame_score_variance": (
                    float(np.var(frame_score, ddof=1)) if frame_score is not None else np.nan
                ),
                "pairs": len(feature),
                "bandwidth_angstrom": bandwidth,
            }
        )
    return rows


def evaluate_task(arguments):
    return evaluate_system(*arguments)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(10000):
        indices = rng.integers(0, len(x), len(x))
        if np.ptp(x[indices]) == 0 or np.ptp(y[indices]) == 0:
            continue
        values.append(float(spearmanr(x[indices], y[indices]).statistic))
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def correlations(rows: pd.DataFrame, matched_systems: set[str]) -> pd.DataFrame:
    output = []
    comparisons = (
        ("pair_feature_variance", "Unscaled pair-feature variance"),
        ("target_variance", "MD target variance"),
    )
    for cohort, cohort_rows in (
        ("available", rows),
        ("matched_openmm", rows[rows.system_id.isin(matched_systems)]),
    ):
        for model in MODELS:
            selected = cohort_rows[cohort_rows.model == model]
            for variance, description in comparisons:
                finite = selected[np.isfinite(selected.alpha) & np.isfinite(selected[variance])]
                if len(finite) < 3:
                    continue
                result = spearmanr(finite[variance], finite.alpha)
                low, high = bootstrap_spearman(
                    finite[variance].to_numpy(),
                    finite.alpha.to_numpy(),
                    2500 + len(output),
                )
                output.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "variance": variance,
                        "variance_description": description,
                        "systems": len(finite),
                        "spearman_rho": float(result.statistic),
                        "spearman_pvalue": float(result.pvalue),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    }
                )
    return pd.DataFrame(output)


def plot_correlation_panels(
    rows: pd.DataFrame,
    stats: pd.DataFrame,
    variance: str,
    cohort: str,
    matched_systems: set[str],
    destination,
) -> None:
    selected_rows = rows if cohort == "available" else rows[rows.system_id.isin(matched_systems)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, model in zip(axes.flat, MODELS):
        selected = selected_rows[selected_rows.model == model]
        axis.scatter(selected[variance], selected.alpha, s=34, alpha=0.75)
        if np.all(selected[variance] > 0):
            axis.set_xscale("log")
        if np.all(selected.alpha > 0):
            axis.set_yscale("log")
        result = stats[
            (stats.cohort == cohort)
            & (stats.model == model)
            & (stats.variance == variance)
        ]
        if len(result):
            value = result.iloc[0]
            axis.set_title(
                f"{LABELS[model]}\n"
                f"Spearman ρ={value.spearman_rho:.2f} "
                f"[{value.bootstrap_ci_low:.2f}, {value.bootstrap_ci_high:.2f}], "
                f"n={int(value.systems)}"
            )
        else:
            axis.set_title(LABELS[model])
        axis.set_xlabel(variance.replace("_", " "))
        axis.set_ylabel("Fitted system α")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle(
        "System α versus system variance\n"
        + (
            "All available systems"
            if cohort == "available"
            else f"Matched OpenMM {len(matched_systems)}-system cohort"
        )
    )
    fig.tight_layout()
    fig.savefig(destination / f"alpha_vs_{variance}_{cohort}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()
    tasks = [(row, config) for row in rows]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(evaluate_task, tasks))
    else:
        evaluated = [evaluate_task(task) for task in tasks]
    diagnostics = pd.DataFrame([item for group in evaluated for item in group])
    matched_systems = set(
        diagnostics.loc[diagnostics.model == "openmm_total", "system_id"]
    )
    stats = correlations(diagnostics, matched_systems)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_parquet(diagnostics, OUTPUT / "alpha_variance_systems.parquet")
    atomic_parquet(stats, OUTPUT / "alpha_variance_correlations.parquet")
    for cohort in ("available", "matched_openmm"):
        for variance in ("pair_feature_variance", "target_variance"):
            plot_correlation_panels(
                diagnostics, stats, variance, cohort, matched_systems, OUTPUT
            )
    atomic_yaml(
        OUTPUT / "checkpoint25_report.yaml",
        {
            "checkpoint": 25,
            "target": "rank-10 MD structural-W1 KDE magnitude; replica A",
            "primary_variance": "variance of exact unscaled fit-pair predictor",
            "pair_cap": PAIR_CAP,
            "available_systems": diagnostics.groupby("model").system_id.nunique().to_dict(),
            "matched_openmm_systems": len(matched_systems),
        },
    )


if __name__ == "__main__":
    main()
