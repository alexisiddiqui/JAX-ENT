"""Checkpoint 21: PF-W1 and variance-scaled PF information-distance pilot."""

from __future__ import annotations

import argparse

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from MDAnalysis.analysis.dihedrals import Dihedral

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
    post_equilibration_indices,
    replica_paths,
)
from jaxent.examples.ATLAS_BV.analysis.contact_difference_pilot_checkpoint20 import regions
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    density_targets,
    mass_metrics,
    scalar_scale,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import pf_pair_distance
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    OUTPUT as CHECKPOINT19_OUTPUT,
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint21_pf_information_pilot"
FIT_REPLICA, TUNE_REPLICA, TEST_REPLICA = 1, 2, 3
VARIANCE_SHRINKAGES = (0.001, 0.01, 0.1)
BASELINE_MODELS = (
    "work_scale",
    "structural_w1_control",
    "backbone_drmsd_control",
    "backbone_zdrmsd_control",
    "backbone_zquadratic_control",
    "legacy_density",
    "absolute_l1",
    "l2",
)
CANDIDATE_MODELS = ("pf_w1_raw", "pf_w1_centered", "information_quadratic", "information_root")
ALL_MODELS = (*BASELINE_MODELS, *CANDIDATE_MODELS)
DIHEDRAL_MODELS = (
    "backbone_drmsd_control",
    "backbone_zdrmsd_control",
    "backbone_zquadratic_control",
)
MODEL_LABELS = {
    "work_scale": "Work Scale",
    "structural_w1_control": "Structural W1 control",
    "backbone_drmsd_control": "Backbone dRMSD control",
    "backbone_zdrmsd_control": "Circular z-dRMSD control",
    "backbone_zquadratic_control": "Circular z-quadratic control",
    "legacy_density": r"Legacy density ($Zq$)",
    "absolute_l1": "Aligned-residue absolute L1",
    "l2": "Aligned-residue L2",
    "pf_w1_raw": "PF-W1 (raw)",
    "pf_w1_centered": "PF-W1 (mean-centred)",
    "information_quadratic": "Variance-scaled quadratic",
    "information_root": "Variance-scaled Mahalanobis",
}


def pf_w1_frame_profiles(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted raw and frame-mean-centred residue logPF profiles."""
    raw = np.sort(np.asarray(z, dtype=float), axis=0)
    centered = raw - np.mean(z, axis=0, keepdims=True)
    return raw, centered


def mean_absolute_pair_distance(
    profiles: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Mean absolute profile distance for frame pairs without a full residue x pair allocation."""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    result = np.empty(len(left), dtype=float)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        result[start:stop] = np.mean(
            np.abs(profiles[:, left[start:stop]] - profiles[:, right[start:stop]]), axis=0
        )
    return result


def training_residue_variance(z: np.ndarray, replicas: np.ndarray, fit_replica: int) -> np.ndarray:
    """Per-residue population variance estimated solely from the declared fit replica."""
    training = np.asarray(z, dtype=float)[:, np.asarray(replicas) == fit_replica]
    if training.shape[1] < 2:
        raise ValueError(f"replica {fit_replica} has fewer than two frames")
    return np.var(training, axis=1, ddof=1)


def regularized_residue_variance(variance: np.ndarray, shrinkage: float) -> tuple[np.ndarray, float]:
    """Add a median-positive-variance ridge and a numerical floor."""
    variance = np.asarray(variance, dtype=float)
    finite_positive = variance[np.isfinite(variance) & (variance > 0)]
    reference = float(np.median(finite_positive)) if len(finite_positive) else 1.0
    floor = max(float(shrinkage) * reference, np.finfo(float).eps * reference)
    regularized = np.where(np.isfinite(variance) & (variance >= 0), variance, 0.0) + floor
    return regularized, reference


def variance_scaled_pair_distance(
    z: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    variance: np.ndarray,
    root: bool,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Diagonal variance-scaled quadratic cost or its Mahalanobis root."""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    inverse_variance = 1.0 / np.asarray(variance, dtype=float)
    result = np.empty(len(left), dtype=float)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        delta = z[:, left[start:stop]] - z[:, right[start:stop]]
        result[start:stop] = np.mean(np.square(delta) * inverse_variance[:, None], axis=0)
    if root:
        np.sqrt(result, out=result)
    return result


def pooled_analysis_frame_indices(
    total_frames: int,
    replicas: int,
    equilibration_ns: float,
    frame_interval_ns: float,
) -> np.ndarray:
    """Trajectory indices in the same replica-major order used by the PF analysis."""
    if total_frames % replicas:
        raise ValueError(f"{total_frames} trajectory frames are not divisible by {replicas} replicas")
    per_replica = total_frames // replicas
    keep = post_equilibration_indices(per_replica, equilibration_ns, frame_interval_ns)
    return np.concatenate([replica * per_replica + keep for replica in range(replicas)])


def load_backbone_dihedrals(
    row: dict[str, str], config: dict,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Load periodic phi/psi angles, in radians, for the analysis frames."""
    universe = mda.Universe(HERE / row["pdb_path"], *replica_paths(row))
    protein_residues = universe.select_atoms("protein").residues
    atomgroups = []
    complete_residues = 0
    for residue in protein_residues:
        phi = residue.phi_selection()
        psi = residue.psi_selection()
        if phi is not None and psi is not None:
            atomgroups.extend((phi, psi))
            complete_residues += 1
    if not atomgroups:
        raise ValueError(f"{row['system_id']}: no residues have both phi and psi angles")
    settings = config["analysis"]
    frame_indices = pooled_analysis_frame_indices(
        universe.trajectory.n_frames,
        3,
        settings["equilibration_ns"],
        settings["frame_interval_ns"],
    )
    angles_degrees = np.asarray(Dihedral(atomgroups).run(frames=frame_indices).results.angles)
    if angles_degrees.shape != (len(frame_indices), len(atomgroups)):
        raise ValueError(
            f"{row['system_id']}: unexpected dihedral shape {angles_degrees.shape}; "
            f"expected {(len(frame_indices), len(atomgroups))}"
        )
    if not np.isfinite(angles_degrees).all():
        raise ValueError(f"{row['system_id']}: non-finite backbone dihedral angles")
    metadata: dict[str, float | int] = {
        "protein_residues": int(len(protein_residues)),
        "dihedral_residues": complete_residues,
        "dihedral_angles": int(len(atomgroups)),
        "dihedral_residue_coverage": float(complete_residues / len(protein_residues)),
    }
    return np.deg2rad(angles_degrees).T, metadata


def periodic_dihedral_pair_distance(
    angles_radians: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Periodic phi/psi RMS difference in degrees for pooled frame pairs."""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    result = np.empty(len(left), dtype=float)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        delta = angles_radians[:, left[start:stop]] - angles_radians[:, right[start:stop]]
        wrapped = np.arctan2(np.sin(delta), np.cos(delta))
        result[start:stop] = np.rad2deg(np.sqrt(np.mean(np.square(wrapped), axis=0)))
    return result


def training_circular_variance(
    angles_radians: np.ndarray, replicas: np.ndarray, fit_replica: int,
) -> np.ndarray:
    """Mean squared wrapped deviations around the fit-replica circular mean."""
    training = np.asarray(angles_radians, dtype=float)[:, np.asarray(replicas) == fit_replica]
    if training.shape[1] < 2:
        raise ValueError(f"replica {fit_replica} has fewer than two dihedral frames")
    mean_angle = np.arctan2(np.mean(np.sin(training), axis=1), np.mean(np.cos(training), axis=1))
    deviation = training - mean_angle[:, None]
    wrapped = np.arctan2(np.sin(deviation), np.cos(deviation))
    return np.mean(np.square(wrapped), axis=1)


def variance_scaled_dihedral_pair_distance(
    angles_radians: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    variance: np.ndarray,
    root: bool,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Circular diagonal quadratic cost or RMS distance in angular z-score space."""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    inverse_variance = 1.0 / np.asarray(variance, dtype=float)
    result = np.empty(len(left), dtype=float)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        delta = angles_radians[:, left[start:stop]] - angles_radians[:, right[start:stop]]
        wrapped = np.arctan2(np.sin(delta), np.cos(delta))
        result[start:stop] = np.mean(np.square(wrapped) * inverse_variance[:, None], axis=0)
    if root:
        np.sqrt(result, out=result)
    return result


def pair_endpoints(pair_frame: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        pair_frame.left_frame.to_numpy()[indices],
        pair_frame.right_frame.to_numpy()[indices],
    )


def fixed_pair_features(
    data: dict,
    sample_indices: dict[int, np.ndarray],
) -> dict[str, dict[int, np.ndarray]]:
    """Baselines and PF-W1 candidates without variance-dependent features."""
    thermo = thermodynamic_pair_features(data["z"], data["pairs"])
    raw_profiles, centered_profiles = pf_w1_frame_profiles(data["z"])
    output = {name: {} for name in (*BASELINE_MODELS, "pf_w1_raw", "pf_w1_centered")}
    for replica, pair_frame in data["pairs"].items():
        selected = sample_indices[replica]
        left, right = pair_endpoints(pair_frame, selected)
        output["work_scale"][replica] = thermo["work_scale"][replica][selected]
        output["structural_w1_control"][replica] = pair_frame.w1.to_numpy()[selected]
        output["legacy_density"][replica] = thermo["work_density_legacy_zq"][replica][selected]
        output["absolute_l1"][replica] = pf_pair_distance(data["z"], left, right, "l1")
        output["l2"][replica] = pf_pair_distance(data["z"], left, right, "l2")
        output["pf_w1_raw"][replica] = mean_absolute_pair_distance(raw_profiles, left, right)
        output["pf_w1_centered"][replica] = mean_absolute_pair_distance(centered_profiles, left, right)
    return output


def information_features(
    data: dict,
    sample_indices: dict[int, np.ndarray],
    shrinkage: float,
) -> tuple[dict[str, dict[int, np.ndarray]], float, float]:
    variance = training_residue_variance(data["z"], data["replicas"], FIT_REPLICA)
    regularized, reference = regularized_residue_variance(variance, shrinkage)
    output = {"information_quadratic": {}, "information_root": {}}
    for replica, pair_frame in data["pairs"].items():
        left, right = pair_endpoints(pair_frame, sample_indices[replica])
        quadratic = variance_scaled_pair_distance(data["z"], left, right, regularized, root=False)
        output["information_quadratic"][replica] = quadratic
        output["information_root"][replica] = np.sqrt(quadratic)
    return output, reference, float(np.min(regularized))


def fit_scalar(feature: np.ndarray, target: np.ndarray) -> float:
    return scalar_scale(feature, target, True)[0]


def select_information_models(
    data: dict,
    sample_indices: dict[int, np.ndarray],
    target: dict[int, np.ndarray],
) -> tuple[dict[str, dict[int, np.ndarray]], list[dict]]:
    """Select the variance shrinkage on B after fitting each scalar alpha on A."""
    candidates: dict[float, tuple[dict[str, dict[int, np.ndarray]], float, float]] = {}
    for shrinkage in VARIANCE_SHRINKAGES:
        candidates[shrinkage] = information_features(data, sample_indices, shrinkage)
    selected_features: dict[str, dict[int, np.ndarray]] = {}
    rows: list[dict] = []
    for model in ("information_quadratic", "information_root"):
        scored = []
        for shrinkage, (features, reference, minimum) in candidates.items():
            alpha = fit_scalar(features[model][FIT_REPLICA], target[FIT_REPLICA])
            tune_prediction = alpha * features[model][TUNE_REPLICA]
            tune_mae = float(np.mean(np.abs(target[TUNE_REPLICA] - tune_prediction)))
            scored.append((tune_mae, -shrinkage, shrinkage, alpha, features, reference, minimum))
        tune_mae, _, shrinkage, alpha, features, reference, minimum = min(scored)
        selected_features[model] = features[model]
        rows.append({
            "model": model,
            "alpha": alpha,
            "variance_shrinkage": shrinkage,
            "tune_mae": tune_mae,
            "median_positive_training_variance": reference,
            "minimum_regularized_variance": minimum,
        })
    return selected_features, rows


def dihedral_zscore_features(
    data: dict,
    angles_radians: np.ndarray,
    sample_indices: dict[int, np.ndarray],
    shrinkage: float,
) -> tuple[dict[str, dict[int, np.ndarray]], float, float]:
    variance = training_circular_variance(angles_radians, data["replicas"], FIT_REPLICA)
    regularized, reference = regularized_residue_variance(variance, shrinkage)
    output = {"backbone_zdrmsd_control": {}, "backbone_zquadratic_control": {}}
    for replica, pair_frame in data["pairs"].items():
        left, right = pair_endpoints(pair_frame, sample_indices[replica])
        quadratic = variance_scaled_dihedral_pair_distance(
            angles_radians, left, right, regularized, root=False
        )
        output["backbone_zquadratic_control"][replica] = quadratic
        output["backbone_zdrmsd_control"][replica] = np.sqrt(quadratic)
    return output, reference, float(np.min(regularized))


def select_dihedral_zscore_models(
    data: dict,
    angles_radians: np.ndarray,
    sample_indices: dict[int, np.ndarray],
    target: dict[int, np.ndarray],
) -> tuple[dict[str, dict[int, np.ndarray]], list[dict]]:
    """Select circular-variance shrinkage on B after each A-only scalar fit."""
    candidates = {
        shrinkage: dihedral_zscore_features(data, angles_radians, sample_indices, shrinkage)
        for shrinkage in VARIANCE_SHRINKAGES
    }
    selected_features: dict[str, dict[int, np.ndarray]] = {}
    rows: list[dict] = []
    for model in ("backbone_zdrmsd_control", "backbone_zquadratic_control"):
        scored = []
        for shrinkage, (features, reference, minimum) in candidates.items():
            alpha = fit_scalar(features[model][FIT_REPLICA], target[FIT_REPLICA])
            tune_prediction = alpha * features[model][TUNE_REPLICA]
            tune_mae = float(np.mean(np.abs(target[TUNE_REPLICA] - tune_prediction)))
            scored.append((tune_mae, -shrinkage, shrinkage, alpha, features, reference, minimum))
        tune_mae, _, shrinkage, alpha, features, reference, minimum = min(scored)
        selected_features[model] = features[model]
        rows.append({
            "model": model,
            "alpha": alpha,
            "variance_shrinkage": shrinkage,
            "tune_mae": tune_mae,
            "median_positive_training_variance": reference,
            "minimum_regularized_variance": minimum,
        })
    return selected_features, rows


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(pd.Series(left).corr(pd.Series(right), method="spearman"))
    return value if np.isfinite(value) else 0.0


def evaluate_system(
    row: dict[str, str], config: dict, edges: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict]]:
    data = system_data(row, config)
    system = data["system"]
    signed_targets, bandwidth = density_targets(data, FIT_REPLICA, PRIMARY_RANK)
    indices = {
        replica: sampled_indices(system, replica, len(values), PAIR_CAP)
        for replica, values in signed_targets.items()
    }
    target = {replica: np.abs(values[indices[replica]]) for replica, values in signed_targets.items()}
    features = fixed_pair_features(data, indices)
    angles_radians, dihedral_metadata = load_backbone_dihedrals(row, config)
    if angles_radians.shape[1] != len(data["replicas"]):
        raise ValueError(
            f"{system}: dihedral/PF frame mismatch "
            f"({angles_radians.shape[1]} != {len(data['replicas'])})"
        )
    for replica, pair_frame in data["pairs"].items():
        left, right = pair_endpoints(pair_frame, indices[replica])
        features["backbone_drmsd_control"][replica] = periodic_dihedral_pair_distance(
            angles_radians, left, right
        )
    selected_information, information_rows = select_information_models(data, indices, target)
    features.update(selected_information)
    selected_dihedral, dihedral_rows = select_dihedral_zscore_models(
        data, angles_radians, indices, target
    )
    features.update(selected_dihedral)
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    result_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    predictions: dict[str, dict[int, np.ndarray]] = {}
    tuned_by_model = {item["model"]: item for item in (*information_rows, *dihedral_rows)}
    for model in ALL_MODELS:
        alpha = tuned_by_model.get(model, {}).get(
            "alpha", fit_scalar(features[model][FIT_REPLICA], target[FIT_REPLICA])
        )
        predictions[model] = {replica: alpha * values for replica, values in features[model].items()}
        coefficient_rows.append({
            "system_id": system,
            "model": model,
            "alpha": alpha,
            "variance_shrinkage": tuned_by_model.get(model, {}).get("variance_shrinkage", np.nan),
            "tune_mae": tuned_by_model.get(model, {}).get("tune_mae", np.nan),
            "median_positive_training_variance": tuned_by_model.get(model, {}).get(
                "median_positive_training_variance", np.nan
            ),
            "minimum_regularized_variance": tuned_by_model.get(model, {}).get(
                "minimum_regularized_variance", np.nan
            ),
            "protein_residues": (
                dihedral_metadata["protein_residues"]
                if model in DIHEDRAL_MODELS else np.nan
            ),
            "dihedral_residues": (
                dihedral_metadata["dihedral_residues"]
                if model in DIHEDRAL_MODELS else np.nan
            ),
            "dihedral_angles": (
                dihedral_metadata["dihedral_angles"]
                if model in DIHEDRAL_MODELS else np.nan
            ),
            "dihedral_residue_coverage": (
                dihedral_metadata["dihedral_residue_coverage"]
                if model in DIHEDRAL_MODELS else np.nan
            ),
        })

    work_scale_residual = target[TUNE_REPLICA] - predictions["work_scale"][TUNE_REPLICA]
    for model in ALL_MODELS:
        diagnostic_rows.append({
            "system_id": system,
            "model": model,
            "work_scale_feature_spearman": finite_spearman(
                features[model][TUNE_REPLICA], features["work_scale"][TUNE_REPLICA]
            ),
            "work_scale_residual_spearman": finite_spearman(
                features[model][TUNE_REPLICA], work_scale_residual
            ),
        })

    test_w1 = data["pairs"][TEST_REPLICA].w1.to_numpy()[indices[TEST_REPLICA]]
    for model in ALL_MODELS:
        prediction = predictions[model][TEST_REPLICA]
        for region, mask in regions(test_w1, edges).items():
            if not mask.any():
                continue
            distribution = mass_metrics(
                target[TEST_REPLICA][mask], prediction[mask], target[FIT_REPLICA],
                settings["distribution_bins"], settings["distribution_smoothing"],
            )
            result_rows.append({
                "system_id": system,
                "model": model,
                "region": region,
                "pairs": int(mask.sum()),
                "bandwidth_angstrom": bandwidth,
                "mae": float(np.mean(np.abs(target[TEST_REPLICA][mask] - prediction[mask]))),
                "spearman": finite_spearman(target[TEST_REPLICA][mask], prediction[mask]),
                **distribution,
            })
    return result_rows, coefficient_rows, diagnostic_rows


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = results[results.model == "work_scale"][
        ["system_id", "region", "distribution_recovery", "mae"]
    ].rename(columns={"distribution_recovery": "baseline_recovery", "mae": "baseline_mae"})
    paired = results.merge(baseline, on=["system_id", "region"], validate="many_to_one")
    paired["recovery_improvement"] = paired.distribution_recovery - paired.baseline_recovery
    paired["mae_improvement"] = paired.baseline_mae - paired.mae
    summary = (
        paired.groupby(["model", "region"], as_index=False)
        .agg(
            median_recovery=("distribution_recovery", "median"),
            recovery_sd=("distribution_recovery", "std"),
            median_improvement=("recovery_improvement", "median"),
            improvement_q25=("recovery_improvement", lambda x: x.quantile(0.25)),
            improvement_q75=("recovery_improvement", lambda x: x.quantile(0.75)),
            contributing_systems=("system_id", "nunique"),
            systems_improved=("recovery_improvement", lambda x: int((x >= 0.03).sum())),
            median_mae_improvement=("mae_improvement", "median"),
        )
    )
    return paired, summary


def make_report(
    results: pd.DataFrame,
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
    edges: np.ndarray,
) -> None:
    paired, summary = summarize(results)
    atomic_parquet(paired, OUTPUT / "pf_information_paired_results.parquet")
    atomic_parquet(summary, OUTPUT / "pf_information_pilot_summary.parquet")
    bands = [f"q{i}" for i in range(6)]
    ticks = [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    for model in ALL_MODELS:
        block = paired[(paired.model == model) & paired.region.isin(bands)]
        grouped = block.groupby("region").distribution_recovery
        median = grouped.median()
        sd = grouped.std()
        recovery = 100 * np.array([median.get(band, np.nan) for band in bands])
        spread = 100 * np.array([sd.get(band, np.nan) for band in bands])
        line = axes[0].plot(range(6), recovery, marker="o", lw=2, label=MODEL_LABELS[model])[0]
        axes[0].fill_between(
            range(6), np.maximum(0, recovery - spread), np.minimum(100, recovery + spread),
            color=line.get_color(), alpha=0.08, linewidth=0,
        )
        if model in CANDIDATE_MODELS:
            grouped = block.groupby("region").recovery_improvement
            median = grouped.median()
            low = grouped.quantile(0.25)
            high = grouped.quantile(0.75)
            improvement = 100 * np.array([median.get(band, np.nan) for band in bands])
            axes[1].plot(range(6), improvement, marker="o", lw=2, label=MODEL_LABELS[model])
            axes[1].fill_between(
                range(6), 100 * np.array([low.get(band, np.nan) for band in bands]),
                100 * np.array([high.get(band, np.nan) for band in bands]), alpha=0.10,
            )
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[0].set_title("Fixed-BV population-magnitude recovery across structural W1")
    axes[1].axhline(3, color="black", ls="--", label="+3 pp gate")
    axes[1].axhline(0, color="grey", lw=1)
    axes[1].set_ylabel("Paired improvement over Work Scale (pp)")
    axes[1].set_xlabel("Global frame-pair structural W1 band")
    axes[1].set_xticks(range(6), ticks)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("PF-W1 and variance-scaled information pilot: 12 systems, replica 1→2→3")
    fig.tight_layout()
    fig.savefig(OUTPUT / "pf_information_recovery_across_global_w1.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    diagnostics.boxplot(column="work_scale_feature_spearman", by="model", ax=axes[0], grid=False)
    diagnostics.boxplot(column="work_scale_residual_spearman", by="model", ax=axes[1], grid=False)
    axes[0].set_title("Metric correlation with Work Scale")
    axes[1].set_title("Correlation with Work Scale residual")
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Replica-B complementarity diagnostics")
    fig.tight_layout()
    fig.savefig(OUTPUT / "pf_information_complementarity.png", dpi=180)
    plt.close(fig)

    tail = summary[
        summary.region.isin(["q4", "q5"]) & summary.model.isin(CANDIDATE_MODELS)
    ].copy()
    tail["required_systems"] = np.ceil(2 * tail.contributing_systems / 3).astype(int)
    passing = tail[
        (tail.median_improvement >= 0.03)
        & (tail.systems_improved >= tail.required_systems)
        & (tail.median_mae_improvement >= 0)
    ]
    atomic_yaml(OUTPUT / "checkpoint21_report.yaml", {
        "checkpoint": 21,
        "status": "pilot_complete",
        "full_run_gate_passed": bool(len(passing)),
        "passing_model_regions": passing[["model", "region"]].to_dict("records"),
        "fit_replica": FIT_REPLICA,
        "tune_replica": TUNE_REPLICA,
        "test_replica": TEST_REPLICA,
        "kde_neighbour_rank": PRIMARY_RANK,
        "variance_shrinkage_grid": list(VARIANCE_SHRINKAGES),
        "gate": ">=3 pp median q4 or q5 improvement, >=2/3 contributing systems improve >=3 pp, MAE not worse",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = load_config()
    by_id = {row["system_id"]: row for row in load_systems()}
    selection = pd.read_parquet(CHECKPOINT19_OUTPUT / "pilot_systems.parquet")
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml") as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for index, system in enumerate(selection.system_id, 1):
        results, coefficients, diagnostics = evaluate_system(by_id[system], config, edges)
        result_rows.extend(results)
        coefficient_rows.extend(coefficients)
        diagnostic_rows.extend(diagnostics)
        print(f"[{index}/{len(selection)}] {system}", flush=True)
    results = pd.DataFrame(result_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    atomic_parquet(results, OUTPUT / "pf_information_pilot_results.parquet")
    atomic_parquet(coefficients, OUTPUT / "pf_information_pilot_coefficients.parquet")
    atomic_parquet(diagnostics, OUTPUT / "pf_information_pilot_diagnostics.parquet")
    make_report(results, coefficients, diagnostics, edges)


if __name__ == "__main__":
    main()
