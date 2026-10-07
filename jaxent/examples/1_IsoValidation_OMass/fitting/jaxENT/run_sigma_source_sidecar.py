#!/usr/bin/env python3
"""Compare uptake and aligned-coordinate Sigma sources with uptake fitting fixed.

Defaults run 2 ensembles x 4 Sigma sources x 9 identity-shrinkage values x
3 non-redundant sequence-cluster splits (216 fits), all at MaxEnt=1000.
Both validation-selected convergence states and the final optimization state are
reported without changing the native JAX-ENT fitting architecture.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxent-matplotlib")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd

from jaxent.examples.common import analysis
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.examples.common.analysis.convergence_labels import (
    iter_labeled_convergence_states,
)
from jaxent.examples.common.analysis.stats import effective_sample_size
from jaxent.examples.common.config import LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import create_data_loaders, run_optimization
from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.utils.hdf import load_optimization_history_from_file

import run_sigma_shrinkage_sidecar as shrinkage
from sidecar_selection import closed_validation_mse, select_best_rows, SELECTION_POLICY
from compute_sigma_synthetic import (
    compute_cluster_weights,
    compute_weighted_covariance,
    construct_covariance,
    ridge_thresholds,
)


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent.parent
DEFAULT_TRAJECTORY_DIR = (
    EXPERIMENT_DIR
    / "data"
    / "_Bradshaw"
    / "Reproducibility_pack_v2"
    / "data"
    / "trajectories"
)
DEFAULT_SOURCES = (
    "gt_uptake",
    "gt_coordinate",
    "closed_uptake",
    "closed_coordinate",
)
SOURCE_LABELS = {
    "gt_uptake": "GT uptake",
    "gt_coordinate": "GT coordinate",
    "closed_uptake": "Closed-only uptake",
    "closed_coordinate": "Closed-only coordinate",
}
SOURCE_COLORS = {
    "gt_uptake": "#009E73",
    "gt_coordinate": "#0072B2",
    "closed_uptake": "#E69F00",
    "closed_coordinate": "#CC79A7",
}
TRAJECTORIES = {
    "ISO_BI": "sliced_trajectories/TeaA_filtered_sliced.xtc",
    "ISO_TRI": "sliced_trajectories/TeaA_initial_sliced.xtc",
}


@dataclasses.dataclass(frozen=True)
class RunSpec:
    ensemble: str
    sigma_source: str
    alpha: float
    split_type: str
    split_idx: int
    sigma_path: str
    output_dir: str
    features_dir: str
    datasplit_dir: str
    clustering_dir: str
    n_steps: int
    learning_rate: float
    ema_alpha: float
    forward_model_scaling: float
    execution_mode: str

    @property
    def run_id(self) -> str:
        return (
            f"{self.ensemble}_Sigma_MSE_uptake_{self.sigma_source}_{self.split_type}_"
            f"split{self.split_idx:03d}_alpha{shrinkage.alpha_token(self.alpha)}_maxent1000_"
            f"{shrinkage.FORWARD_CONSTRUCTION_VERSION}"
        )

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / "fits" / self.split_type / self.run_id

    @property
    def history_path(self) -> Path:
        return self.run_dir / f"{self.run_id}_results.hdf5"

    @property
    def config_path(self) -> Path:
        return self.run_dir / f"{self.run_id}_config.json"


def population_weights(assignments: np.ndarray, population: str) -> np.ndarray:
    assignments = np.asarray(assignments, dtype=int)
    if population == "gt":
        return compute_cluster_weights(assignments, {"open": 0.4, "closed": 0.6})
    if population == "closed":
        closed = assignments == 1
        if not np.any(closed):
            raise ValueError("Closed-only Sigma requires at least one cluster-1 frame")
        weights = closed.astype(np.float64)
        return weights / weights.sum()
    raise ValueError(f"Unknown population: {population}")


def aligned_ca_coordinates(
    ensemble: str,
    feature_topology,
    trajectory_dir: str | Path,
) -> np.ndarray:
    """Return aligned C-alpha positions as frame x feature-residue x xyz."""
    trajectory_dir = Path(trajectory_dir)
    reference_path = trajectory_dir / "TeaA_ref_closed_state.pdb"
    trajectory_path = trajectory_dir / TRAJECTORIES[ensemble]
    reference = mda.Universe(str(reference_path))
    mobile = mda.Universe(str(reference_path), str(trajectory_path), in_memory=True)
    align.AlignTraj(
        mobile,
        reference,
        select="protein and name CA",
        in_memory=True,
    ).run(verbose=False)

    ca = mobile.select_atoms("protein and name CA")
    index_by_resid: dict[int, int] = {}
    for atom in ca:
        resid = int(atom.resid)
        if resid in index_by_resid:
            raise ValueError(f"Multiple protein C-alpha atoms have resid {resid}")
        index_by_resid[resid] = int(atom.index)

    residue_ids = []
    for topology in feature_topology:
        residues = tuple(topology.residues)
        if len(residues) != 1:
            raise ValueError("Coordinate Sigma requires residue-level feature topology")
        residue_ids.append(int(residues[0]))
    missing = sorted(set(residue_ids) - set(index_by_resid))
    if missing:
        raise ValueError(f"Feature residues lack C-alpha coordinates: {missing[:10]}")
    atom_indices = [index_by_resid[resid] for resid in residue_ids]
    coordinates = np.stack(
        [mobile.atoms[atom_indices].positions.copy() for _ in mobile.trajectory]
    ).astype(np.float64)
    expected = (len(mobile.trajectory), len(feature_topology), 3)
    if coordinates.shape != expected or not np.isfinite(coordinates).all():
        raise ValueError(
            f"Invalid aligned coordinate array {coordinates.shape}; expected {expected}"
        )
    return coordinates


def weighted_coordinate_covariance(
    coordinates: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted residue covariance from aligned 3D displacement dot products."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError("Coordinates must have shape (frames, residues, 3)")
    if weights.shape != (coordinates.shape[0],):
        raise ValueError("Coordinate frames and population weights do not align")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive mass")
    weights = weights / weights.sum()
    mean = np.einsum("f,frc->rc", weights, coordinates)
    centered = coordinates - mean[None, :, :]
    covariance = np.einsum("f,fic,fjc->ij", weights, centered, centered) / 3.0
    covariance = (covariance + covariance.T) / 2.0
    if not np.isfinite(covariance).all():
        raise ValueError("Coordinate covariance contains non-finite values")
    return covariance


def regularize_sigma(
    raw_covariance: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    condition_limit: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shrunk = construct_covariance(
        raw_covariance,
        weights,
        correction="population",
        target="identity",
        alpha=float(alpha),
        ridge=0.0,
    )
    thresholds = ridge_thresholds(shrunk, condition_limit=condition_limit)
    ridge = float(thresholds["ridge_stable"])
    covariance = shrunk + np.eye(len(shrunk)) * ridge
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    precision = np.linalg.inv(covariance)
    precision_norm = float(np.linalg.norm(precision))
    condition = float(np.linalg.cond(covariance))
    if (
        eigenvalues[0] <= 0
        or condition > condition_limit * (1 + 1e-8)
        or precision_norm <= 0
        or not np.isfinite(precision).all()
    ):
        raise ValueError("Regularized Sigma failed positive-definite stability checks")
    return (
        {
            "Sigma_raw": np.asarray(raw_covariance, dtype=np.float64),
            "Sigma_shrunk": shrunk,
            "Sigma": covariance,
            "Sigma_inv": precision,
            "Sigma_inv_normalized": precision / precision_norm,
            "frame_weights": weights,
        },
        {
            "alpha": float(alpha),
            "ridge": ridge,
            "ridge_pd": float(thresholds["ridge_pd"]),
            "ridge_stable": ridge,
            "rank_before_ridge": int(np.linalg.matrix_rank(shrunk)),
            "rank_after_ridge": int(np.linalg.matrix_rank(covariance)),
            "eigen_min_before_ridge": float(thresholds["eigen_min"]),
            "eigen_max_before_ridge": float(thresholds["eigen_max"]),
            "eigen_min": float(eigenvalues[0]),
            "eigen_max": float(eigenvalues[-1]),
            "condition_number": condition,
            "trace": float(np.trace(covariance)),
            "precision_frobenius": precision_norm,
        },
    )


def prepare_sigma_artifacts(
    output_dir: Path,
    features_dir: Path,
    clustering_dir: Path,
    trajectory_dir: Path,
    ensembles: Iterable[str],
    sources: Iterable[str],
    alphas: Iterable[float],
    condition_limit: float,
) -> tuple[dict[tuple[str, str, float], Path], pd.DataFrame]:
    sigma_dir = output_dir / "sigma"
    sigma_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, str, float], Path] = {}
    rows = []
    for ensemble in ensembles:
        features, feature_top = shrinkage.load_features(features_dir, ensemble)
        assignments = shrinkage.load_cluster_assignments(clustering_dir, ensemble)
        if features.features_shape[1] != len(assignments):
            raise ValueError(f"Feature/assignment frame mismatch for {ensemble}")
        uptake = shrinkage.oracle_uptake_coordinates(features)
        coordinates = aligned_ca_coordinates(ensemble, feature_top, trajectory_dir)
        if coordinates.shape[:2] != (len(assignments), features.features_shape[0]):
            raise ValueError(f"Coordinate/feature mismatch for {ensemble}")
        for source in sources:
            population, coordinate_type = source.split("_", maxsplit=1)
            weights = population_weights(assignments, population)
            if coordinate_type == "uptake":
                raw = compute_weighted_covariance(uptake, weights)
            elif coordinate_type == "coordinate":
                raw = weighted_coordinate_covariance(coordinates, weights)
            else:
                raise ValueError(f"Unknown Sigma source: {source}")
            for alpha in alphas:
                arrays, metrics = regularize_sigma(raw, weights, alpha, condition_limit)
                path = sigma_dir / (
                    f"{ensemble}_{source}_alpha{shrinkage.alpha_token(alpha)}.npz"
                )
                np.savez_compressed(
                    path,
                    **arrays,
                    cluster_assignments=assignments,
                    ensemble=ensemble,
                    sigma_source=source,
                    population=population,
                    coordinate_type=coordinate_type,
                    alpha=float(alpha),
                )
                paths[(ensemble, source, float(alpha))] = path
                rows.append(
                    {
                        "ensemble": ensemble,
                        "sigma_source": source,
                        "sigma_source_label": SOURCE_LABELS[source],
                        "population": population,
                        "coordinate_type": coordinate_type,
                        **metrics,
                        "path": str(path),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "sigma_metrics.csv", index=False)
    return paths, frame


def run_is_complete(spec: RunSpec) -> bool:
    if not spec.history_path.exists() or not spec.config_path.exists():
        return False
    try:
        load_optimization_history_from_file(str(spec.history_path))
    except Exception:
        return False
    return True


def run_fit(spec: RunSpec) -> None:
    features, feature_top = shrinkage.load_features(spec.features_dir, spec.ensemble)
    assignments = shrinkage.load_cluster_assignments(spec.clustering_dir, spec.ensemble)
    model = shrinkage.configure_model("uptake", assignments)
    prior = shrinkage.build_prior_dataset(model, features, feature_top)
    train_data, val_data = shrinkage.load_split(
        spec.datasplit_dir, spec.split_type, spec.split_idx
    )
    with np.load(spec.sigma_path) as sigma:
        precision = jnp.asarray(sigma["Sigma_inv_normalized"])
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    run_optimization(
        train_data=train_data,
        val_data=val_data,
        prior_data=prior,
        features=features,
        forward_model=model,
        model_parameters=model.params,
        feature_top=feature_top,
        convergence=list(shrinkage.CONVERGENCE_RATES),
        loss_config=LossConfig(
            primary_loss="hdx_uptake_sigma_MSE_loss",
            maxent_scaling=shrinkage.MAXENT,
            optimize_bv_params=False,
        ),
        opt_config=OptimizationConfig(
            n_steps=spec.n_steps,
            learning_rate=spec.learning_rate,
            ema_alpha=spec.ema_alpha,
            convergence_rates=list(shrinkage.CONVERGENCE_RATES),
            optimizer="adam",
            step_chunk_size=100,
            lr_adjustment=True,
            frame_average_impl="tensordot",
            reset_threshold_cooldown_on_oscillation=True,
            forward_model_scaling=spec.forward_model_scaling,
        ),
        name=spec.run_id,
        output_dir=str(spec.run_dir),
        cov_matrix=precision,
        execution_mode=spec.execution_mode,
    )
    config = json.loads(spec.config_path.read_text())
    config["effective_settings"]["frame_averaging_mode"] = "frame_uptake"
    config["sidecar_settings"] = {
        "mode": "uptake",
        "uptake_model": "standard",
        "intrinsic_rate_unit": "min^-1",
        "intrinsic_rate_provider": "jaxent_calculate_HDXrate",
        "uptake_reduction": "frame_uptake",
        "forward_construction_version": shrinkage.FORWARD_CONSTRUCTION_VERSION,
        "sigma_source": spec.sigma_source,
        "shrinkage_alpha": spec.alpha,
        "sigma_path": spec.sigma_path,
        "split_type": spec.split_type,
        "split_idx": spec.split_idx,
    }
    temporary = spec.config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True))
    temporary.replace(spec.config_path)


def _worker_command(script: Path, spec_path: Path, log_path: Path) -> tuple[str, int]:
    with log_path.open("w") as log:
        completed = subprocess.run(
            [sys.executable, str(script), "--worker-spec", str(spec_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )
    return spec_path.stem, completed.returncode


def execute_specs(specs: list[RunSpec], output_dir: Path, jobs: int) -> None:
    spec_dir = output_dir / "specs"
    log_dir = output_dir / "logs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for spec in specs:
        if run_is_complete(spec):
            print(f"[resume] {spec.run_id}")
            continue
        spec_path = spec_dir / f"{spec.run_id}.json"
        spec_path.write_text(
            json.dumps(dataclasses.asdict(spec), indent=2, sort_keys=True)
        )
        pending.append((spec_path, log_dir / f"{spec.run_id}.log"))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _worker_command, Path(__file__).resolve(), spec_path, log_path
            ): (
                spec_path,
                log_path,
            )
            for spec_path, log_path in pending
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            spec_path, log_path = futures[future]
            run_id, returncode = future.result()
            print(f"[{index}/{len(pending)}] {run_id}: return code {returncode}")
            if returncode:
                failures.append((run_id, log_path))
    if failures:
        detail = "\n".join(f"  {run_id}: {path}" for run_id, path in failures)
        raise RuntimeError(f"{len(failures)} fit(s) failed:\n{detail}")


def score_context(spec: RunSpec, cache: dict[tuple, tuple]):
    key = (spec.ensemble, spec.split_type, spec.split_idx)
    if key not in cache:
        features, feature_top = shrinkage.load_features(
            spec.features_dir, spec.ensemble
        )
        assignments = shrinkage.load_cluster_assignments(
            spec.clustering_dir, spec.ensemble
        )
        model = shrinkage.configure_model("uptake", assignments)
        train_data, val_data = shrinkage.load_split(
            spec.datasplit_dir, spec.split_type, spec.split_idx
        )
        loader = create_data_loaders(
            train_data + val_data,
            train_data,
            val_data,
            features,
            feature_top,
            cov_matrix=None,
        )
        cache[key] = (
            features,
            assignments,
            model,
            loader,
            analysis.get_experimental_uptake(val_data),
        )
    return cache[key]


def score_state(
    spec: RunSpec, state, checkpoint: str, context: tuple
) -> tuple[dict, np.ndarray]:
    features, assignments, model, loader, y_true_val = context
    weights = np.asarray(
        validated_frame_weight_simplex(state.params.frame_weight_simplex), dtype=float
    )
    weights /= weights.sum()
    predicted = shrinkage.predict_uptake(model, features, weights)
    mapped = np.asarray(
        [
            apply_sparse_mapping(
                loader.val.residue_feature_ouput_mapping, jnp.asarray(predicted[t])
            )
            for t in range(predicted.shape[0])
        ]
    ).T
    native_val_loss = np.nan
    if state.losses is not None and state.losses.val_losses is not None:
        native_val_loss = float(state.losses.val_losses[0])
    ess = effective_sample_size(weights)
    return (
        {
            "run_id": spec.run_id,
            "ensemble": spec.ensemble,
            "sigma_source": spec.sigma_source,
            "sigma_source_label": SOURCE_LABELS[spec.sigma_source],
            "alpha": spec.alpha,
            "split_type": spec.split_type,
            "split_idx": spec.split_idx,
            "maxent": shrinkage.MAXENT,
            "checkpoint_policy": checkpoint,
            "step": int(np.asarray(state.step)),
            "native_sigma_val_loss": native_val_loss,
            "val_mse": analysis.calculate_mse(mapped, y_true_val),
            "val_closed_sigma_mse": closed_validation_mse(spec, mapped, y_true_val),
            "recovery_percent": calculate_recovery_percentage(
                assignments,
                weights,
                shrinkage.GROUND_TRUTH,
                shrinkage.STATE_MAPPING,
            ),
            "ess": ess,
            "ess_percent": 100.0 * ess / len(weights),
            "n_frames": len(weights),
        },
        weights,
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_mse", "recovery_percent", "ess", "ess_percent"]
    grouped = results.groupby(
        ["ensemble", "sigma_source", "sigma_source_label", "alpha"], sort=False
    )
    summary = grouped[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def plot_metric(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: Path,
    alphas: Iterable[float],
    log_y: bool = False,
) -> None:
    alpha_values = tuple(float(value) for value in alphas)
    positions = {alpha: index for index, alpha in enumerate(alpha_values)}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, ensemble in zip(axes, shrinkage.DEFAULT_ENSEMBLES):
        panel = results[results["ensemble"] == ensemble]
        for source in DEFAULT_SOURCES:
            source_rows = panel[panel["sigma_source"] == source]
            color = SOURCE_COLORS[source]
            for (_, _), trace in source_rows.groupby(
                ["split_type", "split_idx"], sort=False
            ):
                trace = trace.sort_values("alpha")
                linestyle = "--" if trace.iloc[0]["split_type"] == "spatial" else "-"
                axis.plot(
                    [positions[float(value)] for value in trace["alpha"]],
                    trace[metric],
                    color=color,
                    linewidth=0.8,
                    alpha=0.3,
                    linestyle=linestyle,
                )
            mean = (
                source_rows[source_rows["split_type"] == "sequence_cluster"]
                .groupby("alpha", sort=False)[metric]
                .mean()
            )
            if not mean.empty:
                mean = mean.reindex([a for a in alpha_values if a in mean.index])
                axis.plot(
                    [positions[float(value)] for value in mean.index],
                    mean.values,
                    color=color,
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    label=SOURCE_LABELS[source],
                )
        axis.set_title(ensemble.replace("ISO_", "ISO "))
        axis.set_xlabel("Identity shrinkage (alpha)")
        axis.set_xticks(range(len(alpha_values)))
        axis.set_xticklabels([f"{alpha:g}" for alpha in alpha_values], rotation=45)
        axis.grid(axis="y", alpha=0.2)
        if log_y:
            axis.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.subplots_adjust(top=0.82)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def analyze_specs(
    specs: list[RunSpec], output_dir: Path, alphas: Iterable[float]
) -> None:
    convergence_rows = []
    final_rows = []
    convergence_weights: dict[str, np.ndarray] = {}
    final_weights: dict[str, np.ndarray] = {}
    incomplete = []
    cache: dict[tuple, tuple] = {}
    for spec in specs:
        if not run_is_complete(spec):
            incomplete.append(
                {"run_id": spec.run_id, "reason": "missing_or_invalid_history"}
            )
            continue
        history = load_optimization_history_from_file(str(spec.history_path))
        labeled = iter_labeled_convergence_states(history)
        if not labeled:
            incomplete.append(
                {"run_id": spec.run_id, "reason": "no_convergence_states"}
            )
            continue
        if not history.states:
            incomplete.append(
                {"run_id": spec.run_id, "reason": "no_optimization_states"}
            )
            continue
        context = score_context(spec, cache)
        for labeled_state in labeled:
            row, weights = score_state(
                spec, labeled_state.state, "convergence_state", context
            )
            row["convergence_rank"] = labeled_state.rank
            row["convergence_threshold"] = labeled_state.threshold
            convergence_rows.append(row)
            convergence_weights[f"{spec.run_id}__conv{labeled_state.rank:02d}"] = (
                weights
            )
        final_row, weights = score_state(
            spec, history.states[-1], "final_optimization_state", context
        )
        final_rows.append(final_row)
        final_weights[spec.run_id] = weights

    convergence = pd.DataFrame(convergence_rows)
    final = pd.DataFrame(final_rows)
    convergence.to_csv(output_dir / "convergence_scores.csv", index=False)
    selected = select_best_rows(convergence)
    selected.to_csv(output_dir / "selected_results.csv", index=False)
    final.to_csv(output_dir / "final_step_results.csv", index=False)
    summarize(selected).to_csv(output_dir / "selected_summary.csv", index=False)
    summarize(final).to_csv(output_dir / "final_step_summary.csv", index=False)
    selected_weights = {
        f"{row.run_id}__conv{int(row.convergence_rank):02d}": convergence_weights[
            f"{row.run_id}__conv{int(row.convergence_rank):02d}"
        ]
        for row in selected.itertuples()
    }
    np.savez_compressed(output_dir / "selected_frame_weights.npz", **selected_weights)
    np.savez_compressed(output_dir / "final_step_frame_weights.npz", **final_weights)
    pd.DataFrame(incomplete, columns=["run_id", "reason"]).to_csv(
        output_dir / "incomplete_runs.csv", index=False
    )
    if selected.empty or final.empty:
        raise RuntimeError(
            "No complete selected/final results were available for plotting"
        )
    for label, frame in (("selected", selected), ("final_step", final)):
        plot_metric(
            frame,
            "recovery_percent",
            "Recovery (%)",
            output_dir / "plots" / f"recovery_vs_shrinkage_{label}",
            alphas,
        )
        plot_metric(
            frame,
            "ess_percent",
            "ESS (%)",
            output_dir / "plots" / f"ess_percent_vs_shrinkage_{label}",
            alphas,
            log_y=True,
        )


def build_specs(args, sigma_paths: dict[tuple[str, str, float], Path]) -> list[RunSpec]:
    return [
        RunSpec(
            ensemble=ensemble,
            sigma_source=source,
            alpha=float(alpha),
            split_type=args.split_type,
            split_idx=split_idx,
            sigma_path=str(sigma_paths[(ensemble, source, float(alpha))]),
            output_dir=str(args.output_dir),
            features_dir=str(args.features_dir),
            datasplit_dir=str(args.datasplit_dir),
            clustering_dir=str(args.clustering_dir),
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            ema_alpha=args.ema_alpha,
            forward_model_scaling=args.forward_model_scaling,
            execution_mode=args.execution_mode,
        )
        for ensemble in args.ensembles
        for source in args.sources
        for alpha in args.alphas
        for split_idx in range(args.n_splits)
    ]


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(args, specs: list[RunSpec], sigma_metrics: pd.DataFrame) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "maxent": shrinkage.MAXENT,
        "primary_loss": "hdx_uptake_sigma_MSE_loss",
        "frame_averaging_mode": "frame_uptake",
        "forward_construction_version": shrinkage.FORWARD_CONSTRUCTION_VERSION,
        "intrinsic_rate_unit": "min^-1",
        "intrinsic_rate_provider": "jaxent_calculate_HDXrate",
        "checkpoint_policies": [
            SELECTION_POLICY,
            "final_optimization_state",
        ],
        "coordinate_definition": "aligned_ca_displacement_dot_product_covariance_divided_by_3",
        "sample_size_correction": "population",
        "shrinkage_target": "identity",
        "condition_limit": args.condition_limit,
        "ensembles": list(args.ensembles),
        "sigma_sources": list(args.sources),
        "alphas": list(args.alphas),
        "split_type": args.split_type,
        "n_splits": args.n_splits,
        "expected_fits": len(specs),
        "completed_fit_files": sum(run_is_complete(spec) for spec in specs),
        "sigma_rows": len(sigma_metrics),
        "settings": {
            "trajectory_dir": str(args.trajectory_dir.resolve()),
            "n_steps": args.n_steps,
            "learning_rate": args.learning_rate,
            "ema_alpha": args.ema_alpha,
            "forward_model_scaling": args.forward_model_scaling,
            "execution_mode": args.execution_mode,
            "jobs": args.jobs,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare uptake and aligned-coordinate Sigma sources with uptake fitting fixed."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--features-dir", type=Path, default=shrinkage.DEFAULT_FEATURES_DIR
    )
    parser.add_argument(
        "--datasplit-dir", type=Path, default=shrinkage.DEFAULT_DATASPLIT_DIR
    )
    parser.add_argument(
        "--clustering-dir", type=Path, default=shrinkage.DEFAULT_CLUSTERING_DIR
    )
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--ensembles", default=",".join(shrinkage.DEFAULT_ENSEMBLES))
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument(
        "--alphas", default=",".join(f"{alpha:g}" for alpha in shrinkage.DEFAULT_ALPHAS)
    )
    parser.add_argument("--split-type", default="sequence_cluster")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.5)
    parser.add_argument("--forward-model-scaling", type=float, default=1000.0)
    parser.add_argument("--condition-limit", type=float, default=1e8)
    parser.add_argument(
        "--execution-mode", choices=("compiled", "python"), default="compiled"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--phase", choices=("all", "fit", "analyze"), default="all")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.n_splits < 1 or args.n_steps < 1 or args.jobs < 1:
        parser.error("--n-splits, --n-steps, and --jobs must be positive")
    if not np.isfinite(args.condition_limit) or args.condition_limit <= 1:
        parser.error("--condition-limit must be finite and greater than one")
    if any(ensemble not in shrinkage.DEFAULT_ENSEMBLES for ensemble in args.ensembles):
        parser.error(f"--ensembles must be drawn from {shrinkage.DEFAULT_ENSEMBLES}")
    if any(source not in DEFAULT_SOURCES for source in args.sources):
        parser.error(f"--sources must be drawn from {DEFAULT_SOURCES}")
    if not args.alphas or any(
        not np.isfinite(alpha) or alpha < 0 or alpha > 1 for alpha in args.alphas
    ):
        parser.error("--alphas must contain finite values in [0, 1]")
    if args.split_type != "sequence_cluster":
        parser.error("this first experiment is restricted to sequence_cluster")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_spec:
        run_fit(RunSpec(**json.loads(args.worker_spec.read_text())))
        return
    args.ensembles = shrinkage.parse_csv(args.ensembles)
    args.sources = shrinkage.parse_csv(args.sources)
    args.alphas = shrinkage.parse_csv(args.alphas, float)
    validate_args(parser, args)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = HERE / f"_sigma_source_sidecar_{stamp}"
    for name in (
        "output_dir",
        "features_dir",
        "datasplit_dir",
        "clustering_dir",
        "trajectory_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sigma_paths, sigma_metrics = prepare_sigma_artifacts(
        args.output_dir,
        args.features_dir,
        args.clustering_dir,
        args.trajectory_dir,
        args.ensembles,
        args.sources,
        args.alphas,
        args.condition_limit,
    )
    specs = build_specs(args, sigma_paths)
    if args.phase in {"all", "fit"}:
        execute_specs(specs, args.output_dir, args.jobs)
    write_manifest(args, specs, sigma_metrics)
    if args.phase in {"all", "analyze"}:
        analyze_specs(specs, args.output_dir, args.alphas)
        write_manifest(args, specs, sigma_metrics)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
