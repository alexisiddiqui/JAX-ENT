#!/usr/bin/env python3
"""Focused GT-Sigma identity-shrinkage experiment for ISO BI/TRI.

The script deliberately keeps fitting inside the normal JAX-ENT architecture:
each cell calls :func:`jaxent.examples.common.optimization.run_optimization`,
and model selection considers only the native labelled convergence states.

The default campaign is 2 ensembles x 3 uptake models x 9 shrinkage values x
3 non-redundant sequence-cluster splits (162 fits), all at MaxEnt=1000.
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
import numpy as np
import pandas as pd
from scipy import stats

import jaxent.src.interfaces.topology as pt
from jaxent.examples.common import analysis
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.examples.common.analysis.convergence_labels import (
    iter_labeled_convergence_states,
)
from jaxent.examples.common.analysis.stats import effective_sample_size
from jaxent.examples.common.config import LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import create_data_loaders, run_optimization
from jaxent.examples.common.uptake_models import build_uptake_model
from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.utils.hdf import load_optimization_history_from_file
from jaxent.src.utils.jit_fn import jit_Guard

from compute_sigma_synthetic import (
    compute_cluster_weights,
    compute_weighted_covariance,
    construct_covariance,
    ridge_thresholds,
)


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent.parent
SELF_CONSISTENT_DIR = HERE / "_self_consistent_iso"
DEFAULT_FEATURES_DIR = SELF_CONSISTENT_DIR / "fit_features"
DEFAULT_DATASPLIT_DIR = SELF_CONSISTENT_DIR / "datasplits"
DEFAULT_RATE_SOURCE_MANIFEST = (
    EXPERIMENT_DIR / "data" / "_self_consistent_target_features" / "manifest.json"
)
DEFAULT_CLUSTERING_DIR = EXPERIMENT_DIR / "data" / "_clustering_results"
TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=float)
DEFAULT_ALPHAS = (0.0, 1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
DEFAULT_ENSEMBLES = ("ISO_BI", "ISO_TRI")
DEFAULT_MODES = ("rate", "linear", "uptake")
CONVERGENCE_RATES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
MAXENT = 1000.0
GROUND_TRUTH = {"intermediate": 0.0, "open": 0.4, "closed": 0.6}
STATE_MAPPING = {-1: "intermediate", 0: "open", 1: "closed"}
MODE_LABELS = {"rate": "Rate", "linear": "Linear BV", "uptake": "Uptake"}
MODE_COLORS = {"rate": "#0072B2", "linear": "#D55E00", "uptake": "#009E73"}
FORWARD_CONSTRUCTION_VERSION = "jaxent_rates_framewise_v3"
WEIGHT_LABELS = {
    "unweighted": "Unweighted",
    "oracle_weighted": "Oracle weighted (40:60)",
}


@dataclasses.dataclass(frozen=True)
class RunSpec:
    ensemble: str
    mode: str
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
            f"{self.ensemble}_Sigma_MSE_{self.mode}_{self.split_type}_"
            f"split{self.split_idx:03d}_alpha{alpha_token(self.alpha)}_maxent1000_"
            f"{FORWARD_CONSTRUCTION_VERSION}"
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


def alpha_token(alpha: float) -> str:
    """Return a stable filename token without rounding small shrinkages away."""
    return f"{float(alpha):.10g}".replace("-", "m").replace(".", "p")


def parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def load_cluster_assignments(clustering_dir: str | Path, ensemble: str) -> np.ndarray:
    path = Path(clustering_dir) / f"cluster_assignments_{ensemble}.csv"
    table = pd.read_csv(path)
    if "cluster_assignment" not in table:
        raise ValueError(f"Missing cluster_assignment column in {path}")
    assignments = table["cluster_assignment"].to_numpy(dtype=int)
    if not set(np.unique(assignments)).issubset({-1, 0, 1}):
        raise ValueError(
            f"Unexpected cluster labels in {path}: {np.unique(assignments)}"
        )
    return assignments


def load_features(features_dir: str | Path, ensemble: str):
    feature_path = Path(features_dir) / f"features_{ensemble.lower()}.npz"
    topology_path = Path(features_dir) / f"topology_{ensemble.lower()}.json"
    return BV_input_features.load(
        str(feature_path)
    ), pt.PTSerialiser.load_list_from_json(str(topology_path))


def load_split(datasplit_dir: str | Path, split_type: str, split_idx: int):
    split_dir = Path(datasplit_dir) / split_type / f"split_{split_idx:03d}"
    train = HDX_peptide.load_list_from_files(
        json_path=str(split_dir / "train_topology.json"),
        csv_path=str(split_dir / "train_dfrac.csv"),
    )
    val = HDX_peptide.load_list_from_files(
        json_path=str(split_dir / "val_topology.json"),
        csv_path=str(split_dir / "val_dfrac.csv"),
    )
    return train, val


def configure_model(mode: str, assignments: np.ndarray):
    """Build the exact native model used for one of the three comparisons."""
    if mode == "linear":
        return build_uptake_model(
            "linear", TIMEPOINTS, kint_unit="min^-1", time_unit="min"
        )
    if mode not in {"rate", "uptake"}:
        raise ValueError(f"Unknown uptake mode: {mode}")
    model = build_uptake_model("standard", TIMEPOINTS, kint_unit="min^-1")
    forward = model.forward[m_key("HDX_peptide")]
    forward.frame_averaging_mode = "frame_uptake" if mode == "uptake" else mode
    return model


def predict_uptake(model, features, frame_weights, model_parameters=None) -> np.ndarray:
    """Evaluate a fitted simplex through the model's native averaging operation."""
    params = model.params if model_parameters is None else model_parameters
    forward = model.forward[m_key("HDX_peptide")]
    output = forward.average_frames(features, params, jnp.asarray(frame_weights))
    return np.asarray(output.uptake, dtype=float)


def flatten_uptake_observations(uptake: np.ndarray) -> np.ndarray:
    """Return timepoint x flattened peptide/residue observations."""
    uptake = np.asarray(uptake, dtype=np.float64)
    if uptake.ndim < 2 or uptake.shape[0] != len(TIMEPOINTS):
        raise ValueError(
            f"Expected uptake with leading time axis of length {len(TIMEPOINTS)}, "
            f"got {uptake.shape}"
        )
    flattened = uptake.reshape(len(TIMEPOINTS), -1)
    if flattened.shape[1] < 2 or not np.isfinite(flattened).all():
        raise ValueError("Uptake comparison needs at least two finite observations")
    return flattened


def paired_mode_statistics(
    candidate: np.ndarray, baseline: np.ndarray
) -> list[dict[str, float]]:
    """Paired t-test and Cohen's dz for each timepoint."""
    candidate = flatten_uptake_observations(candidate)
    baseline = flatten_uptake_observations(baseline)
    if candidate.shape != baseline.shape:
        raise ValueError("Candidate and uptake baseline shapes must match")
    rows = []
    for time_index, timepoint in enumerate(TIMEPOINTS):
        values = candidate[time_index]
        reference = baseline[time_index]
        difference = values - reference
        difference_sd = float(np.std(difference, ddof=1))
        mean_difference = float(np.mean(difference))
        if difference_sd <= np.finfo(float).eps:
            if abs(mean_difference) <= np.finfo(float).eps:
                statistic, p_value, effect_size = 0.0, 1.0, 0.0
            else:
                statistic = float(np.sign(mean_difference) * np.inf)
                p_value = 0.0
                effect_size = float(np.sign(mean_difference) * np.inf)
        else:
            test = stats.ttest_rel(values, reference)
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            effect_size = mean_difference / difference_sd
        rows.append(
            {
                "timepoint": float(timepoint),
                "n_observations": int(len(values)),
                "mean_uptake": float(np.mean(values)),
                "sd_uptake": float(np.std(values, ddof=1)),
                "baseline_mean_uptake": float(np.mean(reference)),
                "mean_difference": mean_difference,
                "t_statistic": statistic,
                "p_value": p_value,
                "cohen_dz": effect_size,
            }
        )
    return rows


def forward_uptake_tables(
    features_dir: Path,
    clustering_dir: Path,
    ensembles: Iterable[str],
    modes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each forward construction under uniform and oracle weights."""
    modes = tuple(modes)
    if "uptake" not in modes:
        raise ValueError(
            "Forward uptake diagnostics require uptake as the baseline mode"
        )
    curve_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for ensemble in ensembles:
        features, _ = load_features(features_dir, ensemble)
        assignments = load_cluster_assignments(clustering_dir, ensemble)
        n_frames = features.features_shape[1]
        if n_frames != len(assignments):
            raise ValueError(f"Feature/assignment frame mismatch for {ensemble}")
        oracle_weights = compute_cluster_weights(
            assignments,
            {"open": GROUND_TRUTH["open"], "closed": GROUND_TRUTH["closed"]},
        )
        weight_sets = {
            "unweighted": np.full(n_frames, 1.0 / n_frames),
            "oracle_weighted": oracle_weights,
        }
        for weighting, frame_weights in weight_sets.items():
            predictions = {
                mode: flatten_uptake_observations(
                    predict_uptake(
                        configure_model(mode, assignments), features, frame_weights
                    )
                )
                for mode in modes
            }
            baseline = predictions["uptake"]
            for mode, uptake in predictions.items():
                for time_index, timepoint in enumerate(TIMEPOINTS):
                    values = uptake[time_index]
                    curve_rows.append(
                        {
                            "ensemble": ensemble,
                            "weighting": weighting,
                            "weighting_label": WEIGHT_LABELS[weighting],
                            "mode": mode,
                            "mode_label": MODE_LABELS[mode],
                            "timepoint": float(timepoint),
                            "n_observations": int(len(values)),
                            "mean_uptake": float(np.mean(values)),
                            "sd_uptake": float(np.std(values, ddof=1)),
                        }
                    )
                for row in paired_mode_statistics(uptake, baseline):
                    comparison_rows.append(
                        {
                            "ensemble": ensemble,
                            "weighting": weighting,
                            "weighting_label": WEIGHT_LABELS[weighting],
                            "mode": mode,
                            "mode_label": MODE_LABELS[mode],
                            "baseline_mode": "uptake",
                            "test": "two_sided_paired_t_test",
                            "effect_size": "paired_cohen_dz",
                            **row,
                        }
                    )
    return pd.DataFrame(curve_rows), pd.DataFrame(comparison_rows)


def plot_forward_uptake_curves(curves: pd.DataFrame, output_stem: Path) -> None:
    """Mean and SD across residue coordinates for every forward construction."""
    ensembles = tuple(dict.fromkeys(curves["ensemble"]))
    weightings = tuple(WEIGHT_LABELS)
    figure, axes = plt.subplots(
        len(ensembles),
        len(weightings),
        figsize=(11.5, 4.2 * len(ensembles)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, ensemble in enumerate(ensembles):
        for column_index, weighting in enumerate(weightings):
            axis = axes[row_index, column_index]
            panel = curves[
                (curves["ensemble"] == ensemble) & (curves["weighting"] == weighting)
            ]
            for mode in DEFAULT_MODES:
                trace = panel[panel["mode"] == mode].sort_values("timepoint")
                if trace.empty:
                    continue
                timepoints = trace["timepoint"].to_numpy(dtype=float)
                mean = trace["mean_uptake"].to_numpy(dtype=float)
                sd = trace["sd_uptake"].to_numpy(dtype=float)
                axis.plot(
                    timepoints,
                    mean,
                    color=MODE_COLORS[mode],
                    marker="o",
                    linewidth=2.0,
                    label=MODE_LABELS[mode],
                )
                axis.fill_between(
                    timepoints,
                    np.clip(mean - sd, 0.0, 1.0),
                    np.clip(mean + sd, 0.0, 1.0),
                    color=MODE_COLORS[mode],
                    alpha=0.14,
                    linewidth=0,
                )
            axis.set_xscale("log")
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{ensemble.replace('ISO_', 'ISO ')} — {WEIGHT_LABELS[weighting]}"
            )
            if row_index == len(ensembles) - 1:
                axis.set_xlabel("Timepoint (min)")
            if column_index == 0:
                axis.set_ylabel("Uptake (mean ± SD across residues)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=False,
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_forward_uptake_statistics(
    comparisons: pd.DataFrame, ensemble: str, output_stem: Path
) -> None:
    """Weighting columns and p-value/effect-size rows for one ensemble."""
    modes = tuple(mode for mode in DEFAULT_MODES if mode in set(comparisons["mode"]))
    weightings = tuple(WEIGHT_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), squeeze=False)
    effect_values = comparisons.loc[
        (comparisons["ensemble"] == ensemble) & np.isfinite(comparisons["cohen_dz"]),
        "cohen_dz",
    ].to_numpy(dtype=float)
    effect_limit = max(0.25, float(np.max(np.abs(effect_values), initial=0.0)))
    finite_p = comparisons.loc[
        (comparisons["ensemble"] == ensemble) & (comparisons["p_value"] > 0),
        "p_value",
    ].to_numpy(dtype=float)
    log_p_limit = max(
        1.0,
        float(
            np.max(
                -np.log10(np.clip(finite_p, np.finfo(float).tiny, 1.0)),
                initial=0.0,
            )
        ),
    )
    for column_index, weighting in enumerate(weightings):
        panel = comparisons[
            (comparisons["ensemble"] == ensemble)
            & (comparisons["weighting"] == weighting)
        ]
        p_values = np.asarray(
            [
                panel[panel["mode"] == mode]
                .set_index("timepoint")
                .reindex(TIMEPOINTS)["p_value"]
                .to_numpy(dtype=float)
                for mode in modes
            ]
        )
        log_p = -np.log10(np.clip(p_values, np.finfo(float).tiny, 1.0))
        p_axis = axes[0, column_index]
        p_image = p_axis.imshow(
            log_p, aspect="auto", cmap="viridis", vmin=0.0, vmax=log_p_limit
        )
        figure.colorbar(p_image, ax=p_axis, label="−log10(p)")
        for mode_index in range(len(modes)):
            for time_index in range(len(TIMEPOINTS)):
                p_axis.text(
                    time_index,
                    mode_index,
                    f"{p_values[mode_index, time_index]:.1e}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if log_p[mode_index, time_index] > 2 else "black",
                )
        p_axis.set_title(f"{WEIGHT_LABELS[weighting]} — paired p")

        effects = np.asarray(
            [
                panel[panel["mode"] == mode]
                .set_index("timepoint")
                .reindex(TIMEPOINTS)["cohen_dz"]
                .to_numpy(dtype=float)
                for mode in modes
            ]
        )
        d_axis = axes[1, column_index]
        d_image = d_axis.imshow(
            effects,
            aspect="auto",
            cmap="coolwarm",
            vmin=-effect_limit,
            vmax=effect_limit,
        )
        figure.colorbar(d_image, ax=d_axis, label="Cohen's dz")
        for mode_index in range(len(modes)):
            for time_index in range(len(TIMEPOINTS)):
                d_axis.text(
                    time_index,
                    mode_index,
                    f"{effects[mode_index, time_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        d_axis.set_title(f"{WEIGHT_LABELS[weighting]} — paired effect size")

        for axis in (p_axis, d_axis):
            axis.set_xticks(range(len(TIMEPOINTS)))
            axis.set_xticklabels([f"{value:g}" for value in TIMEPOINTS])
            axis.set_yticks(range(len(modes)))
            axis.set_yticklabels([MODE_LABELS[mode] for mode in modes])
        d_axis.set_xlabel("Timepoint (min)")
    axes[0, 0].set_ylabel("Forward construction\n(vs uptake baseline)")
    axes[1, 0].set_ylabel("Forward construction\n(vs uptake baseline)")
    figure.suptitle(
        f"{ensemble.replace('ISO_', 'ISO ')}: paired residue comparisons to uptake averaging",
        y=1.01,
    )
    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def run_forward_uptake_diagnostics(
    output_dir: Path,
    features_dir: Path,
    clustering_dir: Path,
    ensembles: Iterable[str],
    modes: Iterable[str],
) -> None:
    diagnostic_dir = output_dir / "forward_uptake_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    curves, comparisons = forward_uptake_tables(
        features_dir, clustering_dir, ensembles, modes
    )
    curves.to_csv(diagnostic_dir / "uptake_curve_summary.csv", index=False)
    comparisons.to_csv(diagnostic_dir / "uptake_paired_statistics.csv", index=False)
    plot_forward_uptake_curves(curves, diagnostic_dir / "mean_sd_uptake_vs_timepoint")
    for ensemble in dict.fromkeys(curves["ensemble"]):
        plot_forward_uptake_statistics(
            comparisons,
            ensemble,
            diagnostic_dir / f"{ensemble}_uptake_statistics_heatmap",
        )
    metadata = {
        "reduction": "mean_and_sample_sd_across_flattened_residue_coordinates",
        "baseline": "framewise_uptake_mode_within_the_same_weighting_and_ensemble",
        "test": "two_sided_paired_t_test_across_residues_at_each_timepoint",
        "effect_size": "paired_cohen_dz_mean_difference_over_sd_difference",
        "multiple_testing_correction": "none",
        "unweighted": "uniform_over_all_ensemble_frames",
        "oracle_weighted": "0.4_open_0.6_closed_0_intermediate",
        "intrinsic_rate_unit": "min^-1",
        "intrinsic_rate_provider": "jaxent_calculate_HDXrate",
        "intrinsic_rate_source_manifest": str(DEFAULT_RATE_SOURCE_MANIFEST.resolve()),
        "forward_construction_version": FORWARD_CONSTRUCTION_VERSION,
    }
    (diagnostic_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def build_prior_dataset(model, features, feature_top) -> ExpD_Dataloader:
    """Construct the prior dataset exactly as the existing Sigma fitter does."""
    n_frames = features.features_shape[1]
    parameters = Simulation_Parameters.from_frame_weights(
        jnp.ones(n_frames) / n_frames,
        model_parameters=(model.params,),
        forward_model_weights=jnp.ones(2),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
    )
    simulation = Simulation(
        input_features=(features,), forward_models=(model,), params=parameters
    )
    with jit_Guard(simulation, cleanup_on_exit=True) as simulation:
        simulation.initialise()
        simulation.forward(simulation, params=parameters)
        output_features = simulation.outputs[0].y_pred()
    prior_hdx = [
        HDX_peptide._create_from_features(topology=top, features=output_features[index])
        for index, top in enumerate(feature_top)
    ]
    prior = ExpD_Dataloader(data=prior_hdx)
    prior.create_datasets(features=features, feature_topology=feature_top)
    return prior


def oracle_uptake_coordinates(features) -> np.ndarray:
    """Per-residue coordinates used by the current synthetic Sigma generator."""
    model = build_uptake_model("standard", TIMEPOINTS, kint_unit="s^-1")
    per_frame = np.asarray(
        model.forward[m_key("HDX_peptide")](features, model.params).uptake,
        dtype=float,
    )
    if per_frame.ndim != 3:
        raise ValueError(
            f"Expected time x residue x frame uptake, got {per_frame.shape}"
        )
    return per_frame.mean(axis=0)


def construct_oracle_sigma(
    coordinates: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    condition_limit: float = 1e8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct, minimally ridge, invert, and diagnose one oracle covariance."""
    weights = compute_cluster_weights(
        assignments, {"open": GROUND_TRUTH["open"], "closed": GROUND_TRUTH["closed"]}
    )
    raw = compute_weighted_covariance(coordinates, weights)
    shrunk = construct_covariance(
        raw,
        weights,
        correction="population",
        target="identity",
        alpha=float(alpha),
        ridge=0.0,
    )
    thresholds = ridge_thresholds(shrunk, condition_limit=condition_limit)
    ridge = float(thresholds["ridge_stable"])
    covariance = (shrunk + np.eye(len(shrunk)) * ridge).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    precision = np.linalg.inv(covariance)
    precision_norm = float(np.linalg.norm(precision))
    if not np.isfinite(precision).all() or precision_norm <= 0:
        raise ValueError("Oracle covariance produced a non-finite precision matrix")
    normalized_precision = precision / precision_norm
    condition = float(np.linalg.cond(covariance))
    if eigenvalues[0] <= 0 or condition > condition_limit * (1 + 1e-8):
        raise ValueError(
            f"Regularized covariance failed stability check: min={eigenvalues[0]}, "
            f"condition={condition}"
        )
    arrays = {
        "Sigma_raw": raw,
        "Sigma_shrunk": shrunk,
        "Sigma": covariance,
        "Sigma_inv": precision,
        "Sigma_inv_normalized": normalized_precision,
        "frame_weights": weights,
        "cluster_assignments": assignments,
        "uptake_coordinates": coordinates,
    }
    metrics = {
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
    }
    return arrays, metrics


def prepare_sigma_artifacts(
    output_dir: Path,
    features_dir: Path,
    clustering_dir: Path,
    ensembles: Iterable[str],
    alphas: Iterable[float],
    condition_limit: float,
) -> tuple[dict[tuple[str, float], Path], pd.DataFrame]:
    sigma_dir = output_dir / "sigma"
    sigma_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, float], Path] = {}
    rows = []
    for ensemble in ensembles:
        features, _ = load_features(features_dir, ensemble)
        assignments = load_cluster_assignments(clustering_dir, ensemble)
        if features.features_shape[1] != len(assignments):
            raise ValueError(f"Feature/assignment frame mismatch for {ensemble}")
        coordinates = oracle_uptake_coordinates(features)
        for alpha in alphas:
            arrays, metrics = construct_oracle_sigma(
                coordinates, assignments, alpha, condition_limit
            )
            path = sigma_dir / f"{ensemble}_GT_sigma_alpha{alpha_token(alpha)}.npz"
            np.savez_compressed(path, **arrays, alpha=float(alpha))
            paths[(ensemble, float(alpha))] = path
            rows.append({"ensemble": ensemble, **metrics, "path": str(path)})
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
    """Execute one cell using the production fitting entry point."""
    features, feature_top = load_features(spec.features_dir, spec.ensemble)
    assignments = load_cluster_assignments(spec.clustering_dir, spec.ensemble)
    model = configure_model(spec.mode, assignments)
    prior = build_prior_dataset(model, features, feature_top)
    train_data, val_data = load_split(
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
        convergence=list(CONVERGENCE_RATES),
        loss_config=LossConfig(
            primary_loss="hdx_uptake_sigma_MSE_loss",
            maxent_scaling=MAXENT,
            optimize_bv_params=False,
        ),
        opt_config=OptimizationConfig(
            n_steps=spec.n_steps,
            learning_rate=spec.learning_rate,
            ema_alpha=spec.ema_alpha,
            convergence_rates=list(CONVERGENCE_RATES),
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
    config["effective_settings"]["frame_averaging_mode"] = {
        "linear": "linear_uptake",
        "rate": "rate",
        "uptake": "frame_uptake",
    }[spec.mode]
    config["sidecar_settings"] = {
        "mode": spec.mode,
        "uptake_model": "linear" if spec.mode == "linear" else "standard",
        "intrinsic_rate_unit": "min^-1",
        "intrinsic_rate_provider": "jaxent_calculate_HDXrate",
        "intrinsic_rate_source_manifest": str(DEFAULT_RATE_SOURCE_MANIFEST.resolve()),
        "forward_construction_version": FORWARD_CONSTRUCTION_VERSION,
        "shrinkage_alpha": spec.alpha,
        "sigma_path": spec.sigma_path,
        "split_type": spec.split_type,
        "split_idx": spec.split_idx,
    }
    temporary_config = spec.config_path.with_suffix(".json.tmp")
    temporary_config.write_text(json.dumps(config, indent=2, sort_keys=True))
    temporary_config.replace(spec.config_path)


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
    pending: list[tuple[Path, Path]] = []
    for spec in specs:
        if run_is_complete(spec):
            print(f"[resume] {spec.run_id}")
            continue
        spec_path = spec_dir / f"{spec.run_id}.json"
        spec_path.write_text(
            json.dumps(dataclasses.asdict(spec), indent=2, sort_keys=True)
        )
        pending.append((spec_path, log_dir / f"{spec.run_id}.log"))
    if not pending:
        return
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
        details = "\n".join(f"  {run_id}: {path}" for run_id, path in failures)
        raise RuntimeError(f"{len(failures)} fit(s) failed:\n{details}")


from sidecar_selection import closed_validation_mse, select_best_rows, SELECTION_POLICY


def score_history(
    spec: RunSpec,
    context_cache: dict[tuple[str, str, str, int], tuple] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Score native convergence states for fixed closed-coordinate selection."""
    history = load_optimization_history_from_file(str(spec.history_path))
    labeled = iter_labeled_convergence_states(history)
    if not labeled:
        raise ValueError(f"No convergence states recorded for {spec.run_id}")
    cache_key = (spec.ensemble, spec.mode, spec.split_type, spec.split_idx)
    cached = None if context_cache is None else context_cache.get(cache_key)
    if cached is None:
        features, feature_top = load_features(spec.features_dir, spec.ensemble)
        assignments = load_cluster_assignments(spec.clustering_dir, spec.ensemble)
        model = configure_model(spec.mode, assignments)
        train_data, val_data = load_split(
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
        y_true_val = analysis.get_experimental_uptake(val_data)
        cached = features, assignments, model, loader, y_true_val
        if context_cache is not None:
            context_cache[cache_key] = cached
    features, assignments, model, loader, y_true_val = cached
    rows = []
    weights_stack = []
    for labeled_state in labeled:
        state = labeled_state.state
        weights = validated_frame_weight_simplex(state.params.frame_weight_simplex)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        predicted = predict_uptake(model, features, weights)
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
        rows.append(
            {
                "run_id": spec.run_id,
                "ensemble": spec.ensemble,
                "mode": spec.mode,
                "mode_label": MODE_LABELS[spec.mode],
                "alpha": spec.alpha,
                "split_type": spec.split_type,
                "split_idx": spec.split_idx,
                "maxent": MAXENT,
                "convergence_rank": labeled_state.rank,
                "convergence_threshold": labeled_state.threshold,
                "step": int(np.asarray(state.step)),
                "native_sigma_val_loss": native_val_loss,
                "val_mse": analysis.calculate_mse(mapped, y_true_val),
                "val_closed_sigma_mse": closed_validation_mse(spec, mapped, y_true_val),
                "recovery_percent": calculate_recovery_percentage(
                    assignments, weights, GROUND_TRUTH, STATE_MAPPING
                ),
                "ess": effective_sample_size(weights),
                "ess_percent": 100.0 * effective_sample_size(weights) / len(weights),
                "n_frames": len(weights),
            }
        )
        weights_stack.append(weights)
    return rows, np.stack(weights_stack)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_mse", "recovery_percent", "ess", "ess_percent"]
    grouped = selected.groupby(["ensemble", "mode", "mode_label", "alpha"], sort=False)
    summary = grouped[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def plot_metric(
    selected: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: Path,
    alphas: Iterable[float],
    log_y: bool = False,
) -> None:
    alpha_values = tuple(float(value) for value in alphas)
    positions = {alpha: index for index, alpha in enumerate(alpha_values)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, ensemble in zip(axes, DEFAULT_ENSEMBLES):
        panel = selected[selected["ensemble"] == ensemble]
        for mode in DEFAULT_MODES:
            mode_rows = panel[panel["mode"] == mode]
            color = MODE_COLORS[mode]
            for (_, split_idx), trace in mode_rows.groupby(
                ["split_type", "split_idx"], sort=False
            ):
                trace = trace.sort_values("alpha")
                linestyle = "--" if trace.iloc[0]["split_type"] == "spatial" else "-"
                axis.plot(
                    [positions[float(value)] for value in trace["alpha"]],
                    trace[metric],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                    linestyle=linestyle,
                )
            nonredundant = mode_rows[mode_rows["split_type"] == "sequence_cluster"]
            mean = nonredundant.groupby("alpha", sort=False)[metric].mean()
            if not mean.empty:
                mean = mean.reindex([a for a in alpha_values if a in mean.index])
                axis.plot(
                    [positions[float(value)] for value in mean.index],
                    mean.values,
                    color=color,
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    label=MODE_LABELS[mode],
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
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(top=0.82)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def analyze_specs(
    specs: list[RunSpec], output_dir: Path, alphas: Iterable[float]
) -> None:
    score_rows = []
    weight_arrays: dict[str, np.ndarray] = {}
    incomplete = []
    context_cache: dict[tuple[str, str, str, int], tuple] = {}
    for spec in specs:
        if not run_is_complete(spec):
            incomplete.append(
                {"run_id": spec.run_id, "reason": "missing_or_invalid_history"}
            )
            continue
        try:
            rows, weights = score_history(spec, context_cache)
        except ValueError as error:
            incomplete.append({"run_id": spec.run_id, "reason": str(error)})
            continue
        score_rows.extend(rows)
        for row, weights_for_state in zip(rows, weights):
            weight_arrays[f"{spec.run_id}__conv{int(row['convergence_rank']):02d}"] = (
                weights_for_state
            )
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output_dir / "convergence_scores.csv", index=False)
    selected = select_best_rows(scores)
    selected.to_csv(output_dir / "selected_results.csv", index=False)
    summarize_selected(selected).to_csv(output_dir / "summary.csv", index=False)
    selected_weights = {
        f"{row.run_id}__conv{int(row.convergence_rank):02d}": weight_arrays[
            f"{row.run_id}__conv{int(row.convergence_rank):02d}"
        ]
        for row in selected.itertuples()
    }
    np.savez_compressed(output_dir / "selected_frame_weights.npz", **selected_weights)
    pd.DataFrame(incomplete, columns=["run_id", "reason"]).to_csv(
        output_dir / "incomplete_runs.csv", index=False
    )
    if selected.empty:
        raise RuntimeError("No convergence states were available for plotting")
    plot_metric(
        selected,
        "recovery_percent",
        "Recovery (%)",
        output_dir / "plots" / "recovery_vs_shrinkage",
        alphas,
    )
    plot_metric(
        selected,
        "ess_percent",
        "ESS (%)",
        output_dir / "plots" / "ess_percent_vs_shrinkage",
        alphas,
        log_y=True,
    )


def build_specs(args, sigma_paths: dict[tuple[str, float], Path]) -> list[RunSpec]:
    return [
        RunSpec(
            ensemble=ensemble,
            mode=mode,
            alpha=float(alpha),
            split_type=args.split_type,
            split_idx=split_idx,
            sigma_path=str(sigma_paths[(ensemble, float(alpha))]),
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
        for mode in args.modes
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
    completed = sum(run_is_complete(spec) for spec in specs)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "maxent": MAXENT,
        "primary_loss": "hdx_uptake_sigma_MSE_loss",
        "forward_construction_version": FORWARD_CONSTRUCTION_VERSION,
        "intrinsic_rate_unit": "min^-1",
        "intrinsic_rate_provider": "jaxent_calculate_HDXrate",
        "intrinsic_rate_source_manifest": str(DEFAULT_RATE_SOURCE_MANIFEST.resolve()),
        "uptake_reduction": "frame_uptake",
        "selection_metric": SELECTION_POLICY,
        "sample_size_correction": "population",
        "shrinkage_target": "identity",
        "condition_limit": args.condition_limit,
        "timepoints_minutes": TIMEPOINTS.tolist(),
        "ensembles": list(args.ensembles),
        "modes": list(args.modes),
        "alphas": list(args.alphas),
        "split_type": args.split_type,
        "n_splits": args.n_splits,
        "expected_fits": len(specs),
        "completed_fit_files": completed,
        "sigma_rows": len(sigma_metrics),
        "settings": {
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
        description="Run the focused ISO BI/TRI GT-Sigma shrinkage experiment."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--datasplit-dir", type=Path, default=DEFAULT_DATASPLIT_DIR)
    parser.add_argument("--clustering-dir", type=Path, default=DEFAULT_CLUSTERING_DIR)
    parser.add_argument("--ensembles", default=",".join(DEFAULT_ENSEMBLES))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--alphas", default=",".join(f"{a:g}" for a in DEFAULT_ALPHAS))
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
    parser.add_argument(
        "--phase",
        choices=("all", "fit", "analyze", "diagnostics"),
        default="all",
    )
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.n_splits < 1 or args.n_steps < 1 or args.jobs < 1:
        parser.error("--n-splits, --n-steps, and --jobs must be positive")
    if not np.isfinite(args.condition_limit) or args.condition_limit <= 1:
        parser.error("--condition-limit must be finite and greater than one")
    if any(ensemble not in DEFAULT_ENSEMBLES for ensemble in args.ensembles):
        parser.error(f"--ensembles must be drawn from {DEFAULT_ENSEMBLES}")
    if any(mode not in DEFAULT_MODES for mode in args.modes):
        parser.error(f"--modes must be drawn from {DEFAULT_MODES}")
    if not args.alphas or any(
        not np.isfinite(a) or a < 0 or a > 1 for a in args.alphas
    ):
        parser.error("--alphas must contain finite values in [0, 1]")
    if args.split_type != "sequence_cluster":
        parser.error("this first experiment is restricted to sequence_cluster")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_spec:
        spec = RunSpec(**json.loads(args.worker_spec.read_text()))
        run_fit(spec)
        return
    args.ensembles = parse_csv(args.ensembles)
    args.modes = parse_csv(args.modes)
    args.alphas = parse_csv(args.alphas, float)
    validate_args(parser, args)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = HERE / f"_sigma_shrinkage_sidecar_{stamp}"
    args.output_dir = args.output_dir.resolve()
    args.features_dir = args.features_dir.resolve()
    args.datasplit_dir = args.datasplit_dir.resolve()
    args.clustering_dir = args.clustering_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "diagnostics":
        run_forward_uptake_diagnostics(
            args.output_dir,
            args.features_dir,
            args.clustering_dir,
            args.ensembles,
            args.modes,
        )
        print(f"Results: {args.output_dir}")
        return
    sigma_paths, sigma_metrics = prepare_sigma_artifacts(
        args.output_dir,
        args.features_dir,
        args.clustering_dir,
        args.ensembles,
        args.alphas,
        args.condition_limit,
    )
    specs = build_specs(args, sigma_paths)
    if args.phase in {"all", "fit"}:
        execute_specs(specs, args.output_dir, args.jobs)
    write_manifest(args, specs, sigma_metrics)
    if args.phase in {"all", "analyze"}:
        analyze_specs(specs, args.output_dir, args.alphas)
        run_forward_uptake_diagnostics(
            args.output_dir,
            args.features_dir,
            args.clustering_dir,
            args.ensembles,
            args.modes,
        )
        write_manifest(args, specs, sigma_metrics)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
