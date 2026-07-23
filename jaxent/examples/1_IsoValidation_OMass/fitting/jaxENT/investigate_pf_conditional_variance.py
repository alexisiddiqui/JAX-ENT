#!/usr/bin/env python3
"""Controlled PF conditional-variance matching experiment.

The experiment uses the plausible ISO_BI ensemble to define a known 40:60
open/closed target.  Uptake means are fitted with the production average-first BV
semantics.  A separate, optional objective matches the target and predicted
``1 / diag(C_PF**-1)`` profiles.  It does not model replicate or observation noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jaxent.src.analysis.pf_variance import (
    average_after_uptake,
    average_first_uptake,
    conditional_variance_log_ratio_loss,
    conditional_variance_profile,
    covariance_mse,
    kl_to_uniform,
    marginal_variance_profile,
    shrink_covariance,
    trace_normalize_precision,
    weighted_population_covariance,
    weights_from_logits,
)


TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=np.float32)
TARGET_OPEN = 0.40
TARGET_CLOSED = 0.60


@dataclass(frozen=True)
class ExperimentConfig:
    target_modes: tuple[str, ...]
    ensembles: tuple[str, ...]
    split_types: tuple[str, ...]
    split_indices: tuple[int, ...]
    gammas: tuple[float, ...]
    maxent_values: tuple[float, ...]
    alphas: tuple[float, ...]
    starts: int
    steps: int
    learning_rate: float
    bv_bc: float
    bv_bh: float
    variance_profile: str


@dataclass(frozen=True)
class EnsembleData:
    name: str
    log_pf_by_frame: np.ndarray
    k_ints: np.ndarray
    assignments: np.ndarray


def _csv_tuple(value: str, cast: type = str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_weights(assignments: np.ndarray) -> np.ndarray:
    """Assign 40%/60% mass uniformly within the BI open/closed clusters."""

    assignments = np.asarray(assignments)
    weights = np.zeros(assignments.size, dtype=np.float64)
    open_mask = assignments == 0
    closed_mask = assignments == 1
    if not open_mask.any() or not closed_mask.any():
        raise ValueError("Target ensemble must contain both open and closed frames")
    weights[open_mask] = TARGET_OPEN / open_mask.sum()
    weights[closed_mask] = TARGET_CLOSED / closed_mask.sum()
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Ground-truth frame weights do not sum to one")
    return weights.astype(np.float32)


def load_ensemble(features_path: Path, clusters_path: Path, name: str, bc: float, bh: float) -> EnsembleData:
    with np.load(features_path) as features:
        log_pf = bc * features["heavy_contacts"] + bh * features["acceptor_contacts"]
        k_ints = np.asarray(features["k_ints"])
    assignments = pd.read_csv(clusters_path)["cluster_assignment"].to_numpy(dtype=int)
    if log_pf.shape[1] != assignments.size:
        raise ValueError(f"{name}: feature frames and cluster assignments are misaligned")
    return EnsembleData(name=name, log_pf_by_frame=log_pf, k_ints=k_ints, assignments=assignments)


def load_split_indices(topology_path: Path, dimension: int) -> np.ndarray:
    payload = json.loads(topology_path.read_text())
    indices = np.asarray(
        [topology["fragment_index"] for topology in payload["topologies"]], dtype=int
    )
    # The shipped target has one terminal fragment (index 293) absent from BV features.
    indices = indices[(indices >= 0) & (indices < dimension)]
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError(f"Invalid or duplicate feature indices in {topology_path}")
    return indices


def _population_metrics(weights: np.ndarray, assignments: np.ndarray) -> dict[str, float]:
    open_population = float(weights[assignments == 0].sum())
    closed_population = float(weights[assignments == 1].sum())
    decoy_population = float(weights[assignments == -1].sum())
    return {
        "open_population": open_population,
        "closed_population": closed_population,
        "decoy_population": decoy_population,
        "population_l1_error": abs(open_population - TARGET_OPEN)
        + abs(closed_population - TARGET_CLOSED),
    }


def _covariance_diagnostics(
    target_values: np.ndarray,
    target_weights: np.ndarray,
    predicted_values: np.ndarray,
    predicted_weights: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    target_cov = np.asarray(weighted_population_covariance(target_values, target_weights))
    pred_cov = np.asarray(weighted_population_covariance(predicted_values, predicted_weights))
    target_regularized = np.asarray(shrink_covariance(target_cov, alpha=alpha))
    pred_regularized = np.asarray(shrink_covariance(pred_cov, alpha=alpha))
    upper = np.triu_indices(target_cov.shape[0], k=1)
    target_offdiag = target_cov[upper]
    pred_offdiag = pred_cov[upper]
    corr = float(np.corrcoef(target_offdiag, pred_offdiag)[0, 1])
    relative_frobenius = float(
        np.linalg.norm(pred_cov - target_cov) / max(np.linalg.norm(target_cov), 1e-12)
    )
    return {
        "full_covariance_offdiag_correlation": corr,
        "full_covariance_relative_frobenius": relative_frobenius,
        "target_regularized_condition": float(np.linalg.cond(target_regularized)),
        "pred_regularized_condition": float(np.linalg.cond(pred_regularized)),
    }


def optimize_weights(
    *,
    predicted_log_pf: np.ndarray,
    predicted_k_ints: np.ndarray,
    target_log_pf: np.ndarray,
    target_weights: np.ndarray,
    target_uptake: np.ndarray,
    curve_precision: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    assignments: np.ndarray,
    gamma: float,
    maxent_value: float,
    alpha: float,
    steps: int,
    learning_rate: float,
    start_seed: int,
    variance_profile: str,
) -> dict[str, Any]:
    """Fit raw logits and return metrics on a common validation scale."""

    z_pred = jnp.asarray(predicted_log_pf)
    z_target = jnp.asarray(target_log_pf)
    k_pred = jnp.asarray(predicted_k_ints)
    true_weights = jnp.asarray(target_weights)
    target = jnp.asarray(target_uptake)
    timepoints = jnp.asarray(TIMEPOINTS)
    train_idx = jnp.asarray(train_indices)
    val_idx = jnp.asarray(val_indices)

    train_precision = trace_normalize_precision(
        jnp.asarray(curve_precision[np.ix_(train_indices, train_indices)])
    )
    val_precision = trace_normalize_precision(
        jnp.asarray(curve_precision[np.ix_(val_indices, val_indices)])
    )
    profile_function = (
        conditional_variance_profile
        if variance_profile == "conditional"
        else marginal_variance_profile
    )
    target_cond_train = jax.lax.stop_gradient(
        profile_function(z_target[train_idx], true_weights, alpha=alpha)
    )
    target_cond_val = jax.lax.stop_gradient(
        profile_function(z_target[val_idx], true_weights, alpha=alpha)
    )

    uniform_logits = jnp.zeros(z_pred.shape[1], dtype=z_pred.dtype)
    uniform_weights = weights_from_logits(uniform_logits)
    uniform_prediction = average_first_uptake(z_pred, k_pred, timepoints, uniform_weights)
    initial_mean_loss = covariance_mse(
        uniform_prediction[:, train_idx], target[:, train_idx], train_precision
    )
    mean_scale = jnp.maximum(initial_mean_loss, jnp.asarray(1e-12, dtype=z_pred.dtype))

    def components(logits: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        weights = weights_from_logits(logits)
        prediction = average_first_uptake(z_pred, k_pred, timepoints, weights)
        mean_loss = covariance_mse(prediction[:, train_idx], target[:, train_idx], train_precision)
        pred_cond = profile_function(z_pred[train_idx], weights, alpha=alpha)
        variance_loss = conditional_variance_log_ratio_loss(pred_cond, target_cond_train)
        maxent_loss = kl_to_uniform(weights)
        total = mean_loss / mean_scale + gamma * variance_loss + maxent_value * maxent_loss
        return total, (mean_loss, variance_loss, maxent_loss)

    optimizer = optax.adam(learning_rate)
    rng = np.random.default_rng(start_seed)
    initial_logits = np.zeros(z_pred.shape[1], dtype=np.float32)
    if start_seed != 0:
        initial_logits += rng.normal(scale=0.01, size=initial_logits.shape).astype(np.float32)
    logits = jnp.asarray(initial_logits)
    opt_state = optimizer.init(logits)

    @jax.jit
    def step(logits: jax.Array, state: optax.OptState) -> tuple[jax.Array, optax.OptState, jax.Array]:
        (loss, _), gradients = jax.value_and_grad(components, has_aux=True)(logits)
        updates, state = optimizer.update(gradients, state, logits)
        return optax.apply_updates(logits, updates), state, loss

    final_step_loss = jnp.asarray(jnp.nan)
    for _ in range(steps):
        logits, opt_state, final_step_loss = step(logits, opt_state)

    total, (train_mean, train_variance, maxent_loss) = components(logits)
    fitted_weights = weights_from_logits(logits)
    prediction = average_first_uptake(z_pred, k_pred, timepoints, fitted_weights)
    val_mean = covariance_mse(prediction[:, val_idx], target[:, val_idx], val_precision)
    val_mse = jnp.mean(jnp.square(prediction[:, val_idx] - target[:, val_idx]))
    pred_cond_val = profile_function(z_pred[val_idx], fitted_weights, alpha=alpha)
    val_variance = conditional_variance_log_ratio_loss(pred_cond_val, target_cond_val)

    weights_np = np.asarray(fitted_weights)
    result: dict[str, Any] = {
        "train_total": float(total),
        "train_curve_mse": float(train_mean),
        "train_pf_variance_loss": float(train_variance),
        "val_curve_mse": float(val_mean),
        "val_ordinary_mse": float(val_mse),
        "val_pf_variance_loss": float(val_variance),
        "maxent_kl": float(maxent_loss),
        "effective_sample_size": float(1.0 / np.sum(np.square(weights_np))),
        "final_step_loss": float(final_step_loss),
        "finite": bool(np.isfinite(np.asarray(total)) and np.all(np.isfinite(weights_np))),
        "weights": weights_np,
    }
    result.update(_population_metrics(weights_np, assignments))
    return result


def _select_results(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifying = [
        "ensemble",
        "target_mode",
        "split_type",
        "split_index",
        "method",
        "gamma",
        "maxent_value",
        "alpha",
    ]
    best_starts = raw.loc[raw.groupby(identifying)["train_total"].idxmin()].copy()
    selection_groups = ["ensemble", "target_mode", "split_type", "split_index", "method"]
    selected = best_starts.loc[
        best_starts.groupby(selection_groups)["val_curve_mse"].idxmin()
    ].copy()
    return best_starts, selected


def _decision_gate(
    _selected: pd.DataFrame, _best_starts: pd.DataFrame, _config: ExperimentConfig
) -> dict[str, Any]:
    return {
        "status": "not_evaluated",
        "reason": "Run analyze_pf_cluster_recovery.py after fitting selected weights.",
    }


def _write_plots(selected: pd.DataFrame, output_dir: Path) -> None:
    if selected.empty:
        return
    metrics = ["val_curve_mse", "population_l1_error", "decoy_population", "effective_sample_size"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric in zip(axes.ravel(), metrics):
        summary = selected.groupby("method")[metric].agg(["mean", "std"])
        axis.bar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), capsize=4)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_dir / "selected_method_summary.png", dpi=180)
    plt.close(figure)


def merge_shard_outputs(
    shard_root: Path, output_dir: Path, config: ExperimentConfig
) -> None:
    """Merge non-overlapping target/split shards and apply the complete gate."""

    shard_dirs = sorted(path for path in shard_root.iterdir() if path.is_dir())
    if not shard_dirs:
        raise ValueError(f"No shard directories found under {shard_root}")
    raw = pd.concat(
        [pd.read_csv(path / "raw_results.csv") for path in shard_dirs], ignore_index=True
    )
    if raw.run_id.duplicated().any():
        duplicates = raw.loc[raw.run_id.duplicated(), "run_id"].tolist()
        raise ValueError(f"Shard outputs contain duplicate run IDs: {duplicates[:3]}")
    best_starts, selected = _select_results(raw)

    shard_selected = pd.concat(
        [pd.read_csv(path / "selected_results.csv") for path in shard_dirs], ignore_index=True
    )
    diagnostics = [
        "full_covariance_offdiag_correlation",
        "full_covariance_relative_frobenius",
        "target_regularized_condition",
        "pred_regularized_condition",
    ]
    selected = selected.merge(
        shard_selected[["run_id", *diagnostics]], on="run_id", how="left", validate="one_to_one"
    )
    if selected[diagnostics].isna().any(axis=None):
        raise ValueError("Selected shard diagnostics are incomplete")

    selected_weights: dict[str, np.ndarray] = {}
    for path in shard_dirs:
        with np.load(path / "selected_weights.npz") as archive:
            selected_weights.update({name: np.asarray(archive[name]) for name in archive.files})
    missing_weights = set(selected.run_id) - set(selected_weights)
    if missing_weights:
        raise ValueError(f"Missing selected weights for: {sorted(missing_weights)[:3]}")

    first_manifest = json.loads((shard_dirs[0] / "manifest.json").read_text())
    first_manifest["config"] = asdict(config)
    first_manifest["merged_shards"] = [str(path) for path in shard_dirs]
    first_manifest["initialization_note"] = (
        "The final grid used one uniform start after the three-start pilot converged "
        "to matching metrics at approximately 1e-6."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(first_manifest, indent=2))
    raw.to_csv(output_dir / "raw_results.csv", index=False)
    selected.to_csv(output_dir / "selected_results.csv", index=False)
    np.savez_compressed(
        output_dir / "selected_weights.npz",
        **{run_id: selected_weights[run_id] for run_id in selected.run_id},
    )
    decision = _decision_gate(selected, best_starts, config)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    _write_plots(selected, output_dir)


def run_experiment(config: ExperimentConfig, output_dir: Path, repo_root: Path) -> None:
    example_root = repo_root / "jaxent/examples/1_IsoValidation_OMass"
    fitting_root = example_root / "fitting/jaxENT"
    feature_root = fitting_root / "_featurise"
    cluster_root = example_root / "data/_clustering_results"
    split_root = fitting_root / "_datasplits"
    curve_precision_path = fitting_root / "_covariance_matrices/Sigma.npz"

    input_paths = {
        "bi_features": feature_root / "features_iso_bi.npz",
        "tri_features": feature_root / "features_iso_tri.npz",
        "bi_clusters": cluster_root / "cluster_assignments_ISO_BI.csv",
        "tri_clusters": cluster_root / "cluster_assignments_ISO_TRI.csv",
        "curve_precision": curve_precision_path,
    }
    ensembles = {
        name: load_ensemble(
            input_paths[f"{name.lower()}_features"],
            input_paths[f"{name.lower()}_clusters"],
            name,
            config.bv_bc,
            config.bv_bh,
        )
        for name in config.ensembles
    }
    if "BI" not in ensembles:
        ensembles["BI"] = load_ensemble(
            input_paths["bi_features"],
            input_paths["bi_clusters"],
            "BI",
            config.bv_bc,
            config.bv_bh,
        )
    target_ensemble = ensembles["BI"]
    target_weights = cluster_weights(target_ensemble.assignments)
    dimension = target_ensemble.log_pf_by_frame.shape[0]
    if any(data.log_pf_by_frame.shape[0] != dimension for data in ensembles.values()):
        raise ValueError("BI and TRI residue axes are not aligned")
    if any(not np.allclose(data.k_ints, target_ensemble.k_ints) for data in ensembles.values()):
        raise ValueError("BI and TRI intrinsic-rate vectors are not aligned")

    average_first_target = np.asarray(
        average_first_uptake(
            target_ensemble.log_pf_by_frame,
            target_ensemble.k_ints,
            TIMEPOINTS,
            target_weights,
        )
    )
    average_after_target = np.asarray(
        average_after_uptake(
            target_ensemble.log_pf_by_frame,
            target_ensemble.k_ints,
            TIMEPOINTS,
            target_weights,
        )
    )
    targets = {
        "average_first": average_first_target,
        "average_after": average_after_target,
    }
    jensen_gap = np.abs(average_after_target - average_first_target)
    with np.load(curve_precision_path) as covariance_archive:
        curve_precision = np.asarray(covariance_archive["Sigma_inv"])[:dimension, :dimension]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(config),
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in input_paths.items()},
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "target_open": TARGET_OPEN,
        "target_closed": TARGET_CLOSED,
        "jensen_gap_mean_abs": float(jensen_gap.mean()),
        "jensen_gap_max_abs": float(jensen_gap.max()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows: list[dict[str, Any]] = []
    fitted_weights_by_run: dict[str, np.ndarray] = {}
    for target_mode in config.target_modes:
        target_uptake = targets[target_mode]
        for split_type in config.split_types:
            for split_index in config.split_indices:
                split_dir = split_root / split_type / f"split_{split_index:03d}"
                train_indices = load_split_indices(split_dir / "train_topology.json", dimension)
                val_indices = load_split_indices(split_dir / "val_topology.json", dimension)
                for ensemble_name in config.ensembles:
                    ensemble = ensembles[ensemble_name]
                    for gamma in config.gammas:
                        alphas = (0.05,) if gamma == 0.0 else config.alphas
                        for alpha in alphas:
                            for maxent_value in config.maxent_values:
                                for start in range(config.starts):
                                    run_id = (
                                        f"{ensemble_name}_{target_mode}_{split_type}_{split_index:03d}"
                                        f"_g{gamma:g}_m{maxent_value:g}_a{alpha:g}_s{start}"
                                    )
                                    result = optimize_weights(
                                        predicted_log_pf=ensemble.log_pf_by_frame,
                                        predicted_k_ints=ensemble.k_ints,
                                        target_log_pf=target_ensemble.log_pf_by_frame,
                                        target_weights=target_weights,
                                        target_uptake=target_uptake,
                                        curve_precision=curve_precision,
                                        train_indices=train_indices,
                                        val_indices=val_indices,
                                        assignments=ensemble.assignments,
                                        gamma=gamma,
                                        maxent_value=maxent_value,
                                        alpha=alpha,
                                        steps=config.steps,
                                        learning_rate=config.learning_rate,
                                            start_seed=start,
                                            variance_profile=config.variance_profile,
                                    )
                                    fitted_weights_by_run[run_id] = result.pop("weights")
                                    rows.append(
                                        {
                                            "run_id": run_id,
                                            "ensemble": ensemble_name,
                                            "target_mode": target_mode,
                                            "split_type": split_type,
                                            "split_index": split_index,
                                            "method": "baseline" if gamma == 0.0 else "variance_match",
                                            "gamma": gamma,
                                            "maxent_value": maxent_value,
                                            "alpha": alpha,
                                            "start": start,
                                            "variance_profile": config.variance_profile,
                                            **result,
                                        }
                                    )
                                    pd.DataFrame(rows).to_csv(output_dir / "raw_results.csv", index=False)

    raw = pd.DataFrame(rows)
    best_starts, selected = _select_results(raw)
    diagnostic_columns: dict[str, list[float]] = {}
    for _, row in selected.iterrows():
        diagnostics = _covariance_diagnostics(
            target_ensemble.log_pf_by_frame,
            target_weights,
            ensembles[row.ensemble].log_pf_by_frame,
            fitted_weights_by_run[row.run_id],
            float(row.alpha),
        )
        for name, value in diagnostics.items():
            diagnostic_columns.setdefault(name, []).append(value)
    for name, values in diagnostic_columns.items():
        selected[name] = values
    selected.to_csv(output_dir / "selected_results.csv", index=False)
    np.savez_compressed(
        output_dir / "selected_weights.npz",
        **{row.run_id: fitted_weights_by_run[row.run_id] for _, row in selected.iterrows()},
    )
    decision = _decision_gate(selected, best_starts, config)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    _write_plots(selected, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--merge-shards",
        type=Path,
        default=None,
        help="Merge completed non-overlapping shard directories instead of fitting.",
    )
    parser.add_argument("--target-modes", default="average_first,average_after")
    parser.add_argument("--ensembles", default="BI,TRI")
    parser.add_argument("--split-types", default="sequence_cluster,spatial")
    parser.add_argument("--split-indices", default="0,1,2")
    parser.add_argument("--gammas", default="0,0.01,0.1,1,10")
    parser.add_argument("--maxent-values", default="1,10,100,1000")
    parser.add_argument("--alphas", default="0.01,0.05,0.10")
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--bv-bc", type=float, default=0.35)
    parser.add_argument("--bv-bh", type=float, default=2.0)
    parser.add_argument(
        "--variance-profile",
        choices=("conditional", "marginal"),
        default="conditional",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small non-decisive BI/TRI integration check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[5]
    output_dir = args.output_dir or Path(__file__).resolve().parent / "_pf_conditional_variance"
    if args.smoke:
        config = ExperimentConfig(
            target_modes=("average_first",),
            ensembles=("BI", "TRI"),
            split_types=("sequence_cluster",),
            split_indices=(0,),
            gammas=(0.0, 1.0),
            maxent_values=(1.0,),
            alphas=(0.05,),
            starts=1,
            steps=min(args.steps, 50),
            learning_rate=args.learning_rate,
            bv_bc=args.bv_bc,
            bv_bh=args.bv_bh,
            variance_profile=args.variance_profile,
        )
    else:
        config = ExperimentConfig(
            target_modes=_csv_tuple(args.target_modes),
            ensembles=_csv_tuple(args.ensembles),
            split_types=_csv_tuple(args.split_types),
            split_indices=_csv_tuple(args.split_indices, int),
            gammas=_csv_tuple(args.gammas, float),
            maxent_values=_csv_tuple(args.maxent_values, float),
            alphas=_csv_tuple(args.alphas, float),
            starts=args.starts,
            steps=args.steps,
            learning_rate=args.learning_rate,
            bv_bc=args.bv_bc,
            bv_bh=args.bv_bh,
            variance_profile=args.variance_profile,
        )
    if args.merge_shards is not None:
        merge_shard_outputs(args.merge_shards.resolve(), output_dir.resolve(), config)
    else:
        run_experiment(config, output_dir.resolve(), repo_root)
    if not args.smoke:
        from analyze_pf_cluster_recovery import analyze

        analyze(output_dir.resolve(), repo_root, minimum_open_ess_fraction=0.8)


if __name__ == "__main__":
    main()
