"""
compute_sigma_synthetic.py

Computes covariance matrices (sigma) for the synthetic data using clustering results.

Requirements:
    - Clustering results (_clustering_results/)
    - Featurized data (_featurise/)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py \\
        --clustering_dir ... \\
        --features_dir ... \\
        --ensemble_name ISO_BI \\
        --output_dir ...

Output:
    - Sigma matrices in _covariance_matrices_sigma/

Disposable comparison (existing artifacts are not overwritten):
    python -m jaxent.examples.1_IsoValidation_OMass.fitting.jaxENT.compute_sigma_synthetic \\
        --ensemble_name ISO_BI --output_dir /tmp/my_sigma_experiment --diagnostics

Weighted construction options: --sample_size_correction population|weighted_sample,
--shrinkage_target none|identity|diagonal, --shrinkage_alpha (0..1), --ridge.
The unweighted comparator retains its original numpy.cov ddof=1 and 1e-6 ridge.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.linalg import cho_solve
from scipy.stats import rankdata

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import PTSerialiser
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.predict import run_predict


# --- Helper Functions ---
def plot_heatmap(
    matrix,
    title,
    filename,
    output_dir,
    cmap="viridis",
    annot=False,
    fmt=".2f",
    log_scale=False,
    eps=1e-12,
):
    plt.figure(figsize=(10, 8))
    if log_scale:
        matrix_to_plot = np.abs(np.array(matrix, dtype=float))
        matrix_to_plot[matrix_to_plot <= eps] = eps
        norm = LogNorm(vmin=matrix_to_plot.min(), vmax=matrix_to_plot.max())
        sns.heatmap(
            matrix_to_plot,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            norm=norm,
            cbar_kws={"label": "Value (log scale)"},
        )
    else:
        sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, cbar_kws={"label": "Value"})
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_diagonal_bar(matrix, title, filename, output_dir, log_scale=False, eps=1e-12):
    diag = np.diag(matrix).astype(float)
    indices = np.arange(len(diag))
    plt.figure(figsize=(10, 4.5))
    if log_scale:
        diag_plot = np.abs(diag)
        diag_plot[diag_plot <= eps] = eps
        plt.bar(indices, diag_plot)
        plt.yscale("log")
        plt.ylabel("Absolute diagonal value (log scale)")
    else:
        plt.bar(indices, diag)
        plt.ylabel("Diagonal value")
    plt.xlabel("Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def compute_cluster_weights(cluster_assignments, target_ratios):
    """
    Compute frame weights to achieve target cluster ratios.

    Args:
        cluster_assignments (np.ndarray): Cluster assignments (0=open, 1=closed, -1=unclustered)
        target_ratios (dict): Target ratios {'open': 0.4, 'closed': 0.6}

    Returns:
        np.ndarray: Normalized frame weights
    """
    n_frames = len(cluster_assignments)
    frame_weights = np.zeros(n_frames)

    # Count frames in each cluster
    open_mask = cluster_assignments == 0
    closed_mask = cluster_assignments == 1
    unclustered_mask = cluster_assignments == -1

    n_open = np.sum(open_mask)
    n_closed = np.sum(closed_mask)
    n_unclustered = np.sum(unclustered_mask)

    print(f"  Cluster counts - Open: {n_open}, Closed: {n_closed}, Unclustered: {n_unclustered}")

    # Compute weights for each cluster
    target_open = target_ratios["open"]
    target_closed = target_ratios["closed"]

    if n_open > 0:
        frame_weights[open_mask] = target_open / n_open
    if n_closed > 0:
        frame_weights[closed_mask] = target_closed / n_closed

    # Distribute remaining weight to unclustered frames
    if n_unclustered > 0:
        remaining_weight = 1.0 - (target_open + target_closed)
        if remaining_weight > 0:
            frame_weights[unclustered_mask] = remaining_weight / n_unclustered
        else:
            # If target ratios sum to 1, don't weight unclustered frames
            frame_weights[unclustered_mask] = 0.0

    # Normalize weights to sum to 1
    total_weight = np.sum(frame_weights)
    if total_weight > 0:
        frame_weights = frame_weights / total_weight

    return frame_weights


def compute_weighted_covariance(data, weights):
    """
    Compute weighted covariance matrix.

    Args:
        data (np.ndarray): Data matrix (n_variables, n_observations)
        weights (np.ndarray): Weights for each observation (n_observations,)

    Returns:
        np.ndarray: Weighted covariance matrix (n_variables, n_variables)
    """
    # Ensure weights are normalized
    weights = weights / np.sum(weights)

    # Compute weighted mean
    weighted_mean = np.sum(data * weights[np.newaxis, :], axis=1, keepdims=True)

    # Center the data
    centered_data = data - weighted_mean

    # Compute weighted covariance
    weighted_cov = (centered_data * weights[np.newaxis, :]) @ centered_data.T

    return weighted_cov


DEFAULT_ALPHAS = (0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, .001, .01, .05, .1, .25, .5, .75, 1)


def construct_covariance(covariance, weights, correction="population", target="none", alpha=0., ridge=1e-6):
    """Correct, shrink, then add ridge; never diagonalise the precision matrix."""
    if correction not in ("population", "weighted_sample") or target not in ("none", "identity", "diagonal"):
        raise ValueError("Unknown covariance construction")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("alpha must lie in [0, 1] and ridge must be finite and nonnegative")
    if target == "none" and alpha != 0:
        raise ValueError("Nonzero alpha requires a shrinkage target")
    covariance = np.array(covariance, dtype=np.float64, copy=True)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.isfinite(covariance).all():
        raise ValueError("Covariance must be a finite square matrix")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be finite nonnegative probabilities with positive total")
    weights = weights / weights.sum()
    if correction == "weighted_sample":
        denominator = 1 - np.sum(weights**2)
        if denominator <= np.finfo(float).eps:
            raise ValueError("Sample correction requires more than one effective observation")
        covariance /= denominator
    if target == "identity":
        destination = np.eye(len(covariance)) * np.trace(covariance) / len(covariance)
    else:
        destination = np.diag(np.diag(covariance))
    if alpha:
        covariance = (1-alpha)*covariance + alpha*destination
    covariance += np.diag(np.full(len(covariance), ridge))
    return covariance


def ridge_thresholds(covariance, condition_limit=1e8):
    """Numerical PD and conditioning thresholds, not an exact-arithmetic minimum."""
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and > 1")
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    low, high = float(eigenvalues[0]), float(eigenvalues[-1])
    if high <= 0 or not np.isfinite(eigenvalues).all():
        raise ValueError("Ridge thresholds require a covariance with positive spectral scale")
    eps = len(covariance)*np.finfo(float).eps
    pd_bound = max(0., (eps*high-low)/(1-eps))
    stable_bound = max(pd_bound, (high-condition_limit*low)/(condition_limit-1), 0.)
    identity = np.eye(len(covariance))

    def accepted(ridge, stable):
        # Adding a scalar identity shifts every eigenvalue exactly in real arithmetic.
        if low+ridge <= eps*(high+ridge):
            return False
        if stable and (high+ridge)/(low+ridge) > condition_limit:
            return False
        try:
            np.linalg.cholesky(covariance+ridge*identity)
        except np.linalg.LinAlgError:
            return False
        return True

    def refine(bound, stable):
        if accepted(0., stable):
            return 0.
        upper = max(np.nextafter(bound, np.inf), np.finfo(float).eps*high)
        for _ in range(60):
            if accepted(upper, stable):
                break
            upper *= 2
        else:
            raise ValueError("Could not bracket a valid ridge")
        lower = 0.
        for _ in range(45):
            midpoint = (lower+upper)/2
            if accepted(midpoint, stable):
                upper = midpoint
            else:
                lower = midpoint
        return upper

    return {"ridge_pd": refine(pd_bound, False), "ridge_stable": refine(stable_bound, True),
            "eigen_min": low, "eigen_max": high}


def shape_metrics(candidate, reference):
    """Shape-only profile comparisons; constant profiles have undefined correlations."""
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    if candidate.sum() <= 0 or reference.sum() <= 0:
        return {"pearson": np.nan, "spearman": np.nan, "profile_distance": np.nan}
    a, b = candidate/candidate.sum(), reference/reference.sum()
    def correlation(x, y):
        if np.ptp(x) <= 32*np.finfo(float).eps or np.ptp(y) <= 32*np.finfo(float).eps:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    return {"pearson": correlation(a, b), "spearman": correlation(rankdata(a), rankdata(b)),
            "profile_distance": float(np.linalg.norm(a-b)/np.linalg.norm(b))}


def matrix_shape_metrics(candidate, reference, active, permutations=999, seed=0):
    """Mantel association plus distances that detect magnitude changes in correlation."""
    from jaxent.src.analysis.covariance_comparison import mantel_test
    a, b = candidate[np.ix_(active, active)], reference[np.ix_(active, active)]
    # Normalise explicitly before Mantel: the shared helper's variance floor would
    # distort small but positive uptake variances and break scalar invariance.
    ca = a / np.sqrt(np.outer(np.diag(a), np.diag(a)))
    cb = b / np.sqrt(np.outer(np.diag(b), np.diag(b)))
    upper = np.triu_indices(len(a), 1)
    if len(upper[0]) < 3 or np.ptp(ca[upper]) <= 32*np.finfo(float).eps or np.ptp(cb[upper]) <= 32*np.finfo(float).eps:
        r, p = np.nan, np.nan
    else:
        r, p = mantel_test(ca, cb, permutations=permutations, seed=seed)
    return {"mantel_r": r, "mantel_p": p,
            "correlation_distance": float(np.linalg.norm(ca-cb)/np.linalg.norm(cb)),
            "normalized_covariance_distance": float(np.linalg.norm(candidate/np.trace(candidate)-reference/np.trace(reference))/np.linalg.norm(reference/np.trace(reference))),
            "trace_ratio": float(np.trace(candidate)/np.trace(reference))}


def run_diagnostics(raw, weights, log_pf, output_dir, ensemble, inputs, alphas=DEFAULT_ALPHAS,
                    permutations=999, seed=0, condition_limit=1e8):
    """Disposable experiment; all numerical coordinates retained for ridge checks."""
    output = Path(output_dir)/"diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    raw = (np.asarray(raw, float)+np.asarray(raw, float).T)/2
    reference = np.diag(compute_weighted_covariance(np.asarray(log_pf, float), weights))
    active = np.flatnonzero(np.diag(raw) > len(raw)*np.finfo(float).eps*np.max(np.diag(raw)))
    baseline = raw + 1e-6*np.eye(len(raw))
    matrix_rows, profile_rows, correction_rows = [], [], []
    matrices = {"raw_population": raw, "baseline": baseline, "log_pf_marginal": reference,
                "active_indices": active, "frame_weights": weights}
    profiles = {}
    # The same fixed reference and mask are used throughout; no PF shrinkage is applied.
    for target in ("identity", "diagonal"):
        for alpha in alphas:
            print(f"  {ensemble}: {target}, alpha={alpha:g}", flush=True)
            pair = {}
            for correction in ("population", "weighted_sample"):
                shrunk = construct_covariance(raw, weights, correction, target, alpha, ridge=0)
                thresholds = ridge_thresholds(shrunk, condition_limit)
                tag = f"{target}__{alpha:g}__{correction}"
                matrices[tag+"__unregularized"] = shrunk
                stages = {"raw": shrunk, "fixed_ridge": shrunk+1e-6*np.eye(len(raw)),
                          "stable_ridge": shrunk+thresholds["ridge_stable"]*np.eye(len(raw))}
                pair[correction] = stages
                for stage in ("fixed_ridge", "stable_ridge"):
                    covariance = stages[stage]
                    chol = np.linalg.cholesky(covariance)
                    precision = cho_solve((chol, True), np.eye(len(raw)))
                    if not np.isfinite(precision).all():
                        raise ValueError("Nonfinite inverse in diagnostics")
                    matrices[tag+"__"+stage] = covariance
                    matrices[tag+"__"+stage+"__precision"] = precision
                    eigen = np.linalg.eigvalsh(covariance)
                    info = {"ensemble": ensemble, "target": target, "alpha": alpha,
                            "correction": correction, "stage": stage, **thresholds,
                            "condition": float(eigen[-1]/eigen[0]),
                            "inverse_residual": float(np.linalg.norm(covariance@precision-np.eye(len(raw)), ord=np.inf))}
                    # Permutation inference is recorded for fixed-ridge matrix comparisons.
                    metrics = matrix_shape_metrics(covariance, baseline, active, permutations if stage=="fixed_ridge" else 0, seed)
                    if stage != "fixed_ridge":
                        metrics["mantel_p"] = np.nan
                    matrix_rows.append({**info, **metrics})
                    for name, values in (("marginal_uptake_variance", np.diag(covariance)),
                                         ("precision_weight", np.diag(precision)),
                                         ("conditional_uptake_variance", 1/np.diag(precision))):
                        profile_rows.append({**info, "profile": name, **shape_metrics(values, reference)})
                        profiles[(target, alpha, correction, stage, name)] = values
            for stage in ("raw", "fixed_ridge", "stable_ridge"):
                correction_rows.append({"ensemble": ensemble, "target": target, "alpha": alpha, "stage": stage,
                    **matrix_shape_metrics(pair["weighted_sample"][stage], pair["population"][stage], active, permutations, seed)})
    matrix_frame, profile_frame = pd.DataFrame(matrix_rows), pd.DataFrame(profile_rows)
    matrix_frame.to_csv(output/"matrix_metrics.csv", index=False)
    profile_frame.to_csv(output/"profile_metrics.csv", index=False)
    pd.DataFrame(correction_rows).to_csv(output/"sample_correction_metrics.csv", index=False)
    profile_table = {"coordinate_index": np.arange(len(raw)), "log_pf_marginal": reference}
    for key, values in profiles.items():
        profile_table["__".join(map(str, key))] = values
    pd.DataFrame(profile_table).to_csv(output/"diagonal_profiles.csv", index=False)
    np.savez_compressed(output/"matrices.npz", **matrices)
    manifest = {"ensemble": ensemble, "inputs": inputs, "dtype": "float64", "alphas": list(alphas),
                "permutations": permutations, "seed": seed, "condition_limit": condition_limit,
                "correction_factor": float(1/(1-np.sum((weights/weights.sum())**2))),
                "effective_frame_count": float(1/np.sum((weights/weights.sum())**2)),
                "active_indices": active.tolist(), "excluded_indices": np.setdiff1d(np.arange(len(raw)), active).tolist(),
                "reference": "unregularized weighted marginal log-PF variance; same BV parameters and frame weights",
                "ridge_definition": "lambda_min > p * eps64 * lambda_max, Cholesky success; stable also condition <= limit",
                "mantel_interpretation": "association under label permutations, not a test of equality; undefined for constant off-diagonals",
                "unweighted_comparator": "existing numpy.cov ddof=1 unchanged", "fitting_performed": False}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    plot_diagnostics(matrix_frame, profile_frame, profiles, reference, baseline, matrices, active, output, ensemble)
    return matrix_frame, profile_frame


def plot_diagnostics(matrix_frame, profile_frame, profiles, reference, baseline, matrices, active, output, ensemble):
    def save(fig, name):
        fig.suptitle(ensemble)
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output/f"{name}.{extension}", dpi=170)
        plt.close(fig)

    stable = matrix_frame[matrix_frame.stage == "stable_ridge"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (target, correction), group in stable.groupby(["target", "correction"]):
        label = f"{target}, {correction}"
        for axis, column in zip(axes, ("ridge_pd", "ridge_stable")):
            axis.plot(group.alpha, group[column], marker=".", label=label)
            axis.set_xscale("symlog", linthresh=1e-8)
            axis.set_yscale("symlog", linthresh=1e-15)
            axis.set_xlabel("Shrinkage alpha")
            axis.set_ylabel(column.replace("_", " "))
            axis.axhline(1e-6, color="gray", linestyle=":", label="current ridge" if label=="diagonal, population" else None)
    axes[0].legend(fontsize=7)
    save(fig, "ridge_thresholds")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for (target, correction), group in matrix_frame[matrix_frame.stage=="fixed_ridge"].groupby(["target", "correction"]):
        for axis, column in zip(axes, ("condition", "mantel_r", "correlation_distance")):
            axis.plot(group.alpha, group[column], marker=".", label=f"{target}, {correction}")
            axis.set_xscale("symlog", linthresh=1e-8)
            axis.set_xlabel("Shrinkage alpha")
            axis.set_ylabel(column.replace("_", " "))
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    save(fig, "conditioning_and_shape")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for column, name in enumerate(("marginal_uptake_variance", "precision_weight", "conditional_uptake_variance")):
        data = profile_frame[(profile_frame.profile==name)&(profile_frame.stage=="stable_ridge")]
        for row, metric in enumerate(("pearson", "profile_distance")):
            axis = axes[row, column]
            for (target, correction), group in data.groupby(["target", "correction"]):
                axis.plot(group.alpha, group[metric], marker=".", label=f"{target}, {correction}")
            axis.set_xscale("symlog", linthresh=1e-8)
            axis.set_xlabel("Shrinkage alpha")
            axis.set_title(name.replace("_", " "), fontsize=10)
            axis.set_ylabel("Pearson r vs log-PF variance" if row==0 else "Normalised profile distance")
            if row==0:
                axis.set_ylim(-1.05, 1.05)
    axes[0, 0].legend(fontsize=7)
    save(fig, "pf_diagonal_shape")
    correction_frame = pd.read_csv(output/"sample_correction_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for (target, stage), group in correction_frame.groupby(["target", "stage"]):
        for axis, metric in zip(axes, ("mantel_r", "correlation_distance", "normalized_covariance_distance")):
            axis.plot(group.alpha, group[metric], marker=".", label=f"{target}, {stage}")
            axis.set_xscale("symlog", linthresh=1e-8)
            axis.set_xlabel("Shrinkage alpha")
            axis.set_title(metric.replace("_", " "), fontsize=10)
    axes[0].set_ylim(.99, 1.001)
    axes[1].set_yscale("symlog", linthresh=1e-15)
    axes[2].set_yscale("symlog", linthresh=1e-15)
    axes[0].legend(fontsize=7)
    save(fig, "sample_correction_shape")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, name in zip(axes, ("marginal_uptake_variance", "precision_weight", "conditional_uptake_variance")):
        axis.plot(reference/reference.sum(), color="black", linewidth=1.5, label="marginal log-PF variance")
        for target, alpha in (("identity", 0.), ("identity", .05), ("diagonal", .05), ("diagonal", 1.)):
            key = (target, alpha, "population", "fixed_ridge", name)
            if key in profiles:
                values = profiles[key]
                axis.plot(values/values.sum(), alpha=.7, linewidth=.8, label=f"{target}, alpha={alpha:g}")
        axis.set_ylabel("Profile / sum")
        axis.set_title(name.replace("_", " "))
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Residue feature index")
    save(fig, "diagonal_profile_overlays")
    from jaxent.src.analysis.covariance_comparison import to_correlation
    original = to_correlation(baseline[np.ix_(active, active)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, target in zip(axes, ("identity", "diagonal")):
        key = f"{target}__0.05__population__fixed_ridge"
        if key in matrices:
            delta = to_correlation(matrices[key][np.ix_(active, active)])-original
            limit = max(float(np.max(np.abs(delta))), 1e-12)
            plot = axis.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit)
            fig.colorbar(plot, ax=axis, label="Change in correlation")
        axis.set_title(f"{target}, alpha=0.05")
    save(fig, "correlation_differences")


def main():
    parser = argparse.ArgumentParser(
        description="Compute weighted Sigma covariance matrices from clustering and ensemble predictions."
    )

    script_dir = os.path.dirname(__file__)

    # Default paths - matching existing scripts
    default_clustering_dir = os.path.join(script_dir, "../../data/_clustering_results")
    default_features_dir = os.path.join(script_dir, "_featurise")
    default_output_dir = os.path.join(script_dir, "_covariance_matrices_sigma")

    parser.add_argument(
        "--clustering_dir",
        type=str,
        default=default_clustering_dir,
        help="Directory containing clustering results.",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=default_features_dir,
        help="Directory containing featurized data.",
    )
    parser.add_argument(
        "--ensemble_name",
        type=str,
        default="ISO_BI",
        help="Name of the ensemble to use (default: ISO_BI).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory to save the output weighted Sigma matrices and plots. Defaults to '_covariance_matrices'.",
    )

    parser.add_argument("--sample_size_correction", choices=("population", "weighted_sample"), default="population")
    parser.add_argument("--shrinkage_target", choices=("none", "identity", "diagonal"), default="none")
    parser.add_argument("--shrinkage_alpha", type=float, default=0.)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--diagnostics", action="store_true", help="Run disposable construction comparisons in output_dir/diagnostics.")
    parser.add_argument("--diagnostic_alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition_limit", type=float, default=1e8)
    args = parser.parse_args()
    if args.diagnostics and (not any(arg == "--output_dir" or arg.startswith("--output_dir=") for arg in sys.argv[1:])
                             or Path(args.output_dir).resolve() == Path(default_output_dir).resolve()):
        parser.error("Diagnostics require an explicit output_dir separate from the existing covariance artifacts")
    if args.permutations < 0 or not np.isfinite(args.condition_limit) or args.condition_limit <= 1:
        parser.error("permutations must be nonnegative and condition_limit finite and > 1")
    try:
        construct_covariance(np.eye(2), np.ones(2), args.sample_size_correction,
                             args.shrinkage_target, args.shrinkage_alpha, args.ridge)
        if any(not np.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in args.diagnostic_alphas):
            raise ValueError("Diagnostic alphas must lie in [0, 1]")
    except ValueError as error:
        parser.error(str(error))

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("WEIGHTED SIGMA COMPUTATION")
    print("=" * 80)

    # --- Load Clustering Results ---
    print(f"\n--- Loading Clustering Results for {args.ensemble_name} ---")
    cluster_file = os.path.join(
        args.clustering_dir, f"cluster_assignments_{args.ensemble_name}.csv"
    )

    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Clustering file not found: {cluster_file}")

    cluster_df = pd.read_csv(cluster_file)
    cluster_assignments = cluster_df["cluster_assignment"].values
    n_frames = len(cluster_assignments)

    print(f"  Loaded {n_frames} cluster assignments from {cluster_file}")

    # --- Compute Cluster Weights ---
    print("\n--- Computing Cluster-Based Weights ---")
    target_ratios = {"open": 0.4, "closed": 0.6}
    print(
        f"  Target ratios - Open: {target_ratios['open']:.1%}, Closed: {target_ratios['closed']:.1%}"
    )

    frame_weights = compute_cluster_weights(cluster_assignments, target_ratios)

    # Verify achieved ratios
    open_mask = cluster_assignments == 0
    closed_mask = cluster_assignments == 1
    achieved_open = np.sum(frame_weights[open_mask])
    achieved_closed = np.sum(frame_weights[closed_mask])

    print(f"  Achieved ratios - Open: {achieved_open:.1%}, Closed: {achieved_closed:.1%}")

    # Save weights
    weights_df = pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "cluster_assignment": cluster_assignments,
            "frame_weight": frame_weights,
        }
    )
    weights_path = os.path.join(args.output_dir, f"{args.ensemble_name}_frame_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    print(f"  Saved frame weights to: {weights_path}")

    # Plot weight distribution
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(n_frames), frame_weights, width=1.0, edgecolor="none")
    plt.xlabel("Frame")
    plt.ylabel("Weight")
    plt.title(
        f"{args.ensemble_name} Frame Weights (Open: {achieved_open:.1%}, Closed: {achieved_closed:.1%})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.ensemble_name}_frame_weights.png"))
    plt.close()

    # --- Load Features and Topology ---
    print(f"\n--- Loading Features and Topology for {args.ensemble_name} ---")
    topology_file = os.path.join(args.features_dir, f"topology_{args.ensemble_name.lower()}.json")
    features_file = os.path.join(args.features_dir, f"features_{args.ensemble_name.lower()}.npz")

    if not os.path.exists(topology_file):
        raise FileNotFoundError(f"Topology file not found: {topology_file}")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")

    topology = PTSerialiser.load_list_from_json(topology_file)
    features = BV_input_features.load(features_file)

    print(f"  Loaded topology with {len(topology)} peptides")
    print(f"  Loaded features with shape: {features.features_shape}")

    # --- Initialize BV_model ---
    print("\n--- Initializing BV_model ---")
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params

    print(f"  BV_model initialized with {len(bv_config.timepoints)} timepoints")

    # --- Predict Uptake for All Frames ---
    print("\n--- Predicting HDX Uptake for All Frames ---")

    # Create dummy Simulation_Parameters
    dummy_sim_params = Simulation_Parameters.from_frame_weights(
        jnp.ones(n_frames) / n_frames,
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([1.0]),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )

    predictions_output_features = run_predict(
        input_features=[features],
        forward_models=[bv_model],
        model_parameters=dummy_sim_params,
        validate=False,
    )

    # Extract predictions: (num_timepoints, num_residues, num_frames)
    y_pred_all_frames = predictions_output_features[0].y_pred()
    print(f"  Prediction shape: {y_pred_all_frames.shape}")

    if y_pred_all_frames.ndim != 3 or y_pred_all_frames.shape[-1] != n_frames:
        raise ValueError("Expected per-frame uptake predictions aligned with cluster assignments")
    # Average across timepoints to get (num_residues, num_frames)
    y_pred_avg_timepoints = np.array(np.mean(y_pred_all_frames, axis=0))
    print(f"  Time-averaged prediction shape: {y_pred_avg_timepoints.shape}")

    # --- Compute Unweighted Covariance ---
    print("\n--- Computing Unweighted Sigma ---")
    Sigma_unweighted = np.cov(y_pred_avg_timepoints) + np.diag(
        np.full(y_pred_avg_timepoints.shape[0], 1e-6)
    )

    print(f"  Unweighted Sigma shape: {Sigma_unweighted.shape}")

    plot_heatmap(
        Sigma_unweighted,
        f"{args.ensemble_name} Unweighted Sigma",
        f"{args.ensemble_name}_Sigma_unweighted_heatmap.png",
        args.output_dir,
    )
    plot_heatmap(
        np.linalg.inv(Sigma_unweighted),
        f"{args.ensemble_name} Inverse Unweighted Sigma",
        f"{args.ensemble_name}_Sigma_unweighted_inv_heatmap.png",
        args.output_dir,
        cmap="magma",
        log_scale=True,
    )
    plot_diagonal_bar(
        Sigma_unweighted,
        f"{args.ensemble_name} Unweighted Sigma Diagonal",
        f"{args.ensemble_name}_Sigma_unweighted_diagonal_bar.png",
        args.output_dir,
    )

    np.savez(
        os.path.join(args.output_dir, f"{args.ensemble_name}_Sigma_unweighted.npz"),
        Sigma=Sigma_unweighted,
        Sigma_inv=np.linalg.inv(Sigma_unweighted),
    )

    # --- Compute Weighted Covariance ---
    print("\n--- Computing Weighted Sigma (Cluster-Based) ---")
    raw_covariance = compute_weighted_covariance(y_pred_avg_timepoints, frame_weights)
    Sigma_weighted = construct_covariance(raw_covariance, frame_weights, args.sample_size_correction,
                                         args.shrinkage_target, args.shrinkage_alpha, args.ridge)
    try:
        weighted_precision = np.linalg.inv(Sigma_weighted)
    except np.linalg.LinAlgError as error:
        raise ValueError("Selected weighted covariance is singular; increase ridge or shrinkage") from error
    if not np.isfinite(weighted_precision).all():
        raise ValueError("Selected covariance has a nonfinite inverse")

    print(f"  Weighted Sigma shape: {Sigma_weighted.shape}")

    plot_heatmap(
        Sigma_weighted,
        f"{args.ensemble_name} Weighted Sigma (40:60 Open:Closed)",
        f"{args.ensemble_name}_Sigma_weighted_heatmap.png",
        args.output_dir,
    )
    plot_heatmap(
        weighted_precision,
        f"{args.ensemble_name} Inverse Weighted Sigma",
        f"{args.ensemble_name}_Sigma_weighted_inv_heatmap.png",
        args.output_dir,
        cmap="magma",
        log_scale=True,
    )
    plot_diagonal_bar(
        Sigma_weighted,
        f"{args.ensemble_name} Weighted Sigma Diagonal",
        f"{args.ensemble_name}_Sigma_weighted_diagonal_bar.png",
        args.output_dir,
    )

    np.savez(
        os.path.join(args.output_dir, f"{args.ensemble_name}_Sigma_weighted.npz"),
        Sigma=Sigma_weighted,
        Sigma_inv=weighted_precision,
        frame_weights=frame_weights,
        cluster_assignments=cluster_assignments,
        target_ratios=target_ratios,
        achieved_ratios={"open": achieved_open, "closed": achieved_closed},
        sample_size_correction=args.sample_size_correction,
        shrinkage_target=args.shrinkage_target,
        shrinkage_alpha=args.shrinkage_alpha,
        ridge=args.ridge,
    )

    print(f"  Weighted Sigma computed and saved to: {args.output_dir}")

    # --- Compute Difference Between Weighted and Unweighted ---
    print("\n--- Computing Difference Matrix ---")
    Sigma_diff = Sigma_weighted - Sigma_unweighted

    plot_heatmap(
        Sigma_diff,
        f"{args.ensemble_name} Sigma Difference (Weighted - Unweighted)",
        f"{args.ensemble_name}_Sigma_diff_heatmap.png",
        args.output_dir,
        cmap="RdBu_r",
    )
    plot_diagonal_bar(
        Sigma_diff,
        f"{args.ensemble_name} Sigma Difference Diagonal",
        f"{args.ensemble_name}_Sigma_diff_diagonal_bar.png",
        args.output_dir,
    )

    np.savez(
        os.path.join(args.output_dir, f"{args.ensemble_name}_Sigma_diff.npz"),
        Sigma_diff=Sigma_diff,
    )

    # --- Summary Statistics ---
    print("\n--- Summary Statistics ---")
    print("  Unweighted Sigma:")
    print(f"    Trace: {np.trace(Sigma_unweighted):.6f}")
    print(f"    Frobenius norm: {np.linalg.norm(Sigma_unweighted, 'fro'):.6f}")
    print("  Weighted Sigma:")
    print(f"    Trace: {np.trace(Sigma_weighted):.6f}")
    print(f"    Frobenius norm: {np.linalg.norm(Sigma_weighted, 'fro'):.6f}")
    print("  Difference:")
    print(f"    Max absolute difference: {np.max(np.abs(Sigma_diff)):.6f}")
    print(f"    Mean absolute difference: {np.mean(np.abs(Sigma_diff)):.6f}")

    # Save summary
    summary_data = {
        "ensemble": args.ensemble_name,
        "n_frames": n_frames,
        "n_peptides": y_pred_avg_timepoints.shape[0],
        "target_open_ratio": target_ratios["open"],
        "target_closed_ratio": target_ratios["closed"],
        "achieved_open_ratio": achieved_open,
        "achieved_closed_ratio": achieved_closed,
        "unweighted_trace": np.trace(Sigma_unweighted),
        "weighted_trace": np.trace(Sigma_weighted),
        "unweighted_frobenius": np.linalg.norm(Sigma_unweighted, "fro"),
        "weighted_frobenius": np.linalg.norm(Sigma_weighted, "fro"),
        "max_abs_diff": np.max(np.abs(Sigma_diff)),
        "mean_abs_diff": np.mean(np.abs(Sigma_diff)),
    }

    summary_df = pd.DataFrame([summary_data])
    summary_path = os.path.join(args.output_dir, f"{args.ensemble_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to: {summary_path}")

    if args.diagnostics:
        # Calculate z using exactly the same model parameters and residue/frame order.
        log_pf = (np.asarray(model_parameters.bv_bc, float)*np.asarray(features.heavy_contacts, float)
                  + np.asarray(model_parameters.bv_bh, float)*np.asarray(features.acceptor_contacts, float))
        print("Running disposable diagnostics (float64; marginal log-PF reference)", flush=True)
        run_diagnostics(raw_covariance, frame_weights, log_pf, args.output_dir, args.ensemble_name,
                        {"features": str(Path(features_file).resolve()), "topology": str(Path(topology_file).resolve()),
                         "clusters": str(Path(cluster_file).resolve()),
                         "timepoints": np.asarray(bv_config.timepoints, float).tolist(),
                         "bv_bc": np.asarray(model_parameters.bv_bc, float).tolist(),
                         "bv_bh": np.asarray(model_parameters.bv_bh, float).tolist()},
                        alphas=args.diagnostic_alphas, permutations=args.permutations,
                        seed=args.seed, condition_limit=args.condition_limit)

    print("\n" + "=" * 80)
    print("WEIGHTED SIGMA COMPUTATION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
