"""Checkpoint 2: separate common PF support from explicit endpoint extrapolation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import (
    align_to_structure,
    make_fold_pairs,
    pf_pair_distance,
    transform_logpf,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    intraframe_distance_distributions,
    target_bands,
    w1_pair_distance,
)


ARMS = (
    ("raw", "l1"),
    ("raw", "l2"),
    ("frame_centered", "l2"),
    ("raw", "cosine"),
    ("raw", "correlation"),
)


def probability_distribution_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    low: float,
    high: float,
    bins: int,
    smoothing: float,
) -> dict[str, float]:
    """Dimensionless errors between predicted and target probability masses."""
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("distribution-fit support must be finite and increasing")
    edges = np.linspace(low, high, bins + 1)
    # Preserve all probability mass by assigning values beyond the training band
    # limits to the corresponding boundary bin.
    predicted_counts = np.histogram(np.clip(prediction, low, high), bins=edges)[0]
    target_counts = np.histogram(np.clip(target, low, high), bins=edges)[0]
    predicted_mass = predicted_counts.astype(float) + smoothing
    target_mass = target_counts.astype(float) + smoothing
    predicted_mass /= predicted_mass.sum()
    target_mass /= target_mass.sum()
    midpoint = 0.5 * (predicted_mass + target_mass)
    delta = predicted_mass - target_mass
    jsd = 0.5 * (
        np.sum(predicted_mass * np.log(predicted_mass / midpoint))
        + np.sum(target_mass * np.log(target_mass / midpoint))
    )
    kld_target_to_prediction = np.sum(
        target_mass * np.log(target_mass / predicted_mass)
    )
    return {
        "distribution_l1": float(np.sum(np.abs(delta))),
        "distribution_l2": float(np.sqrt(np.sum(delta**2))),
        "distribution_jsd": float(jsd),
        "distribution_sqrt_jsd": float(np.sqrt(jsd)),
        "distribution_kld_target_to_prediction": float(kld_target_to_prediction),
        "distribution_recovery": float(1.0 - np.sqrt(jsd)),
    }


def endpoint_slopes(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float,
    minimum: int,
) -> tuple[float, float]:
    """Non-negative least-squares slopes in the lower and upper x tails."""
    count = max(minimum, int(np.ceil(len(x) * fraction)))
    count = min(count, len(x))
    order = np.argsort(x)

    def slope(indices: np.ndarray) -> float:
        local_x = x[indices]
        local_y = y[indices]
        centered_x = local_x - local_x.mean()
        denominator = float(np.dot(centered_x, centered_x))
        if denominator <= 0:
            return 0.0
        value = float(np.dot(centered_x, local_y - local_y.mean()) / denominator)
        return max(0.0, value) if np.isfinite(value) else 0.0

    return slope(order[:count]), slope(order[-count:])


def boundary_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    tail_fraction: float,
    tail_minimum: int,
) -> dict:
    """Historical clipped and continuous linear-tail isotonic predictions."""
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(train_x, train_y)
    clipped_train = model.predict(train_x)
    clipped_test = model.predict(test_x)
    low_x, high_x = float(train_x.min()), float(train_x.max())
    low_y = float(model.predict([low_x])[0])
    high_y = float(model.predict([high_x])[0])
    low_slope, high_slope = endpoint_slopes(
        train_x, train_y, tail_fraction, tail_minimum
    )
    extrapolated = clipped_test.copy()
    below = test_x < low_x
    above = test_x > high_x
    extrapolated[below] = low_y + low_slope * (test_x[below] - low_x)
    extrapolated[above] = high_y + high_slope * (test_x[above] - high_x)
    extrapolated = np.maximum(0.0, extrapolated)
    return {
        "clipped_train": clipped_train,
        "clipped_test": clipped_test,
        "extrapolated_test": extrapolated,
        "in_pf_support": ~(below | above),
        "below_pf_support": below,
        "above_pf_support": above,
        "low_x": low_x,
        "high_x": high_x,
        "low_y": low_y,
        "high_y": high_y,
        "low_slope": low_slope,
        "high_slope": high_slope,
    }


def residual_intervals(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_prediction: np.ndarray,
    test_x: np.ndarray,
    test_prediction: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Historical same-fit conditional residual intervals used in Checkpoints 1–2."""
    edges = np.unique(np.quantile(train_x, np.linspace(0, 1, bins + 1)))
    low = np.full(len(test_x), np.nan)
    high = np.full(len(test_x), np.nan)
    if len(edges) < 2:
        return low, high
    train_labels = np.clip(np.digitize(train_x, edges[1:-1]), 0, len(edges) - 2)
    test_labels = np.clip(np.digitize(test_x, edges[1:-1]), 0, len(edges) - 2)
    residual = train_y - train_prediction
    for label in range(len(edges) - 1):
        train_mask = train_labels == label
        if np.count_nonzero(train_mask) < 2:
            continue
        qlow, qhigh = np.quantile(residual[train_mask], [0.05, 0.95])
        test_mask = test_labels == label
        low[test_mask] = test_prediction[test_mask] + qlow
        high[test_mask] = test_prediction[test_mask] + qhigh
    return low, high


def metric_rows(
    system: str,
    heldout: int,
    target: str,
    representation: str,
    metric: str,
    train_target: np.ndarray,
    test_target: np.ndarray,
    predictions: dict,
    clipped_interval: tuple[np.ndarray, np.ndarray],
    extrapolated_interval: tuple[np.ndarray, np.ndarray],
    distribution_bins: int,
    distribution_smoothing: float,
) -> list[dict]:
    methods = (
        ("clipped_all", np.ones(len(test_target), dtype=bool), predictions["clipped_test"], clipped_interval),
        ("common_pf_support", predictions["in_pf_support"], predictions["clipped_test"], clipped_interval),
        ("extrapolation_only", ~predictions["in_pf_support"], predictions["extrapolated_test"], extrapolated_interval),
        ("extrapolated_all", np.ones(len(test_target), dtype=bool), predictions["extrapolated_test"], extrapolated_interval),
    )
    rows = []
    null_mae = float(np.mean(np.abs(test_target - np.median(train_target))))
    for method, support_mask, prediction, (lower, upper) in methods:
        for band, band_mask, band_low, band_high in target_bands(target, train_target, test_target):
            mask = support_mask & band_mask & np.isfinite(lower) & np.isfinite(upper)
            if not mask.any():
                continue
            mae = float(np.mean(np.abs(prediction[mask] - test_target[mask])))
            fit_low = band_low if np.isfinite(band_low) else float(train_target.min())
            fit_high = band_high if np.isfinite(band_high) else float(train_target.max())
            if fit_high > fit_low:
                distribution_errors = probability_distribution_errors(
                    prediction[mask],
                    test_target[mask],
                    fit_low,
                    fit_high,
                    distribution_bins,
                    distribution_smoothing,
                )
            else:
                distribution_errors = {
                    "distribution_l1": np.nan,
                    "distribution_l2": np.nan,
                    "distribution_jsd": np.nan,
                    "distribution_sqrt_jsd": np.nan,
                    "distribution_kld_target_to_prediction": np.nan,
                    "distribution_recovery": np.nan,
                }
            coverage = (test_target[mask] >= lower[mask]) & (test_target[mask] <= upper[mask])
            rows.append(
                {
                    "system_id": system,
                    "heldout_replica": heldout,
                    "target": target,
                    "representation": representation,
                    "metric": metric,
                    "method": method,
                    "band": band,
                    "band_low": band_low,
                    "band_high": band_high,
                    "pairs": int(mask.sum()),
                    "mae": mae,
                    **distribution_errors,
                    "skill_vs_train_median": 1.0 - mae / null_mae if null_mae > 0 else np.nan,
                    "interval_90_coverage": float(coverage.mean()),
                    "median_interval_width": float(np.median(upper[mask] - lower[mask])),
                    "below_pf_support_fraction": float(np.mean(predictions["below_pf_support"][band_mask])),
                    "above_pf_support_fraction": float(np.mean(predictions["above_pf_support"][band_mask])),
                    "low_slope": predictions["low_slope"],
                    "high_slope": predictions["high_slope"],
                }
            )
    return rows


def analyse_system(row: dict[str, str], config: dict) -> list[dict]:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    boundary = settings["boundary_audit"]
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms(config["analysis"]["basins"]["atom_selection"]).positions.copy()
    aligned = align_to_structure(coordinates, reference)
    distributions = intraframe_distance_distributions(coordinates)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    rmsd_to_start = np.sqrt(
        np.mean(np.sum((aligned - (reference - reference.mean(axis=0))) ** 2, axis=2), axis=1)
    )
    g = z.sum(axis=0)
    theiler = {}
    for replica in (1, 2, 3):
        mask = replicas == replica
        theiler[replica] = max(
            integrated_autocorrelation_frames(rmsd_to_start[mask]),
            integrated_autocorrelation_frames(g[mask]),
        )
    rows = []
    for heldout in (1, 2, 3):
        train_frames = np.flatnonzero(replicas != heldout)
        train_pairs, test_pairs = make_fold_pairs(
            aligned, replicas, frames, heldout, theiler,
            settings["train_pairs"], settings["test_pairs"], config["analysis"]["seed"]
        )
        train_w1 = w1_pair_distance(
            distributions, train_pairs.left, train_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        test_w1 = w1_pair_distance(
            distributions, test_pairs.left, test_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        targets = {"rmsd": (train_pairs.rmsd, test_pairs.rmsd), "w1": (train_w1, test_w1)}
        for representation, metric in ARMS:
            transformed, _ = transform_logpf(z, train_frames, representation, settings["sigma_floor"])
            train_pf = pf_pair_distance(
                transformed, train_pairs.left, train_pairs.right, metric, settings["distance_chunk_size"]
            )
            test_pf = pf_pair_distance(
                transformed, test_pairs.left, test_pairs.right, metric, settings["distance_chunk_size"]
            )
            for target, (train_target, test_target) in targets.items():
                predictions = boundary_predictions(
                    train_pf, train_target, test_pf,
                    boundary["tail_fraction"], boundary["tail_minimum_pairs"],
                )
                clipped_interval = residual_intervals(
                    train_pf, train_target, predictions["clipped_train"], test_pf,
                    predictions["clipped_test"], settings["interval_bins"],
                )
                extrapolated_interval = residual_intervals(
                    train_pf, train_target, predictions["clipped_train"], test_pf,
                    predictions["extrapolated_test"], settings["interval_bins"],
                )
                rows.extend(
                    metric_rows(
                        system, heldout, target, representation, metric,
                        train_target, test_target,
                        predictions, clipped_interval, extrapolated_interval,
                        boundary["distribution_bins"],
                        boundary["distribution_smoothing"],
                    )
                )
    return rows


def aggregate(results: pd.DataFrame) -> dict:
    summary = []
    for keys, group in results.groupby(
        ["target", "representation", "metric", "method", "band"]
    ):
        target, representation, metric, method, band = keys
        summary.append(
            {
                "target": target,
                "representation": representation,
                "metric": metric,
                "method": method,
                "band": band,
                "system_folds": len(group),
                "median_pairs": float(group.pairs.median()),
                "median_mae": float(group.mae.median()),
                "median_distribution_l1": float(group.distribution_l1.median()),
                "median_distribution_l2": float(group.distribution_l2.median()),
                "median_distribution_jsd": float(group.distribution_jsd.median()),
                "median_distribution_sqrt_jsd": float(
                    group.distribution_sqrt_jsd.median()
                ),
                "median_distribution_kld_target_to_prediction": float(
                    group.distribution_kld_target_to_prediction.median()
                ),
                "median_distribution_recovery": float(
                    group.distribution_recovery.median()
                ),
                "median_skill_vs_train_median": float(group.skill_vs_train_median.median()),
                "median_interval_90_coverage": float(group.interval_90_coverage.median()),
                "median_interval_width": float(group.median_interval_width.median()),
                "median_below_pf_support_fraction": float(group.below_pf_support_fraction.median()),
                "median_above_pf_support_fraction": float(group.above_pf_support_fraction.median()),
            }
        )
    return {
        "checkpoint": 2,
        "status": "measurement_complete",
        "decision": "pause_for_review",
        "systems": int(results.system_id.nunique()),
        "common_support": "test PF distance within training PF-distance range",
        "extrapolation": "continuous non-negative linear endpoint slope from outer training decile",
        "intervals": "historical same-fit residual-decile intervals; replaced at Checkpoint 3",
        "distribution_fit": "dimensionless probability-mass L1/L2/JSD/KLD in training-defined target bands",
        "summary": summary,
    }


def write_plots(results: pd.DataFrame, output_dir) -> None:
    import matplotlib.pyplot as plt

    selected = results[
        (results.representation == "frame_centered") & (results.metric == "l2")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, target in zip(axes, ("rmsd", "w1")):
        sub = selected[selected.target == target]
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else sorted(sub.band.unique())
        for method, label in (("clipped_all", "Clipped all"), ("common_pf_support", "Common PF support"), ("extrapolated_all", "Linear-tail all")):
            group = sub[sub.method == method].groupby("band").interval_90_coverage.median()
            axis.plot(order, [100 * group.get(band, np.nan) for band in order], marker="o", label=label)
        axis.axhline(90, color="black", linestyle="--", linewidth=1)
        axis.set_title(target.upper())
        axis.set_ylabel("Nominal 90% interval coverage (%)")
        axis.tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_coverage_comparison.png", dpi=180)
    plt.close(fig)

    tail = selected[
        ((selected.target == "rmsd") & (selected.band == "global"))
        | ((selected.target == "w1") & (selected.band == "q5"))
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, values = [], []
    for (target, method), group in tail.groupby(["target", "method"]):
        labels.append(f"{target.upper()}\n{method}")
        values.append(group.distribution_sqrt_jsd.to_numpy())
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Distribution fit error: sqrt(JSD)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "tail_distribution_fit_methods.png", dpi=180)
    plt.close(fig)

    w1 = selected[selected.target == "w1"]
    order = [f"q{index}" for index in range(6)]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method, label, color in (
        ("clipped_all", "Clipped all", "#4477AA"),
        ("common_pf_support", "Common PF support", "#EE6677"),
        ("extrapolated_all", "Linear-tail all", "#228833"),
    ):
        group = w1[w1.method == method]
        medians = group.groupby("band").distribution_sqrt_jsd.median().reindex(order)
        lower = group.groupby("band").distribution_sqrt_jsd.quantile(0.25).reindex(order)
        upper = group.groupby("band").distribution_sqrt_jsd.quantile(0.75).reindex(order)
        x = np.arange(len(order))
        ax.plot(x, medians, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(x, lower, upper, color=color, alpha=0.10)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_xlabel("Held-out pairwise-coordinate W1 band (training quantiles)")
    ax.set_ylabel("Distribution fit error: sqrt(JSD)")
    ax.set_title("Distribution fit error across structural W1 bands")
    ax.text(
        0.02,
        0.97,
        "Lines: median across system × replica folds\nBands: interquartile range",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.79))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "w1_fit_error_across_bands.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fit_metrics = (
        ("distribution_sqrt_jsd", "sqrt(JSD)"),
        ("distribution_l1", "Probability-mass L1"),
        ("distribution_l2", "Probability-mass L2"),
        ("distribution_kld_target_to_prediction", "KLD(target || predicted)"),
    )
    clipped = w1[w1.method == "clipped_all"]
    for axis, (column, label) in zip(axes.flat, fit_metrics):
        grouped = clipped.groupby("band")[column]
        median = grouped.median().reindex(order)
        lower = grouped.quantile(0.25).reindex(order)
        upper = grouped.quantile(0.75).reindex(order)
        x = np.arange(len(order))
        axis.plot(x, median, marker="o", color="#4477AA", linewidth=2)
        axis.fill_between(x, lower, upper, color="#4477AA", alpha=0.16)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, order)
    fig.suptitle("Probability-distribution fit errors across structural W1 bands")
    fig.supxlabel("Held-out pairwise-coordinate W1 band (training quantiles)")
    fig.tight_layout()
    fig.savefig(output_dir / "distribution_fit_metrics_across_w1_bands.png", dpi=180)
    plt.close(fig)

    comparison = results[
        (results.target == "w1")
        & (results.method == "clipped_all")
        & (
            ((results.representation == "frame_centered") & (results.metric == "l2"))
            | (
                (results.representation == "raw")
                & results.metric.isin(["l1", "l2", "cosine", "correlation"])
            )
        )
    ].copy()
    comparison["arm"] = np.select(
        [
            comparison.metric == "l1",
            (comparison.metric == "l2") & (comparison.representation == "raw"),
            (comparison.metric == "l2") & (comparison.representation == "frame_centered"),
            comparison.metric == "cosine",
            comparison.metric == "correlation",
        ],
        ["Absolute-L1", "Raw L2", "Frame-centred L2", "Cosine", "Correlation"],
        default="Unknown",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for arm, color in (
        ("Absolute-L1", "#AA3377"),
        ("Raw L2", "#66CCEE"),
        ("Frame-centred L2", "#4477AA"),
        ("Cosine", "#228833"),
        ("Correlation", "#CCBB44"),
    ):
        group = comparison[comparison.arm == arm].groupby("band").distribution_sqrt_jsd
        median = group.median().reindex(order)
        lower = group.quantile(0.25).reindex(order)
        upper = group.quantile(0.75).reindex(order)
        x = np.arange(len(order))
        ax.plot(x, median, marker="o", linewidth=2, color=color, label=arm)
        ax.fill_between(x, lower, upper, color=color, alpha=0.12)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_xlabel("Held-out pairwise-coordinate W1 band (training quantiles)")
    ax.set_ylabel("Distribution fit error: sqrt(JSD)")
    ax.set_title("PF distance-function comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pf_distance_function_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for arm, color in (
        ("Absolute-L1", "#AA3377"),
        ("Raw L2", "#66CCEE"),
        ("Frame-centred L2", "#4477AA"),
        ("Cosine", "#228833"),
        ("Correlation", "#CCBB44"),
    ):
        group = comparison[comparison.arm == arm].groupby("band").distribution_recovery
        median = 100 * group.median().reindex(order)
        lower = 100 * group.quantile(0.25).reindex(order)
        upper = 100 * group.quantile(0.75).reindex(order)
        x = np.arange(len(order))
        ax.plot(x, median, marker="o", linewidth=2, color=color, label=arm)
        ax.fill_between(x, lower, upper, color=color, alpha=0.12)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_xlabel("Held-out pairwise-coordinate W1 band (training quantiles)")
    ax.set_ylabel("Distribution recovery: 100 × (1 - sqrt(JSD)) (%)")
    ax.set_title("Probability-distribution recovery by PF distance function")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pf_distance_function_recovery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    systems = load_systems()[: args.limit]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in systems}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            rows.extend(future.result())
            print(f"[{index}/{len(systems)}] {row['system_id']} boundary audit complete", flush=True)
    results = pd.DataFrame(rows)
    output_dir = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint2_boundary"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_dir / "boundary_results.parquet", index=False)
    report = aggregate(results)
    atomic_yaml(output_dir / "report.yaml", report)
    write_plots(results, output_dir)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
