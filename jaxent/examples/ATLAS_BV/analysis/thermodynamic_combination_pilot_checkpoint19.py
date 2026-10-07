"""Checkpoint 19 pilot: test whether thermodynamic work metrics add to Work Scale."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_systems
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK, density_targets, mass_metrics, system_data,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    THERMODYNAMIC_METRICS, thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint19_thermodynamic_combination_pilot"
FIT_REPLICA, TUNE_REPLICA, TEST_REPLICA = 1, 2, 3
PAIR_CAP = 10_000
SEED = 20260902
RIDGE_ALPHAS = (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0, 10_000.0)
COMBINATIONS = {
    "work_scale": ("work_scale",),
    "scale_plus_legacy": ("work_scale", "work_density_legacy_zq"),
    "scale_plus_normalized": ("work_scale", "work_density_normalized_q_over_z"),
    "scale_plus_legacy_plus_shape": ("work_scale", "work_density_legacy_zq", "work_shape"),
    "all_work_metrics": THERMODYNAMIC_METRICS,
}


def stable_rng(system: str, replica: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{system}:{replica}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def sampled_indices(system: str, replica: int, count: int, cap: int = PAIR_CAP) -> np.ndarray:
    if count <= cap:
        return np.arange(count)
    return np.sort(stable_rng(system, replica).choice(count, size=cap, replace=False))


def choose_systems(rows: list[dict[str, str]], fits: pd.DataFrame, count: int = 12) -> list[dict[str, str]]:
    """Select three seeded systems per protein-size quartile without looking at recovery."""
    sizes = fits.groupby("system_id", as_index=False).n_residues.first()
    sizes["size_quartile"] = pd.qcut(sizes.n_residues.rank(method="first"), 4, labels=False)
    rng = np.random.default_rng(SEED); chosen: list[str] = []
    per_quartile = count // 4
    for quartile in range(4):
        candidates = sizes.loc[sizes.size_quartile == quartile, "system_id"].sort_values().to_numpy()
        chosen.extend(rng.choice(candidates, size=min(per_quartile, len(candidates)), replace=False).tolist())
    if len(chosen) < count:
        remaining = sizes.loc[~sizes.system_id.isin(chosen), "system_id"].sort_values().to_numpy()
        chosen.extend(rng.choice(remaining, size=count - len(chosen), replace=False).tolist())
    by_id = {row["system_id"]: row for row in rows}
    return [by_id[system] for system in chosen]


def matrix(features: dict[str, dict[int, np.ndarray]], names: tuple[str, ...], replica: int,
           indices: np.ndarray) -> np.ndarray:
    return np.column_stack([features[name][replica][indices] for name in names])


def tune_nonnegative_ridge(x_fit: np.ndarray, y_fit: np.ndarray, x_tune: np.ndarray,
                           y_tune: np.ndarray) -> tuple[Ridge, float, np.ndarray]:
    rms = np.sqrt(np.mean(np.square(x_fit), axis=0))
    rms = np.where(rms > np.finfo(float).eps, rms, 1.0)
    scored = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=False, positive=True).fit(x_fit / rms, y_fit)
        loss = float(np.mean(np.abs(y_tune - model.predict(x_tune / rms))))
        scored.append((loss, -alpha, alpha, model))
    _, _, alpha, model = min(scored, key=lambda item: (item[0], item[1]))
    return model, float(alpha), rms


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    value = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return value if np.isfinite(value) else 0.0


def evaluate_system(row: dict[str, str], config: dict, w1_edges: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    data = system_data(row, config); system = data["system"]
    features = thermodynamic_pair_features(data["z"], data["pairs"])
    signed_targets, bandwidth = density_targets(data, FIT_REPLICA, PRIMARY_RANK)
    targets = {replica: np.abs(values) for replica, values in signed_targets.items()}
    indices = {replica: sampled_indices(system, replica, len(targets[replica])) for replica in (1, 2, 3)}
    y = {replica: targets[replica][indices[replica]] for replica in (1, 2, 3)}
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    results: list[dict] = []; coefficient_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for combination, names in COMBINATIONS.items():
        x_fit = matrix(features, names, FIT_REPLICA, indices[FIT_REPLICA])
        x_tune = matrix(features, names, TUNE_REPLICA, indices[TUNE_REPLICA])
        x_test = matrix(features, names, TEST_REPLICA, indices[TEST_REPLICA])
        model, alpha, rms = tune_nonnegative_ridge(x_fit, y[FIT_REPLICA], x_tune, y[TUNE_REPLICA])
        prediction = np.maximum(0.0, model.predict(x_test / rms)); predictions[combination] = prediction
        for name, coefficient, scale in zip(names, model.coef_, rms):
            coefficient_rows.append({"system_id": system, "combination": combination, "feature": name,
                                     "ridge_alpha": alpha, "standardized_coefficient": float(coefficient),
                                     "original_scale_coefficient": float(coefficient / scale),
                                     "training_rms": float(scale)})
        test_w1 = data["pairs"][TEST_REPLICA].w1.to_numpy()[indices[TEST_REPLICA]]
        regions = {
            "all": np.ones(len(test_w1), dtype=bool),
            "q4_q5": test_w1 >= w1_edges[4],
        }
        for band in range(len(w1_edges) - 1):
            regions[f"q{band}"] = ((test_w1 >= w1_edges[band])
                                    & (test_w1 < w1_edges[band + 1]))
        regions[f"q{len(w1_edges) - 2}"] = test_w1 >= w1_edges[-2]
        for region, mask in regions.items():
            if not mask.any():
                continue
            distribution = mass_metrics(y[TEST_REPLICA][mask], prediction[mask], y[FIT_REPLICA],
                                        settings["distribution_bins"], settings["distribution_smoothing"])
            results.append({"system_id": system, "combination": combination, "region": region,
                            "pairs": int(mask.sum()), "bandwidth_angstrom": bandwidth,
                            "mae": float(np.mean(np.abs(y[TEST_REPLICA][mask] - prediction[mask]))),
                            "spearman": correlation(y[TEST_REPLICA][mask], prediction[mask]), **distribution})

    scale_names = COMBINATIONS["work_scale"]
    x_fit = matrix(features, scale_names, FIT_REPLICA, indices[FIT_REPLICA])
    x_tune = matrix(features, scale_names, TUNE_REPLICA, indices[TUNE_REPLICA])
    scale_model, _, scale_rms = tune_nonnegative_ridge(x_fit, y[FIT_REPLICA], x_tune, y[TUNE_REPLICA])
    residual = y[TUNE_REPLICA] - scale_model.predict(x_tune / scale_rms)
    diagnostic_rows = []
    for metric in THERMODYNAMIC_METRICS:
        values = features[metric][TUNE_REPLICA][indices[TUNE_REPLICA]]
        diagnostic_rows.append({"system_id": system, "feature": metric,
                                "residual_spearman": correlation(values, residual)})
    for left_index, left in enumerate(THERMODYNAMIC_METRICS):
        for right in THERMODYNAMIC_METRICS[left_index:]:
            a = features[left][FIT_REPLICA][indices[FIT_REPLICA]]
            b = features[right][FIT_REPLICA][indices[FIT_REPLICA]]
            diagnostic_rows.append({"system_id": system, "feature": f"correlation:{left}:{right}",
                                    "residual_spearman": correlation(a, b)})
    return results, coefficient_rows, diagnostic_rows


def make_report(results: pd.DataFrame, coefficients: pd.DataFrame, diagnostics: pd.DataFrame,
                w1_edges: np.ndarray) -> None:
    baseline = results[results.combination == "work_scale"][["system_id", "region", "distribution_recovery", "mae"]]
    baseline = baseline.rename(columns={"distribution_recovery": "baseline_recovery", "mae": "baseline_mae"})
    paired = results.merge(baseline, on=["system_id", "region"], validate="many_to_one")
    paired["recovery_improvement"] = paired.distribution_recovery - paired.baseline_recovery
    paired["mae_improvement"] = paired.baseline_mae - paired.mae
    atomic_parquet(paired, OUTPUT / "pilot_paired_results.parquet")
    summary = (paired.groupby(["combination", "region"], as_index=False)
               .agg(median_recovery=("distribution_recovery", "median"),
                    median_recovery_improvement=("recovery_improvement", "median"),
                    systems_improved=("recovery_improvement", lambda x: int((x >= .03).sum())),
                    median_mae_improvement=("mae_improvement", "median")))
    atomic_parquet(summary, OUTPUT / "pilot_summary.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    order = list(COMBINATIONS)
    overall = paired[paired.region == "all"]
    axes[0, 0].boxplot([100 * overall.loc[overall.combination == name, "distribution_recovery"] for name in order],
                       tick_labels=order, showfliers=False)
    axes[0, 0].set_ylabel("Held-out recovery (%)"); axes[0, 0].set_title("Overall replica-3 recovery")
    tail = paired[paired.region == "q4_q5"]
    axes[0, 1].boxplot([100 * tail.loc[tail.combination == name, "recovery_improvement"] for name in order],
                       tick_labels=order, showfliers=False)
    axes[0, 1].axhline(3, color="black", ls="--", lw=1, label="+3 pp gate")
    axes[0, 1].axhline(0, color="grey", lw=1); axes[0, 1].set_ylabel("Recovery improvement (pp)")
    axes[0, 1].set_title("q4–q5 improvement over Work Scale"); axes[0, 1].legend()
    residual = diagnostics[~diagnostics.feature.str.startswith("correlation:")]
    axes[1, 0].boxplot([residual.loc[residual.feature == name, "residual_spearman"]
                        for name in THERMODYNAMIC_METRICS], tick_labels=THERMODYNAMIC_METRICS,
                       showfliers=False)
    axes[1, 0].axhline(0, color="grey", lw=1); axes[1, 0].set_ylabel("Spearman correlation")
    axes[1, 0].set_title("Metric correlation with replica-2 Work Scale residual")
    matrix_values = np.eye(len(THERMODYNAMIC_METRICS))
    corr = diagnostics[diagnostics.feature.str.startswith("correlation:")]
    for i, left in enumerate(THERMODYNAMIC_METRICS):
        for j, right in enumerate(THERMODYNAMIC_METRICS):
            key = f"correlation:{left}:{right}" if i <= j else f"correlation:{right}:{left}"
            values = corr.loc[corr.feature == key, "residual_spearman"]
            matrix_values[i, j] = values.median() if len(values) else np.nan
    image = axes[1, 1].imshow(matrix_values, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1, 1].set_xticks(range(len(THERMODYNAMIC_METRICS)), THERMODYNAMIC_METRICS, rotation=35, ha="right")
    axes[1, 1].set_yticks(range(len(THERMODYNAMIC_METRICS)), THERMODYNAMIC_METRICS)
    axes[1, 1].set_title("Median training-feature Spearman correlation"); fig.colorbar(image, ax=axes[1, 1])
    for axis in axes.flat[:3]:
        axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Thermodynamic combination pilot: 12 size-stratified systems, 1→2→3 split")
    fig.tight_layout(); fig.savefig(OUTPUT / "thermodynamic_combination_pilot.png", dpi=180); plt.close(fig)

    labels = {
        "work_scale": "Work Scale",
        "scale_plus_legacy": "Scale + legacy density",
        "scale_plus_normalized": "Scale + normalized density",
        "scale_plus_legacy_plus_shape": "Scale + legacy density + Shape",
        "all_work_metrics": "All Work metrics",
    }
    bands = [f"q{index}" for index in range(6)]
    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    for combination in COMBINATIONS:
        block = paired[(paired.combination == combination) & paired.region.isin(bands)]
        medians = block.groupby("region").distribution_recovery.median()
        spread = block.groupby("region").distribution_recovery.std()
        y = 100 * np.array([medians.get(band, np.nan) for band in bands])
        sd = 100 * np.array([spread.get(band, np.nan) for band in bands])
        line = axis.plot(range(6), y, marker="o", lw=2, label=labels[combination])[0]
        axis.fill_between(range(6), np.maximum(0, y - sd), np.minimum(100, y + sd),
                          color=line.get_color(), alpha=.10, linewidth=0)
    distance_labels = [f"q{index}\n{w1_edges[index]:.3f}–{w1_edges[index + 1]:.3f} Å"
                       for index in range(6)]
    axis.set_xticks(range(6), distance_labels)
    axis.set_xlabel("Global frame-pair W1 band")
    axis.set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    axis.set_title("Combination-pilot recovery across structural distance\n"
                   "Replica-3 system medians; shading ±1 system SD")
    axis.grid(alpha=.25); axis.legend(fontsize=9); fig.tight_layout()
    fig.savefig(OUTPUT / "pilot_recovery_across_global_w1.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    for combination in tuple(COMBINATIONS)[1:]:
        block = paired[(paired.combination == combination) & paired.region.isin(bands)]
        grouped = block.groupby("region").recovery_improvement
        median = grouped.median(); lower = grouped.quantile(.25); upper = grouped.quantile(.75)
        y = 100 * np.array([median.get(band, np.nan) for band in bands])
        lo = 100 * np.array([lower.get(band, np.nan) for band in bands])
        hi = 100 * np.array([upper.get(band, np.nan) for band in bands])
        line = axis.plot(range(6), y, marker="o", lw=2, label=labels[combination])[0]
        axis.fill_between(range(6), lo, hi, color=line.get_color(), alpha=.12, linewidth=0)
    axis.axhline(3, color="black", ls="--", lw=1.2, label="+3 pp gate")
    axis.axhline(0, color="grey", lw=1)
    axis.set_xticks(range(6), distance_labels)
    axis.set_xlabel("Global frame-pair W1 band")
    axis.set_ylabel("Paired recovery improvement over Work Scale (percentage points)")
    axis.set_title("Within-system combination benefit across structural distance\n"
                   "Lines: median; shading: interquartile range")
    axis.grid(alpha=.25); axis.legend(fontsize=9); fig.tight_layout()
    fig.savefig(OUTPUT / "pilot_paired_improvement_across_global_w1.png", dpi=180); plt.close(fig)

    gate = summary[(summary.region.isin(["all", "q4_q5"])) & (summary.combination != "work_scale")]
    passed = gate[(gate.median_recovery_improvement >= .03) & (gate.systems_improved >= 8)]
    atomic_yaml(OUTPUT / "checkpoint19_report.yaml", {
        "checkpoint": 19, "status": "pilot_complete", "full_run_gate_passed": bool(len(passed)),
        "passing_combination_regions": passed[["combination", "region"]].to_dict("records"),
        "gate": "median recovery improvement >= 0.03 and >=8/12 systems improve by >=0.03",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=int, default=12)
    args = parser.parse_args()
    config = load_config(); rows = load_systems()
    checkpoint18 = HERE / "outputs/analysis/pairwise_geometry/checkpoint18_thermodynamic_population/thermodynamic_population_fits.parquet"
    fits = pd.read_parquet(checkpoint18)
    selected = choose_systems(rows, fits, args.systems)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = pd.DataFrame([{"system_id": row["system_id"],
                               "n_residues": int(fits.loc[fits.system_id == row["system_id"], "n_residues"].iloc[0])}
                              for row in selected])
    atomic_parquet(selection, OUTPUT / "pilot_systems.parquet")
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml") as handle:
        w1_edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    result_rows = []; coefficient_rows = []; diagnostic_rows = []
    for index, row in enumerate(selected, 1):
        results, coefficients, diagnostics = evaluate_system(row, config, w1_edges)
        result_rows.extend(results); coefficient_rows.extend(coefficients); diagnostic_rows.extend(diagnostics)
        print(f"[{index}/{len(selected)}] {row['system_id']}", flush=True)
    result_frame = pd.DataFrame(result_rows); coefficient_frame = pd.DataFrame(coefficient_rows)
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    atomic_parquet(result_frame, OUTPUT / "pilot_results.parquet")
    atomic_parquet(coefficient_frame, OUTPUT / "pilot_coefficients.parquet")
    atomic_parquet(diagnostic_frame, OUTPUT / "pilot_diagnostics.parquet")
    make_report(result_frame, coefficient_frame, diagnostic_frame, w1_edges)


if __name__ == "__main__":
    main()
