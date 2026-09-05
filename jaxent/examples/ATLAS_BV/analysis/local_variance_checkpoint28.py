"""Checkpoint 28: two-sided W1-neighbour variance scaling of cached predictors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.alpha_variance_checkpoint25 import (
    OPENMM_TOTAL_ONLY_DIR,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    density_targets,
    mass_metrics,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import (
    pair_endpoints,
)
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_population_checkpoint24 import (
    load_score_frames,
)
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_graph_checkpoint26 import (
    FIT,
    TUNE,
    TEST,
    finite_spearman,
    fitted_scale,
    global_to_local,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    PAIR_CAP,
    RIDGE_ALPHAS,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_frame_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)

OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint28_local_variance"
METRICS = (
    "work_scale",
    "work_shape",
    "work_density_legacy_zq",
    "pf_l1",
    "pf_l2",
    "pyro_ref2015",
    "openmm_total",
)
K_VALUES = (5, 10, 20, 50)
SHRINKAGES = (0.001, 0.01, 0.1)
MODELS = ("pooled", "two_sided_z", "variance_magnitude", "variance_signed_magnitude")


def nearest(matrix: np.ndarray, k: int) -> np.ndarray:
    raw = np.argpartition(matrix, min(k, len(matrix) - 1), axis=1)[
        :, : min(k + 1, len(matrix))
    ]
    return np.asarray(
        [[j for j in row if j != i][:k] for i, row in enumerate(raw)], dtype=int
    )


def local_statistics(
    values: np.ndarray,
    neighbours: np.ndarray,
    reference: np.ndarray | None,
    shrinkage: float,
):
    values = np.atleast_2d(values)
    gathered = values[:, neighbours]
    mean = gathered.mean(axis=2)
    variance = gathered.var(axis=2)
    if reference is None:
        reference = np.median(np.where(variance > 0, variance, np.nan), axis=1)
        reference = np.where(np.isfinite(reference) & (reference > 0), reference, 1.0)
    variance = variance + shrinkage * reference[:, None]
    return mean, variance, reference


def pair_local_features(values, mean, variance, left, right, metric_kind):
    values = np.atleast_2d(values)
    delta = values[:, left] - values[:, right]
    pooled_variance = 0.5 * (variance[:, left] + variance[:, right])
    if metric_kind == "l2":
        pooled = np.sqrt(np.mean(delta * delta / pooled_variance, axis=0))
    else:
        pooled = np.mean(np.abs(delta) / np.sqrt(pooled_variance), axis=0)
    zl = (values[:, left] - mean[:, left]) / np.sqrt(variance[:, left])
    zr = (values[:, right] - mean[:, right]) / np.sqrt(variance[:, right])
    zdist = (
        np.sqrt(np.mean((zl - zr) ** 2, axis=0))
        if metric_kind == "l2"
        else np.mean(np.abs(zl - zr), axis=0)
    )
    volume = 0.5 * np.mean(np.log(variance), axis=0)
    contrast = volume[right] - volume[left]
    magnitude = np.abs(contrast)
    signed_magnitude = np.sign(contrast) * pooled
    potential = np.mean(values, axis=0)
    direct_signed = potential[left] - potential[right]
    return {
        "pooled": pooled,
        "two_sided_z": zdist,
        "variance_magnitude": magnitude,
        "variance_signed_magnitude": signed_magnitude,
        "variance_contrast": contrast,
        "direct_signed": direct_signed,
    }


def load_openmm(system: str, config: dict) -> np.ndarray:
    parts = []
    for replica in (1, 2, 3):
        with np.load(
            OPENMM_TOTAL_ONLY_DIR / system / f"{system}_R{replica}.energies.npz"
        ) as z:
            frames = z["frame"]
            keep = (
                frames * config["analysis"]["frame_interval_ns"]
                > config["analysis"]["equilibration_ns"]
            )
            parts.append(np.asarray(z["energy_total_kj_mol"])[keep])
    return np.concatenate(parts)


def representations(data, system, config):
    thermo = thermodynamic_frame_features(data["z"])
    return {
        "work_scale": (thermo["work_scale"], "l1"),
        "work_shape": (thermo["work_shape"], "l1"),
        "work_density_legacy_zq": (thermo["work_density_legacy_zq"], "l1"),
        "pf_l1": (data["z"], "l1"),
        "pf_l2": (data["z"], "l2"),
        "pyro_ref2015": (load_score_frames(system, config)["ref2015__total"], "l1"),
        "openmm_total": (load_openmm(system, config), "l1"),
    }


def direct_distance(values, left, right, kind):
    values = np.atleast_2d(values)
    delta = values[:, left] - values[:, right]
    return (
        np.sqrt(np.mean(delta * delta, axis=0))
        if kind == "l2"
        else np.mean(np.abs(delta), axis=0)
    )


def fit_ridge(xa, ya, xb, yb, positive):
    scale = np.sqrt(np.mean(xa * xa, axis=0))
    scale[scale <= np.finfo(float).eps] = 1.0
    scored = []
    for penalty in RIDGE_ALPHAS:
        model = Ridge(alpha=penalty, fit_intercept=False, positive=positive).fit(
            xa / scale, ya
        )
        scored.append(
            (
                np.mean(np.abs(yb - model.predict(xb / scale))),
                -penalty,
                model,
                scale,
                penalty,
            )
        )
    return min(scored, key=lambda x: (x[0], x[1]))


def fitted_signed_scale(feature: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.dot(feature, feature))
    return float(np.dot(feature, target) / denominator) if denominator > 0 else 0.0


def evaluate_system(row, config, edges):
    started = time.perf_counter()
    data = system_data(row, config)
    system = data["system"]
    signed, bandwidth = density_targets(data, FIT, 10)
    take = {r: sampled_indices(system, r, len(v), PAIR_CAP) for r, v in signed.items()}
    targets = {
        "signed": {r: signed[r][take[r]] for r in (1, 2, 3)},
        "magnitude": {r: np.abs(signed[r][take[r]]) for r in (1, 2, 3)},
    }
    endpoints = {}
    local_endpoints = {}
    for r, pairs in data["pairs"].items():
        endpoints[r] = pair_endpoints(pairs, take[r])
        gidx = data["matrices"][r][0]
        local_endpoints[r] = tuple(global_to_local(gidx, x) for x in endpoints[r])
    predictions = []
    fits = []
    for metric, (values, kind) in representations(data, system, config).items():
        direct = {r: direct_distance(values, *endpoints[r], kind) for r in (1, 2, 3)}
        alpha = fitted_scale(direct[FIT], targets["magnitude"][FIT])
        predictions.append((metric, "direct", "magnitude", alpha * direct[TEST]))
        fits.append(
            (
                metric,
                "direct",
                "magnitude",
                alpha,
                np.nan,
                np.nan,
                np.nan,
                float(np.var(direct[TEST])),
            )
        )
        candidates = []
        for k in K_VALUES:
            stats = {}
            reference = None
            for r, (gidx, matrix) in data["matrices"].items():
                local = (
                    values[gidx] if np.asarray(values).ndim == 1 else values[:, gidx]
                )
                stats[r] = (local, nearest(matrix, k))
            for shrink in SHRINKAGES:
                features = {}
                for r in (FIT, TUNE, TEST):
                    local, nn = stats[r]
                    mean, var, reference = local_statistics(
                        local, nn, reference if r != FIT else None, shrink
                    )
                    features[r] = pair_local_features(
                        local, mean, var, *local_endpoints[r], kind
                    )
                candidates.append((k, shrink, features))
        for model_name in MODELS[:3]:
            scored = []
            for k, shrink, features in candidates:
                a = fitted_scale(features[FIT][model_name], targets["magnitude"][FIT])
                loss = np.mean(
                    np.abs(targets["magnitude"][TUNE] - a * features[TUNE][model_name])
                )
                scored.append((loss, k, shrink, a, features))
            loss, k, shrink, a, features = min(scored, key=lambda x: (x[0], x[1], x[2]))
            predictions.append(
                (metric, model_name, "magnitude", a * features[TEST][model_name])
            )
            fits.append(
                (
                    metric,
                    model_name,
                    "magnitude",
                    a,
                    k,
                    shrink,
                    loss,
                    float(np.var(features[TEST][model_name])),
                )
            )
        ridge_scored = []
        for k, shrink, features in candidates:
            matrices = {
                r: np.column_stack(
                    [
                        direct[r],
                        features[r]["pooled"],
                        features[r]["two_sided_z"],
                        features[r]["variance_magnitude"],
                    ]
                )
                for r in (1, 2, 3)
            }
            loss, _, model, scale, penalty = fit_ridge(
                matrices[FIT],
                targets["magnitude"][FIT],
                matrices[TUNE],
                targets["magnitude"][TUNE],
                True,
            )
            ridge_scored.append((loss, k, shrink, penalty, model, scale, matrices))
        loss, k, shrink, penalty, model, scale, matrices = min(
            ridge_scored, key=lambda x: (x[0], x[1], x[2], -x[3])
        )
        pred = np.maximum(0, model.predict(matrices[TEST] / scale))
        predictions.append((metric, "ridge", "magnitude", pred))
        fits.append(
            (metric, "ridge", "magnitude", np.nan, k, shrink, loss, float(np.var(pred)))
        )
        signed_scored = []
        for k, shrink, features in candidates:
            x = {
                r: np.column_stack(
                    [
                        features[r]["direct_signed"],
                        features[r]["variance_contrast"],
                        features[r]["variance_signed_magnitude"],
                    ]
                )
                for r in (1, 2, 3)
            }
            loss, _, model, scale, penalty = fit_ridge(
                x[FIT], targets["signed"][FIT], x[TUNE], targets["signed"][TUNE], False
            )
            signed_scored.append((loss, k, shrink, penalty, model, scale, x))
        loss, k, shrink, penalty, model, scale, x = min(
            signed_scored, key=lambda q: (q[0], q[1], q[2], -q[3])
        )
        pred = model.predict(x[TEST] / scale)
        predictions.append((metric, "signed_ridge", "signed", pred))
        fits.append(
            (
                metric,
                "signed_ridge",
                "signed",
                np.nan,
                k,
                shrink,
                loss,
                float(np.var(pred)),
            )
        )
        for model_name in (
            "direct_signed",
            "variance_contrast",
            "variance_signed_magnitude",
        ):
            scored = []
            for k, shrink, features in candidates:
                alpha = fitted_signed_scale(
                    features[FIT][model_name], targets["signed"][FIT]
                )
                loss = np.mean(
                    np.abs(targets["signed"][TUNE] - alpha * features[TUNE][model_name])
                )
                scored.append((loss, k, shrink, alpha, features))
            loss, k, shrink, alpha, features = min(
                scored, key=lambda q: (q[0], q[1], q[2])
            )
            pred = alpha * features[TEST][model_name]
            predictions.append((metric, model_name, "signed", pred))
            fits.append(
                (
                    metric,
                    model_name,
                    "signed",
                    alpha,
                    k,
                    shrink,
                    loss,
                    float(np.var(pred)),
                )
            )
    test_w1 = data["pairs"][TEST].w1.to_numpy()[take[TEST]]
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    result = []
    for metric, model, kind, pred in predictions:
        target = targets[kind][TEST]
        for band in range(6):
            mask = (test_w1 >= edges[band]) & (
                (test_w1 < edges[band + 1]) if band < 5 else True
            )
            if mask.sum() < 30:
                continue
            result.append(
                {
                    "system_id": system,
                    "metric": metric,
                    "model": model,
                    "target_kind": kind,
                    "band": f"q{band}",
                    "pairs": int(mask.sum()),
                    "mae": float(np.mean(np.abs(target[mask] - pred[mask]))),
                    "spearman": finite_spearman(target[mask], pred[mask]),
                    "sign_accuracy": float(
                        np.mean(np.sign(target[mask]) == np.sign(pred[mask]))
                    )
                    if kind == "signed"
                    else np.nan,
                    **mass_metrics(
                        target[mask],
                        pred[mask],
                        targets[kind][FIT],
                        settings["distribution_bins"],
                        settings["distribution_smoothing"],
                    ),
                }
            )
    fit_rows = [
        {
            "system_id": system,
            "metric": m,
            "model": m2,
            "target_kind": t,
            "alpha": a,
            "k": k,
            "shrinkage": s,
            "tune_mae": loss,
            "feature_variance": v,
        }
        for m, m2, t, a, k, s, loss, v in fits
    ]
    return (
        result,
        fit_rows,
        {
            "system_id": system,
            "runtime_seconds": time.perf_counter() - started,
            "bandwidth_angstrom": bandwidth,
        },
    )


def task(x):
    return evaluate_system(*x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()
    pilot_path = (
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint26_pyrosetta_graph/pilot_systems.parquet"
    )
    if not args.full:
        ids = set(pd.read_parquet(pilot_path).query("pilot").system_id)
        rows = [r for r in rows if r["system_id"] in ids]
    destination = OUTPUT / ("full" if args.full else "pilot")
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    with (
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml"
    ).open() as f:
        edges = np.asarray(yaml.safe_load(f)["edges_angstrom"])
    pending = [
        r
        for r in rows
        if args.force
        or not all(
            (parts / f"{r['system_id']}.{n}.parquet").exists()
            for n in ("results", "fits", "runtime")
        )
    ]
    ex = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    evaluated = (
        ex.map(task, [(r, config, edges) for r in pending])
        if ex
        else map(task, [(r, config, edges) for r in pending])
    )
    for i, (row, value) in enumerate(zip(pending, evaluated), 1):
        for name, records in zip(("results", "fits", "runtime"), value):
            atomic_parquet(
                pd.DataFrame(records if isinstance(records, list) else [records]),
                parts / f"{row['system_id']}.{name}.parquet",
            )
        print(f"[{i}/{len(pending)}] {row['system_id']}", flush=True)
    if ex:
        ex.shutdown()
    combined = {}
    for name in ("results", "fits", "runtime"):
        combined[name] = pd.concat(
            [pd.read_parquet(parts / f"{r['system_id']}.{name}.parquet") for r in rows],
            ignore_index=True,
        )
        atomic_parquet(combined[name], destination / f"local_variance_{name}.parquet")
    summary = (
        combined["results"]
        .groupby(["target_kind", "metric", "model", "band"], as_index=False)
        .agg(
            recovery_mean=("distribution_recovery", "mean"),
            recovery_median=("distribution_recovery", "median"),
            recovery_sd=("distribution_recovery", "std"),
            sign_accuracy=("sign_accuracy", "mean"),
            spearman=("spearman", "mean"),
            systems=("system_id", "nunique"),
        )
    )
    atomic_parquet(summary, destination / "local_variance_summary.parquet")
    magnitude = combined["results"].query("target_kind == 'magnitude'")
    balanced = (
        magnitude.groupby(["system_id", "metric", "model"], as_index=False)
        .distribution_recovery.mean()
        .rename(columns={"distribution_recovery": "balanced_recovery"})
    )
    comparison = (
        balanced.groupby(["metric", "model"], as_index=False)
        .balanced_recovery.agg(["mean", "std", "count"])
        .reset_index()
    )
    baseline = comparison.query("metric == 'work_scale' and model == 'direct'").iloc[0]
    comparison["mean_gain_points_vs_work_scale"] = 100 * (
        comparison["mean"] - baseline["mean"]
    )
    comparison["sd_reduction_fraction_vs_work_scale"] = 1 - (
        comparison["std"] / baseline["std"]
    )
    comparison["passes_practical_threshold"] = (
        comparison.mean_gain_points_vs_work_scale >= 2
    ) | (
        (comparison.sd_reduction_fraction_vs_work_scale >= 0.10)
        & (comparison.mean_gain_points_vs_work_scale >= -2)
    )
    atomic_parquet(comparison, destination / "local_variance_success_table.parquet")
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, point in comparison.iterrows():
        ax.scatter(100 * point["std"], 100 * point["mean"], s=28)
        if point["passes_practical_threshold"]:
            ax.annotate(
                f"{point['metric']}:{point['model']}",
                (100 * point["std"], 100 * point["mean"]),
                fontsize=6,
            )
    ax.axhline(100 * baseline["mean"], color="black", linestyle="--", linewidth=1)
    ax.axvline(100 * baseline["std"], color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Between-system SD of balanced recovery (%)",
        ylabel="Balanced mean recovery (%)",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination / "local_variance_mean_sd_pareto.png", dpi=180)
    plt.close(fig)
    mag = summary[summary.target_kind == "magnitude"]
    band_labels = [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)
    for ax, metric in zip(axes.flat, METRICS):
        for model in ("direct", "pooled", "two_sided_z", "variance_magnitude", "ridge"):
            b = mag[(mag.metric == metric) & (mag.model == model)].set_index("band")
            y = np.array([b.recovery_mean.get(f"q{i}", np.nan) for i in range(6)])
            sd = np.array([b.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            ax.plot(range(6), 100 * y, marker="o", label=model)
            ax.fill_between(range(6), 100 * (y - sd), 100 * (y + sd), alpha=0.05)
        ax.set_title(metric)
        ax.set_xticks(range(6), band_labels, fontsize=7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Mean recovery ± system SD (%)")
    axes[1, 0].set_ylabel("Mean recovery ± system SD (%)")
    fig.tight_layout()
    fig.savefig(destination / "local_variance_recovery.png", dpi=180)
    plt.close(fig)
    signed_summary = summary[summary.target_kind == "signed"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)
    signed_models = (
        "direct_signed",
        "variance_contrast",
        "variance_signed_magnitude",
        "signed_ridge",
    )
    for ax, metric in zip(axes.flat, METRICS):
        for model_name in signed_models:
            banded = signed_summary[
                (signed_summary.metric == metric) & (signed_summary.model == model_name)
            ].set_index("band")
            y = np.array([banded.recovery_mean.get(f"q{i}", np.nan) for i in range(6)])
            sd = np.array([banded.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            ax.plot(range(6), 100 * y, marker="o", label=model_name)
            ax.fill_between(range(6), 100 * (y - sd), 100 * (y + sd), alpha=0.05)
        ax.set_title(metric)
        ax.set_xticks(range(6), band_labels, fontsize=7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Signed recovery ± system SD (%)")
    axes[1, 0].set_ylabel("Signed recovery ± system SD (%)")
    fig.tight_layout()
    fig.savefig(destination / "local_variance_signed_recovery.png", dpi=180)
    plt.close(fig)
    atomic_yaml(
        destination / "checkpoint28_report.yaml",
        {
            "systems": len(rows),
            "assignment": "A-fit/B-tune/C-test",
            "metrics": list(METRICS),
            "k_values": list(K_VALUES),
            "shrinkages": list(SHRINKAGES),
            "median_runtime_seconds": float(
                combined["runtime"].runtime_seconds.median()
            ),
        },
    )


if __name__ == "__main__":
    main()
