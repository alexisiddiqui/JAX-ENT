"""Fit positive BV coefficients to within-basin Absolute-L1 occupancy profiles."""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from multiprocessing import get_context

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.within_basin_stage1 import (
    binned,
    fixed_width_edges,
    structural_vectors,
)
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.models.HDX.forward import BV_ForwardPass


def absolute_l1_numpy(
    heavy: np.ndarray,
    acceptor: np.ndarray,
    reference: int,
    bc: float,
    bh: float,
) -> np.ndarray:
    z = bc * heavy + bh * acceptor
    return np.abs(z - z[:, reference, None]).sum(axis=0)


def absolute_l1_jax(
    heavy: np.ndarray,
    acceptor: np.ndarray,
    reference: int,
    bc: float,
    bh: float,
) -> jnp.ndarray:
    features = BV_input_features(heavy_contacts=heavy, acceptor_contacts=acceptor)
    parameters = BV_Model_Parameters(bv_bc=jnp.asarray(bc), bv_bh=jnp.asarray(bh))
    z = BV_ForwardPass()(features, parameters).log_Pf
    return jnp.abs(z - z[:, reference, None]).sum(axis=0)


def profiled_scale(q: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = weights / weights.sum()
    qc = q - np.sum(weights * q)
    yc = y - np.sum(weights * y)
    denominator = np.sum(weights * qc * qc)
    scale = 0.0 if denominator <= 0 else max(0.0, -float(np.sum(weights * qc * yc) / denominator))
    loss = float(np.sum(weights * (yc + scale * qc) ** 2))
    return scale, loss


def fit_coefficients(
    delta_heavy: np.ndarray,
    delta_acceptor: np.ndarray,
    labels: np.ndarray,
    common: np.ndarray,
    log_density: np.ndarray,
    counts: np.ndarray,
    grid_points: int = 129,
) -> dict[str, float | bool]:
    if grid_points < 3:
        raise ValueError("grid_points must be at least three")

    def objective(theta: float) -> tuple[float, float]:
        direction = np.cos(theta) * delta_heavy + np.sin(theta) * delta_acceptor
        distances = np.abs(direction).sum(axis=0)
        means, _, _ = binned(distances, labels.astype(float), np.arange(common.size + 1) - 0.5)
        scale, loss = profiled_scale(
            means[common], log_density[common], counts[common].astype(float)
        )
        return loss, scale

    grid = np.linspace(0.0, np.pi / 2, grid_points)
    losses = np.asarray([objective(theta)[0] for theta in grid])
    best = int(np.argmin(losses))
    low = grid[max(0, best - 1)]
    high = grid[min(grid_points - 1, best + 1)]
    if low == high:
        theta = float(grid[best])
    else:
        refined = minimize_scalar(
            lambda value: objective(float(value))[0],
            bounds=(float(low), float(high)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(refined.x)
    loss, scale = objective(theta)
    bc = scale * np.cos(theta)
    bh = scale * np.sin(theta)
    return {
        "bc": float(bc),
        "bh": float(bh),
        "bh_over_bc": float(bh / bc) if bc > 1e-12 else float("inf"),
        "coefficient_norm": float(scale),
        "theta_radians": theta,
        "train_loss": loss,
        "boundary_solution": bool(theta < 1e-6 or theta > np.pi / 2 - 1e-6 or scale == 0),
    }


def density(counts: np.ndarray, widths: np.ndarray) -> np.ndarray:
    result = np.full(len(counts), np.nan)
    populated = counts > 0
    result[populated] = np.log(counts[populated] / (counts.sum() * widths[populated]))
    return result


def system_inputs(row: dict[str, str], config: dict) -> dict | None:
    system = row["system_id"]
    summary = yaml.safe_load(
        (HERE / "outputs" / "analysis" / "basins" / system / "summary.yaml").read_text()
    )
    if summary["usable_basins"] < 1:
        return None
    counts = summary["counts_by_replica"]
    basin = max(summary["usable_labels"], key=lambda label: sum(counts[label].values()))
    assignments = pd.read_parquet(HERE / summary["assignments"])
    mask = assignments["basin"].to_numpy() == basin
    coordinates, replicas, _ = load_ca_coordinates(row, config)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)[:, mask]
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)[:, mask]
    return {
        "system_id": system,
        "vectors": structural_vectors(coordinates[mask]),
        "replicas": replicas[mask],
        "heavy": heavy,
        "acceptor": acceptor,
        "cath_class": row["cath_class"],
        "rmsf_tercile": row["rmsf_tercile"],
    }


def fit_fold(data: dict, heldout: int, config: dict) -> dict:
    settings = config["analysis"]["stage1"]
    stage2 = config["analysis"]["stage2"]
    train = data["replicas"] != heldout
    test = ~train
    pca = PCA(n_components=1, random_state=config["analysis"]["seed"])
    pc_train = pca.fit_transform(data["vectors"][train]).ravel()
    pc_test = pca.transform(data["vectors"][test]).ravel()
    edges = fixed_width_edges(pc_train, settings)
    widths = np.diff(edges)
    _, train_counts, train_labels = binned(pc_train, pc_train, edges)
    _, test_counts, test_labels = binned(pc_test, pc_test, edges)
    common = (train_counts >= settings["min_frames_per_bin"]) & (
        test_counts >= settings["min_frames_per_bin"]
    )
    valid = int(common.sum()) >= settings["min_common_bins_per_fold"]
    if not valid:
        return {"heldout_replica": heldout, "valid": False, "common_bins": int(common.sum())}
    train_indices = np.flatnonzero(train)
    train_log_density = density(train_counts, widths)
    reference_bin = int(np.nanargmax(np.where(common, train_log_density, np.nan)))
    candidates = np.flatnonzero(train_labels == reference_bin)
    center = (edges[reference_bin] + edges[reference_bin + 1]) / 2
    reference = int(train_indices[candidates[np.argmin(np.abs(pc_train[candidates] - center))]])
    delta_h = data["heavy"] - data["heavy"][:, reference, None]
    delta_o = data["acceptor"] - data["acceptor"][:, reference, None]
    fit = fit_coefficients(
        delta_h[:, train],
        delta_o[:, train],
        train_labels,
        common,
        train_log_density,
        train_counts,
        stage2["direction_grid_points"],
    )
    observed = density(test_counts, widths)[common]
    weights = test_counts[common] / test_counts[common].sum()
    observed -= np.sum(weights * observed)
    rt = config["analysis"]["gas_constant_kcal_mol_k"] * config["protocol"]["temperature_k"]

    def evaluate(bc: float, bh: float) -> tuple[float, float, bool]:
        z_test = bc * data["heavy"][:, test] + bh * data["acceptor"][:, test]
        z_reference = bc * data["heavy"][:, reference] + bh * data["acceptor"][:, reference]
        distances = np.abs(z_test - z_reference[:, None]).sum(axis=0)
        means, _, _ = binned(distances, pc_test, edges)
        predicted = -means[common]
        predicted -= np.sum(weights * predicted)
        constant = bool(np.ptp(predicted) <= 1e-12 or np.ptp(observed) <= 1e-12)
        rho = 0.0 if constant else float(spearmanr(predicted, observed).statistic)
        return rho, float(rt * np.sum(weights * np.abs(predicted - observed))), constant

    fitted_rho, fitted_mae, fitted_constant = evaluate(fit["bc"], fit["bh"])
    default_rho, default_mae, default_constant = evaluate(
        config["protocol"]["bv_bc"], config["protocol"]["bv_bh"]
    )
    return {
        "heldout_replica": heldout,
        "valid": True,
        "common_bins": int(common.sum()),
        **fit,
        "fitted_heldout_rho": fitted_rho,
        "default_heldout_rho": default_rho,
        "delta_heldout_rho": fitted_rho - default_rho,
        "fitted_heldout_pmf_mae_kcal_mol": fitted_mae,
        "default_heldout_pmf_mae_kcal_mol": default_mae,
        "delta_heldout_pmf_mae_kcal_mol": fitted_mae - default_mae,
        "fitted_constant_prediction": fitted_constant,
        "default_constant_prediction": default_constant,
    }


def moving_block_indices(indices: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    if block >= len(indices):
        return indices.copy()
    starts = rng.integers(0, len(indices) - block + 1, size=int(np.ceil(len(indices) / block)))
    return np.concatenate([indices[start : start + block] for start in starts])[: len(indices)]


def fit_pooled(data: dict, config: dict, bootstrap_samples: int) -> tuple[dict, list[dict]]:
    stage1 = config["analysis"]["stage1"]
    stage2 = config["analysis"]["stage2"]
    pca = PCA(n_components=1, random_state=config["analysis"]["seed"])
    pc1 = pca.fit_transform(data["vectors"]).ravel()
    edges = fixed_width_edges(pc1, stage1)
    widths = np.diff(edges)
    _, counts, labels = binned(pc1, pc1, edges)
    common = counts >= stage1["min_frames_per_bin"]
    log_density = density(counts, widths)
    reference_bin = int(np.nanargmax(np.where(common, log_density, np.nan)))
    candidates = np.flatnonzero(labels == reference_bin)
    center = (edges[reference_bin] + edges[reference_bin + 1]) / 2
    reference = int(candidates[np.argmin(np.abs(pc1[candidates] - center))])
    delta_h = data["heavy"] - data["heavy"][:, reference, None]
    delta_o = data["acceptor"] - data["acceptor"][:, reference, None]
    fit = fit_coefficients(
        delta_h,
        delta_o,
        labels,
        common,
        log_density,
        counts,
        stage2["direction_grid_points"],
    )
    default_distance = np.abs(
        config["protocol"]["bv_bc"] * delta_h + config["protocol"]["bv_bh"] * delta_o
    ).sum(axis=0)
    block = integrated_autocorrelation_frames(default_distance)
    rng = np.random.default_rng(config["analysis"]["seed"])
    draws = []
    replica_indices = [np.flatnonzero(data["replicas"] == replica) for replica in (1, 2, 3)]
    bootstrap_started = time.monotonic()
    for draw in range(bootstrap_samples):
        sampled = np.concatenate(
            [moving_block_indices(indices, block, rng) for indices in replica_indices]
        )
        sample_counts = np.bincount(labels[sampled], minlength=len(counts))
        sample_common = sample_counts >= stage1["min_frames_per_bin"]
        if sample_common.sum() < stage1["min_common_bins_per_fold"]:
            continue
        sample_density = density(sample_counts, widths)
        fitted = fit_coefficients(
            delta_h[:, sampled],
            delta_o[:, sampled],
            labels[sampled],
            sample_common,
            sample_density,
            sample_counts,
            stage2["direction_grid_points"],
        )
        draws.append({"draw": draw, "bc": fitted["bc"], "bh": fitted["bh"]})
    fit.update(
        {
            "common_bins": int(common.sum()),
            "autocorrelation_frames": block,
            "bootstrap_requested": bootstrap_samples,
            "bootstrap_successful": len(draws),
            "bootstrap_elapsed_seconds": time.monotonic() - bootstrap_started,
        }
    )
    if draws:
        for coefficient in ("bc", "bh"):
            interval = np.quantile([draw[coefficient] for draw in draws], [0.025, 0.975])
            fit[f"{coefficient}_ci95"] = [float(interval[0]), float(interval[1])]
    return fit, draws


def analyse_system(row: dict[str, str], config: dict, bootstrap_samples: int = 0) -> tuple[dict, list[dict]]:
    started = time.monotonic()
    data = system_inputs(row, config)
    if data is None:
        return {"system_id": row["system_id"], "eligible": False}, []
    folds = [fit_fold(data, heldout, config) for heldout in (1, 2, 3)]
    valid = [fold for fold in folds if fold["valid"]]
    result = {
        "system_id": row["system_id"],
        "eligible": len(valid) == 3,
        "folds": folds,
        "cath_class": row["cath_class"],
        "rmsf_tercile": row["rmsf_tercile"],
    }
    if len(valid) == 3:
        for name in (
            "bc", "bh", "bh_over_bc", "fitted_heldout_rho", "default_heldout_rho",
            "delta_heldout_rho", "fitted_heldout_pmf_mae_kcal_mol",
            "default_heldout_pmf_mae_kcal_mol", "delta_heldout_pmf_mae_kcal_mol",
        ):
            result[f"mean_{name}"] = float(np.mean([fold[name] for fold in valid]))
        pooled, draws = fit_pooled(data, config, bootstrap_samples)
        result["pooled_fit"] = pooled
        for draw in draws:
            draw["system_id"] = row["system_id"]
        result["elapsed_seconds"] = time.monotonic() - started
        return result, draws
    return result, []


def require_authorized(exploratory_override: bool) -> tuple[dict, dict]:
    path = HERE / "outputs" / "analysis" / "stage1_decision.yaml"
    decision = yaml.safe_load(path.read_text())
    if decision.get("stage2_authorized", False):
        return decision, {"mode": "stage1_gate", "exploratory": False}
    if not exploratory_override:
        raise SystemExit(
            "Stage 2 refused: corrected Absolute-L1 Stage 1 gate did not pass. "
            f"Blocking gate: {decision.get('blocking_gate', 'unknown')}. "
            "Pass --exploratory-override to record an explicit exploratory continuation."
        )
    primary = decision["thermodynamic_test"]["coordinate_results"]["absolute_l1"]
    return decision, {
        "mode": "user_exploratory_override",
        "exploratory": True,
        "recorded_date": date.today().isoformat(),
        "stage1_pass_preserved": bool(decision.get("stage1_pass", False)),
        "blocking_gate": decision.get("blocking_gate"),
        "rationale": (
            "Absolute-L1 is above its population null with a consistent expected sign; "
            "the predeclared compactness-majority threshold alone was missed."
        ),
        "absolute_l1_population_p": primary["population_permutation_p_one_sided"],
        "absolute_l1_beats_compactness_fraction": primary[
            "fraction_beats_best_compactness_baseline"
        ],
    }


def paired_improvement(
    deltas: list[float], samples: int, seed: int
) -> dict[str, float | bool]:
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(samples, len(values)), replace=True), axis=1
    )
    lower = float(np.quantile(draws, 0.05))
    return {
        "median_delta_heldout_rho": float(np.median(values)),
        "one_sided_95pct_lower_bound": lower,
        "bootstrap_samples": samples,
        "success": bool(lower > 0),
    }


def compactness_comparators() -> dict[str, float]:
    path = HERE / "outputs" / "analysis" / "within_basin_results.yaml"
    systems = yaml.safe_load(path.read_text())["systems"]
    comparators = {}
    for system in systems:
        folds = [fold for fold in system.get("folds", []) if fold.get("valid")]
        if len(folds) != 3:
            continue
        per_coordinate = np.mean(
            [
                [fold["metrics"][name]["rho"] for name in ("H", "Rg", "RMSD", "native_contacts")]
                for fold in folds
            ],
            axis=0,
        )
        comparators[system["system_id"]] = float(np.max(per_coordinate))
    return comparators


def aggregate_results(results: list[dict], config: dict, authorization: dict) -> dict:
    eligible = [result for result in results if result.get("eligible")]

    def distribution(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        return {
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
        }

    summary = {
        "systems": len(results),
        "eligible_systems": len(eligible),
        "authorization": authorization,
    }
    if eligible:
        summary["distributions"] = {
            "pooled_bc": distribution([result["pooled_fit"]["bc"] for result in eligible]),
            "pooled_bh": distribution([result["pooled_fit"]["bh"] for result in eligible]),
            "pooled_bh_over_bc": distribution(
                [result["pooled_fit"]["bh_over_bc"] for result in eligible]
            ),
            "fitted_heldout_rho": distribution(
                [result["mean_fitted_heldout_rho"] for result in eligible]
            ),
            "default_heldout_rho": distribution(
                [result["mean_default_heldout_rho"] for result in eligible]
            ),
            "delta_heldout_rho": distribution(
                [result["mean_delta_heldout_rho"] for result in eligible]
            ),
            "fitted_heldout_pmf_mae_kcal_mol": distribution(
                [result["mean_fitted_heldout_pmf_mae_kcal_mol"] for result in eligible]
            ),
            "delta_heldout_pmf_mae_kcal_mol": distribution(
                [result["mean_delta_heldout_pmf_mae_kcal_mol"] for result in eligible]
            ),
        }
        summary["stage2_predictive_decision"] = paired_improvement(
            [result["mean_delta_heldout_rho"] for result in eligible],
            config["analysis"]["stage2"]["population_bootstrap_samples"],
            config["analysis"]["seed"],
        )
        summary["boundary_solution_fraction"] = float(
            np.mean([result["pooled_fit"]["boundary_solution"] for result in eligible])
        )
        summary["fraction_fitted_beats_compactness"] = float(
            np.mean(
                [
                    result["mean_fitted_heldout_rho"] > result["best_compactness_rho"]
                    for result in eligible
                ]
            )
        )
        summary["strata"] = {}
        for field in ("rmsf_tercile", "cath_class"):
            summary["strata"][field] = {
                str(value): {
                    "systems": len(group),
                    "median_bc": float(np.median([item["pooled_fit"]["bc"] for item in group])),
                    "median_bh": float(np.median([item["pooled_fit"]["bh"] for item in group])),
                    "median_delta_heldout_rho": float(
                        np.median([item["mean_delta_heldout_rho"] for item in group])
                    ),
                }
                for value in sorted({item[field] for item in eligible})
                if (group := [item for item in eligible if item[field] == value])
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--exploratory-override", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    decision, authorization = require_authorized(args.exploratory_override)
    config = load_config()
    workers = args.workers or config["analysis"]["stage2"]["workers"]
    if workers < 1:
        parser.error("--workers must be positive")
    selected = set(decision["informative_system_ids"])
    rows = [row for row in load_systems() if row["system_id"] in selected]
    if args.limit:
        rows = rows[: args.limit]
    if args.benchmark:
        rows.sort(key=lambda row: int(row["length"]))
        rows = [rows[len(rows) // 2]]
    started = time.monotonic()
    results = []
    bootstrap_draws = []
    samples = (
        config["analysis"]["stage2"]["benchmark_bootstrap_samples"]
        if args.benchmark
        else config["analysis"]["stage2"]["bootstrap_samples"]
    )
    if args.benchmark or workers == 1:
        for index, row in enumerate(rows, 1):
            result, draws = analyse_system(row, config, samples)
            results.append(result)
            bootstrap_draws.extend(draws)
            print(
                f"[{index}/{len(rows)}] {row['system_id']} eligible={result['eligible']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(analyse_system, row, config, samples): row for row in rows
            }
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                result, draws = future.result()
                results.append(result)
                bootstrap_draws.extend(draws)
                print(
                    f"[{index}/{len(rows)}] {row['system_id']} eligible={result['eligible']}",
                    flush=True,
                )
        results.sort(key=lambda result: result["system_id"])
    elapsed = time.monotonic() - started
    output_dir = HERE / "outputs" / "fitting"
    if args.benchmark:
        report = {
            "authorization": authorization,
            "system_id": rows[0]["system_id"],
            "elapsed_seconds": elapsed,
            "benchmark_bootstrap_samples": config["analysis"]["stage2"]["benchmark_bootstrap_samples"],
            "full_bootstrap_samples": config["analysis"]["stage2"]["bootstrap_samples"],
            "base_elapsed_seconds": elapsed
            - results[0]["pooled_fit"]["bootstrap_elapsed_seconds"],
            "bootstrap_elapsed_seconds": results[0]["pooled_fit"]["bootstrap_elapsed_seconds"],
            "systems": len(selected),
        }
        per_draw = report["bootstrap_elapsed_seconds"] / report["benchmark_bootstrap_samples"]
        report["projected_full_seconds"] = len(selected) * (
            report["base_elapsed_seconds"] + per_draw * report["full_bootstrap_samples"]
        )
        report["projected_cpu_hours"] = report["projected_full_seconds"] / 3600
        draw_frame = pd.DataFrame(bootstrap_draws)
        bytes_per_draw = (
            float(draw_frame.memory_usage(index=False, deep=True).sum() / len(draw_frame))
            if len(draw_frame)
            else 0.0
        )
        projected_draw_bytes = int(
            bytes_per_draw * len(selected) * report["full_bootstrap_samples"]
        )
        projected_yaml_bytes = len(yaml.safe_dump({"systems": results}).encode()) * len(selected)
        report["projected_output_bytes"] = projected_draw_bytes + projected_yaml_bytes
        report["external_service_cost"] = 0
        report["full_run_started"] = False
        atomic_yaml(output_dir / "absolute_l1_benchmark.yaml", report)
        print(yaml.safe_dump(report, sort_keys=False))
        return
    comparators = compactness_comparators()
    for result in results:
        if result.get("eligible"):
            result["best_compactness_rho"] = comparators[result["system_id"]]
            result["fitted_minus_best_compactness_rho"] = (
                result["mean_fitted_heldout_rho"] - result["best_compactness_rho"]
            )
    atomic_yaml(
        output_dir / "absolute_l1_results.yaml",
        {"authorization": authorization, "systems": results},
    )
    atomic_yaml(
        output_dir / "absolute_l1_aggregate.yaml",
        aggregate_results(results, config, authorization),
    )
    pd.DataFrame(bootstrap_draws).to_parquet(
        output_dir / "absolute_l1_bootstrap.parquet", index=False
    )
    summary_path = output_dir / "absolute_l1_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system_id", "eligible", "mean_bc", "mean_bh", "mean_bh_over_bc",
        "mean_fitted_heldout_rho", "mean_default_heldout_rho", "mean_delta_heldout_rho",
        "mean_fitted_heldout_pmf_mae_kcal_mol", "mean_default_heldout_pmf_mae_kcal_mol",
        "mean_delta_heldout_pmf_mae_kcal_mol", "best_compactness_rho",
        "fitted_minus_best_compactness_rho",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})
    print(f"Completed {len(results)} systems in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
