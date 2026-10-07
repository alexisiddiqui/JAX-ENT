from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from jaxent.examples.ATLAS_BV.analysis.basin_census import (
    align_to_reference,
    load_ca_coordinates,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    distribution_distances,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)


COORDINATES = ("G", "d1", "d2", "H", "O", "Rg", "RMSD", "native_contacts")
DISTANCE_COORDINATES = ("G", "d1", "d2")
COORDINATE_NAMES = {"G": "signed_l1", "d1": "absolute_l1", "d2": "l2"}


def structural_vectors(coordinates: np.ndarray) -> np.ndarray:
    return np.asarray([pdist(frame) for frame in coordinates], dtype=np.float32)


def fixed_width_edges(values: np.ndarray, settings: dict) -> np.ndarray:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    width = 2 * (q75 - q25) / np.cbrt(len(values))
    bins = settings["pc1_bins_min"] if width <= 0 else int(np.ceil(np.ptp(values) / width))
    bins = int(np.clip(bins, settings["pc1_bins_min"], settings["pc1_bins_max"]))
    return np.linspace(values.min(), values.max(), bins + 1)


def weighted_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones(len(x))])
    root = np.sqrt(weights / weights.sum())
    coefficient = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return float(coefficient[0]), float(coefficient[1])


def binned(values: np.ndarray, pc1: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.digitize(pc1, edges[1:-1])
    means, counts = [], []
    for label in range(len(edges) - 1):
        mask = labels == label
        means.append(float(np.mean(values[mask])) if mask.any() else np.nan)
        counts.append(int(mask.sum()))
    return np.asarray(means), np.asarray(counts), labels


def shuffle_blocks(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    chunks = [values[start : start + block] for start in range(0, len(values), block)]
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[index] for index in order])


def native_contacts(coordinates: np.ndarray, reference_index: int, config: dict) -> np.ndarray:
    settings = config["analysis"]["stage1"]
    reference = coordinates[reference_index]
    i, j = np.triu_indices(len(reference), k=settings["native_sequence_separation"])
    reference_distances = np.linalg.norm(reference[i] - reference[j], axis=1)
    native = reference_distances <= settings["native_ca_cutoff_angstrom"]
    i, j, reference_distances = i[native], j[native], reference_distances[native]
    if not len(i):
        return np.zeros(len(coordinates))
    distances = np.linalg.norm(coordinates[:, i] - coordinates[:, j], axis=2)
    return np.sum(distances <= settings["native_distance_multiplier"] * reference_distances, axis=1)


def load_rg(system: str, equilibration_ns: float) -> np.ndarray:
    table = np.genfromtxt(
        HERE / "data" / "raw" / system / f"{system}_gyrate.tsv", names=True, delimiter="\t"
    )
    keep = table["time"] > equilibration_ns
    return np.concatenate([table[f"gyr_R{replica}"][keep] for replica in (1, 2, 3)])


def fold_metrics(
    frame_table: pd.DataFrame,
    vectors: np.ndarray,
    z: np.ndarray,
    fold: int,
    config: dict,
) -> dict:
    settings = config["analysis"]["stage1"]
    train = frame_table["replica"].to_numpy() != fold
    test = ~train
    pca = PCA(n_components=1, random_state=config["analysis"]["seed"])
    pc_train = pca.fit_transform(vectors[train]).ravel()
    pc_test = pca.transform(vectors[test]).ravel()
    edges = fixed_width_edges(pc_train, settings)
    out_of_range = float(np.mean((pc_test < edges[0]) | (pc_test > edges[-1])))
    widths = np.diff(edges)
    _, train_counts, train_labels = binned(pc_train, pc_train, edges)
    _, test_counts, test_labels = binned(pc_test, pc_test, edges)
    common = (train_counts >= settings["min_frames_per_bin"]) & (
        test_counts >= settings["min_frames_per_bin"]
    )
    valid = int(common.sum()) >= settings["min_common_bins_per_fold"] and out_of_range <= settings[
        "max_heldout_out_of_range_fraction"
    ]
    result = {
        "fold": fold,
        "valid": bool(valid),
        "common_bins": int(common.sum()),
        "out_of_range_fraction": out_of_range,
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
    }
    if not valid:
        return result

    log_density_train = np.full(len(widths), np.nan)
    log_density_test = np.full(len(widths), np.nan)
    train_populated = train_counts > 0
    test_populated = test_counts > 0
    log_density_train[train_populated] = np.log(
        train_counts[train_populated] / (train_counts.sum() * widths[train_populated])
    )
    log_density_test[test_populated] = np.log(
        test_counts[test_populated] / (test_counts.sum() * widths[test_populated])
    )
    reference_bin = int(np.nanargmax(np.where(common, log_density_train, np.nan)))
    train_indices = np.flatnonzero(train)
    candidates = np.flatnonzero(train_labels == reference_bin)
    center = (edges[reference_bin] + edges[reference_bin + 1]) / 2
    reference_global = int(train_indices[candidates[np.argmin(np.abs(pc_train[candidates] - center))]])
    zref = z[:, reference_global]
    local = frame_table.copy()
    local["d1"] = np.abs(z - zref[:, None]).sum(axis=0)
    local["d2"] = np.linalg.norm(z - zref[:, None], axis=0)
    metrics = {}
    for coordinate in COORDINATES:
        train_means, _, _ = binned(local.loc[train, coordinate].to_numpy(), pc_train, edges)
        test_means, _, _ = binned(local.loc[test, coordinate].to_numpy(), pc_test, edges)
        slope, intercept = weighted_line(
            train_means[common], log_density_train[common], train_counts[common]
        )
        predicted = slope * test_means[common] + intercept
        observed = log_density_test[common]
        rho = float(spearmanr(predicted, observed).statistic)
        metrics[coordinate] = {
            "slope": slope,
            "rho": rho,
            # F_conf = -RT ln(p), while DeltaG_open = RT ln(PF).  Thus the
            # thermodynamic comparison is F_conf against -DeltaG_open; the
            # two sign changes leave the rank correlation equal to `rho`.
            "free_energy_rho_vs_negative_opening_free_energy": rho,
            "pmf_mae_kcal_mol": float(
                config["analysis"]["gas_constant_kcal_mol_k"]
                * config["protocol"]["temperature_k"]
                * np.mean(np.abs(predicted - observed))
            ),
        }

    for coordinate_index, coordinate in enumerate(DISTANCE_COORDINATES):
        train_values = local.loc[train, coordinate].to_numpy()
        test_values = local.loc[test, coordinate].to_numpy()
        block = max(1, integrated_autocorrelation_frames(train_values))
        rng = np.random.default_rng(
            config["analysis"]["seed"] + 1000 * coordinate_index + fold
        )
        null = np.empty(settings["permutation_samples"])
        for index in range(len(null)):
            shuffled_train = shuffle_blocks(train_values, block, rng)
            shuffled_test = shuffle_blocks(test_values, block, rng)
            train_means, _, _ = binned(shuffled_train, pc_train, edges)
            test_means, _, _ = binned(shuffled_test, pc_test, edges)
            slope, intercept = weighted_line(
                train_means[common], log_density_train[common], train_counts[common]
            )
            null[index] = spearmanr(
                slope * test_means[common] + intercept, log_density_test[common]
            ).statistic
        metrics[coordinate].update(
            {
                "autocorrelation_frames": block,
                "permutation_rho_95": float(np.quantile(null, 0.95)),
                "permutation_rhos": null.tolist(),
            }
        )
    result.update(
        {
            "reference_global_index": reference_global,
            "pmf_range_kcal_mol": float(
                config["analysis"]["gas_constant_kcal_mol_k"]
                * config["protocol"]["temperature_k"]
                * np.ptp(log_density_test[common])
            ),
            "metrics": metrics,
        }
    )
    return result


def analyse_system(row: dict[str, str], config: dict) -> dict:
    system = row["system_id"]
    summary_path = HERE / "outputs" / "analysis" / "basins" / system / "summary.yaml"
    summary = yaml.safe_load(summary_path.read_text())
    if summary["usable_basins"] < 1:
        return {"system_id": system, "eligible": False, "reason": "no_shared_dominant_basin"}
    counts = summary["counts_by_replica"]
    label = max(
        summary["usable_labels"],
        key=lambda candidate: sum(counts[candidate].values()),
    )
    assignments = pd.read_parquet(HERE / summary["assignments"])
    mask = assignments["basin"].to_numpy() == label
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    coordinates = coordinates[mask]
    aligned = align_to_reference(coordinates)
    vectors = structural_vectors(coordinates)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)[:, mask]
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)[:, mask]
    h = heavy.sum(axis=0)
    o = acceptor.sum(axis=0)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    g = z.sum(axis=0)
    rg = load_rg(system, config["analysis"]["equilibration_ns"])[mask]
    pooled_pca = PCA(n_components=1, random_state=config["analysis"]["seed"])
    pooled_pc1 = pooled_pca.fit_transform(vectors).ravel()
    reference_index = int(np.argmin(np.linalg.norm(aligned.reshape(len(aligned), -1), axis=1)))
    rmsd = np.linalg.norm(aligned - aligned[reference_index], axis=(1, 2)) / np.sqrt(aligned.shape[1])
    native = native_contacts(coordinates, reference_index, config)
    table = pd.DataFrame(
        {
            "replica": replicas[mask], "frame": frames[mask], "basin": label,
            "PC1_pooled_descriptive": pooled_pc1, "H": h, "O": o, "G": g,
            "Rg": rg, "RMSD": rmsd, "native_contacts": native,
        }
    )
    output_dir = HERE / "outputs" / "analysis" / "within_basin" / system
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_dir / "frame_coords.parquet", index=False)

    convergence = []
    thresholds = config["analysis"]["convergence"]
    for coordinate in ("G", "PC1_pooled_descriptive", "Rg"):
        values = [table.loc[table.replica == replica, coordinate].to_numpy() for replica in (1, 2, 3)]
        for left, right in ((0, 1), (0, 2), (1, 2)):
            ks, js = distribution_distances(
                values[left], values[right], thresholds["histogram_bins_min"], thresholds["histogram_bins_max"]
            )
            convergence.append(ks <= thresholds["ks_max"] and js <= thresholds["js_bits_max"])
    folds = [fold_metrics(table, vectors, z, fold, config) for fold in (1, 2, 3)]
    return {
        "system_id": system, "eligible": True, "frames": len(table),
        "within_basin_converged": bool(all(convergence)), "folds": folds,
        "cath_class": row["cath_class"], "rmsf_tercile": row["rmsf_tercile"],
    }


def decision(results: list[dict], config: dict) -> dict:
    settings = config["analysis"]["stage1"]
    informative = []
    coordinate_data = {
        coordinate: {"rhos": [], "nulls": [], "beats": [], "expected_sign": []}
        for coordinate in DISTANCE_COORDINATES
    }
    eligible_count = sum(result.get("eligible", False) for result in results)
    converged_count = sum(result.get("within_basin_converged", False) for result in results)
    three_fold_count = sum(
        result.get("eligible", False)
        and sum(fold["valid"] for fold in result.get("folds", [])) == 3
        for result in results
    )
    for result in results:
        folds = result.get("folds", [])
        valid = [fold for fold in folds if fold["valid"]]
        if len(valid) != 3:
            continue
        if np.mean([fold["pmf_range_kcal_mol"] for fold in valid]) < settings["min_delta_f_range_kcal_mol"]:
            continue
        informative.append(result["system_id"])
        baselines = np.mean(
            [[fold["metrics"][name]["rho"] for name in ("H", "Rg", "RMSD", "native_contacts")] for fold in valid], axis=0
        )
        for coordinate in DISTANCE_COORDINATES:
            rho = float(np.mean([fold["metrics"][coordinate]["rho"] for fold in valid]))
            null_draws = np.mean(
                [fold["metrics"][coordinate]["permutation_rhos"] for fold in valid], axis=0
            )
            slope = float(np.mean([fold["metrics"][coordinate]["slope"] for fold in valid]))
            coordinate_data[coordinate]["rhos"].append(rho)
            coordinate_data[coordinate]["nulls"].append(null_draws)
            coordinate_data[coordinate]["beats"].append(rho > np.max(baselines))
            coordinate_data[coordinate]["expected_sign"].append(
                slope > 0 if coordinate == "G" else slope < 0
            )
    n = len(informative)
    coordinate_results = {}
    for coordinate, data in coordinate_data.items():
        observed = float(np.median(data["rhos"])) if n else float("nan")
        population_null = np.median(np.asarray(data["nulls"]), axis=0) if n else np.asarray([])
        null_95 = float(np.quantile(population_null, 0.95)) if n else float("nan")
        p_value = (
            float((1 + np.count_nonzero(population_null >= observed)) / (len(population_null) + 1))
            if n else float("nan")
        )
        individual = (
            np.mean(
                [rho > np.quantile(null, 0.95) for rho, null in zip(data["rhos"], data["nulls"])]
            )
            if n else 0.0
        )
        coordinate_results[COORDINATE_NAMES[coordinate]] = {
            "median_heldout_rho": observed,
            "population_permutation_rho_95": null_95,
            "population_permutation_p_one_sided": p_value,
            "fraction_individually_above_95pct_null_diagnostic": float(individual),
            "fraction_beats_best_compactness_baseline": float(np.mean(data["beats"])) if n else 0.0,
            "fraction_expected_slope_sign": float(np.mean(data["expected_sign"])) if n else 0.0,
            "expected_slope_sign": "positive" if coordinate == "G" else "negative",
        }
    primary = coordinate_results["absolute_l1"]
    passed = (
        n >= settings["min_informative_systems"]
        and primary["population_permutation_p_one_sided"] <= settings["population_null_alpha"]
        and primary["fraction_beats_best_compactness_baseline"] >= settings["majority_fraction"]
        and primary["fraction_expected_slope_sign"] >= settings["sign_consistency_fraction"]
    )
    return {
        "redesign_version": config["analysis"]["redesign_version"],
        "historical_multi_basin_decision": "failed",
        "primary_coordinate": "absolute_l1",
        "stage1_pass": bool(passed), "stage2_authorized": bool(passed),
        "informative_systems": n, "informative_system_ids": informative,
        "thermodynamic_test": {
            "experimental_relation": "DeltaG_open = RT ln(PF)",
            "tested_relation": "ln p versus negative absolute residue-wise Delta log(PF)",
            "coordinate_results": coordinate_results,
        },
        "diagnostics": {
            "total_systems": len(results),
            "shared_basin_eligible_systems": eligible_count,
            "within_basin_converged_systems": converged_count,
            "systems_with_three_valid_pc1_folds": three_fold_count,
            "convergence_role": "reported diagnostic; not an exclusion gate",
            "null_role": "population-level block permutation; individual 95% exceedance is diagnostic only",
        },
        "blocking_gate": None if passed else "within_basin_stage1_absolute_l1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()[: args.limit]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in rows}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{count}/{len(rows)}] {result['system_id']} eligible={result['eligible']}", flush=True)
    results.sort(key=lambda item: item["system_id"])
    output = HERE / "outputs" / "analysis" / "within_basin_results.yaml"
    atomic_yaml(output, {"systems": results})
    gate = decision(results, config)
    atomic_yaml(HERE / "outputs" / "analysis" / "stage1_decision.yaml", gate)
    with (HERE / "outputs" / "analysis" / "within_basin_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system_id", "eligible", "within_basin_converged", "frames"])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})
    print(yaml.safe_dump(gate, sort_keys=False))


if __name__ == "__main__":
    main()
