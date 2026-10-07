from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    bootstrap_distance_ci,
    distribution_distances,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)


def load_atlas_trace(system: str, suffix: str, prefix: str, equilibration_ns: float) -> list[np.ndarray]:
    path = HERE / "data" / "raw" / system / f"{system}_{suffix}.tsv"
    table = np.genfromtxt(path, names=True, delimiter="\t")
    keep = table["time"] > equilibration_ns
    return [np.asarray(table[f"{prefix}_R{replica}"][keep], dtype=float) for replica in (1, 2, 3)]


def analyse_system(row: dict[str, str], config: dict) -> dict:
    system = row["system_id"]
    analysis = config["analysis"]
    settings = analysis["convergence"]
    series = {
        "G": [load_contact_coordinates(system, replica, config)["G"] for replica in (1, 2, 3)],
        "RMSD": load_atlas_trace(system, "RMSD", "RMSD", analysis["equilibration_ns"]),
        "Rg": load_atlas_trace(system, "gyrate", "gyr", analysis["equilibration_ns"]),
    }
    tau = max(integrated_autocorrelation_frames(values) for values in series["G"])
    comparisons = []
    comparison_index = 0
    for coordinate, replicas in series.items():
        pairs = []
        for replica, values in enumerate(replicas, 1):
            midpoint = len(values) // 2
            pairs.append((f"R{replica}_halves", values[:midpoint], values[midpoint:]))
        pairs.extend(
            (f"R{left}_R{right}", replicas[left - 1], replicas[right - 1])
            for left, right in ((1, 2), (1, 3), (2, 3))
        )
        for label, left, right in pairs:
            ks, js = distribution_distances(
                left, right, settings["histogram_bins_min"], settings["histogram_bins_max"]
            )
            ks_ci, js_ci = bootstrap_distance_ci(
                left,
                right,
                tau,
                settings["bootstrap_samples"],
                analysis["seed"] + comparison_index,
                settings["histogram_bins_min"],
                settings["histogram_bins_max"],
            )
            comparisons.append(
                {
                    "coordinate": coordinate,
                    "comparison": label,
                    "ks": ks,
                    "ks_ci_low": ks_ci[0],
                    "ks_ci_high": ks_ci[1],
                    "js_bits": js,
                    "js_ci_low": js_ci[0],
                    "js_ci_high": js_ci[1],
                    "pass": ks <= settings["ks_max"] and js <= settings["js_bits_max"],
                }
            )
            comparison_index += 1
    return {
        "system_id": system,
        "length": int(row["length"]),
        "cath_class": row["cath_class"],
        "rmsf_tercile": row["rmsf_tercile"],
        "avg_RMSF": float(row["avg_RMSF"]),
        "autocorrelation_frames": tau,
        "pass": all(item["pass"] for item in comparisons),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()[: args.limit]
    output = HERE / "data" / "convergence_report.csv"
    fields = [
        "system_id", "length", "cath_class", "rmsf_tercile", "avg_RMSF",
        "autocorrelation_frames", "pass", "max_ks", "max_js_bits",
    ]
    detail_path = HERE / "outputs" / "analysis" / "convergence_details.yaml"
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in rows}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            atomic_yaml(
                detail_path,
                {"thresholds": config["analysis"]["convergence"], "systems": results},
            )
            print(
                f"[{count}/{len(rows)}] {result['system_id']}: "
                f"{'pass' if result['pass'] else 'fail'}",
                flush=True,
            )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **{key: result[key] for key in fields if key in result},
                    "max_ks": max(item["ks"] for item in result["comparisons"]),
                    "max_js_bits": max(item["js_bits"] for item in result["comparisons"]),
                }
            )
    atomic_yaml(
        detail_path,
        {"thresholds": config["analysis"]["convergence"], "systems": results},
    )
    passed = sum(result["pass"] for result in results)
    print(f"Convergence complete: {passed}/{len(results)} systems passed; report={output}")


if __name__ == "__main__":
    main()
