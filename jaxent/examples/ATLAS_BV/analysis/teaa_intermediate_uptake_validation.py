"""Validate BV uptake candidates on TeaA open/closed/intermediate structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.bv_uptake_model_validation import (
    EnsembleData,
    calibrate_models,
    evaluate_case,
    load_features,
    markdown_table,
)


REPO = Path(__file__).resolve().parents[4]
TEAA = REPO / "jaxent/examples/1_IsoValidation_OMass"
CONTROL = TEAA / "analysis/omc_control"
DEFAULT_OUTPUT = CONTROL / "bv_uptake_intermediate_validation"
TRUTH = np.asarray([0.4, 0.6, 0.0])


def topology_keys(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    records = json.loads(path.read_text())["topologies"]
    keys = [(record["chain"], tuple(record["residues"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate residue identifiers in {path}")
    return keys


def load_intermediate_candidate() -> tuple[EnsembleData, np.ndarray, dict[str, int]]:
    feature_dir = CONTROL / "features"
    heavy, acceptor, kint = load_features(feature_dir / "features_iso_tri.npz")
    reference_keys = topology_keys(
        TEAA / "data/_self_consistent_target_features/topology_open_closed.json"
    )
    candidate_keys = topology_keys(feature_dir / "topology_iso_tri.json")
    if set(reference_keys) != set(candidate_keys):
        raise ValueError("TeaA reference and ISO_TRI residue layouts differ")
    alignment = np.asarray([candidate_keys.index(key) for key in reference_keys])
    heavy, acceptor, kint = heavy[alignment], acceptor[alignment], kint[alignment]

    prepared = np.load(CONTROL / "ISO_TRI/input.npz")
    raw_groups = np.asarray(prepared["groups"], dtype=int)
    mapping = {0: 0, 1: 1, -1: 2}
    states = np.asarray([mapping[value] for value in raw_groups])
    target_data = np.load(CONTROL / "target.npz")
    target = np.asarray(target_data["target"], dtype=np.float64)
    times = np.asarray(target_data["times"], dtype=np.float64)
    np.testing.assert_allclose(kint, target_data["kints"], rtol=1e-6, atol=1e-10)
    counts = {
        "open": int(np.sum(states == 0)),
        "closed": int(np.sum(states == 1)),
        "intermediate": int(np.sum(states == 2)),
    }
    data = EnsembleData(
        system_id="TeaA_ISO_TRI",
        heavy=heavy,
        acceptor=acceptor,
        log_kint=np.log(kint),
        states=states,
        state_names=("open", "closed", "intermediate"),
        times=times,
        metadata={"candidate": "ISO_TRI", **{f"n_{key}": value for key, value in counts.items()}},
    )
    return data, target, counts


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data, target, counts = load_intermediate_candidate()
    calibrated = calibrate_models(data, TRUTH, target)
    results = pd.DataFrame(
        evaluate_case(
            data,
            TRUTH,
            "TeaA_TRI",
            calibrated_parameters=calibrated,
            target_uptake=target,
        )
    )
    populations = np.stack(
        results.fitted_population.map(lambda value: json.loads(value)).to_numpy()
    )
    results["recovered_open"] = populations[:, 0]
    results["recovered_closed"] = populations[:, 1]
    results["recovered_intermediate"] = populations[:, 2]
    results.to_parquet(output / "results.parquet", index=False)
    summary = (
        results.groupby(["model", "noise_sigma"])
        .agg(
            heldout_uptake_mae=("test_mae", "median"),
            population_max_abs=("population_max_abs", "median"),
            population_recovery_percent=("population_recovery_percent", "median"),
            recovered_open=("recovered_open", "median"),
            recovered_closed=("recovered_closed", "median"),
            recovered_intermediate=("recovered_intermediate", "median"),
            converged=("converged", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    lines = [
        "# TeaA intermediate-structure BV validation",
        "",
        "The independent target is exact frame-wise uptake from the reference open/closed",
        "ensembles at 40% open, 60% closed, and 0% intermediate. The ISO_TRI candidate",
        f"contains {counts['open']} open, {counts['closed']} closed, and",
        f"{counts['intermediate']} intermediate frames classified by the established",
        "1-Angstrom RMSD rule. Parameters are calibrated on alternating training peptide",
        "windows and frozen before population recovery. Gamma and Q4 calibration uses",
        "32 balanced frames per state; final predictions and metrics use all frames.",
        "",
        markdown_table(summary),
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "status": "complete",
        "candidate": "ISO_TRI",
        "states": list(data.state_names),
        "frame_counts": counts,
        "target_population": TRUTH.tolist(),
        "target": "independent exact frame-wise open/closed reference",
        "records": len(results),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
