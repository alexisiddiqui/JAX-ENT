"""Strict fixed-BV scalar distance likelihoods across coordinate-W1 bands."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_checkpoint11 import summarize_model
from jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 import frame_disjoint_pair_split, _scalar_fit_predict
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import pf_pair_distance
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, mondrian_quantiles, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import GAUSSIAN_Q90, _pair_sets_from_audit, gaussian_mass
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


METRICS = ("l1", "l2", "cosine", "correlation")
TARGETS = ("rmsd", "w1")


def analyse_system(row: dict[str, str], config: dict, parts: Path) -> str:
    system = row["system_id"]; settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    audit = pd.read_parquet(source); pairs = _pair_sets_from_audit(audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    log_pf = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    distances = {metric: {
        replica: pf_pair_distance(log_pf, frame.left_frame.to_numpy(), frame.right_frame.to_numpy(), metric)
        for replica, frame in pairs.items()
    } for metric in METRICS}
    summaries = []
    for fit_replica, calibration_replica, test_replica in ordered_assignments():
        fit_pairs = pairs[fit_replica]; calibration_pairs = pairs[calibration_replica]; test_pairs = pairs[test_replica]
        fit_mask, validation_mask = frame_disjoint_pair_split(
            fit_pairs.left_frame.to_numpy(), fit_pairs.right_frame.to_numpy()
        )
        for target_name in TARGETS:
            fit_target = fit_pairs[target_name].to_numpy()
            calibration_target = calibration_pairs[target_name].to_numpy()
            test_target = test_pairs[target_name].to_numpy()
            assignment_audit = audit[
                (audit.fit_replica == fit_replica) & (audit.calibration_replica == calibration_replica)
                & (audit.test_replica == test_replica) & (audit.target == target_name)
            ].reset_index(drop=True)
            for metric, distance in distances.items():
                candidate = _scalar_fit_predict(
                    distance[fit_replica], distance[calibration_replica], distance[test_replica],
                    fit_target, fit_mask, validation_mask, settings,
                )
                scores = np.abs(calibration_target - candidate.calibration_prediction)
                marginal_q = finite_conformal_quantile(scores, strict["coverage"])
                mondrian_q, _ = mondrian_quantiles(
                    candidate.fit_prediction, candidate.calibration_prediction, scores,
                    candidate.test_prediction, strict["mondrian_bins"], strict["coverage"], marginal_q,
                )
                for calibration, width in (("marginal", np.full(len(test_target), marginal_q)),
                                           ("mondrian", mondrian_q)):
                    sigma = np.maximum(width / GAUSSIAN_Q90, np.finfo(float).eps)
                    density = lambda mask, low, high, bins, p=candidate.test_prediction, s=sigma: gaussian_mass(
                        p[mask], s[mask], low, high, bins
                    )
                    summaries.extend(summarize_model(
                        assignment_audit, f"fixed_logpf_scalar_{metric}_gaussian", calibration,
                        candidate.test_prediction, np.maximum(0.0, candidate.test_prediction - width),
                        candidate.test_prediction + width, density, settings,
                    ))
    atomic_parquet(pd.DataFrame(summaries), parts / f"{system}.summary.parquet")
    return system


def valid_checkpoint(parts: Path, system: str) -> bool:
    path = parts / f"{system}.summary.parquet"
    if not path.exists(): return False
    try: frame = pd.read_parquet(path)
    except Exception: return False
    expected = {f"fixed_logpf_scalar_{metric}_gaussian" for metric in METRICS}
    assignments = frame[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates()
    return set(frame.system_id) == {system} and set(frame.model) == expected and len(assignments) == 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint14_fixed_metrics"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{resumed + index}/{len(systems)}] {future.result()} fixed metrics checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing fixed-metric systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / "fixed_metric_assignment_summary.parquet", index=False)
    atomic_yaml(output / "checkpoint14_run.yaml", {"checkpoint": "14", "status": "measurement_complete", "systems": len(systems)})


if __name__ == "__main__":
    main()
