"""Fixed-BV metric recovery in pooled global coordinate-W1 bands."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_checkpoint11 import summarize_model
from jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 import frame_disjoint_pair_split, _ridge_fit_predict, _scalar_fit_predict
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import RMSD_QUANTILES, pf_pair_distance
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, mondrian_quantiles, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import GAUSSIAN_Q90, _pair_sets_from_audit, gaussian_mass
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import absolute_change_vectors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


METRICS = ("l1", "l2", "cosine", "correlation")
RIDGE_MODEL = "fixed_logpf_per_residue_ridge_gaussian"


def global_target_edges(systems: list[dict[str, str]], target: str) -> tuple[np.ndarray, int]:
    """Pool every unique sampled structural pair once; assignments do not add weight."""
    values = []
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts"
    for row in systems:
        frame = pd.read_parquet(
            source / f"{row['system_id']}.pairs.parquet",
            columns=["fit_replica", "calibration_replica", "test_replica", "target", "left_frame", "right_frame", "target_value"],
        )
        frame = frame[frame.target == target].drop_duplicates(
            ["test_replica", "left_frame", "right_frame"]
        )
        values.append(frame.target_value.to_numpy())
    pooled = np.concatenate(values)
    return np.quantile(pooled, RMSD_QUANTILES), len(pooled)


def apply_global_bands(audit: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    result = audit.copy()
    labels = np.clip(np.digitize(result.target_value.to_numpy(), edges[1:-1]), 0, len(edges) - 2)
    result["band"] = [f"q{label}" for label in labels]
    result["band_low"] = edges[labels]
    result["band_high"] = edges[labels + 1]
    return result


def analyse_system(row: dict[str, str], config: dict, edges: np.ndarray, target: str, parts: Path) -> str:
    system = row["system_id"]; settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    full_audit = pd.read_parquet(source); pairs = _pair_sets_from_audit(full_audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    log_pf = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    distances = {metric: {
        replica: pf_pair_distance(log_pf, frame.left_frame.to_numpy(), frame.right_frame.to_numpy(), metric)
        for replica, frame in pairs.items()
    } for metric in METRICS}
    vectors = {
        replica: absolute_change_vectors(log_pf, frame.left_frame.to_numpy(), frame.right_frame.to_numpy())
        for replica, frame in pairs.items()
    }
    summaries = []
    for fit_replica, calibration_replica, test_replica in ordered_assignments():
        fit_pairs = pairs[fit_replica]; calibration_pairs = pairs[calibration_replica]; test_pairs = pairs[test_replica]
        fit_mask, validation_mask = frame_disjoint_pair_split(
            fit_pairs.left_frame.to_numpy(), fit_pairs.right_frame.to_numpy()
        )
        fit_target = fit_pairs[target].to_numpy(); calibration_target = calibration_pairs[target].to_numpy(); test_target = test_pairs[target].to_numpy()
        assignment_audit = full_audit[
            (full_audit.fit_replica == fit_replica) & (full_audit.calibration_replica == calibration_replica)
            & (full_audit.test_replica == test_replica) & (full_audit.target == target)
        ].reset_index(drop=True)
        assignment_audit = apply_global_bands(assignment_audit, edges)
        candidates = {
            f"fixed_logpf_scalar_{metric}_gaussian": _scalar_fit_predict(
                distance[fit_replica], distance[calibration_replica], distance[test_replica],
                fit_target, fit_mask, validation_mask, settings,
            ) for metric, distance in distances.items()
        }
        candidates[RIDGE_MODEL] = _ridge_fit_predict(
            vectors[fit_replica], vectors[calibration_replica], vectors[test_replica],
            fit_target, fit_mask, validation_mask, settings["vector_audit"]["ridge_alphas"],
        )
        for model, candidate in candidates.items():
            scores = np.abs(calibration_target - candidate.calibration_prediction)
            marginal_q = finite_conformal_quantile(scores, strict["coverage"])
            mondrian_q, _ = mondrian_quantiles(
                candidate.fit_prediction, candidate.calibration_prediction, scores,
                candidate.test_prediction, strict["mondrian_bins"], strict["coverage"], marginal_q,
            )
            for calibration, width in (("marginal", np.full(len(test_target), marginal_q)), ("mondrian", mondrian_q)):
                sigma = np.maximum(width / GAUSSIAN_Q90, np.finfo(float).eps)
                density = lambda mask, low, high, bins, p=candidate.test_prediction, s=sigma: gaussian_mass(
                    p[mask], s[mask], low, high, bins
                )
                summaries.extend(summarize_model(
                    assignment_audit, model, calibration, candidate.test_prediction,
                    np.maximum(0.0, candidate.test_prediction - width), candidate.test_prediction + width,
                    density, settings,
                ))
    atomic_parquet(pd.DataFrame(summaries), parts / f"{system}.summary.parquet")
    return system


def valid_checkpoint(parts: Path, system: str) -> bool:
    path = parts / f"{system}.summary.parquet"
    if not path.exists(): return False
    try: frame = pd.read_parquet(path)
    except Exception: return False
    expected = {f"fixed_logpf_scalar_{metric}_gaussian" for metric in METRICS} | {RIDGE_MODEL}
    assignments = frame[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates()
    return set(frame.system_id) == {system} and set(frame.model) == expected and len(assignments) == 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--target", choices=("w1", "rmsd"), default="w1")
    args = parser.parse_args(); config = load_config(); all_systems = load_systems(); systems = all_systems[:args.limit]
    edges, pairs = global_target_edges(all_systems, args.target)
    checkpoint = "checkpoint15_global_w1" if args.target == "w1" else "checkpoint16_global_rmsd"
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / checkpoint
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyse_system, row, config, edges, args.target, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[{resumed + index}/{len(systems)}] {future.result()} global-W1 checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing global-W1 systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / f"global_{args.target}_assignment_summary.parquet", index=False)
    atomic_yaml(output / f"global_{args.target}_edges.yaml", {
        "target": args.target,
        "quantile_probabilities": RMSD_QUANTILES.tolist(), "edges_angstrom": edges.tolist(),
        "unique_sampled_pairs": pairs, "systems": len(all_systems),
    })


if __name__ == "__main__":
    main()
