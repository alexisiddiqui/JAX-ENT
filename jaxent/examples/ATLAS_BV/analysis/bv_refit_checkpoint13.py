"""Checkpoint 13: strict per-system BV coefficient refit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_checkpoint11 import TARGETS, select_ridge, summarize_model
from jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 import frame_disjoint_pair_split
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, mondrian_quantiles, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import GAUSSIAN_Q90, _pair_sets_from_audit, gaussian_mass
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import normalized_mae


MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)
MODEL = "system_refit_logpf_ridge_gaussian"
COEFFICIENT_SELECTION_ALPHA = 10000.0


def contact_change_vectors(
    heavy: np.ndarray, acceptor: np.ndarray, pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    left = pairs.left_frame.to_numpy(); right = pairs.right_frame.to_numpy()
    return (
        (heavy[:, left] - heavy[:, right]).T.astype(np.float32),
        (acceptor[:, left] - acceptor[:, right]).T.astype(np.float32),
    )


def bv_change_vectors(
    heavy_change: np.ndarray, acceptor_change: np.ndarray, bc: float, bh: float,
) -> np.ndarray:
    return np.abs(bc * heavy_change + bh * acceptor_change)


def select_coefficients(
    heavy_change: np.ndarray, acceptor_change: np.ndarray,
    targets: dict[str, np.ndarray], fit_mask: np.ndarray, validation_mask: np.ndarray,
    default_bc: float, default_bh: float,
) -> tuple[float, float, float, str]:
    """Select one physical coefficient pair using joint A-only RMSD+W1 loss."""
    if not validation_mask.any():
        return default_bc, default_bh, float("nan"), "prespecified_no_disjoint_pair_split"
    scored = []
    for bc_multiplier in MULTIPLIERS:
        for bh_multiplier in MULTIPLIERS:
            bc = default_bc * bc_multiplier; bh = default_bh * bh_multiplier
            features = bv_change_vectors(heavy_change, acceptor_change, bc, bh)
            losses = []
            for target_name in TARGETS:
                target = targets[target_name]
                model = Ridge(alpha=COEFFICIENT_SELECTION_ALPHA).fit(
                    features[fit_mask], target[fit_mask]
                )
                losses.append(normalized_mae(
                    target[validation_mask], model.predict(features[validation_mask]), target[fit_mask]
                ))
            default_distance = float(np.hypot(np.log(bc_multiplier), np.log(bh_multiplier)))
            scored.append((float(np.mean(losses)), default_distance, bc, bh))
    loss, _, bc, bh = min(scored, key=lambda item: (item[0], item[1], item[2], item[3]))
    return bc, bh, loss, "frame_disjoint_joint_target"


def analyse_system(row: dict[str, str], config: dict, parts: Path) -> str:
    system = row["system_id"]; settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]; vector = settings["vector_audit"]
    default_bc = float(config["protocol"]["bv_bc"]); default_bh = float(config["protocol"]["bv_bh"])
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    audit = pd.read_parquet(source); pairs = _pair_sets_from_audit(audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    changes = {
        replica: contact_change_vectors(heavy, acceptor, frame) for replica, frame in pairs.items()
    }
    summaries, hyperparameters = [], []
    for assignment_index, (fit_replica, calibration_replica, test_replica) in enumerate(ordered_assignments()):
        fit_pairs = pairs[fit_replica]; calibration_pairs = pairs[calibration_replica]
        test_pairs = pairs[test_replica]
        fit_mask, validation_mask = frame_disjoint_pair_split(
            fit_pairs.left_frame.to_numpy(), fit_pairs.right_frame.to_numpy()
        )
        fit_targets = {target: fit_pairs[target].to_numpy() for target in TARGETS}
        bc, bh, coefficient_loss, selection_mode = select_coefficients(
            *changes[fit_replica], fit_targets, fit_mask, validation_mask, default_bc, default_bh
        )
        fit_x = bv_change_vectors(*changes[fit_replica], bc, bh)
        calibration_x = bv_change_vectors(*changes[calibration_replica], bc, bh)
        test_x = bv_change_vectors(*changes[test_replica], bc, bh)
        for target_index, target_name in enumerate(TARGETS):
            fit_target = fit_targets[target_name]
            calibration_target = calibration_pairs[target_name].to_numpy()
            test_target = test_pairs[target_name].to_numpy()
            assignment_audit = audit[
                (audit.fit_replica == fit_replica) & (audit.calibration_replica == calibration_replica)
                & (audit.test_replica == test_replica) & (audit.target == target_name)
            ].reset_index(drop=True)
            alpha, ridge_loss = select_ridge(
                fit_x, fit_target, fit_mask, validation_mask, vector["ridge_alphas"]
            )
            ridge = Ridge(alpha=alpha).fit(fit_x, fit_target)
            fit_prediction = np.maximum(0.0, ridge.predict(fit_x))
            calibration_prediction = np.maximum(0.0, ridge.predict(calibration_x))
            test_prediction = np.maximum(0.0, ridge.predict(test_x))
            scores = np.abs(calibration_target - calibration_prediction)
            marginal_q = finite_conformal_quantile(scores, strict["coverage"])
            mondrian_q, _ = mondrian_quantiles(
                fit_prediction, calibration_prediction, scores, test_prediction,
                strict["mondrian_bins"], strict["coverage"], marginal_q,
            )
            for calibration, width in (("marginal", np.full(len(test_target), marginal_q)),
                                       ("mondrian", mondrian_q)):
                sigma = np.maximum(width / GAUSSIAN_Q90, np.finfo(float).eps)
                density = lambda mask, low, high, bins, p=test_prediction, s=sigma: gaussian_mass(
                    p[mask], s[mask], low, high, bins
                )
                summaries.extend(summarize_model(
                    assignment_audit, MODEL, calibration, test_prediction,
                    np.maximum(0.0, test_prediction - width), test_prediction + width,
                    density, settings,
                ))
            hyperparameters.append({
                "system_id": system, "fit_replica": fit_replica,
                "calibration_replica": calibration_replica, "test_replica": test_replica,
                "target": target_name, "bc": bc, "bh": bh,
                "bc_multiplier": bc / default_bc, "bh_multiplier": bh / default_bh,
                "coefficient_joint_validation_loss": coefficient_loss,
                "ridge_alpha": alpha, "ridge_validation_loss": ridge_loss,
                "selection_mode": selection_mode,
            })
    atomic_parquet(pd.DataFrame(summaries), parts / f"{system}.summary.parquet")
    atomic_parquet(pd.DataFrame(hyperparameters), parts / f"{system}.hyperparameters.parquet")
    return system


def valid_checkpoint(parts: Path, system: str) -> bool:
    paths = [parts / f"{system}.{suffix}.parquet" for suffix in ("summary", "hyperparameters")]
    if not all(path.exists() for path in paths): return False
    try:
        summary = pd.read_parquet(paths[0], columns=["distribution_recovery", "coverage_90"])
        hyper = pd.read_parquet(paths[1])
    except Exception: return False
    assignments = set(map(tuple, hyper[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates().to_numpy()))
    return (np.isfinite(summary.to_numpy()).all() and set(hyper.system_id) == {system}
            and assignments == set(ordered_assignments()) and set(hyper.target) == set(TARGETS)
            and len(hyper) == 12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint13_bv_refit"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            system = future.result(); print(f"[{resumed + index}/{len(systems)}] {system} BV refit checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoint-13 systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    hyper = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.hyperparameters.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / "bv_refit_assignment_summary.parquet", index=False)
    hyper.to_parquet(output / "bv_refit_hyperparameters.parquet", index=False)
    atomic_yaml(output / "checkpoint13_run.yaml", {"checkpoint": "13", "status": "measurement_complete", "systems": len(systems)})


if __name__ == "__main__":
    main()
