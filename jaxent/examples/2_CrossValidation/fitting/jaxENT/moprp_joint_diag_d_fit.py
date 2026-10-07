#!/usr/bin/env python3
"""Stage 6: joint-BV fitting with a frozen per-ensemble ``diag(D)`` target.

This is the scoped reopening of the BV mean model after the D-only Stage-5 run:
shared non-negative ``(Bc, Bh)`` are fitted jointly with one frame-weight vector
per ensemble.  The target is loaded once from the selected 2026-07-24
``scaled_published`` candidate and is never re-inferred during optimization.

The fit is blind.  NMR states and target populations are revealed only after all
losses, mean gates, and Stage-5 cliff comparisons have been computed; recovery,
decoy mass, and ESS are post-fit diagnostics only.  The refined run uses three
held-out interleaved peptide-by-timepoint split pairs and writes both split-level
rows and their mean/std aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import _moprp_recovery_common as common
import moprp_covariance_recovery as R
from joint_diag_d_diagnostics import (
    fixed_ess_weights,
    load_frame_cluster_labels,
    per_state_weight_diagnostics,
    plot_joint_diag_d_fit,
)
from jaxent.src.analysis.pf_variance import (
    kl_to_uniform,
    scale_free_log_ratio_profile_loss,
    weighted_population_covariance,
)
from jaxent.src.analysis.state_population import (
    FULL_STATE_SUPPORT,
    log_ratio_profile_loss,
    state_populations,
    strict_recovery_percent,
)


HERE = Path(__file__).resolve().parent
TARGET_ARTIFACT = HERE / "_moprp_target_variance_scaled_published_20260724"
STAGE5_RAW = HERE / "_moprp_diag_d_reweighting" / "diag_d_reweighting_raw.csv"

REF_BC, REF_BH = 0.35, 2.0
GAMMAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
ETAS = (0.0, 0.01, 0.022, 0.046, 0.1)
ESS_TARGETS = (1.0, 5.0, 15.0, 30.0, 60.0)
FIXED_ESS_GAMMAS = GAMMAS
MEAN_GATE_FACTOR = 1.05
STEPS, LR, N_START = 2000, 0.03, 5
TARGET_STATES = ("Folded", "PUF1", "PUF2")
ARMS = ("diag_d_absolute", "diag_d_scalefree")
N_SPLITS = 3
PEPTIDE_FOLDS = 3
TIMEPOINT_FOLDS = 5


def _coeffs_from_theta(theta):
    """Softplus -> strictly positive, auditable BV coefficients."""

    return jax.nn.softplus(theta[0]), jax.nn.softplus(theta[1])


def _inv_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def _interleaved_folds(size: int, count: int) -> tuple[np.ndarray, ...]:
    """Return the repository's interleaved fold convention."""

    if size < count:
        raise ValueError("cannot create more non-empty folds than observations")
    return tuple(np.arange(offset, size, count, dtype=int) for offset in range(count))


def _split_specs(n_peptides: int, n_timepoints: int) -> tuple[dict[str, object], ...]:
    """Build three disjoint diagonal peptide×timepoint holdouts.

    The peptide axis follows the target-variance sweep's three interleaved folds;
    the time axis follows its five interleaved folds.  The first three diagonal
    pairs provide three independent split replicates while preserving the exact
    Cartesian train/validation mask convention used by that sweep.
    """

    peptide_folds = _interleaved_folds(n_peptides, PEPTIDE_FOLDS)
    time_folds = _interleaved_folds(n_timepoints, TIMEPOINT_FOLDS)
    specs = []
    for split_index in range(N_SPLITS):
        val_peptides = peptide_folds[split_index]
        val_times = time_folds[split_index]
        specs.append(
            {
                "split": split_index,
                "peptide_fold": split_index,
                "time_fold": split_index,
                "train_peptides": np.setdiff1d(np.arange(n_peptides), val_peptides),
                "val_peptides": val_peptides,
                "train_times": np.setdiff1d(np.arange(n_timepoints), val_times),
                "val_times": val_times,
            }
        )
    return tuple(specs)


def _load_cell(ensemble: str, artifact_dir: Path) -> dict:
    """Load one blind cell and its frozen target; no NMR fields are read."""

    inputs = common.load_blinded_ensemble_inputs(ensemble)
    target, target_info = common.load_selected_diag_d_target(
        artifact_dir, ensemble, inputs.feature_residue_ids
    )

    # Peptide 1 is held out exactly as in the D-only runners before any split is
    # constructed.  The target is a residue profile, so it retains the full
    # feature-residue shape.
    keep = np.ones(inputs.mapping.shape[0], dtype=bool)
    keep[common.PEPTIDE1_INDEX] = False
    mapping = jnp.asarray(inputs.mapping[keep])
    observed = jnp.asarray(inputs.observed_uptake[keep])
    k_ints = jnp.asarray(inputs.k_ints)
    timepoints = jnp.asarray(inputs.timepoints)

    return {
        "ensemble": ensemble,
        "inputs": inputs,
        "heavy": jnp.asarray(inputs.heavy_contacts),
        "acceptor": jnp.asarray(inputs.acceptor_contacts),
        "mapping": mapping,
        "observed": observed,
        "k_ints": k_ints,
        "timepoints": timepoints,
        "target": jnp.asarray(target),
        "target_info": target_info,
        "peptide_ids": np.asarray(inputs.peptide_ids[keep]),
        "n_frames": inputs.n_frames,
    }


def _split_cell(cell: dict, spec: dict[str, object]) -> dict:
    """Materialize one blind train/held-out cell from a peptide×time split."""

    train_peptides = np.asarray(spec["train_peptides"], dtype=int)
    val_peptides = np.asarray(spec["val_peptides"], dtype=int)
    train_times = np.asarray(spec["train_times"], dtype=int)
    val_times = np.asarray(spec["val_times"], dtype=int)
    train_mapping = cell["mapping"][train_peptides]
    val_mapping = cell["mapping"][val_peptides]
    train_observed = cell["observed"][np.ix_(train_peptides, train_times)]
    val_observed = cell["observed"][np.ix_(val_peptides, val_times)]
    train_timepoints = cell["timepoints"][train_times]
    val_timepoints = cell["timepoints"][val_times]
    uniform = jnp.full(cell["n_frames"], 1.0 / cell["n_frames"])
    ref_log_pf = REF_BC * cell["heavy"] + REF_BH * cell["acceptor"]
    mean_ref = float(
        R._mean_mse(
            R._predict_uptake(
                ref_log_pf,
                cell["k_ints"],
                train_timepoints,
                train_mapping,
                uniform,
            ).T,
            train_observed,
        )
    )
    if not np.isfinite(mean_ref) or mean_ref <= 0.0:
        raise ValueError(f"{cell['ensemble']}: invalid split mean reference {mean_ref}")
    return {
        **cell,
        "split": int(spec["split"]),
        "peptide_fold": int(spec["peptide_fold"]),
        "time_fold": int(spec["time_fold"]),
        "train_mapping": train_mapping,
        "train_observed": train_observed,
        "train_timepoints": train_timepoints,
        "val_mapping": val_mapping,
        "val_observed": val_observed,
        "val_timepoints": val_timepoints,
        "train_peptides": train_peptides,
        "val_peptides": val_peptides,
        "train_times": train_times,
        "val_times": val_times,
        "mean_ref": mean_ref,
    }


def _split_cells(base_cells: dict[str, dict], spec: dict[str, object]) -> dict[str, dict]:
    return {name: _split_cell(cell, spec) for name, cell in base_cells.items()}


def _predict_mean(cell: dict, log_pf, weights, partition: str = "train"):
    mapping = cell.get(f"{partition}_mapping", cell["mapping"])
    timepoints = cell.get(f"{partition}_timepoints", cell["timepoints"])
    return R._predict_uptake(
        log_pf, cell["k_ints"], timepoints, mapping, weights
    ).T


def _diag_d(cell: dict, log_pf, weights):
    rates = cell["k_ints"][:, None] * jnp.exp(-log_pf)
    return jnp.diag(weighted_population_covariance(rates, weights))


def _diag_d_loss(predicted_d, target, arm: str, *, finite_floor: bool = False):
    """Evaluate the selected profile loss, optionally flooring zero variances."""

    if finite_floor:
        # At the nominal E=1 open boundary, finite-precision softmax may retain
        # only two frames. Contact-identical residues can then have exact zero
        # variance even though total ESS is 1+δ. The log-profile objective is
        # undefined at zero, so use the same absolute scale as the covariance
        # utilities' numerical ridge.
        predicted_d = jnp.maximum(predicted_d, 1e-12)
    if arm == "diag_d_absolute":
        return log_ratio_profile_loss(predicted_d, target)
    return scale_free_log_ratio_profile_loss(predicted_d, target)


def _loss_fn(
    cells: dict[str, dict],
    gamma: float,
    eta: float,
    arm: str,
    diversity_control: str = "kl",
):
    """Build the differentiable joint objective used by both arms."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if diversity_control not in {"kl", "fixed_ess"}:
        raise ValueError(f"unknown diversity control {diversity_control!r}")
    if gamma < 0.0 or eta < 0.0:
        raise ValueError("gamma and eta must be non-negative")
    names = tuple(cells)

    def loss(params):
        bc, bh = _coeffs_from_theta(params["theta"])
        total = jnp.asarray(0.0)
        for name in names:
            cell = cells[name]
            weights = (
                fixed_ess_weights(params["logits"][name], eta)
                if diversity_control == "fixed_ess"
                else jax.nn.softmax(params["logits"][name])
            )
            log_pf = bc * cell["heavy"] + bh * cell["acceptor"]
            mean = R._mean_mse(
                _predict_mean(cell, log_pf, weights), cell.get("train_observed", cell["observed"])
            )
            mean = mean / cell["mean_ref"]
            predicted_d = _diag_d(cell, log_pf, weights)
            d_loss = _diag_d_loss(
                predicted_d,
                cell["target"],
                arm,
                finite_floor=diversity_control == "fixed_ess",
            )
            total = total + mean + gamma * d_loss
            if diversity_control == "kl":
                total = total + eta * kl_to_uniform(weights)
        return total

    return loss


def _initial_params(cells: dict[str, dict], seed: int):
    rng = np.random.default_rng(100 + seed)
    theta0 = np.asarray([_inv_softplus(REF_BC), _inv_softplus(REF_BH)])
    return {
        "theta": jnp.asarray(theta0 if seed == 0 else theta0 + rng.normal(scale=0.1, size=2)),
        "logits": {
            name: jnp.asarray(
                np.zeros(cell["n_frames"])
                if seed == 0
                else rng.normal(scale=0.01, size=cell["n_frames"])
            )
            for name, cell in cells.items()
        },
    }


def _run_with_diagnostics(
    cells: dict[str, dict],
    gamma: float,
    eta: float,
    arm: str,
    steps: int = STEPS,
    lr: float = LR,
    n_start: int = N_START,
    diversity_control: str = "kl",
):
    """Fit one arm/grid cell and retain the five-start ESS diagnostic."""

    loss = _loss_fn(cells, gamma, eta, arm, diversity_control)
    grad_fn = jax.vmap(jax.grad(loss))
    objective_fn = jax.vmap(loss)
    optimizer = optax.adam(lr)

    @jax.jit
    def optimize_all_starts(params):
        state = optimizer.init(params)

        def step(carry, _):
            current_params, current_state = carry
            gradients = grad_fn(current_params)
            updates, current_state = optimizer.update(
                gradients, current_state, current_params
            )
            current_params = optax.apply_updates(current_params, updates)
            return (current_params, current_state), None

        (params, _), _ = jax.lax.scan(step, (params, state), None, length=steps)
        return params, objective_fn(params)

    start_params = [_initial_params(cells, seed) for seed in range(n_start)]
    params = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values), *start_params
    )
    params, objectives = optimize_all_starts(params)
    objectives = np.asarray(objectives)
    best_start = int(np.argmin(objectives))
    best_params = jax.tree_util.tree_map(lambda value: value[best_start], params)
    restart_rows = []
    for seed in range(n_start):
        objective = float(objectives[seed])
        for name in cells:
            weights = np.asarray(
                fixed_ess_weights(params["logits"][name][seed], eta)
                if diversity_control == "fixed_ess"
                else jax.nn.softmax(params["logits"][name][seed])
            )
            restart_rows.append(
                {
                    "restart": seed,
                    "ensemble": name,
                    "objective": objective,
                    "ess": float(1.0 / np.sum(np.square(weights))),
                }
            )
    diagnostics = {}
    for name in cells:
        ess = np.asarray(
            [row["ess"] for row in restart_rows if row["ensemble"] == name], dtype=float
        )
        best_ess = float(ess[best_start])
        diagnostics[name] = {
            "best_start": best_start,
            "best_objective": float(objectives[best_start]),
            "restart_ess_min": float(ess.min()),
            "restart_ess_max": float(ess.max()),
            "restart_ess_spread": float(ess.max() - ess.min()),
            "restart_ess_std": float(ess.std()),
            "best_objective_is_lowest_ess": bool(
                np.isclose(best_ess, ess.min(), rtol=0.0, atol=1e-10)
            ),
        }
    return {"params": best_params, "diagnostics": diagnostics, "restart_rows": restart_rows}


def _run(
    cells: dict[str, dict],
    gamma: float,
    eta: float,
    arm: str,
    steps: int = STEPS,
    lr: float = LR,
    n_start: int = N_START,
):
    """Fit one arm/grid cell and return the best parameter tree."""

    return _run_with_diagnostics(cells, gamma, eta, arm, steps, lr, n_start)["params"]


def _payload_id(
    split: int,
    arm: str,
    gamma: float,
    control: float,
    ensemble: str,
) -> str:
    """Return the stable row key used by first-party fitted-array artifacts."""

    return (
        f"split_{int(split)}__{arm}__gamma_{float(gamma):.17g}"
        f"__control_{float(control):.17g}__{ensemble}"
    )


def _report(
    cells: dict[str, dict],
    fit: dict,
    gamma: float,
    eta: float,
    arm: str,
    diversity_control: str = "kl",
    payload: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Report blind fit metrics; recovery fields are filled after NMR reveal."""

    params = fit["params"]
    bc, bh = (float(value) for value in _coeffs_from_theta(params["theta"]))
    rows = []
    for name, cell in cells.items():
        weights = np.asarray(
            fixed_ess_weights(params["logits"][name], eta)
            if diversity_control == "fixed_ess"
            else jax.nn.softmax(params["logits"][name])
        )
        log_pf = bc * cell["heavy"] + bh * cell["acceptor"]
        val_prediction = _predict_mean(
            cell, log_pf, jnp.asarray(weights), partition="val"
        )
        val_mse = float(
            R._mean_mse(
                val_prediction, cell.get("val_observed", cell["observed"])
            )
        )
        predicted_d = _diag_d(cell, log_pf, jnp.asarray(weights))
        d_loss = _diag_d_loss(
            predicted_d,
            cell["target"],
            arm,
            finite_floor=diversity_control == "fixed_ess",
        )
        payload_id = _payload_id(
            cell.get("split", 0), arm, gamma, eta, name
        )
        if payload is not None:
            payload[f"{payload_id}__weights"] = weights
            payload[f"{payload_id}__val_prediction"] = np.asarray(val_prediction)
        rows.append(
            {
                "arm": arm,
                "gamma": gamma,
                "eta": eta,
                "diversity_axis": eta,
                "ensemble": name,
                "payload_id": payload_id,
                "split": cell.get("split", 0),
                "peptide_fold": cell.get("peptide_fold", 0),
                "time_fold": cell.get("time_fold", 0),
                "bc": bc,
                "bh": bh,
                "val_mse": val_mse,
                "val_diag_d_loss": float(d_loss),
                "ess": float(1.0 / np.sum(np.square(weights))),
                **fit["diagnostics"][name],
            }
        )
        if diversity_control == "fixed_ess":
            rows[-1]["ess_target"] = eta
    return rows


def _add_recovery_metrics(
    rows: list[dict],
    base_cells: dict[str, dict],
    fits_by_cell: dict[tuple, dict],
    diversity_control: str = "kl",
) -> None:
    """Add NMR-derived diagnostics only after blind rows and gates exist."""

    revealed = {}
    for name, cell in base_cells.items():
        inputs = cell["inputs"]
        states, support, targets, reference_weights = common.reveal_nmr_reference(
            name, expected_frames=inputs.n_frames
        )
        cluster_labels = load_frame_cluster_labels(
            common.CLUSTER_CSV,
            common.ENSEMBLES[name],
            expected_frames=inputs.n_frames,
        )
        revealed[name] = (states, support, targets, reference_weights, cluster_labels)

    for row in rows:
        fit = fits_by_cell[(row["split"], row["arm"], row["gamma"], row["eta"])]
        params = fit["params"]
        name = row["ensemble"]
        states, support, targets, _, cluster_labels = revealed[name]
        weights = np.asarray(
            fixed_ess_weights(params["logits"][name], row["eta"])
            if diversity_control == "fixed_ess"
            else jax.nn.softmax(params["logits"][name])
        )
        populations = np.asarray(state_populations(weights, states, support))
        row["recovery"] = float(strict_recovery_percent(weights, states, support, targets))
        row["decoy"] = float(
            sum(
                populations[FULL_STATE_SUPPORT.index(state)]
                for state in support
                if state not in TARGET_STATES
            )
        )
        row.update(
            per_state_weight_diagnostics(
                weights,
                states,
                cluster_labels=cluster_labels,
                support=FULL_STATE_SUPPORT,
            )
        )


def _aggregate_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split rows while retaining pass fractions and finite error bars."""

    keys = ["arm", "gamma", "eta", "ensemble"]
    if "ess_target" in frame.columns:
        keys.insert(3, "ess_target")
    if "diversity_axis" in frame.columns:
        keys.insert(-1, "diversity_axis")
    metrics = [
        "bc",
        "bh",
        "val_mse",
        "val_diag_d_loss",
        "recovery",
        "ess",
        "decoy",
        "mean_gate_reference_mse",
        "mean_gate_passed",
        "restart_ess_spread",
        "restart_ess_std",
        "best_objective_is_lowest_ess",
    ]
    metrics.extend(
        field
        for state in FULL_STATE_SUPPORT
        for field in (f"ess_{state}", f"mass_{state}")
        if field in frame.columns
    )
    metrics = tuple(metrics)
    if "dominant_weight" in frame.columns:
        metrics = (*metrics, "dominant_weight")
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_replicates"] = int(len(group))
        row["split_indices"] = ",".join(
            str(int(value)) for value in sorted(group["split"].unique())
        )
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        if "dominant_cluster" in group:
            row["dominant_cluster"] = group["dominant_cluster"].mode(dropna=False).iloc[0]
        row["mean_gate_pass_fraction"] = row["mean_gate_passed"]
        rows.append(row)
    return pd.DataFrame(rows)


def _stage5_cliff_comparison(rows: pd.DataFrame, stage5_raw: Path) -> pd.DataFrame:
    """Compare the joint pass boundary to Stage-5's fixed scaled-published sweep."""

    joint = {}
    for (arm, ensemble), group in rows.groupby(["arm", "ensemble"]):
        # The reported cliff is conservative: all three held-out replicates
        # must pass. Partial pass fractions remain visible in the aggregate CSV.
        eligible = group[group["mean_gate_passed"] >= 1.0 - 1e-12]
        joint[(arm, ensemble)] = float(eligible.gamma.max()) if not eligible.empty else np.nan

    output = []
    if stage5_raw.exists():
        fixed_all = pd.read_csv(stage5_raw)
        fixed_all = fixed_all[fixed_all["coefficient"] == "scaled_published"]
        fixed = fixed_all[fixed_all["arm"].isin(ARMS)]
        for (arm, ensemble), group in fixed.groupby(["arm", "ensemble"]):
            # Stage 5's raw rows carry absolute MSE and its baseline row carries
            # the baseline MSE; use the explicit baseline cell when available.
            base_rows = fixed_all[
                (fixed_all.arm == "baseline") & (fixed_all.ensemble == ensemble)
            ]
            baseline_mse = float(base_rows["val_mse"].iloc[0])
            eligible = group[group["val_mse"] <= MEAN_GATE_FACTOR * baseline_mse]
            stage5_boundary = float(eligible.gamma.max()) if not eligible.empty else np.nan
            output.append(
                {
                    "arm": arm,
                    "ensemble": ensemble,
                    "joint_gamma_pass_boundary": joint.get((arm, ensemble), np.nan),
                    "stage5_fixed_gamma_pass_boundary": stage5_boundary,
                    "joint_minus_stage5_gamma": joint.get((arm, ensemble), np.nan) - stage5_boundary,
                    "joint_gate_requirement": "all_replicates_pass",
                    "stage5_source": str(stage5_raw),
                }
            )
    else:
        for key, boundary in joint.items():
            arm, ensemble = key
            output.append(
                {
                    "arm": arm,
                    "ensemble": ensemble,
                    "joint_gamma_pass_boundary": boundary,
                    "stage5_fixed_gamma_pass_boundary": np.nan,
                    "joint_minus_stage5_gamma": np.nan,
                    "joint_gate_requirement": "all_replicates_pass",
                    "stage5_source": str(stage5_raw),
                }
            )
    return pd.DataFrame(output)


def _fixed_ess_frontier(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the definitive Experiment-1a gate-ratio frontier and verdict."""

    rows = frame.copy()
    rows["gate_ratio"] = rows["val_mse"] / rows["mean_gate_reference_mse"]
    frontier = (
        rows.groupby(["arm", "gamma", "ess_target", "ensemble"], sort=True)["gate_ratio"]
        .agg(["mean", lambda values: values.std(ddof=0)])
        .reset_index()
        .rename(columns={"mean": "gate_ratio", "<lambda_0>": "gate_ratio_std"})
    )
    groups = []
    for key, group in frontier.groupby(["arm", "gamma", "ensemble"], sort=True):
        ordered = group.sort_values("ess_target")
        ratios = ordered["gate_ratio"].to_numpy()
        controls = ordered["ess_target"].to_numpy()
        monotone = bool(np.all(np.diff(ratios) >= -1e-10))
        passing = ordered[ordered["gate_ratio"] <= MEAN_GATE_FACTOR]
        max_passing = float(passing["ess_target"].max()) if not passing.empty else None
        crossing = None
        above = np.flatnonzero(ratios > MEAN_GATE_FACTOR)
        if above.size:
            right = int(above[0])
            if right == 0:
                crossing = float(controls[0])
            else:
                left = right - 1
                fraction = (MEAN_GATE_FACTOR - ratios[left]) / (
                    ratios[right] - ratios[left]
                )
                crossing = float(
                    controls[left] + fraction * (controls[right] - controls[left])
                )
        groups.append(
            {
                "arm": key[0],
                "gamma": float(key[1]),
                "ensemble": key[2],
                "monotone_non_decreasing": monotone,
                "maximum_gate_valid_ess": max_passing,
                "interpolated_1p05_crossing_ess": crossing,
            }
        )
    healthy = frontier[
        (frontier["ess_target"] >= 5.0)
        & (frontier["gate_ratio"] <= MEAN_GATE_FACTOR)
    ]
    anchor = rows[(rows["ess_target"] == 1.0) & (rows["gamma"] == 0.0)]
    anchor_all_folded = bool((anchor["dominant_cluster"] == 0).all())
    anchor_all_gate_valid = bool(anchor["mean_gate_passed"].all())
    anchor_maximum_decoy = float(anchor["decoy"].max())
    all_monotone = bool(
        all(group["monotone_non_decreasing"] for group in groups)
    )
    continuity_verified = bool(
        anchor_all_gate_valid and anchor_all_folded and anchor_maximum_decoy <= 0.01
    )
    healthy_exists = bool(not healthy.empty)
    definitive = bool(all_monotone and continuity_verified)
    summary = {
        "gate_threshold": MEAN_GATE_FACTOR,
        "healthy_ess_definition": "ess_target >= 5",
        "healthy_gate_valid_point_exists": healthy_exists,
        "healthy_gate_valid_cells": healthy.to_dict(orient="records"),
        "all_frontiers_monotone_non_decreasing": all_monotone,
        "frontier_groups": groups,
        "ess_1_anchor": {
            "all_realized_ess_exact": bool(
                np.all(np.abs(anchor["ess"] - 1.0) <= 1e-3)
            ),
            "gamma": 0.0,
            "all_gate_valid": anchor_all_gate_valid,
            "all_dominant_clusters_folded": anchor_all_folded,
            "maximum_decoy_mass": anchor_maximum_decoy,
            "minimum_folded_mass": (
                float(anchor["mass_Folded"].min())
                if "mass_Folded" in anchor.columns
                else None
            ),
            "continuity_with_experiment_1_verified": continuity_verified,
        },
        "definitive_frontier_checks_passed": definitive,
        "scientific_verdict": (
            "A healthy gate-valid ESS>=5 point was observed."
            if healthy_exists
            else (
                "No healthy gate-valid ESS>=5 aggregate point was observed, but the "
                "hard Pareto wall is not established because the preregistered "
                "monotonicity and/or E=1 continuity checks failed."
                if not definitive
                else "No healthy gate-valid ESS>=5 point exists on the validated frontier."
            )
        ),
    }
    return frontier, summary


def _write_final_artifacts(
    frame: pd.DataFrame,
    restart_rows,
    payload: dict[str, np.ndarray],
    base_cells: dict[str, dict],
    specs: tuple[dict[str, object], ...],
    args: argparse.Namespace,
) -> None:
    """Add held-out gates and write the merged, post-fit artifact set."""

    references = frame[frame.gamma == 0.0].groupby(["split", "ensemble"])["val_mse"].min()
    frame["mean_gate_reference_mse"] = [
        references.loc[(row.split, row.ensemble)] for row in frame.itertuples()
    ]
    frame["mean_gate_passed"] = (
        frame["val_mse"] <= MEAN_GATE_FACTOR * frame["mean_gate_reference_mse"]
    )
    axis_column = "ess_target" if "ess_target" in frame.columns else "eta"
    frame = frame.sort_values(
        ["split", "arm", "gamma", axis_column, "ensemble"]
    ).reset_index(drop=True)
    frame.to_csv(args.output_dir / "joint_diag_d_fit_replicates.csv", index=False)
    aggregate = _aggregate_replicates(frame)
    aggregate.to_csv(args.output_dir / "joint_diag_d_fit.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(args.output_dir / "restart_diagnostics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "joint_diag_d_fit_payload.npz",
        **payload,
    )

    comparison = _stage5_cliff_comparison(aggregate, args.stage5_raw)
    comparison.to_csv(args.output_dir / "cliff_comparison.csv", index=False)
    frontier_summary = None
    if args.diversity_control == "fixed_ess":
        frontier, frontier_summary = _fixed_ess_frontier(frame)
        frontier.to_csv(args.output_dir / "fixed_ess_frontier.csv", index=False)
        (args.output_dir / "fixed_ess_frontier_summary.json").write_text(
            json.dumps(frontier_summary, indent=2) + "\n"
        )
    first_cell = next(iter(base_cells.values()))
    manifest = {
        "artifact_type": "moprp_joint_bv_diag_d_fit_replicated",
        "phase": "Stage 6 joint-BV phase",
        "guardrail_relaxed": "shared BV (Bc, Bh) coefficient freeze",
        "guardrails_retained": [
            "blind inference and selection",
            "per-ensemble cells, never pooled",
            "peptide 1 held out",
            "frozen scaled_published 2026-07-24 target",
            "recovery, decoy, and ESS post-hoc only",
        ],
        "nmr_used_for_loss_or_selection": False,
        "nmr_used_for_validation_diagnostics": True,
        "target_artifact": str(args.target_artifact),
        "target_source": "selected scaled_published 2026-07-24 HDX-only candidate; residue profile retained",
        "target_manifest": [
            {"ensemble": name, **cell["target_info"]} for name, cell in base_cells.items()
        ],
        "arms": list(args.arms),
        "gammas": list(args.gammas),
        "diversity_control": args.diversity_control,
        "diversity_axis_column": axis_column,
        "diversity_axis_values": list(args.control_values),
        "mean_gate": (
            f"per split val_mse <= {MEAN_GATE_FACTOR} * minimum gamma=0 free-coefficient "
            "held-out val_mse per ensemble; aggregate mean_gate_passed is the 3-split pass fraction"
        ),
        "split_scheme": {
            "name": "interleaved peptide×timepoint diagonal pairs",
            "n_replicates": N_SPLITS,
            "peptide_folds": PEPTIDE_FOLDS,
            "timepoint_folds": TIMEPOINT_FOLDS,
            "peptide_1_held_out_before_splitting": True,
            "splits": [
                {
                    "split": int(spec["split"]),
                    "peptide_fold": int(spec["peptide_fold"]),
                    "time_fold": int(spec["time_fold"]),
                    "train_peptide_indices": np.asarray(spec["train_peptides"]).tolist(),
                    "validation_peptide_indices": np.asarray(spec["val_peptides"]).tolist(),
                    "train_peptide_ids": [
                        int(first_cell["peptide_ids"][index])
                        for index in np.asarray(spec["train_peptides"])
                    ],
                    "validation_peptide_ids": [
                        int(first_cell["peptide_ids"][index])
                        for index in np.asarray(spec["val_peptides"])
                    ],
                    "train_time_indices": np.asarray(spec["train_times"]).tolist(),
                    "validation_time_indices": np.asarray(spec["val_times"]).tolist(),
                }
                for spec in specs
            ],
            "cliff_boundary_requires": "all three held-out replicates pass",
        },
        "optimization": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "n_start": args.n_start,
            "shared_coefficients": True,
            "per_ensemble_logits": True,
            "batched_restarts": True,
            "fixed_ess_projection": (
                "48-step log-temperature bisection with stop-gradient temperature; no KL term"
                if args.diversity_control == "fixed_ess"
                else None
            ),
        },
        "stage5_raw_source": str(args.stage5_raw),
        "outputs": {
            "fit_rows": "joint_diag_d_fit.csv",
            "replicate_rows": "joint_diag_d_fit_replicates.csv",
            "restart_diagnostics": "restart_diagnostics.csv",
            "fitted_arrays": (
                "joint_diag_d_fit_payload.npz; weights and held-out predictions "
                "keyed by each row's payload_id"
            ),
            "cliff_comparison": "cliff_comparison.csv",
            "fixed_ess_frontier": (
                "fixed_ess_frontier.csv"
                if args.diversity_control == "fixed_ess"
                else None
            ),
            "fixed_ess_frontier_summary": (
                "fixed_ess_frontier_summary.json"
                if args.diversity_control == "fixed_ess"
                else None
            ),
            "figures": [
                "{arm}_{metric}.png for each persisted arm and metric",
                f"{{arm}}_gate_ratio_vs_{axis_column}.png",
                "{arm}_per_cluster_ess.png",
            ],
        },
        "fixed_ess_verdict": frontier_summary,
    }
    (args.output_dir / "joint_diag_d_fit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for _, row in comparison.iterrows():
        print(
            "{:16s}/{:16s} joint γ*={}  Stage-5 γ*={}  Δγ={}".format(
                row.arm,
                row.ensemble,
                row.joint_gamma_pass_boundary,
                row.stage5_fixed_gamma_pass_boundary,
                row.joint_minus_stage5_gamma,
            )
        )
    print(f"wrote {args.output_dir / 'joint_diag_d_fit.csv'}")
    plot_joint_diag_d_fit(args.output_dir, control_axis=axis_column)


def merge_worker_outputs(worker_dirs: list[Path], args: argparse.Namespace) -> None:
    """Merge independent blind worker outputs and compute the final gate once."""

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(
        [pd.read_csv(directory / "joint_diag_d_fit_replicates.csv") for directory in worker_dirs],
        ignore_index=True,
    )
    expected = (
        N_SPLITS
        * len(args.arms)
        * len(args.gammas)
        * len(args.control_values)
        * len(common.ENSEMBLES)
    )
    key_columns = ["split", "arm", "gamma", "eta", "ensemble"]
    if len(raw) != expected or raw.duplicated(key_columns).any():
        raise ValueError(
            f"worker rows are incomplete or duplicated: got {len(raw)}, expected {expected}"
        )
    restart = pd.concat(
        [pd.read_csv(directory / "restart_diagnostics.csv") for directory in worker_dirs],
        ignore_index=True,
    )
    payload: dict[str, np.ndarray] = {}
    for directory in worker_dirs:
        with np.load(directory / "joint_diag_d_fit_payload.npz") as worker_payload:
            overlap = set(payload).intersection(worker_payload.files)
            if overlap:
                raise ValueError(
                    f"duplicated fitted-array payload keys: {sorted(overlap)[:3]}"
                )
            payload.update(
                {key: np.asarray(worker_payload[key]) for key in worker_payload.files}
            )
    expected_payload_keys = {
        f"{payload_id}__{kind}"
        for payload_id in raw["payload_id"]
        for kind in ("weights", "val_prediction")
    }
    if set(payload) != expected_payload_keys:
        raise ValueError(
            "worker fitted-array payload is incomplete or has unexpected keys"
        )
    base_cells = {name: _load_cell(name, args.target_artifact) for name in common.ENSEMBLES}
    specs = _split_specs(
        next(iter(base_cells.values()))["mapping"].shape[0],
        next(iter(base_cells.values()))["timepoints"].shape[0],
    )
    _write_final_artifacts(raw, restart, payload, base_cells, specs, args)


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.steps < 1 or args.n_start < 1:
        raise ValueError("steps and n_start must be positive")
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("require worker_count >= 1 and 0 <= worker_index < worker_count")

    base_cells = {name: _load_cell(name, args.target_artifact) for name in common.ENSEMBLES}
    if args.diversity_control == "fixed_ess":
        maximum_target = max(args.control_values)
        minimum_frames = min(cell["n_frames"] for cell in base_cells.values())
        if maximum_target >= minimum_frames:
            raise ValueError(
                f"ESS targets must be below every ensemble size; got {maximum_target:g} "
                f"with minimum n_frames={minimum_frames}"
            )
    specs = _split_specs(
        next(iter(base_cells.values()))["mapping"].shape[0],
        next(iter(base_cells.values()))["timepoints"].shape[0],
    )
    cells_by_split = {
        int(spec["split"]): _split_cells(base_cells, spec) for spec in specs
    }
    rows: list[dict] = []
    fits_by_cell: dict[tuple, dict] = {}
    restart_rows: list[dict] = []
    payload: dict[str, np.ndarray] = {}

    # γ=0 is arm-invariant by construction. Reuse its complete fit record so
    # fitted coefficients, mean metrics, and restart diagnostics match between
    # arms; the reported arm-specific d-loss remains a diagnostic. Workers
    # partition unique fits, not report rows, so the γ=0 dependency stays local.
    for split_index, cells in cells_by_split.items():
        jobs = [
            (gamma, eta, arm)
            for gamma in args.gammas
            for eta in args.control_values
            for arm in args.arms
            if gamma != 0.0 or arm == args.arms[0]
        ]
        for job_index, (gamma, eta, fit_arm) in enumerate(jobs):
            if job_index % args.worker_count != args.worker_index:
                continue
            fit_key = (split_index, fit_arm, gamma, eta)
            fit = _run_with_diagnostics(
                cells,
                gamma,
                eta,
                fit_arm,
                args.steps,
                args.learning_rate,
                args.n_start,
                args.diversity_control,
            )
            fits_by_cell[fit_key] = fit
            report_arms = args.arms if gamma == 0.0 else (fit_arm,)
            for arm in report_arms:
                key = (split_index, arm, gamma, eta)
                fits_by_cell[key] = fit
                for restart in fit["restart_rows"]:
                    restart_rows.append(
                        {
                            "split": split_index,
                            "arm": arm,
                            "gamma": gamma,
                            "eta": eta,
                            "diversity_axis": eta,
                            **(
                                {"ess_target": eta}
                                if args.diversity_control == "fixed_ess"
                                else {}
                            ),
                            **restart,
                        }
                    )
                rows.extend(
                    _report(
                        cells,
                        fit,
                        gamma,
                        eta,
                        arm,
                        args.diversity_control,
                        payload,
                    )
                )

    # The blind rows are complete before the post-fit NMR reveal.
    _add_recovery_metrics(
        rows, base_cells, fits_by_cell, args.diversity_control
    )
    frame = pd.DataFrame(rows)
    if args.diversity_control == "fixed_ess":
        errors = np.abs(frame["ess"].to_numpy() - frame["ess_target"].to_numpy())
        if np.max(errors) > 1e-3:
            worst = int(np.argmax(errors))
            raise AssertionError(
                "fixed-ESS projection missed its target: "
                f"target={frame.iloc[worst].ess_target:g}, "
                f"realized={frame.iloc[worst].ess:.8g}, error={errors[worst]:.3g}"
            )
    if args.worker_count > 1:
        frame["mean_gate_reference_mse"] = np.nan
        frame["mean_gate_passed"] = np.nan
        frame.to_csv(args.output_dir / "joint_diag_d_fit_replicates.csv", index=False)
        pd.DataFrame(restart_rows).to_csv(args.output_dir / "restart_diagnostics.csv", index=False)
        np.savez_compressed(
            args.output_dir / "joint_diag_d_fit_payload.npz",
            **payload,
        )
        print(f"wrote worker {args.worker_index} rows to {args.output_dir}")
        return
    _write_final_artifacts(frame, restart_rows, payload, base_cells, specs, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument("--target-artifact", type=Path, default=TARGET_ARTIFACT)
    parser.add_argument("--stage5-raw", type=Path, default=STAGE5_RAW)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--n-start", type=int, default=N_START)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--merge-workers", nargs="+", type=Path)
    parser.add_argument(
        "--diversity-control",
        choices=("kl", "fixed_ess"),
        default="kl",
        help="KL-weight sweep (Experiment 1) or exact ESS projection (Experiment 1a)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        help="arms to run; fixed-ESS defaults to the primary scale-free arm",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = HERE / (
            "_moprp_joint_diag_d_fit_fixedess"
            if args.diversity_control == "fixed_ess"
            else "_moprp_joint_diag_d_fit_replicated"
        )
    args.gammas = FIXED_ESS_GAMMAS if args.diversity_control == "fixed_ess" else GAMMAS
    args.control_values = ESS_TARGETS if args.diversity_control == "fixed_ess" else ETAS
    args.arms = tuple(args.arms) if args.arms else (
        ("diag_d_scalefree",) if args.diversity_control == "fixed_ess" else ARMS
    )
    if args.merge_workers:
        merge_worker_outputs(args.merge_workers, args)
    else:
        run(args)


if __name__ == "__main__":
    main()
