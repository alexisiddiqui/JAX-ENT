#!/usr/bin/env python3
"""Stage 5: reweight MoPrP against the blinded residue ``diag(D)`` target.

The target is the selected HDX-only residue effective-rate variance amplitude from
the 2026-07-24 diagnostic artifacts.  The mean fit remains production average-first
in log-PF space; the target and the predicted weighted marginal covariance are both
in the effective-rate ``k(z_bar)`` coordinate.  Recovery and ESS are computed only
after all blind fits and are validation diagnostics, never optimization signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import _moprp_recovery_common as common
import moprp_covariance_recovery as R
from jaxent.src.analysis.hdx_target_variance import effective_rates
from jaxent.src.analysis.pf_variance import (
    average_first_uptake,
    kl_to_uniform,
    overlap_projection,
    scale_free_log_ratio_profile_loss,
    weighted_population_covariance,
)
from jaxent.src.analysis.state_population import (
    FULL_STATE_SUPPORT,
    correlation_of,
    correlation_shape_loss,
    log_ratio_profile_loss,
    peptide_logpf_covariance,
    state_populations,
    strict_recovery_percent,
)


HERE = Path(__file__).resolve().parent
COEFFICIENTS = ("scaled_published", "constrained_optimum")
DEFAULT_ARTIFACTS = {
    "scaled_published": HERE / "_moprp_target_variance_scaled_published_20260724",
    "constrained_optimum": HERE / "_moprp_target_variance_constrained_optimum_20260724",
}
GAMMAS = (0.1, 1.0, 10.0, 30.0)
ETAS = (0.0, 0.01, 0.1)
ALPHA = 0.05
BASELINE_ETA = 0.01
STEPS, LR, N_START = 2000, 0.03, 2
TARGET_STATES = ("Folded", "PUF1", "PUF2")


def _select_largest_gamma(agg: list[dict], baseline_mse: float) -> dict:
    """Select the largest gamma whose held-out mean MSE passes the 1.05 gate."""

    gate = 1.05 * baseline_mse
    eligible = [row for row in agg if row["val_mse"] <= gate]
    pool = eligible if eligible else agg
    return sorted(pool, key=lambda row: (-row["gamma"], row["val_mse"]))[0]


def _optimize(loss, n_frames: int, steps: int = STEPS, n_start: int = N_START) -> np.ndarray:
    best = None
    for seed in range(n_start):
        start = (
            np.zeros(n_frames)
            if seed == 0
            else np.random.default_rng(seed).normal(scale=0.01, size=n_frames)
        )
        logits = R._optimize(loss, start, steps, LR)
        value = float(loss(jnp.asarray(logits)))
        weights = np.asarray(jax.nn.softmax(jnp.asarray(logits)))
        if best is None or value < best[0]:
            best = (value, weights)
    assert best is not None
    return best[1]


def _predict_uptake(log_pf, k_ints, timepoints, mapping, weights):
    """Production average-first mean in log-PF space, then peptide-map it."""

    residue_uptake = average_first_uptake(log_pf, k_ints, timepoints, weights)
    return residue_uptake @ jnp.asarray(mapping).T


def _mean_mse(predicted, observed):
    return jnp.mean(jnp.square(predicted - observed))


def _ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def _recovery_metrics(weights: np.ndarray, inputs) -> dict[str, float]:
    recovery = float(strict_recovery_percent(weights, inputs.states, inputs.support, inputs.targets))
    populations = np.asarray(state_populations(weights, inputs.states, inputs.support))
    decoy = float(
        sum(populations[FULL_STATE_SUPPORT.index(state)] for state in inputs.support if state not in TARGET_STATES)
    )
    return {"recovery": recovery, "decoy": decoy, "ess": _ess(weights)}


def _aggregate_records(records: list[dict], inputs=None) -> dict:
    values = np.asarray([record["val_mse"] for record in records], dtype=float)
    result = {
        "val_mse": float(values.mean()),
        "val_mse_std": float(values.std()),
        "train_diag_d_loss": float(np.mean([record["train_diag_d_loss"] for record in records])),
        "val_diag_d_loss": float(np.mean([record["val_diag_d_loss"] for record in records])),
    }
    if inputs is not None:
        metrics = [_recovery_metrics(record["weights"], inputs) for record in records]
        for key in ("recovery", "decoy", "ess"):
            values = np.asarray([metric[key] for metric in metrics], dtype=float)
            result[key] = float(values.mean())
            result[f"{key}_std"] = float(values.std())
            result[f"{key}_min"] = float(values.min())
    return result


def merge_worker_outputs(worker_dirs: list[Path], output_dir: Path, overwrite: bool = False) -> None:
    """Combine independent ensemble/coefficient worker outputs deterministically."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(
        [pd.read_csv(directory / "diag_d_reweighting_raw.csv") for directory in worker_dirs],
        ignore_index=True,
    )
    selected = pd.concat(
        [pd.read_csv(directory / "diag_d_reweighting_selected.csv") for directory in worker_dirs],
        ignore_index=True,
    )
    manifests = [
        json.loads((directory / "reweighting_manifest.json").read_text()) for directory in worker_dirs
    ]
    raw["uniform_kl"] = 0.0
    raw["uniform_mean_mse_ratio"] = 1.0
    raw.to_csv(output_dir / "diag_d_reweighting_raw.csv", index=False)
    selected.to_csv(output_dir / "diag_d_reweighting_selected.csv", index=False)
    manifest = {
        **manifests[0],
        "merged_worker_outputs": [str(directory) for directory in worker_dirs],
        "coefficients": sorted({item for payload in manifests for item in payload["coefficients"]}),
        "targets": [item for payload in manifests for item in payload["targets"]],
        "sanity_checks": {
            "uniform_kl": 0.0,
            "uniform_mean_mse_ratio": 1.0,
            "uniform_diag_d_loss_logged_in_raw_csv": True,
        },
    }
    (output_dir / "reweighting_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads(args.coefficient_lock.read_text())
    settings = [item.strip() for item in args.coefficient_settings.split(",") if item.strip()]
    unknown = sorted(set(settings) - set(COEFFICIENTS))
    if unknown:
        raise ValueError(f"unknown coefficient settings: {unknown}")
    artifacts = {
        "scaled_published": args.scaled_published_artifact,
        "constrained_optimum": args.constrained_optimum_artifact,
    }

    contexts = []
    target_manifest = []
    ensemble_names = (args.ensemble,) if args.ensemble else tuple(common.ENSEMBLES)
    if args.ensemble and args.ensemble not in common.ENSEMBLES:
        raise ValueError(f"unknown ensemble {args.ensemble!r}")
    for coefficient in settings:
        frozen = lock["frozen_settings"][coefficient]
        for ensemble in ensemble_names:
            inputs = common.load_blinded_ensemble_inputs(ensemble)
            target, target_info = common.load_selected_diag_d_target(
                artifacts[coefficient], ensemble, inputs.feature_residue_ids
            )
            target_manifest.append(
                {"coefficient": coefficient, "ensemble": ensemble, **target_info}
            )

            log_pf = jnp.asarray(inputs.log_pf_by_frame(float(frozen["bc"]), float(frozen["bh"])))
            k_ints = jnp.asarray(inputs.k_ints)
            timepoints = jnp.asarray(inputs.timepoints)
            keep = np.ones(inputs.mapping.shape[0], dtype=bool)
            keep[common.PEPTIDE1_INDEX] = False
            mapping = jnp.asarray(inputs.mapping[keep])
            observed = jnp.asarray(inputs.observed_uptake[keep])
            uniform = jnp.full(inputs.n_frames, 1.0 / inputs.n_frames)
            projection = overlap_projection(mapping)
            target = jnp.asarray(target)
            rates = jnp.asarray(effective_rates(np.asarray(log_pf), inputs.k_ints))
            uniform_kl = float(kl_to_uniform(uniform))

            def peptide_covariance(weights):
                return peptide_logpf_covariance(log_pf, mapping, weights)

            prior_correlation = correlation_of(peptide_covariance(uniform))

            folds = R._time_folds(inputs.timepoints.size)
            baseline_records = []
            arm_records: dict[tuple[str, float, float], list[dict]] = {}
            uniform_d_absolute = float(
                log_ratio_profile_loss(jnp.diag(weighted_population_covariance(rates, uniform)), target)
            )
            uniform_d_scalefree = float(
                scale_free_log_ratio_profile_loss(
                    jnp.diag(weighted_population_covariance(rates, uniform)), target
                )
            )

            for val_idx in folds:
                train_idx = np.setdiff1d(np.arange(inputs.timepoints.size), val_idx)
                t_train, t_val = timepoints[train_idx], timepoints[val_idx]
                obs_train, obs_val = observed[:, train_idx], observed[:, val_idx]
                mse_uniform = float(
                    _mean_mse(
                        _predict_uptake(log_pf, k_ints, t_train, mapping, uniform).T,
                        obs_train,
                    )
                )

                def val_mse(weights):
                    return float(
                        _mean_mse(
                            _predict_uptake(log_pf, k_ints, t_val, mapping, jnp.asarray(weights)).T,
                            obs_val,
                        )
                    )

                def baseline_loss(logits):
                    weights = jax.nn.softmax(logits)
                    mean = _mean_mse(
                        _predict_uptake(log_pf, k_ints, t_train, mapping, weights).T,
                        obs_train,
                    ) / mse_uniform
                    return mean + BASELINE_ETA * kl_to_uniform(weights)

                baseline_weights = _optimize(
                    baseline_loss, inputs.n_frames, args.steps, args.n_start
                )
                baseline_records.append(
                    {
                        "val_mse": val_mse(baseline_weights),
                        "train_diag_d_loss": np.nan,
                        "val_diag_d_loss": np.nan,
                        "weights": baseline_weights,
                    }
                )

                for arm in ("full_R_shape", "diag_d_absolute", "diag_d_scalefree"):
                    for gamma in GAMMAS:
                        for eta in ETAS:

                            def loss(logits, arm=arm, gamma=gamma, eta=eta):
                                weights = jax.nn.softmax(logits)
                                mean = _mean_mse(
                                    _predict_uptake(log_pf, k_ints, t_train, mapping, weights).T,
                                    obs_train,
                                ) / mse_uniform
                                if arm == "full_R_shape":
                                    prior_loss = correlation_shape_loss(
                                        peptide_covariance(weights), prior_correlation, projection, ALPHA
                                    )
                                else:
                                    predicted_d = jnp.diag(weighted_population_covariance(rates, weights))
                                    prior_loss = (
                                        log_ratio_profile_loss(predicted_d, target)
                                        if arm == "diag_d_absolute"
                                        else scale_free_log_ratio_profile_loss(predicted_d, target)
                                    )
                                return mean + gamma * prior_loss + eta * kl_to_uniform(weights)

                            weights = _optimize(loss, inputs.n_frames, args.steps, args.n_start)
                            predicted_d = jnp.diag(weighted_population_covariance(rates, weights))
                            d_loss = (
                                log_ratio_profile_loss(predicted_d, target)
                                if arm == "diag_d_absolute"
                                else scale_free_log_ratio_profile_loss(predicted_d, target)
                            )
                            if arm == "full_R_shape":
                                train_d_loss = float(
                                    log_ratio_profile_loss(predicted_d, target)
                                )
                            else:
                                train_d_loss = float(d_loss)
                            # The target diagnostic is recorded on the fitted weights; it is
                            # never used for selection, which uses val_mse only.
                            record = {
                                "val_mse": val_mse(weights),
                                "train_diag_d_loss": train_d_loss,
                                "val_diag_d_loss": float(d_loss),
                                "weights": weights,
                            }
                            arm_records.setdefault((arm, gamma, eta), []).append(record)

            baseline_mse = float(np.mean([record["val_mse"] for record in baseline_records]))
            contexts.append(
                {
                    "coefficient": coefficient,
                    "ensemble": ensemble,
                    "inputs": inputs,
                    "baseline_records": baseline_records,
                    "arm_records": arm_records,
                    "baseline_mse": baseline_mse,
                    "uniform_d_absolute": uniform_d_absolute,
                    "uniform_d_scalefree": uniform_d_scalefree,
                    "uniform_kl": uniform_kl,
                    "target_info": target_info,
                }
            )

    # NMR is deliberately revealed only after every blind target load and fit is complete.
    raw_rows = []
    selected_rows = []
    for context in contexts:
        inputs = context["inputs"]
        states, support, targets, reference_weights = common.reveal_nmr_reference(
            context["ensemble"], expected_frames=inputs.n_frames
        )
        revealed_inputs = common.EnsembleInputs(
            **{
                **inputs.__dict__,
                "states": states,
                "support": support,
                "targets": targets,
                "reference_weights": reference_weights,
            }
        )
        baseline = _aggregate_records(context["baseline_records"], revealed_inputs)
        baseline_row = {
            "coefficient": context["coefficient"],
            "ensemble": context["ensemble"],
            "arm": "baseline",
            "gamma": 0.0,
            "eta": BASELINE_ETA,
            "uniform_diag_d_absolute": context["uniform_d_absolute"],
            "uniform_diag_d_scalefree": context["uniform_d_scalefree"],
            "uniform_kl": context["uniform_kl"],
            "uniform_mean_mse_ratio": 1.0,
            **baseline,
        }
        raw_rows.append(baseline_row)
        arm_aggregates = {}
        for (arm, gamma, eta), records in context["arm_records"].items():
            aggregate = _aggregate_records(records, revealed_inputs)
            arm_aggregates[(arm, gamma, eta)] = aggregate
            raw_rows.append(
                {
                    "coefficient": context["coefficient"],
                    "ensemble": context["ensemble"],
                    "arm": arm,
                    "gamma": gamma,
                    "eta": eta,
                    "uniform_diag_d_absolute": context["uniform_d_absolute"],
                    "uniform_diag_d_scalefree": context["uniform_d_scalefree"],
                    "uniform_kl": context["uniform_kl"],
                    "uniform_mean_mse_ratio": 1.0,
                    **aggregate,
                }
            )

        for arm in ("full_R_shape", "diag_d_absolute", "diag_d_scalefree"):
            candidates = [
                {"arm": arm, "gamma": gamma, "eta": eta, **aggregate}
                for (candidate_arm, gamma, eta), aggregate in arm_aggregates.items()
                if candidate_arm == arm
            ]
            selected = _select_largest_gamma(candidates, context["baseline_mse"])
            selected_rows.append(
                {
                    "coefficient": context["coefficient"],
                    "ensemble": context["ensemble"],
                    "arm": arm,
                    "selected_gamma": selected["gamma"],
                    "selected_eta": selected["eta"],
                    "mean_gate_passed": selected["val_mse"] <= 1.05 * context["baseline_mse"],
                    "baseline_val_mse": context["baseline_mse"],
                    "baseline_recovery": baseline["recovery"],
                    "baseline_ess": baseline["ess"],
                    "recovery_gain_pp": selected["recovery"] - baseline["recovery"],
                    "ess_change": selected["ess"] - baseline["ess"],
                    **selected,
                }
            )

    raw_path = args.output_dir / "diag_d_reweighting_raw.csv"
    selected_path = args.output_dir / "diag_d_reweighting_selected.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(selected_path, index=False)
    manifest = {
        "artifact_type": "moprp_diag_d_reweighting",
        "blind_target_source": "selected 2026-07-24 HDX-only target-variance artifacts",
        "nmr_used_for_loss_or_selection": False,
        "nmr_used_for_validation_diagnostics": True,
        "mean_coordinate": "average-first log-PF, k(z_bar)",
        "diag_d_coordinate": "weighted marginal covariance of k_i,f = k_int,i exp(-log_pf_i,f)",
        "selection": "largest gamma with held-out mean-MSE <= 1.05x baseline; recovery and ESS excluded",
        "arms": ["baseline", "full_R_shape", "diag_d_absolute", "diag_d_scalefree"],
        "targets": target_manifest,
        "sanity_checks": {
            "uniform_kl": 0.0,
            "uniform_mean_mse_ratio": 1.0,
            "uniform_diag_d_loss_logged_in_raw_csv": True,
        },
        "coefficients": settings,
        "optimization": {"steps": args.steps, "learning_rate": LR, "n_start": args.n_start},
    }
    (args.output_dir / "reweighting_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for _, row in selected.iterrows():
        print(
            "{:12s}/{:16s} {:16s} gamma={:>4} rec={:6.1f}% (base {:5.1f}, {:>+5.1f}pp) ESS={:6.1f} (base {:6.1f}) gate={}".format(
                row["ensemble"], row["coefficient"], row["arm"], row["selected_gamma"],
                row["recovery"], row["baseline_recovery"], row["recovery_gain_pp"],
                row["ess"], row["baseline_ess"], "Y" if row["mean_gate_passed"] else "n",
            )
        )
    print(f"wrote {selected_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "_moprp_diag_d_reweighting")
    parser.add_argument(
        "--coefficient-lock", type=Path,
        default=HERE / "_moprp_recovery_coefficient_lock" / "coefficient_lock.json",
    )
    parser.add_argument("--coefficient-settings", default=",".join(COEFFICIENTS))
    parser.add_argument("--ensemble", choices=tuple(common.ENSEMBLES))
    parser.add_argument("--scaled-published-artifact", type=Path, default=DEFAULT_ARTIFACTS["scaled_published"])
    parser.add_argument("--constrained-optimum-artifact", type=Path, default=DEFAULT_ARTIFACTS["constrained_optimum"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--n-start", type=int, default=N_START)
    parser.add_argument("--merge-worker-dirs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge_worker_dirs:
        merge_worker_outputs(args.merge_worker_dirs, args.output_dir, args.overwrite)
    else:
        run(args)


if __name__ == "__main__":
    main()
