#!/usr/bin/env python3
"""2D PF recovery analysis for 4_aSyn condition sweeps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.utils.jax_fn import frame_average_features
import jaxent.src.interfaces.topology as pt

BV_REG_FUNCTIONS = ["L1", "L2"]


def _resolve_path(script_dir: Path, value: str | None, absolute_paths: bool) -> Path | None:
    if value is None:
        return None
    return Path(value) if absolute_paths else (script_dir / value).resolve()


def _resolve_results_dir(script_dir: Path, exp_dir: Path, cfg: ExperimentConfig, args) -> Path:
    if args.results_dir:
        return _resolve_path(script_dir, args.results_dir, args.absolute_paths)
    if cfg.results_prefix:
        prefix_path = Path(cfg.results_prefix)
        base_path = exp_dir / prefix_path.parent
        found = find_most_recent_dir(base_path, prefix_path.name)
        if found is None:
            raise FileNotFoundError(
                f"No directory matching prefix '{prefix_path.name}' found in {base_path}"
            )
        return found
    if cfg.results_dir:
        return (exp_dir / cfg.results_dir).resolve()
    raise ValueError("Provide --results-dir or set results_prefix/results_dir in config")


def _load_val_pf(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    val_csv = split_dir / "val_dfrac.csv"
    val_top = split_dir / "val_topology.json"
    if not val_csv.exists() or not val_top.exists():
        raise FileNotFoundError(f"Missing validation split files in {split_dir}")

    val_data = ExpD_Datapoint.load_list_from_files(
        json_path=val_top,
        csv_path=val_csv,
        datapoint_class=HDX_protection_factor,
    )
    val_residue_ids = np.array([int(dp.top.residues[0]) for dp in val_data], dtype=int)
    val_pf = np.array([float(dp.extract_features().reshape(-1)[0]) for dp in val_data], dtype=float)
    return val_residue_ids, val_pf


def _pf_jsd(pred_ln_pf: np.ndarray, true_pf: np.ndarray) -> float:
    pred_ln = np.asarray(pred_ln_pf, dtype=float).reshape(-1)
    true_ln = np.log(np.clip(np.asarray(true_pf, dtype=float).reshape(-1), 1e-12, None))

    pred_prob = np.exp(pred_ln - np.max(pred_ln))
    true_prob = np.exp(true_ln - np.max(true_ln))
    pred_prob = pred_prob / np.sum(pred_prob)
    true_prob = true_prob / np.sum(true_prob)

    m = 0.5 * (pred_prob + true_prob)
    return float(0.5 * (analysis.kl_divergence(pred_prob, m) + analysis.kl_divergence(true_prob, m)))


def main() -> None:
    parser = argparse.ArgumentParser(description="4_aSyn PF-recovery analysis")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--datasplit-dir", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--absolute-paths", action="store_true", default=False)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir.parent
    cfg = ExperimentConfig.from_yaml(
        Path(args.config).resolve() if args.config else exp_dir / "config.yaml"
    )

    results_dir = _resolve_results_dir(script_dir, exp_dir, cfg, args)
    datasplit_dir = (
        _resolve_path(script_dir, args.datasplit_dir, args.absolute_paths)
        if args.datasplit_dir
        else (exp_dir / cfg.datasplit_dir).resolve()
    )
    features_dir = (
        _resolve_path(script_dir, args.features_dir, args.absolute_paths)
        if args.features_dir
        else (exp_dir / cfg.features_dir).resolve()
    )

    if args.output_dir:
        output_dir = _resolve_path(script_dir, args.output_dir, args.absolute_paths)
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = script_dir / f"_analysis_{base_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory : {results_dir}")
    print(f"Datasplit directory: {datasplit_dir}")
    print(f"Features directory : {features_dir}")
    print(f"Output directory   : {output_dir}")

    features = BV_input_features.load(str(features_dir / "features.npz"))
    feature_top = pt.PTSerialiser.load_list_from_json(str(features_dir / "topology.json"))
    residue_to_idx = {int(top.residues[0]): i for i, top in enumerate(feature_top)}

    model = BV_model(config=BV_model_Config(num_timepoints=0))
    forward = model.forward[m_key("HDX_resPF")]

    results = loading.load_all_optimization_results_2d(
        results_dir=str(results_dir),
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        bv_reg_functions=BV_REG_FUNCTIONS,
        num_splits=cfg.num_splits,
        EMA=args.ema,
        verbose=True,
    )

    rows: list[dict] = []
    val_cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}

    for split_type, ensemble_data in results.items():
        for ensemble, loss_data in ensemble_data.items():
            for loss_name, reg_data in loss_data.items():
                for bv_reg_fn, maxent_data in reg_data.items():
                    for maxent_val, bvreg_data in maxent_data.items():
                        for bv_reg_val, split_histories in bvreg_data.items():
                            for split_idx, history in split_histories.items():
                                if history is None or not history.states:
                                    continue

                                cache_key = (ensemble, split_type, int(split_idx))
                                if cache_key not in val_cache:
                                    split_dir = datasplit_dir / ensemble / split_type / f"split_{int(split_idx):03d}"
                                    val_res, val_pf = _load_val_pf(split_dir)
                                    val_idx = np.array([residue_to_idx[int(r)] for r in val_res], dtype=int)
                                    val_cache[cache_key] = (val_idx, val_pf)

                                val_idx, val_pf = val_cache[cache_key]
                                final_state = history.states[-1]
                                if final_state.params is None or final_state.params.frame_weights is None:
                                    continue

                                frame_weights = np.asarray(final_state.params.frame_weights, dtype=float).reshape(-1)
                                if frame_weights.size == 0 or np.sum(frame_weights) <= 0:
                                    continue
                                frame_weights = frame_weights / np.sum(frame_weights)

                                avg_features = frame_average_features(features, jnp.asarray(frame_weights))
                                model_params = final_state.params.model_parameters[0]
                                pred_ln_pf = np.asarray(forward(avg_features, model_params).log_Pf).reshape(-1)

                                jsd = _pf_jsd(pred_ln_pf[val_idx], val_pf)
                                rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "split_type": split_type,
                                        "loss_function": loss_name,
                                        "bv_reg_function": bv_reg_fn,
                                        "split": int(split_idx),
                                        "maxent_value": float(maxent_val),
                                        "bv_reg_value": float(bv_reg_val),
                                        "convergence_step": len(history.states),
                                        "js_divergence": jsd,
                                        "js_distance": float(np.sqrt(jsd)),
                                        "recovery_percent": float((1.0 - np.sqrt(jsd)) * 100.0),
                                    }
                                )

    recovery_df = pd.DataFrame(rows)
    if recovery_df.empty:
        print("No recovery data extracted.")
        return

    out_csv = output_dir / "recovery_2d_sweep_data.csv"
    recovery_df.to_csv(out_csv, index=False)
    print(f"Recovery data saved to: {out_csv}")

    plotting.plot_2d_heatmaps_grid(
        recovery_df,
        str(output_dir),
        metric="recovery_percent",
        metric_label="Recovery (%)",
    )
    plotting.plot_1d_slices_2d_sweep(
        recovery_df,
        str(output_dir),
        metric="recovery_percent",
        metric_label="Recovery (%)",
    )
    plotting.plot_best_hyperparameters(
        recovery_df,
        str(output_dir),
        metric="recovery_percent",
    )

    print("Analysis completed.")


if __name__ == "__main__":
    main()
