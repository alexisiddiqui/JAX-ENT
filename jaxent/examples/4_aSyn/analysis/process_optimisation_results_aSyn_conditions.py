#!/usr/bin/env python3
"""Process 4_aSyn optimization outputs into per-run numpy artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxent.examples.common import analysis
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import derive_processed_output_dir, find_most_recent_dir
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.utils.hdf import load_optimization_history_from_file
from jaxent.src.utils.jax_fn import frame_average_features
import jaxent.src.interfaces.topology as pt


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


def _parse_run_filename(filename: str, split_type: str):
    pattern = re.compile(
        rf"(?P<ensemble>.+?)_(?P<loss>[A-Za-z0-9]+)_{re.escape(split_type)}"
        r"_split(?P<split_idx>\d+)_maxent(?P<maxent>[\d.]+)_bvreg(?P<bvreg>[\d.]+)_bvregfn(?P<bvfn>[A-Za-z0-9]+)_results(?:_EMA)?\.hdf5$"
    )
    m = pattern.match(filename)
    if not m:
        return None
    g = m.groupdict()
    return {
        "ensemble": g["ensemble"],
        "loss_function": g["loss"],
        "split_type": split_type,
        "split_idx": int(g["split_idx"]),
        "maxent_value": float(g["maxent"]),
        "bv_reg_value": float(g["bvreg"]),
        "bv_reg_function": g["bvfn"],
    }


def _run_id_from_filename(filename: str) -> str:
    if filename.endswith("_results_EMA.hdf5"):
        return filename[: -len("_results_EMA.hdf5")]
    if filename.endswith("_results.hdf5"):
        return filename[: -len("_results.hdf5")]
    return filename.rsplit(".", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Process 4_aSyn optimization outputs")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--datasplit-dir", default=None, help="Accepted for CLI compatibility")
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
    features_dir = (
        _resolve_path(script_dir, args.features_dir, args.absolute_paths)
        if args.features_dir
        else (exp_dir / cfg.features_dir).resolve()
    )
    output_dir = (
        _resolve_path(script_dir, args.output_dir, args.absolute_paths)
        if args.output_dir
        else derive_processed_output_dir(results_dir)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory : {results_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Output directory  : {output_dir}")
    print(f"EMA               : {args.ema}")

    features_path = features_dir / "features.npz"
    topology_path = features_dir / "topology.json"
    if not features_path.exists() or not topology_path.exists():
        raise FileNotFoundError(f"Expected {features_path} and {topology_path}")

    features = BV_input_features.load(str(features_path))
    feature_top = pt.PTSerialiser.load_list_from_json(str(topology_path))
    residue_ids = np.array([int(top.residues[0]) for top in feature_top], dtype=int)

    model = BV_model(config=BV_model_Config(num_timepoints=0))
    forward = model.forward[m_key("HDX_resPF")]

    n_frames = features.features_shape[1]
    uniform_weights = jnp.ones(n_frames) / n_frames
    prior_features = frame_average_features(features, uniform_weights)
    prior_ln_pf = np.asarray(forward(prior_features, model.params).log_Pf).reshape(-1)

    suffix = "_results_EMA.hdf5" if args.ema else "_results.hdf5"

    processed_runs = 0

    split_type_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    for split_type_dir in split_type_dirs:
        split_type = split_type_dir.name
        split_output_root = output_dir / split_type
        split_output_root.mkdir(parents=True, exist_ok=True)

        for history_file in sorted(split_type_dir.glob(f"*{suffix}")):
            meta = _parse_run_filename(history_file.name, split_type)
            if meta is None:
                continue
            if meta["ensemble"] not in cfg.ensembles:
                continue

            history = load_optimization_history_from_file(str(history_file))
            if history is None or not history.states:
                continue

            run_id = _run_id_from_filename(history_file.name)
            run_dir = split_output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            pred_ln_pf_stack = []
            frame_weights_stack = []
            kl_stack = []
            val_loss_stack = []
            bv_bc_stack = []
            bv_bh_stack = []
            convergence_steps = []

            for state in history.states:
                if state.params is None or state.params.frame_weights is None:
                    continue

                frame_weights = np.asarray(state.params.frame_weights, dtype=float).reshape(-1)
                if frame_weights.size == 0 or np.sum(frame_weights) <= 0:
                    continue
                frame_weights = frame_weights / np.sum(frame_weights)

                avg_features = frame_average_features(features, jnp.asarray(frame_weights))
                model_params = state.params.model_parameters[0]
                pred_ln_pf = np.asarray(forward(avg_features, model_params).log_Pf).reshape(-1)

                pred_ln_pf_stack.append(pred_ln_pf)
                frame_weights_stack.append(frame_weights)

                uniform = np.ones_like(frame_weights) / len(frame_weights)
                kl_stack.append(float(analysis.kl_divergence(frame_weights, uniform)))

                if state.losses is not None and state.losses.val_losses is not None:
                    val_loss_stack.append(float(state.losses.val_losses[0]))
                else:
                    val_loss_stack.append(np.nan)

                bv_bc_stack.append(float(np.asarray(model_params.bv_bc).reshape(-1)[0]))
                bv_bh_stack.append(float(np.asarray(model_params.bv_bh).reshape(-1)[0]))
                convergence_steps.append(float(state.step))

            if not pred_ln_pf_stack:
                continue

            np.save(run_dir / "pred_ln_pf.npy", np.asarray(pred_ln_pf_stack))
            np.save(run_dir / "frame_weights.npy", np.asarray(frame_weights_stack))
            np.save(run_dir / "kl_divergence.npy", np.asarray(kl_stack))
            np.save(run_dir / "val_loss.npy", np.asarray(val_loss_stack))
            np.save(run_dir / "bv_bc.npy", np.asarray(bv_bc_stack))
            np.save(run_dir / "bv_bh.npy", np.asarray(bv_bh_stack))
            np.save(run_dir / "prior_ln_pf.npy", np.asarray(prior_ln_pf))
            np.save(run_dir / "residue_ids.npy", residue_ids)

            with open(run_dir / "convergence_thresholds.txt", "w", encoding="utf-8") as f:
                for value in convergence_steps:
                    f.write(f"{value}\n")

            processed_runs += 1
            print(f"Processed: {run_id}")

    print(f"\nProcessed runs: {processed_runs}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
