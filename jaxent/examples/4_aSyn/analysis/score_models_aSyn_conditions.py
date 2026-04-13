#!/usr/bin/env python3
"""Score processed 4_aSyn runs with PF-centric metrics and recovery."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.common import analysis, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import derive_processed_output_dir, find_most_recent_dir
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.custom_types.datapoint import ExpD_Datapoint


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


def _parse_run_id(run_id: str, split_type: str):
    pattern = re.compile(
        rf"(?P<ensemble>.+?)_(?P<loss>[A-Za-z0-9]+)_{re.escape(split_type)}"
        r"_split(?P<split_idx>\d+)_maxent(?P<maxent>[\d.]+)_bvreg(?P<bvreg>[\d.]+)_bvregfn(?P<bvfn>[A-Za-z0-9]+)$"
    )
    m = pattern.match(run_id)
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


def _load_pf_split(split_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    csv_path = split_dir / f"{prefix}_dfrac.csv"
    top_path = split_dir / f"{prefix}_topology.json"
    if not csv_path.exists() or not top_path.exists():
        raise FileNotFoundError(f"Missing split files: {csv_path} / {top_path}")

    datapoints = ExpD_Datapoint.load_list_from_files(
        json_path=top_path,
        csv_path=csv_path,
        datapoint_class=HDX_protection_factor,
    )
    residue_ids = np.array([int(dp.top.residues[0]) for dp in datapoints], dtype=int)
    pf_values = np.array([float(dp.extract_features().reshape(-1)[0]) for dp in datapoints], dtype=float)
    return residue_ids, pf_values


def _pf_jsd_and_recovery(pred_ln_pf: np.ndarray, true_pf: np.ndarray) -> tuple[float, float]:
    pred_ln = np.asarray(pred_ln_pf, dtype=float).reshape(-1)
    true_ln = np.log(np.clip(np.asarray(true_pf, dtype=float).reshape(-1), 1e-12, None))

    pred_prob = np.exp(pred_ln - np.max(pred_ln))
    true_prob = np.exp(true_ln - np.max(true_ln))
    pred_prob = pred_prob / np.sum(pred_prob)
    true_prob = true_prob / np.sum(true_prob)

    m = 0.5 * (pred_prob + true_prob)
    jsd = 0.5 * (analysis.kl_divergence(pred_prob, m) + analysis.kl_divergence(true_prob, m))
    recovery = (1.0 - np.sqrt(jsd)) * 100.0
    return float(jsd), float(recovery)

def _safe_exp_ln_pf(values: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(values, dtype=float), -100.0, 100.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score 4_aSyn processed runs")
    parser.add_argument("--processed-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--datasplit-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--absolute-paths", action="store_true", default=False)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir.parent
    cfg = ExperimentConfig.from_yaml(
        Path(args.config).resolve() if args.config else exp_dir / "config.yaml"
    )

    results_dir = _resolve_results_dir(script_dir, exp_dir, cfg, args)
    processed_dir = (
        _resolve_path(script_dir, args.processed_data_dir, args.absolute_paths)
        if args.processed_data_dir
        else derive_processed_output_dir(results_dir)
    )
    datasplit_dir = (
        _resolve_path(script_dir, args.datasplit_dir, args.absolute_paths)
        if args.datasplit_dir
        else (exp_dir / cfg.datasplit_dir).resolve()
    )

    if args.output_dir:
        output_scores_dir = _resolve_path(script_dir, args.output_dir, args.absolute_paths)
    else:
        output_scores_dir = processed_dir / f"_scores_{processed_dir.name}"
    output_scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"processed_data_dir: {processed_dir}")
    print(f"datasplit_dir:      {datasplit_dir}")
    print(f"output_scores_dir:  {output_scores_dir}")

    all_scores: list[dict] = []
    cache: dict[tuple[str, str, int, bytes], dict] = {}

    split_type_dirs = sorted([p for p in processed_dir.iterdir() if p.is_dir()])
    for split_type_dir in split_type_dirs:
        split_type = split_type_dir.name

        for run_dir in sorted([p for p in split_type_dir.iterdir() if p.is_dir()]):
            meta = _parse_run_id(run_dir.name, split_type)
            if meta is None:
                continue
            if meta["ensemble"] not in cfg.ensembles:
                continue

            pred_ln_pf_stack = np.load(run_dir / "pred_ln_pf.npy")
            frame_weights_stack = np.load(run_dir / "frame_weights.npy")
            kl_stack = np.load(run_dir / "kl_divergence.npy")
            prior_ln_pf = np.load(run_dir / "prior_ln_pf.npy")
            residue_ids = np.load(run_dir / "residue_ids.npy")

            val_loss_stack = (
                np.load(run_dir / "val_loss.npy")
                if (run_dir / "val_loss.npy").exists()
                else np.full(len(pred_ln_pf_stack), np.nan)
            )
            bv_bc_stack = (
                np.load(run_dir / "bv_bc.npy")
                if (run_dir / "bv_bc.npy").exists()
                else np.full(len(pred_ln_pf_stack), np.nan)
            )
            bv_bh_stack = (
                np.load(run_dir / "bv_bh.npy")
                if (run_dir / "bv_bh.npy").exists()
                else np.full(len(pred_ln_pf_stack), np.nan)
            )

            convergence_vals = np.arange(1, len(pred_ln_pf_stack) + 1, dtype=float)
            conv_file = run_dir / "convergence_thresholds.txt"
            if conv_file.exists():
                with open(conv_file, encoding="utf-8") as f:
                    loaded = [float(line.strip()) for line in f if line.strip()]
                if len(loaded) == len(pred_ln_pf_stack):
                    convergence_vals = np.asarray(loaded, dtype=float)

            cache_key = (
                meta["ensemble"],
                split_type,
                meta["split_idx"],
                np.asarray(residue_ids, dtype=int).tobytes(),
            )
            if cache_key not in cache:
                split_dir = datasplit_dir / meta["ensemble"] / split_type / f"split_{meta['split_idx']:03d}"
                train_res, y_train = _load_pf_split(split_dir, "train")
                val_res, y_val = _load_pf_split(split_dir, "val")

                residue_to_idx = {int(r): i for i, r in enumerate(np.asarray(residue_ids, dtype=int))}
                train_idx = np.array([residue_to_idx[int(r)] for r in train_res], dtype=int)
                val_idx = np.array([residue_to_idx[int(r)] for r in val_res], dtype=int)

                prior_pf = _safe_exp_ln_pf(prior_ln_pf)
                prior_train_mse = float(np.mean((prior_pf[train_idx] - y_train) ** 2))
                prior_val_mse = float(np.mean((prior_pf[val_idx] - y_val) ** 2))

                cache[cache_key] = {
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "y_train": y_train,
                    "y_val": y_val,
                    "prior_train_mse": prior_train_mse,
                    "prior_val_mse": prior_val_mse,
                }

            cached = cache[cache_key]

            for i in range(len(pred_ln_pf_stack)):
                pred_ln_pf = np.asarray(pred_ln_pf_stack[i], dtype=float).reshape(-1)
                pred_pf = _safe_exp_ln_pf(pred_ln_pf)

                train_pred_pf = pred_pf[cached["train_idx"]]
                val_pred_pf = pred_pf[cached["val_idx"]]

                train_mse = float(np.mean((train_pred_pf - cached["y_train"]) ** 2))
                val_mse = float(np.mean((val_pred_pf - cached["y_val"]) ** 2))

                d_mse_train = train_mse - cached["prior_train_mse"]
                d_mse_val = val_mse - cached["prior_val_mse"]

                train_jsd, train_recovery = _pf_jsd_and_recovery(
                    pred_ln_pf[cached["train_idx"]], cached["y_train"]
                )
                val_jsd, val_recovery = _pf_jsd_and_recovery(
                    pred_ln_pf[cached["val_idx"]], cached["y_val"]
                )

                work = analysis.calculate_work_metrics(pred_ln_pf, prior_ln_pf)

                score_row = {
                    "ensemble": meta["ensemble"],
                    "loss_function": meta["loss_function"],
                    "bv_reg_function": meta["bv_reg_function"],
                    "split_type": split_type,
                    "split_idx": meta["split_idx"],
                    "maxent_value": meta["maxent_value"],
                    "bv_reg_value": meta["bv_reg_value"],
                    "convergence_value": float(convergence_vals[i]),
                    "kl_divergence": float(kl_stack[i]) if i < len(kl_stack) else np.nan,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "test_mse": val_mse,
                    "d_mse_train": d_mse_train,
                    "d_mse_val": d_mse_val,
                    "d_mse_test": d_mse_val,
                    "work_scale_kj": work.get("work_scale_kj", np.nan),
                    "work_shape_kj": work.get("work_shape_kj", np.nan),
                    "work_density_kj": work.get("work_density_kj", np.nan),
                    "work_fitting_kj": work.get("work_fitting_kj", np.nan),
                    "work_magnitude_kj": work.get("work_magnitude_kj", np.nan),
                    "work_scale": work.get("work_scale_kj", np.nan),
                    "work_shape": work.get("work_shape_kj", np.nan),
                    "work_density": work.get("work_density_kj", np.nan),
                    "work_fitting": work.get("work_fitting_kj", np.nan),
                    "recovery_percent": val_recovery,
                    "train_recovery_percent": train_recovery,
                    "train_jsd": train_jsd,
                    "val_jsd": val_jsd,
                    "val_loss": float(val_loss_stack[i]) if i < len(val_loss_stack) else np.nan,
                    "bv_bc": float(bv_bc_stack[i]) if i < len(bv_bc_stack) else np.nan,
                    "bv_bh": float(bv_bh_stack[i]) if i < len(bv_bh_stack) else np.nan,
                }
                all_scores.append(score_row)

    if not all_scores:
        print("No scores were generated.")
        return

    scores_df = pd.DataFrame(all_scores)
    scores_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    output_csv = output_scores_dir / "model_scores.csv"
    scores_df.to_csv(output_csv, index=False)
    print(f"Saved {len(scores_df)} rows to: {output_csv}")

    plotting_df = scores_df.copy()
    plotting_metrics = [
        "train_mse",
        "val_mse",
        "test_mse",
        "d_mse_train",
        "d_mse_val",
        "d_mse_test",
        "work_scale_kj",
        "work_shape_kj",
        "work_density_kj",
        "work_fitting_kj",
        "kl_divergence",
        "recovery_percent",
        "val_loss",
    ]
    finite_mask = np.ones(len(plotting_df), dtype=bool)
    for col in plotting_metrics:
        if col in plotting_df.columns:
            finite_mask &= np.isfinite(plotting_df[col].to_numpy(dtype=float, copy=False))
    plotting_df = plotting_df.loc[finite_mask].copy()

    if plotting_df.empty:
        print("Skipping violin plots (no finite rows across plotting metrics).")
        return

    plotting.create_violin_plots(
        plotting_df,
        str(output_scores_dir),
        metric_columns=plotting_metrics,
    )


if __name__ == "__main__":
    main()
