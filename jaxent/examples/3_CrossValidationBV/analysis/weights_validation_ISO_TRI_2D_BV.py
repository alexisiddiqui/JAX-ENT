"""
weights_validation_ISO_TRI_BI_2D_BV.py  (Exp3 — CrossValidationBV)

Validates frame weights from a 2D hyperparameter sweep (MaxEnt × BV regularization).
Computes KL divergence and ESS, pairwise split consistency, and generates 2D heatmaps
and 1D slice plots.

Requirements:
    - Optimization results (results.hdf5 / results_EMA.hdf5)

Usage:
    python jaxent/examples/3_CrossValidationBV/analysis/weights_validation_ISO_TRI_BI_2D_BV.py \
        [--results-dir <dir>] [--output-dir <dir>] [--ema]
"""

import argparse
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir

# BV regularization function types (not stored in config; hardcoded here)
BV_REG_FUNCTIONS = ["L1"]


def main():
    parser = argparse.ArgumentParser(
        description="2D hyperparameter sweep weights validation analysis (Exp3)."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. If omitted, auto-discovered from config results_prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, derived from results-dir basename.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA results (results_EMA.hdf5).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to ../config.yaml relative to this script.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret --results-dir / --output-dir as absolute paths.",
    )
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = script_dir.parent

    # Load config
    config_path = Path(args.config) if args.config else script_dir / ".." / "config.yaml"
    cfg = ExperimentConfig.from_yaml(config_path)

    # Resolve results directory
    if args.results_dir:
        results_dir = (
            Path(args.results_dir)
            if args.absolute_paths
            else (script_dir / args.results_dir).resolve()
        )
    elif cfg.results_prefix:
        prefix_path = Path(cfg.results_prefix)
        base_path = exp_dir / prefix_path.parent
        prefix = prefix_path.name
        found = find_most_recent_dir(base_path, prefix)
        if found is None:
            raise FileNotFoundError(
                f"No directory matching prefix '{prefix}' found in {base_path}"
            )
        results_dir = found
    elif cfg.results_dir:
        results_dir = exp_dir / cfg.results_dir
    else:
        raise ValueError("Provide --results-dir or set results_prefix/results_dir in config.")

    results_dir = str(results_dir)

    # Resolve output directory
    if args.output_dir:
        output_dir = (
            args.output_dir
            if args.absolute_paths
            else str((script_dir / args.output_dir).resolve())
        )
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = str(script_dir / f"_analysis_weights_2d_{base_name}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Results directory : {results_dir}")
    print(f"Output directory  : {output_dir}")
    print(f"EMA               : {args.ema}")
    print(f"Ensembles         : {cfg.ensembles}")
    print(f"Loss functions    : {cfg.loss_functions}")
    print(f"BV reg functions  : {BV_REG_FUNCTIONS}")
    print("-" * 60)

    # Load optimization results
    print("\nLoading 2D optimization results...")
    results = loading.load_all_optimization_results_2d(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        bv_reg_functions=BV_REG_FUNCTIONS,
        num_splits=cfg.num_splits,
        EMA=args.ema,
        verbose=True,
    )

    # Extract final weights and compute metrics
    print("\n" + "=" * 60)
    print("EXTRACTING WEIGHTS AND COMPUTING METRICS")
    print("=" * 60)

    weights_df = analysis.extract_final_weights_2d(results)

    if not weights_df.empty:
        print(f"Extracted {len(weights_df)} weight distributions")

        weights_path = os.path.join(output_dir, "weights_2d_sweep_data.csv")
        weights_df.to_csv(weights_path, index=False)
        print(f"Weights data saved to: {weights_path}")

        # 2D heatmaps
        print("\nGenerating 2D heatmaps...")
        plotting.plot_2d_heatmaps_grid(
            weights_df, output_dir,
            metric="effective_sample_size", metric_label="Effective Sample Size",
        )
        plotting.plot_2d_heatmaps_grid(
            weights_df, output_dir,
            metric="kl_divergence", metric_label="KL Divergence",
        )

        # 1D slices
        print("\nGenerating 1D slices...")
        plotting.plot_1d_slices_2d_sweep(
            weights_df, output_dir,
            metric="effective_sample_size", metric_label="Effective Sample Size",
        )
        plotting.plot_1d_slices_2d_sweep(
            weights_df, output_dir,
            metric="kl_divergence", metric_label="KL Divergence",
        )

        # Summary statistics
        print("\n" + "=" * 60)
        print("WEIGHTS SUMMARY — 2D SWEEP")
        print("=" * 60)
        for ensemble in cfg.ensembles:
            ens_data = weights_df[weights_df["ensemble"] == ensemble]
            if not ens_data.empty:
                print(f"\n{ensemble}:")
                print(f"  Mean KL divergence : {ens_data['kl_divergence'].mean():.3f}")
                print(f"  Mean ESS           : {ens_data['effective_sample_size'].mean():.1f}")
                print(
                    f"  Mean weight range  : [{ens_data['weight_min'].mean():.4f}, "
                    f"{ens_data['weight_max'].mean():.4f}]"
                )
    else:
        print("No weights data extracted — check results directory.")

    # Compute KLD between splits
    print("\n" + "=" * 60)
    print("COMPUTING KLD BETWEEN SPLITS")
    print("=" * 60)

    kld_df = analysis.compute_pairwise_kld_between_splits_2d(results)

    if not kld_df.empty:
        print(f"Computed KLD for {len(kld_df)} parameter combinations")

        kld_path = os.path.join(output_dir, "kld_between_splits_2d_sweep_data.csv")
        kld_df.to_csv(kld_path, index=False)
        print(f"KLD data saved to: {kld_path}")

        print("\nGenerating KLD between splits 2D heatmaps...")
        plotting.plot_2d_heatmaps_grid(
            kld_df, output_dir,
            metric="mean_kld_between_splits",
            metric_label="Mean KLD Between Splits",
            cmap="Blues",
        )

        # KLD summary
        print("\n" + "=" * 60)
        print("KLD BETWEEN SPLITS SUMMARY")
        print("=" * 60)
        for ensemble in cfg.ensembles:
            ens_kld = kld_df[kld_df["ensemble"] == ensemble]
            if not ens_kld.empty:
                print(f"\n{ensemble}:")
                print(
                    f"  Mean KLD between splits : "
                    f"{ens_kld['mean_kld_between_splits'].mean():.3f}"
                )
                print(
                    f"  Std  KLD between splits : "
                    f"{ens_kld['mean_kld_between_splits'].std():.3f}"
                )
    else:
        print("No KLD data computed — need ≥2 splits per parameter combination.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
