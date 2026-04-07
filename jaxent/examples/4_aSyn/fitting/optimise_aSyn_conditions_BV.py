"""
Optimisation Sweep for aSyn using BV Forward Model (Cross-Validation Splits)

Function:
Performs 2D hyperparameter sweep over MaxEnt scaling and Best-Vendruscolo (BV) regularization.
Supports MSE loss function with L1 or L2 BV regularization.
The script iterates through data splits and conditions (Tris_only, Extracellular,
Intracellular, Lysosomal) to find optimal parameters for conformational ensemble reweighting.

Data type: HDX_protection_factor (log_Pf stored per residue).
BV model outputs log_Pf via BV_ForwardPass (num_timepoints=None).

Requirements:
- fitting/_datasplits/{condition}/{split_type}/ : Per-condition train/val splits
- data/_cluster_aSyn/features/features.npz    : Pre-computed BV features (shared)
- data/_cluster_aSyn/features/topology.json   : Feature topology (shared)

Usage Examples:
# Run a specific condition (default 3 replicates per split)
python optimise_aSyn_conditions_BV.py \\
    --condition Tris_only \\
    --loss-function MSE \\
    --maxent-range 1,10 \\
    --bvreg-range 0.0,1.0 \\
    --bv-reg-function L1 \\
    --output-dir ./results

# Quick smoke test
python optimise_aSyn_conditions_BV.py \\
    --condition Tris_only \\
    --loss-function MSE \\
    --maxent-range 1,1 \\
    --bvreg-range 0.5,0.5 \\
    --n-steps 10 \\
    --output-dir /tmp/test_aSyn_opt
"""

import argparse
import os
import time
from datetime import datetime
from typing import List

import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.utils.jit_fn import jit_Guard

from jaxent.examples.common.optimization import run_optimization
from jaxent.examples.common.config import LossConfig, OptimizationConfig


CONDITIONS = ["Tris_only", "Extracellular", "Intracellular", "Lysosomal"]

LOSS_NAME_MAP = {
    "MSE": "hdx_pf_l2_loss",
}


def load_data_splits(datasplit_dir: str, condition: str, split_type: str, num_splits: int) -> List[tuple]:
    """
    Load all data splits for a given condition and split type.

    Args:
        datasplit_dir: Base directory containing per-condition data splits
        condition: Experimental condition (e.g. 'Tris_only')
        split_type: Type of split (e.g. 'sequence_cluster', 'spatial')
        num_splits: Number of splits to load

    Returns:
        List of (train_data, val_data) tuples of HDX_protection_factor objects
    """
    splits = []
    split_type_dir = os.path.join(datasplit_dir, condition, split_type)

    if not os.path.exists(split_type_dir):
        raise FileNotFoundError(f"Split type directory not found: {split_type_dir}")

    for split_idx in range(num_splits):
        split_path = os.path.join(split_type_dir, f"split_{split_idx:03d}")

        train_data = HDX_protection_factor.load_list_from_files(
            json_path=os.path.join(split_path, "train_topology.json"),
            csv_path=os.path.join(split_path, "train_dfrac.csv"),
        )
        val_data = HDX_protection_factor.load_list_from_files(
            json_path=os.path.join(split_path, "val_topology.json"),
            csv_path=os.path.join(split_path, "val_dfrac.csv"),
        )
        splits.append((train_data, val_data))

    return splits


def run_conditions_sweep(
    condition: str,
    loss_name: str,
    split_types_arg: str,
    maxent_values: List[float],
    bv_reg_values: List[float],
    bv_reg_name: str,
    n_steps: int = 10000,
    num_splits: int = 3,
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    output_base_dir: str = None,
    model_parameters_lr_scale: float = 1.0,
) -> dict:
    """
    Run optimization sweep across maxent and BV regularization values for one condition.

    Args:
        condition: Experimental condition to fit (one of CONDITIONS)
        loss_name: Loss function name ('MSE')
        split_types_arg: Comma-separated split types or 'all'
        maxent_values: List of maxent scaling values to sweep
        bv_reg_values: List of BV regularization scaling values to sweep
        bv_reg_name: BV regularization type ('L1' or 'L2')
        n_steps: Number of optimization steps per run
        num_splits: Number of replicates per split type
        output_base_dir: Base directory for outputs (timestamped default if None)

    Returns:
        dict: Results summary
    """
    start_time = time.time()
    print(f"Starting sweep for condition={condition}, loss={loss_name}")
    print(f"Maxent values: {maxent_values}")
    print(f"BV reg values: {bv_reg_values}")

    convergence_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasplit_dir = os.path.join(script_dir, "_datasplits")
    feature_path = os.path.join(script_dir, "../data/_cluster_aSyn/features/features.npz")
    topology_path = os.path.join(script_dir, "../data/_cluster_aSyn/features/topology.json")

    if output_base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = os.path.join(script_dir, f"_optimise_quick_test_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.exists(datasplit_dir):
        raise FileNotFoundError(f"Datasplit directory not found: {datasplit_dir}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Features file not found: {feature_path}")
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    features = BV_input_features.load(feature_path)
    feature_top = pt.PTSerialiser.load_list_from_json(topology_path)

    bv_config = BV_model_Config(num_timepoints=None)
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params

    # Discover split types from the condition directory
    condition_dir = os.path.join(datasplit_dir, condition)
    if not os.path.exists(condition_dir):
        raise FileNotFoundError(f"Condition directory not found: {condition_dir}")

    split_types = [
        d for d in os.listdir(condition_dir)
        if os.path.isdir(os.path.join(condition_dir, d))
    ]
    split_types.sort()

    if split_types_arg != "all":
        requested = [s.strip() for s in split_types_arg.split(",")]
        split_types = [s for s in split_types if s in requested]
        if not split_types:
            raise ValueError(f"No valid split types selected from: {requested}")

    results = {
        "condition": condition,
        "loss_name": loss_name,
        "maxent_values": maxent_values,
        "bv_reg_values": bv_reg_values,
        "split_types": split_types,
        "completed_runs": 0,
        "failed_runs": 0,
        "total_runs": len(split_types) * num_splits * len(maxent_values) * len(bv_reg_values),
        "start_time": start_time,
        "run_details": [],
    }

    print(f"Processing {len(split_types)} split types with {num_splits} replicates each")
    print(f"Total runs: {results['total_runs']}")

    n_frames = features.features_shape[1]

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.ones(3),
        normalise_loss_functions=jnp.ones(3),
        forward_model_scaling=jnp.ones(3)*forward_model_scaling,    
    )
    sim = Simulation(input_features=(features,), forward_models=(bv_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()
        sim.forward(sim, params=parameters)
        output_features = sim.outputs[0].y_pred()  # shape (n_residues,)

    prior_HDX = []
    for idx, top in enumerate(feature_top):
        prior_HDX.append(
            HDX_protection_factor._create_from_features(
                topology=top, features=output_features[idx:idx+1]
            )
        )

    prior_dataset = ExpD_Dataloader(data=prior_HDX)
    prior_dataset.create_datasets(features=features, feature_topology=feature_top)

    for split_type in split_types:
        print(f"\nProcessing split type: {split_type}")

        try:
            splits = load_data_splits(datasplit_dir, condition, split_type, num_splits)
        except FileNotFoundError as e:
            print(f"Could not load splits for {condition}/{split_type}. Skipping. Error: {e}")
            continue

        output_dir = os.path.join(output_base_dir, split_type)
        os.makedirs(output_dir, exist_ok=True)

        for split_idx, (train_data, val_data) in enumerate(splits):
            print(f"  Processing split {split_idx:03d}")
            print(f"    Train samples: {len(train_data)}, Val samples: {len(val_data)}")

            for maxent_value in maxent_values:
                for bv_reg_value in bv_reg_values:
                    run_name = (
                        f"{condition}_{loss_name}_{split_type}_split{split_idx:03d}"
                        f"_maxent{maxent_value:.1f}_bvreg{bv_reg_value:.1f}_bvregfn{bv_reg_name}"
                    )
                    print(f"    Running maxent={maxent_value:.1f}, bv_reg={bv_reg_value:.1f}: {run_name}")
                    run_start_time = time.time()

                    try:
                        loss_config = LossConfig(
                            optimize_bv_params=True,
                            primary_loss=LOSS_NAME_MAP[loss_name],
                            regularization_losses=[{"name": f"model_params_{bv_reg_name}_loss"}],
                            bv_reg_scaling=bv_reg_value,
                            maxent_scaling=maxent_value,
                            normalize_bv_reg=False,
                        )
                        opt_config = OptimizationConfig(
                            n_steps=n_steps,
                            learning_rate=learning_rate,
                            initial_learning_rate=initial_learning_rate,
                            initial_steps=initial_steps,
                            ema_alpha=ema_alpha,
                            forward_model_scaling=forward_model_scaling,
                            convergence_rates=convergence_rates,
                            model_parameters_lr_scale=model_parameters_lr_scale,
                            optimizer="adam",
                        )

                        run_optimization(
                            train_data=train_data,
                            val_data=val_data,
                            prior_data=prior_dataset,
                            features=features,
                            forward_model=bv_model,
                            model_parameters=model_parameters,
                            feature_top=feature_top,
                            convergence=convergence_rates,
                            loss_config=loss_config,
                            opt_config=opt_config,
                            name=run_name,
                            output_dir=output_dir,
                            cov_matrix=None,
                        )

                        run_elapsed = time.time() - run_start_time
                        print(f"    Completed: {run_name} ({run_elapsed:.2f} s)")

                        results["completed_runs"] += 1
                        results["run_details"].append(
                            {
                                "split_type": split_type,
                                "split_idx": split_idx,
                                "maxent_value": maxent_value,
                                "bv_reg_value": bv_reg_value,
                                "status": "success",
                                "elapsed_time": run_elapsed,
                                "run_name": run_name,
                            }
                        )

                    except Exception as e:
                        run_elapsed = time.time() - run_start_time
                        print(f"    Failed: {run_name} - Error: {str(e)} ({run_elapsed:.2f} s)")
                        raise RuntimeError(f"Run failed for {run_name} with error: {str(e)}") from e

    total_elapsed = time.time() - start_time
    results["total_elapsed"] = total_elapsed
    results["output_base_dir"] = output_base_dir

    print(f"\n{'=' * 60}")
    print("Sweep completed!")
    print(f"Successful runs: {results['completed_runs']}/{results['total_runs']}")
    print(f"Failed runs: {results['failed_runs']}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")

    print("\nSummary by maxent value:")
    for maxent_value in maxent_values:
        successful = sum(
            1 for r in results["run_details"]
            if r["maxent_value"] == maxent_value and r["status"] == "success"
        )
        total = sum(1 for r in results["run_details"] if r["maxent_value"] == maxent_value)
        success_rate = 100 * successful / total if total > 0 else 0
        print(f"  Maxent {maxent_value:.1f}: {successful}/{total} runs ({success_rate:.1f}%)")

    print("\nSummary by BV reg value:")
    for bv_reg_value in bv_reg_values:
        successful = sum(
            1 for r in results["run_details"]
            if r["bv_reg_value"] == bv_reg_value and r["status"] == "success"
        )
        total = sum(1 for r in results["run_details"] if r["bv_reg_value"] == bv_reg_value)
        success_rate = 100 * successful / total if total > 0 else 0
        print(f"  BV Reg {bv_reg_value:.1f}: {successful}/{total} runs ({success_rate:.1f}%)")

    return results


def run_all_conditions(
    split_types_arg: str,
    maxent_values: List[float],
    bv_reg_values: List[float],
    bv_reg_name: str,
    n_steps: int,
    num_splits: int,
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    output_base_dir: str = None,
    model_parameters_lr_scale: float = 1.0,
) -> List[dict]:
    """Run sweep for all conditions with MSE loss."""
    loss_names = ["MSE"]
    combinations = [(cond, loss) for cond in CONDITIONS for loss in loss_names]

    print(f"Running sweep for all {len(combinations)} condition-loss combinations:")
    print(f"Combinations: {combinations}")
    print(f"Maxent values: {maxent_values}")
    print(f"BV reg values: {bv_reg_values}")
    print(f"Steps per run: {n_steps}")
    print(f"Replicates per split: {num_splits}")
    print("=" * 60)

    all_results = []
    total_start_time = time.time()

    for i, (condition, loss_name) in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Running: {condition}-{loss_name}")
        print("-" * 40)

        try:
            result = run_conditions_sweep(
                condition=condition,
                loss_name=loss_name,
                split_types_arg=split_types_arg,
                maxent_values=maxent_values,
                bv_reg_values=bv_reg_values,
                bv_reg_name=bv_reg_name,
                n_steps=n_steps,
                num_splits=num_splits,
                learning_rate=learning_rate,
                initial_learning_rate=initial_learning_rate,
                initial_steps=initial_steps,
                ema_alpha=ema_alpha,
                forward_model_scaling=forward_model_scaling,
                output_base_dir=output_base_dir,
                model_parameters_lr_scale=model_parameters_lr_scale,
            )
            all_results.append(result)
            print(f"Completed: {condition}-{loss_name}")

        except Exception as e:
            print(f"Failed: {condition}-{loss_name} - Error: {str(e)}")
            continue

    total_elapsed = time.time() - total_start_time

    print(f"\n{'=' * 60}")
    print("All conditions completed!")

    total_runs = sum(r["completed_runs"] + r["failed_runs"] for r in all_results)
    successful_runs = sum(r["completed_runs"] for r in all_results)
    failed_runs = sum(r["failed_runs"] for r in all_results)

    print(f"Total combinations processed: {len(all_results)}/{len(combinations)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    success_rate = 100 * successful_runs / total_runs if total_runs > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total elapsed time: {total_elapsed:.2f} s")

    print("\nPer-condition summary:")
    for result in all_results:
        sr = (
            100 * result["completed_runs"] / result["total_runs"]
            if result["total_runs"] > 0 else 0
        )
        print(
            f"  {result['condition']}-{result['loss_name']}: "
            f"{result['completed_runs']}/{result['total_runs']} runs ({sr:.1f}%) "
            f"in {result['total_elapsed']:.1f}s"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run BV optimization sweep for aSyn conditions."
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=CONDITIONS,
        help="Specific condition to run (optional - if not provided, runs all conditions).",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        choices=["MSE"],
        default="MSE",
        help="Loss function to use (default: MSE).",
    )
    parser.add_argument(
        "--bv-reg-function",
        type=str,
        default="L1",
        choices=["L1", "L2"],
        help="BV regularization type (default: L2).",
    )
    parser.add_argument(
        "--split-types",
        type=str,
        default="all",
        help="Comma-separated split types to run (e.g. 'sequence_cluster,spatial'). Use 'all' for all types.",
    )
    parser.add_argument(
        "--maxent-range",
        type=str,
        default="1,10",
        help="Range of maxent values as 'start,end' (inclusive integers). Default: '1,10'.",
    )
    parser.add_argument(
        "--bvreg-range",
        type=str,
        default="1,10",
        help="Range of BV reg values as 'start,end' (floats). Default: '1,10'.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Number of optimization steps per run (default: 500).",
    )
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=3,
        help="Number of replicates (splits) per split type (default: 3).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-1,
        help="Learning rate for optimizer (default: 1e-1).",
    )
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=1e0,
        help="Initial learning rate (default: 1e0).",
    )
    parser.add_argument(
        "--initial-steps",
        type=int,
        default=2,
        help="Number of initial steps with higher learning rate (default: 2).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.5,
        help="EMA alpha for optimizer (default: 0.5).",
    )
    parser.add_argument(
        "--forward-model-scaling",
        type=float,
        default=1000.0,
        help="Forward model scaling factor (default: 100.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: <script>/_optimise_quick_test_<timestamp>).",
    )
    parser.add_argument(
        "--model-parameters-lr-scale",
        type=float,
        default=1.0,
        help="Multiplier for model-parameter learning rate (default: 1.0).",
    )

    args = parser.parse_args()

    try:
        start_val, end_val = map(int, args.maxent_range.split(","))
        maxent_values = list(range(start_val, end_val + 1))
    except ValueError:
        raise ValueError("maxent-range must be 'start,end' integers (e.g. '1,10')")

    try:
        start_val, end_val = map(float, args.bvreg_range.split(","))
        bvreg_values = [start_val, end_val] if start_val != end_val else [start_val]
    except ValueError:
        raise ValueError("bvreg-range must be 'start,end' floats (e.g. '0.0,1.0')")

    print(f"  Split types: {args.split_types}")
    print(f"  Maxent values: {maxent_values}")
    print(f"  BV reg values: {bvreg_values}")
    print(f"  Steps per run: {args.n_steps}")
    print(f"  Replicates per split: {args.n_replicates}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Initial learning rate: {args.initial_learning_rate}")
    print(f"  Initial steps: {args.initial_steps}")
    print(f"  EMA alpha: {args.ema_alpha}")
    print(f"  Forward model scaling: {args.forward_model_scaling}")
    print(f"  Model parameter LR scale: {args.model_parameters_lr_scale}")

    if args.condition is not None:
        print(f"Running sweep for condition: {args.condition}")
        print(f"  Loss function: {args.loss_function}")
        print(f"  BV reg function: {args.bv_reg_function}")
        print("-" * 60)

        results = run_conditions_sweep(
            condition=args.condition,
            loss_name=args.loss_function,
            split_types_arg=args.split_types,
            maxent_values=maxent_values,
            bv_reg_values=bvreg_values,
            bv_reg_name=args.bv_reg_function,
            n_steps=args.n_steps,
            num_splits=args.n_replicates,
            learning_rate=args.learning_rate,
            initial_learning_rate=args.initial_learning_rate,
            initial_steps=args.initial_steps,
            ema_alpha=args.ema_alpha,
            forward_model_scaling=args.forward_model_scaling,
            output_base_dir=args.output_dir,
            model_parameters_lr_scale=args.model_parameters_lr_scale,
        )

    else:
        all_results = run_all_conditions(
            split_types_arg=args.split_types,
            maxent_values=maxent_values,
            bv_reg_values=bvreg_values,
            bv_reg_name=args.bv_reg_function,
            n_steps=args.n_steps,
            num_splits=args.n_replicates,
            learning_rate=args.learning_rate,
            initial_learning_rate=args.initial_learning_rate,
            initial_steps=args.initial_steps,
            ema_alpha=args.ema_alpha,
            forward_model_scaling=args.forward_model_scaling,
            output_base_dir=args.output_dir,
            model_parameters_lr_scale=args.model_parameters_lr_scale,
        )

    if args.output_dir:
        print(f"\nResults saved in: {args.output_dir}")
    else:
        if "results" in locals():
            print(f"\nResults saved in: {results.get('output_base_dir')}")
        elif "all_results" in locals():
            unique_dirs = sorted({r.get("output_base_dir") for r in all_results if r})
            print("\nResults saved in the following directories:")
            for d in unique_dirs:
                print(f"  {d}")


if __name__ == "__main__":
    main()
