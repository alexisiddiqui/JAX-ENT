"""HDX-only MaxEnt reweighting for CaM pulldown data.

Fits CaM ensemble HDX curves against experimental data for two conditions
(CaM+CDZ and CaM-CDZ) using:
  - Sigma MSE loss between predicted and experimental HDX uptake
  - KL divergence regularisation (MaxEnt) to prevent overfitting

CLI-driven for bash-level parallelisation (one call per combination).

Usage:
    python fit_CaM_HDX_KLD.py \
        --target CaM+CDZ \
        --split-type sequence-cluster \
        --split-index 0 \
        --maxent-strength 0.01 \
        --n-steps 50000 \
        --output-dir _optimise_CaM_HDX_KLD
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np

# Register MaxEnt and HDX losses
import jaxent.src.opt.loss.weights  # noqa: F401
from jaxent.examples.common.losses import get_loss_function_by_name

from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.opt.run import run_optimise
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.loss.base import LossRegistry
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.custom_types.key import m_key
from jaxent.src.custom_types.HDX import HDX_peptide
import jaxent.src.interfaces.topology as pt
from jaxent.src.utils.hdf import save_optimization_history_to_file
from paths import HDX_FEATURES_PATH, HDX_TOPOLOGY_PATH, FITTING_DIR



def load_hdx_split(target: str, split_type: str, split_idx: int) -> tuple[list[HDX_peptide], list[HDX_peptide], jnp.ndarray]:
    """Load train/val HDX data and the full covariance matrix for the target."""
    splits_dir = FITTING_DIR / f"_datasplits_HDX_{target}"
    split_path = splits_dir / split_type / f"split_{split_idx:03d}"
    
    train_data = HDX_peptide.load_list_from_files(
        json_path=split_path / "train_topology.json",
        csv_path=split_path / "train_dfrac.csv",
    )
    val_data = HDX_peptide.load_list_from_files(
        json_path=split_path / "val_topology.json",
        csv_path=split_path / "val_dfrac.csv",
    )
    
    cov_path = splits_dir / "_covariance_matrices" / "Sigma.npz"
    # Load Sigma_inv and normalise for numerical stability
    cov_matrix_data = jnp.load(cov_path)["Sigma_inv"]
    cov_matrix_data = cov_matrix_data / jnp.linalg.norm(cov_matrix_data)
    
    return train_data, val_data, cov_matrix_data


def main():
    parser = argparse.ArgumentParser(description="HDX-only MaxEnt reweighting for CaM pulldown data")
    parser.add_argument("--target", choices=["CaM+CDZ", "CaM-CDZ"], required=True)
    parser.add_argument("--split-type", required=True)
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument("--maxent-strength", type=float, required=True)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # --- Load Data ---
    train_data, val_data, cov_matrix_data = load_hdx_split(args.target, args.split_type, args.split_index)
    
    features = BV_input_features.load(HDX_FEATURES_PATH)
    feature_top = pt.PTSerialiser.load_list_from_json(HDX_TOPOLOGY_PATH)
    n_frames = features.features_shape[1]

    hdx_dataloader = ExpD_Dataloader(
        data=train_data + val_data,
        covariance_matrix=cov_matrix_data,
        key=m_key("HDX_peptide")
    )
    hdx_dataloader.create_datasets(
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_topology=feature_top
    )
    
    # --- Model Setup ---
    bv_config = BV_model_Config(num_timepoints=7)
    bv_config.timepoints = jnp.array([0.167, 0.333, 0.500, 1.0, 10.0, 30.0, 120.0])
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params
    
    # --- Loss functions ---
    hdx_loss_fn = get_loss_function_by_name("hdx_uptake_sigma_MSE_loss")
    maxent_loss = LossRegistry.get("maxent_convex_kl")
    
    # --- Parameters Setup ---
    uniform_weights = jnp.ones(n_frames) / n_frames
    init_params = Simulation_Parameters(
        frame_weights=uniform_weights,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([1.0, args.maxent_strength]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * 1000.0,  # Based on other HDX optimise scales
    )
    prior_params = Simulation_Parameters(
        frame_weights=uniform_weights,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([1.0, args.maxent_strength]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * 100.0,
    )
    
    sim = Simulation(
        input_features=[features],
        forward_models=[bv_model],
        params=init_params,
    )
    sim.initialise()
    
    # --- Optimizer ---
    run_name = (
        f"HDX_{args.target}_{args.split_type}"
        f"_split{args.split_index:03d}_maxent{args.maxent_strength}"
    )
    config = OptimiserSettings(
        name=run_name,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        convergence=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        tolerance=1e-10,
        ema_alpha=0.5,
        min_steps_per_threshold=2,
    )
    optimizer = OptaxOptimizer(
        learning_rate=args.learning_rate,
        parameter_partition_masks={Optimisable_Parameters.frame_weights},
        initial_learning_rate=1.0,
        initial_steps=2,
    )
    
    # --- Run Optimization ---
    _sim, history = run_optimise(
        simulation=sim,
        data_to_fit=[hdx_dataloader, prior_params],
        config=config,
        forward_models=[bv_model],
        indexes=[0, 0],
        loss_functions=[hdx_loss_fn, maxent_loss],
        optimizer=optimizer,
    )
    
    # --- Save results ---
    output_path = (
        Path(args.output_dir) / "HDX" / args.target
        / args.split_type / f"split_{args.split_index:03d}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        "scenario": "HDX",
        "target": args.target,
        "split_type": args.split_type,
        "split_index": args.split_index,
        "maxent_strength": args.maxent_strength,
        "n_steps": args.n_steps,
        "learning_rate": args.learning_rate,
    }
    with open(output_path / f"{run_name}_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    save_optimization_history_to_file(
        str(output_path / f"{run_name}_results.hdf5"),
        history,
    )
    if optimizer.ema_history is not None:
        save_optimization_history_to_file(
            str(output_path / f"{run_name}_results_EMA.hdf5"),
            optimizer.ema_history,
        )

    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()