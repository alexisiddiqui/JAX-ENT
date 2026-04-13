"""Combined SAXS and HDX-MS MaxEnt reweighting for CaM pulldown data.

Fits CaM ensemble SAXS and HDX curves against experimental data for two conditions
(CaM+CDZ and CaM-CDZ) using:
  - Chi-squared loss between predicted and experimental SAXS intensities
  - Sigma MSE loss between predicted and experimental HDX uptake
  - KL divergence regularisation (MaxEnt) to prevent overfitting

CLI-driven for bash-level parallelisation.

Usage:
    python fit_CaM_HDX_SAXS_KLD.py \
        --target CaM+CDZ \
        --hdx-split-index 0 \
        --saxs-split-index 0 \
        --hdx-weight 1.0 \
        --saxs-weight 1.0 \
        --maxent-strength 0.01 \
        --n-steps 50000 \
        --output-dir _optimise_CaM_HDX_SAXS_KLD
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

from jaxent.src.models.SAXS.forwardmodel import SAXS_direct_model
from jaxent.src.models.SAXS.config import SAXS_direct_Config
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters

from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.HDX.BV.features import BV_input_features

from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.opt.run import run_optimise
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.loss.base import LossRegistry
from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.custom_types.key import m_key
from jaxent.src.custom_types.HDX import HDX_peptide
import jaxent.src.interfaces.topology as pt
from jaxent.src.utils.hdf import save_optimization_history_to_file
from jaxent.src.data.splitting.mapping import QSubsetMapping

from common import load_dat, find_q_indices, create_chi2_loss
from paths import (
    EXPERIMENTAL_DATA, SAXS_FEATURES_PATH,
    HDX_FEATURES_PATH, HDX_TOPOLOGY_PATH,
    FITTING_DIR
)


HDX_SPLIT_TYPE = "sequence_cluster"
SAXS_SPLIT_TYPE = "random-stratified"


def load_saxs_features() -> np.ndarray:
    """Load per-frame SAXS features, shape (n_q, n_frames)."""
    return np.load(SAXS_FEATURES_PATH)["intensities"]


def load_saxs_datasplit(target: str, split_idx: int) -> tuple:
    """Load train/val q-indices and experimental intensities for SAXS split."""
    splits_dir = FITTING_DIR / f"_datasplits_{target}" / SAXS_SPLIT_TYPE / f"split_{split_idx:03d}"
    train_npz = np.load(splits_dir / "train.npz")
    val_npz = np.load(splits_dir / "val.npz")

    full_q, full_I, _ = load_dat(EXPERIMENTAL_DATA[target])
    train_idx = find_q_indices(full_q, train_npz["q_values"])
    val_idx = find_q_indices(full_q, val_npz["q_values"])
    return train_idx, full_I[train_idx], val_idx, full_I[val_idx]


def load_hdx_split(target: str, split_idx: int) -> tuple[list[HDX_peptide], list[HDX_peptide], jnp.ndarray]:
    """Load train/val HDX data and the full covariance matrix for the target."""
    splits_dir = FITTING_DIR / f"_datasplits_HDX_{target}"
    split_path = splits_dir / HDX_SPLIT_TYPE / f"split_{split_idx:03d}"
    
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
    parser = argparse.ArgumentParser(description="Combined SAXS and HDX MaxEnt reweighting")
    parser.add_argument("--target", choices=["CaM+CDZ", "CaM-CDZ"], required=True)
    parser.add_argument("--hdx-split-index", type=int, default=None)
    parser.add_argument("--saxs-split-index", type=int, default=None)
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument("--saxs-weight", type=float, default=1.0)
    parser.add_argument("--hdx-weight", type=float, default=1.0)
    parser.add_argument("--maxent-strength", type=float, required=True)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Determine indices
    hdx_idx = args.split_index if args.split_index is not None else args.hdx_split_index
    saxs_idx = args.split_index if args.split_index is not None else args.saxs_split_index

    if hdx_idx is None or saxs_idx is None:
        parser.error("Must provide either --split-index or both --hdx-split-index and --saxs-split-index")

    # --- Load SAXS Data ---
    full_q, _, _ = load_dat(EXPERIMENTAL_DATA[args.target])
    saxs_curve_matrix = load_saxs_features()  # (n_q, n_frames)
    saxs_train_idx, saxs_train_I, saxs_val_idx, saxs_val_I = load_saxs_datasplit(
        args.target, saxs_idx
    )
    n_frames = saxs_curve_matrix.shape[1]

    saxs_train_dataset = Dataset(
        data=[],
        y_true=jnp.array(saxs_train_I),
        data_mapping=QSubsetMapping(jnp.array(saxs_train_idx)),
    )
    saxs_val_dataset = Dataset(
        data=[],
        y_true=jnp.array(saxs_val_I),
        data_mapping=QSubsetMapping(jnp.array(saxs_val_idx)),
    )
    saxs_dataloader = ExpD_Dataloader(
        train=saxs_train_dataset,
        val=saxs_val_dataset,
        key=m_key("SAXS_Iq"),
    )

    # --- Load HDX Data ---
    hdx_train_data, hdx_val_data, hdx_cov_matrix_data = load_hdx_split(args.target, hdx_idx)
    
    hdx_features = BV_input_features.load(HDX_FEATURES_PATH)
    hdx_feature_top = pt.PTSerialiser.load_list_from_json(HDX_TOPOLOGY_PATH)

    hdx_dataloader = ExpD_Dataloader(
        data=hdx_train_data + hdx_val_data,
        covariance_matrix=hdx_cov_matrix_data,
        key=m_key("HDX_peptide")
    )
    hdx_dataloader.create_datasets(
        train_data=hdx_train_data,
        val_data=hdx_val_data,
        features=hdx_features,
        feature_topology=hdx_feature_top
    )
    
    # --- SAXS Model Setup ---
    saxs_features_obj = SAXS_curve_input_features(intensities=jnp.array(saxs_curve_matrix))
    saxs_config = SAXS_direct_Config(q_values=jnp.array(full_q))
    saxs_model = SAXS_direct_model(config=saxs_config)
    
    # --- HDX Model Setup ---
    bv_config = BV_model_Config(num_timepoints=7)
    bv_config.timepoints = jnp.array([0.167, 0.333, 0.500, 1.0, 10.0, 30.0, 120.0])
    bv_model = BV_model(config=bv_config)

    # --- Loss functions ---
    chi2_loss = create_chi2_loss()
    hdx_loss_fn = get_loss_function_by_name("hdx_uptake_sigma_MSE_loss")
    maxent_loss = LossRegistry.get("maxent_convex_kl")
    
    # --- Parameters Setup ---
    uniform_weights = jnp.ones(n_frames) / n_frames
    
    model_weights = jnp.array([args.saxs_weight, args.hdx_weight, args.maxent_strength])
    model_scaling = jnp.ones(3) 
    
    init_params = Simulation_Parameters(
        frame_weights=uniform_weights,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(SAXS_direct_Model_Parameters(), bv_model.params,),
        forward_model_weights=model_weights,
        normalise_loss_functions=jnp.ones(3),
        forward_model_scaling=model_scaling,
    )
    
    prior_params = Simulation_Parameters(
        frame_weights=uniform_weights,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(SAXS_direct_Model_Parameters(), bv_model.params,),
        forward_model_weights=model_weights,
        normalise_loss_functions=jnp.ones(3),
        forward_model_scaling=model_scaling,
    )
    
    sim = Simulation(
        input_features=[saxs_features_obj, hdx_features],
        forward_models=[saxs_model, bv_model],
        params=init_params,
    )
    sim.initialise()
    
    # --- Optimizer ---
    scenario_str = f"HDX:SAXS_{args.hdx_weight}:{args.saxs_weight}"
    run_name = (
        f"{scenario_str}_{args.target}_combined"
        f"_split{hdx_idx:03d}_maxent{args.maxent_strength}"
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
        data_to_fit=[saxs_dataloader, hdx_dataloader, prior_params],
        config=config,
        forward_models=[saxs_model, bv_model],
        indexes=[0, 1, None],
        loss_functions=[chi2_loss, hdx_loss_fn, maxent_loss],
        optimizer=optimizer,
    )
    
    # --- Save results ---
    output_path = (
        Path(args.output_dir) / "HDX_SAXS" / args.target
        / "combined" / f"split_{hdx_idx:03d}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        "scenario": scenario_str,
        "target": args.target,
        "hdx_split_type": HDX_SPLIT_TYPE,
        "saxs_split_type": SAXS_SPLIT_TYPE,
        "hdx_split_index": hdx_idx,
        "saxs_split_index": saxs_idx,
        "hdx_weight": args.hdx_weight,
        "saxs_weight": args.saxs_weight,
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