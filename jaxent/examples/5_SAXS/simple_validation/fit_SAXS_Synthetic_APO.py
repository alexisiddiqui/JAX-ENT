"""
Reweights the FOXS curves (features) in jaxent/examples/5_SAXS/simple_validation/_synthetic_SAXS_features
fits against jaxent/examples/5_SAXS/FOXS/1CLL_apo.pdb.dat

Uses datasplits from jaxent/examples/5_SAXS/simple_validation/_datasplits

Fits 3 replicates for each split type (random, stratified, data-cluster)

Loss functions:
- MSE
- Chi2 (hardcode into the script for now)

Vary maxENT strength (similar to IsoValidation example)

Due to the limitations of the current implementation - parallel processing is performed at the bash script level.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np

# Register maxent loss
import jaxent.src.opt.loss.weights

from jaxent.src.models.SAXS.forwardmodel import SAXS_direct_model
from jaxent.src.models.SAXS.config import SAXS_direct_Config
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.opt.run import run_optimise
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.loss.base import create_functional_loss, LossRegistry
from jaxent.src.data.loader import Dataset
from jaxent.src.utils.hdf import save_optimization_history_to_file
from jaxent.src.data.splitting.mapping import QSubsetMapping
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.custom_types.key import m_key
SCRIPT_DIR = Path(__file__).resolve().parent





def find_q_indices(full_q, subset_q, tol=1e-7):
    """Find integer indices of subset_q values in full_q array."""
    return np.array([np.where(np.abs(full_q - q) < tol)[0][0] for q in subset_q])


def load_experimental_curve(curve_type: str) -> tuple:
    """Load experimental curve (q, I, err) from FOXS directory."""
    FOXS_DIR = SCRIPT_DIR.parent.parent / "5_SAXS" / "FOXS" / "missing_residues"
    fname = "1CLL_apo.pdb.dat" if curve_type == "APO" else "1CLL_nosol.pdb.dat"
    dat = np.loadtxt(FOXS_DIR / fname, comments="#")
    return dat[:, 0], dat[:, 1], dat[:, 2]


def load_foxs_features() -> np.ndarray:
    """Load per-frame FOXS curves, shape (501, 12700)."""
    FOXS_DIR = SCRIPT_DIR.parent.parent / "5_SAXS" / "FOXS"
    foxs_data = np.load(FOXS_DIR / "CaM_SAXS_ordered.npz")
    # Load (12700, 501) and transpose to (501, 12700)
    return foxs_data["saxs"].T


def load_datasplit(curve_type: str, split_type: str, split_idx: int) -> tuple:
    """Load train/val datasplit, return q-indices and intensity arrays."""
    SPLITS_DIR = SCRIPT_DIR / f"_datasplits_{curve_type}"
    split_path = SPLITS_DIR / split_type / f"split_{split_idx:03d}"

    train_data = np.load(split_path / "train.npz")
    val_data = np.load(split_path / "val.npz")

    full_q, _, _ = load_experimental_curve(curve_type)

    train_idx = find_q_indices(full_q, train_data["q_values"])
    val_idx = find_q_indices(full_q, val_data["q_values"])

    return (train_idx, train_data["intensities"],
            val_idx, val_data["intensities"])


def create_mse_loss():
    """Create MSE loss function."""
    return create_functional_loss(
        lambda p, t: jnp.mean((p - t)**2),
        post_mean=False,
        flatten=True
    )


def create_chi2_loss():
    """Create Chi2 loss function."""
    def chi2_fn(p, t):
        var_est = jnp.var(t) + 1e-10
        return jnp.mean((p - t)**2 / var_est)

    return create_functional_loss(
        chi2_fn,
        post_mean=False,
        flatten=True
    )


def main():
    parser = argparse.ArgumentParser(description="SAXS reweighting fitting")
    parser.add_argument("--split-type",
                        choices=["random", "stratified", "random-stratified"],
                        required=True)
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument("--maxent-strength", type=float, required=True)
    parser.add_argument("--loss-function",
                        choices=["MSE", "Chi2"],
                        required=True)
    parser.add_argument("--target-curve",
                        choices=["APO", "nosol"],
                        required=True)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    # Load data
    full_q, full_I, full_err = load_experimental_curve(args.target_curve)
    curve_matrix = load_foxs_features()
    train_idx, train_I_split, val_idx, val_I_split = load_datasplit(args.target_curve, args.split_type, args.split_index)

    # Use intensities from the full curve indexed by split indices to ensure match
    train_I = full_I[train_idx]
    val_I = full_I[val_idx]

    n_frames = curve_matrix.shape[1]

    # Build datasets with q-subset mappings
    train_dataset = Dataset(
        data=[],
        y_true=jnp.array(train_I),
        data_mapping=QSubsetMapping(jnp.array(train_idx))
    )
    val_dataset = Dataset(
        data=[],
        y_true=jnp.array(val_I),
        data_mapping=QSubsetMapping(jnp.array(val_idx))
    )
    saxs_dataloader = ExpD_Dataloader(train=train_dataset, val=val_dataset, key=m_key("SAXS_Iq"))

    # Select loss function
    if args.loss_function == "MSE":
        active_loss = create_mse_loss()
    else:
        active_loss = create_chi2_loss()

    maxent_loss = LossRegistry.get("maxent_convex_kl")

    # Initialize simulation parameters
    init_params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(SAXS_direct_Model_Parameters(),),
        forward_model_weights=jnp.array([1.0, args.maxent_strength]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2)*1000.0,
    )

    # Prior params for MaxEnt (same as init but with uniform weights)
    prior_params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(SAXS_direct_Model_Parameters(),),
        forward_model_weights=jnp.array([1.0, args.maxent_strength]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2)*1000.0,
    )

    # Setup SAXS model
    saxs_features = SAXS_curve_input_features(intensities=jnp.array(curve_matrix))
    saxs_config = SAXS_direct_Config(q_values=jnp.array(full_q))
    saxs_model = SAXS_direct_model(config=saxs_config)

    # Create simulation
    sim = Simulation(
        input_features=[saxs_features],
        forward_models=[saxs_model],
        params=init_params
    )
    sim.initialise()

    # Setup optimizer
    optimizer = OptaxOptimizer(
        learning_rate=args.learning_rate,
        parameter_partition_masks={Optimisable_Parameters.frame_weights},
        initial_learning_rate=1.0,
        initial_steps=2,
    )

    # Configure optimizer settings
    run_name = f"{args.target_curve}_{args.loss_function}_{args.split_type}_split{args.split_index:03d}_maxent{args.maxent_strength}"
    config = OptimiserSettings(
        name=run_name,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        convergence=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        tolerance=1e-10,
        ema_alpha=0.5,
        min_steps_per_threshold=2,
    )

    # Run optimization
    _sim, history = run_optimise(
        simulation=sim,
        data_to_fit=[saxs_dataloader, prior_params],
        config=config,
        forward_models=[saxs_model],
        indexes=["SAXS_Iq", None],
        loss_functions=[active_loss, maxent_loss],
        optimizer=optimizer,
    )

    # Prepare output directory
    output_path = Path(args.output_dir) / args.target_curve / args.split_type / f"split_{args.split_index:03d}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config as JSON
    config_dict = {
        "split_type": args.split_type,
        "split_index": args.split_index,
        "maxent_strength": args.maxent_strength,
        "loss_function": args.loss_function,
        "target_curve": args.target_curve,
        "n_steps": args.n_steps,
        "learning_rate": args.learning_rate,
    }
    with open(output_path / f"{run_name}_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save optimization history
    save_optimization_history_to_file(
        str(output_path / f"{run_name}_results.hdf5"),
        history
    )

    # Save EMA history if available
    if optimizer.ema_history is not None:
        save_optimization_history_to_file(
            str(output_path / f"{run_name}_results_EMA.hdf5"),
            optimizer.ema_history
        )

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
