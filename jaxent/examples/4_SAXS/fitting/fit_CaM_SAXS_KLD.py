"""SAXS-only MaxEnt reweighting for CaM pulldown data.

Fits CaM ensemble SAXS curves against experimental data for two conditions
(CaM+CDZ and CaM-CDZ) using:
  - Chi-squared loss between predicted and experimental SAXS intensities
  - KL divergence regularisation (MaxEnt) to prevent overfitting

CLI-driven for bash-level parallelisation (one call per combination).

Usage:
    python fit_CaM_SAXS_KLD.py \\
        --target CaM+CDZ \\
        --split-type random-stratified \\
        --split-index 0 \\
        --maxent-strength 0.01 \\
        --n-steps 50000 \\
        --output-dir _optimise_CaM_SAXS_KLD
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

# Register MaxEnt loss
import jaxent.src.opt.loss.weights  # noqa: F401

from jaxent.src.models.SAXS.forwardmodel import SAXS_direct_model
from jaxent.src.models.SAXS.config import SAXS_direct_Config
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.opt.run import run_optimise
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.loss.base import LossRegistry
from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.utils.hdf import save_optimization_history_to_file
from jaxent.src.data.splitting.mapping import QSubsetMapping
from jaxent.src.custom_types.key import m_key

from common import load_dat, find_q_indices, create_chi2_loss
from paths import EXPERIMENTAL_DATA, SAXS_FEATURES_PATH, FITTING_DIR



def load_saxs_features() -> np.ndarray:
    """Load per-frame SAXS features, shape (n_q, n_frames)."""
    return np.load(SAXS_FEATURES_PATH)["intensities"]


def load_datasplit(target: str, split_type: str, split_idx: int) -> tuple:
    """Load train/val q-indices and experimental intensities for one split.

    Returns:
        (train_idx, train_I, val_idx, val_I) — integer index arrays and
        intensity arrays extracted from the full experimental curve.
    """
    splits_dir = FITTING_DIR / f"_datasplits_{target}" / split_type / f"split_{split_idx:03d}"
    train_npz = np.load(splits_dir / "train.npz")
    val_npz = np.load(splits_dir / "val.npz")

    full_q, full_I, _ = load_dat(EXPERIMENTAL_DATA[target])

    train_idx = find_q_indices(full_q, train_npz["q_values"])
    val_idx = find_q_indices(full_q, val_npz["q_values"])

    return train_idx, full_I[train_idx], val_idx, full_I[val_idx]


def main():
    parser = argparse.ArgumentParser(description="SAXS-only MaxEnt reweighting for CaM pulldown data")
    parser.add_argument("--target", choices=["CaM+CDZ", "CaM-CDZ"], required=True)
    parser.add_argument("--split-type", choices=["random", "stratified", "random-stratified"], required=True)
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument("--maxent-strength", type=float, required=True)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # --- Load data ---
    full_q, _, _ = load_dat(EXPERIMENTAL_DATA[args.target])
    curve_matrix = load_saxs_features()  # (n_q, n_frames)
    train_idx, train_I, val_idx, val_I = load_datasplit(
        args.target, args.split_type, args.split_index
    )
    n_frames = curve_matrix.shape[1]

    # --- Build datasets ---
    train_dataset = Dataset(
        data=[],
        y_true=jnp.array(train_I),
        data_mapping=QSubsetMapping(jnp.array(train_idx)),
    )
    val_dataset = Dataset(
        data=[],
        y_true=jnp.array(val_I),
        data_mapping=QSubsetMapping(jnp.array(val_idx)),
    )
    saxs_dataloader = ExpD_Dataloader(
        train=train_dataset,
        val=val_dataset,
        key=m_key("SAXS_Iq"),
    )

    # --- Loss functions ---
    chi2_loss = create_chi2_loss()
    maxent_loss = LossRegistry.get("maxent_convex_kl")

    # --- Model ---
    saxs_features = SAXS_curve_input_features(intensities=jnp.array(curve_matrix))
    saxs_config = SAXS_direct_Config(q_values=jnp.array(full_q))
    saxs_model = SAXS_direct_model(config=saxs_config)

    # --- Parameters ---
    uniform_weights = jnp.ones(n_frames) / n_frames
    init_params = Simulation_Parameters(
        frame_weights=uniform_weights,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(SAXS_direct_Model_Parameters(),),
        forward_model_weights=jnp.array([1.0, args.maxent_strength]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * 1000.0,
    )
    prior_params = Simulation_Parameters(
        frame_weights=uniform_weights,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(SAXS_direct_Model_Parameters(),),
        forward_model_weights=jnp.array([1.0, args.maxent_strength]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * 1000.0,
    )

    # --- Simulation ---
    sim = Simulation(
        input_features=[saxs_features],
        forward_models=[saxs_model],
        params=init_params,
    )
    sim.initialise()

    # --- Optimizer ---
    run_name = (
        f"SAXS_{args.target}_{args.split_type}"
        f"_split{args.split_index:03d}_maxent{args.maxent_strength}"
    )
    config = OptimiserSettings(
        name=run_name,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        convergence=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
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

    # --- Run optimisation ---
    _sim, history = run_optimise(
        simulation=sim,
        data_to_fit=[saxs_dataloader, prior_params],
        config=config,
        forward_models=[saxs_model],
        indexes=["SAXS_Iq", None],
        loss_functions=[chi2_loss, maxent_loss],
        optimizer=optimizer,
    )

    # --- Save results ---
    output_path = (
        Path(args.output_dir) / "SAXS" / args.target
        / args.split_type / f"split_{args.split_index:03d}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "scenario": "SAXS",
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
