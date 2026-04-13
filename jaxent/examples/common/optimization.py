"""
Shared optimization/fitting functions for JAX-ENT example scripts.

Replaces 5+ near-identical ``run_optimise_*`` variants and the duplicated
``create_data_loaders`` / ``optimise_sweep`` functions from ``optimise_fn.py``.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
import jaxent.src.interfaces.topology as pt
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features, uptake_BV_output_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_Model_Parameters
from jaxent.src.opt.base import InitialisedSimulation, JaxEnt_Loss, OptimizationHistory
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState
from jaxent.src.opt.run import _optimise
from jaxent.src.utils.hdf import save_optimization_history_to_file
from jaxent.src.utils.jit_fn import jit_Guard

from .config import ExperimentConfig, LossConfig, OptimizationConfig
from .losses import get_loss_function_by_name, maxent_convexKL_loss


# ---------------------------------------------------------------------------
# Data loader creation
# ---------------------------------------------------------------------------


def create_data_loaders(
    hdx_data: List[HDX_peptide],
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    feature_top: list[pt.Partial_Topology],
    cov_matrix: Array | None = None,
) -> ExpD_Dataloader:
    """Create data loaders for training and validation datasets.

    Identical across all 3 active copies of ``optimise_fn.py``.
    """
    train_indices = [data.top.fragment_index for data in train_data]
    val_indices = [data.top.fragment_index for data in val_data]
    if cov_matrix is not None:
        print("\n=== Debug Info ===")
        print(f"Full covariance matrix shape: {cov_matrix.shape}")
        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Number of train samples: {len(train_data)}")
        print(f"Number of val samples: {len(val_data)}")

    loader = ExpD_Dataloader(data=hdx_data, covariance_matrix=cov_matrix)
    loader.create_datasets(
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_topology=feature_top,
    )
    return loader


# ---------------------------------------------------------------------------
# EMA-based optimisation sweep (Consolidated to _optimise)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unified run_optimization  (replaces 6+ run_optimise_ISO_TRI_BI_* variants)
# ---------------------------------------------------------------------------


@jit_Guard.test_isolation()
def run_optimization(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_config: LossConfig,
    opt_config: OptimizationConfig | None = None,
    maxent_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "optimization_run",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    """Single entry point replacing all ``run_optimise_ISO_TRI_BI_*`` variants.

    The ``loss_config`` determines:
    - Which primary loss function to use
    - Which regularisation losses and their weights
    - Whether to optimise BV model parameters
    - MaxEnt scaling factor

    When ``opt_config`` is provided, its fields override the corresponding
    keyword arguments.
    """
    # Apply OptimizationConfig overrides
    if opt_config is not None:
        n_steps = opt_config.n_steps
        learning_rate = opt_config.learning_rate
        initial_learning_rate = opt_config.initial_learning_rate
        initial_steps = opt_config.initial_steps
        ema_alpha = opt_config.ema_alpha
        forward_model_scaling = opt_config.forward_model_scaling
        model_parameters_lr_scale = opt_config.model_parameters_lr_scale
        if opt_config.convergence_rates is not None:
            convergence = opt_config.convergence_rates
        if opt_config.covariance_matrix_path and cov_matrix is None:
            import numpy as np
            cov_data = np.load(opt_config.covariance_matrix_path)
            cov_matrix = jnp.array(cov_data[list(cov_data.keys())[0]])

    # Use loss_config maxent_scaling if set
    if loss_config.maxent_scaling != 1.0:
        maxent_scaling = loss_config.maxent_scaling

    # Build loss function list from config
    primary_loss = get_loss_function_by_name(loss_config.primary_loss)
    loss_fn_list: list[JaxEnt_Loss] = [primary_loss, maxent_convexKL_loss]
    data_targets_list: list = []
    indexes_list: list[int] = [0, 0]

    n_loss_slots = 2  # primary + maxent

    for reg in loss_config.regularization_losses:
        reg_loss = get_loss_function_by_name(reg["name"])
        loss_fn_list.append(reg_loss)
        indexes_list.append(0)
        n_loss_slots += 1

    # Create data loaders
    loader = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]

    # Build Simulation_Parameters
    n_reg = len(loss_config.regularization_losses)
    if n_reg > 0:
        _fwd_weights = (
            [maxent_scaling]
            + [1.0] * (n_loss_slots - 1 - n_reg)
            + [loss_config.bv_reg_scaling] * n_reg
        )
    else:
        _fwd_weights = [maxent_scaling] + [1.0] * (n_loss_slots - 1)

    _norm_fns = jnp.ones(n_loss_slots)
    if not loss_config.normalize_bv_reg and n_reg > 0:
        _norm_fns = _norm_fns.at[-1].set(0.0)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array(_fwd_weights),
        normalise_loss_functions=_norm_fns,
        forward_model_scaling=jnp.ones(n_loss_slots) * forward_model_scaling,
    )

    # Build data_to_fit tuple
    data_targets_list = [loader, parameters]
    if loss_config.regularization_losses:
        data_targets_list.append(parameters)

    # Create simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        # Build optimizer with appropriate parameter masks
        partition_masks = {Optimisable_Parameters.frame_weights}
        if loss_config.optimize_bv_params:
            partition_masks.add(Optimisable_Parameters.model_parameters)

        optimizer_type = opt_config.optimizer if opt_config is not None else "adam"
        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            parameter_partition_masks=partition_masks,
            clip_value=None,
            optimizer=optimizer_type,
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
        )
        opt_state = optimizer.initialise(
            model=sim,
            _jit_test_args=(
                tuple(data_targets_list),
                tuple(loss_fn_list),
                tuple(indexes_list),
            ),
        )

        # Run optimisation sweep
        sim, optimizer = _optimise(
            _simulation=sim,
            data_to_fit=tuple(data_targets_list),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=indexes_list,
            loss_functions=loss_fn_list,
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save config as JSON
        config_dict = {
            "loss_config": loss_config.__dict__,
        }
        if opt_config is not None:
            config_dict["opt_config"] = opt_config.__dict__
        
        config_path = os.path.join(output_dir, f"{name}_config.json")
        with open(config_path, "w") as f:
            json.dump(
                config_dict, 
                f, 
                indent=2, 
                sort_keys=True, 
                default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)
            )
            
        # 2. Save HDF5 Histories
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path_ema = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        save_optimization_history_to_file(filename=output_path_ema, history=optimizer.ema_history)


# ---------------------------------------------------------------------------
# BV_uptake_ForwardPass_frames (used in process_optimisation_results.py)
# ---------------------------------------------------------------------------


class BV_uptake_ForwardPass_frames(
    ForwardPass[BV_input_features, uptake_BV_output_features, BV_Model_Parameters]
):
    """Per-frame uptake forward pass (not frame-averaged).

    Used in ``process_optimisation_results.py`` to compute per-frame predictions
    that can be reweighted later. Identical across 3 active copies.
    """

    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        bc, bh = parameters.bv_bc, parameters.bv_bh
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        kints = jnp.asarray(input_features.k_ints)
        time_points = parameters.timepoints.reshape(-1)

        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        pf = jnp.exp(log_pf)

        def compute_uptake_for_timepoint(timepoint):
            kints_reshaped = kints.reshape(-1, 1)
            uptake = 1 - jnp.exp(-kints_reshaped * timepoint / pf)
            return uptake

        uptake_per_timepoint = jax.vmap(compute_uptake_for_timepoint)(time_points)
        return uptake_BV_output_features(uptake_per_timepoint)
