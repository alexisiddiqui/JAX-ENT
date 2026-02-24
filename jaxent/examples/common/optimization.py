"""
Shared optimization/fitting functions for JAX-ENT example scripts.

Replaces 5+ near-identical ``run_optimise_*`` variants and the duplicated
``create_data_loaders`` / ``optimise_sweep`` functions from ``optimise_fn.py``.
"""

from __future__ import annotations

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
# EMA-based optimisation sweep
# ---------------------------------------------------------------------------


def optimise_sweep(
    _simulation: InitialisedSimulation,
    data_to_fit: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features],
    n_steps: int,
    tolerance: float,
    convergence: list[float],
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    opt_state: OptimizationState,
    optimizer: OptaxOptimizer,
    ema_alpha: float = 0.5,
    min_steps_per_threshold: int = 2,
) -> Tuple[InitialisedSimulation, OptaxOptimizer]:
    """EMA-only approach with relative convergence thresholds.

    This is the canonical version extracted from the 3 identical copies
    in ``optimise_fn.py`` across experiments 1, 2, and 3.
    """
    convergence_thresholds = sorted(convergence, reverse=True)
    convergence_thresholds = [ct * optimizer.learning_rate for ct in convergence_thresholds]
    current_threshold_idx = 0
    current_threshold = convergence_thresholds[current_threshold_idx]

    ema_loss_delta = None
    ema_params = None
    steps_since_threshold_start = 0
    optimizer.history = OptimizationHistory()
    save_state = None

    try:
        previous_loss = None
        prev_opt_state = None
        for step in range(n_steps):
            prev_grads = (
                opt_state.gradients
                if opt_state.gradients is not None
                else jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), opt_state.params)
            )

            opt_state, current_loss, save_state, _simulation = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=_simulation,
                data_targets=tuple(data_to_fit),
                loss_functions=tuple(loss_functions),
                indexes=tuple(indexes),
            )

            if previous_loss is not None:
                raw_loss_delta = jnp.abs(previous_loss - current_loss)
                if ema_loss_delta is None or ema_params is None:
                    ema_loss_delta = raw_loss_delta
                    ema_params = save_state.params
                else:
                    ema_loss_delta = ema_alpha * raw_loss_delta + (1 - ema_alpha) * ema_loss_delta
                    ema_params = ema_alpha * save_state.params + (1 - ema_alpha) * ema_params
            else:
                raw_loss_delta = 0.0

            relative_convergence = (
                ema_loss_delta / current_loss
                if ema_loss_delta is not None and current_loss > 0
                else 0.0
            )

            previous_loss = current_loss
            steps_since_threshold_start += 1

            _opt_state = OptimizationState(
                params=opt_state.params,
                opt_state=opt_state.opt_state,
                step=opt_state.step,
            )
            if prev_opt_state is None:
                zero_params = jax.tree_util.tree_map(
                    lambda x: jnp.full_like(x, 1e0), opt_state.params
                )
                prev_opt_state = OptimizationState(
                    params=zero_params,
                    opt_state=opt_state.opt_state,
                    step=opt_state.step,
                )

            opt_state_param_frameweight_delta = jnp.linalg.norm(
                _opt_state.params.frame_weights - prev_opt_state.params.frame_weights
            )
            grad_dot_product = jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(
                    lambda a, b: jnp.vdot(a, b),
                    prev_grads,
                    opt_state.gradients,
                ),
            )

            prev_opt_state = _opt_state

            jax.debug.print(
                fmt=" ".join([
                    "Step {step}/{n_steps}",
                    "Loss: {current_loss:.6e}",
                    "EMA Δ: {ema_delta:.4e}",
                    "Raw Δ: {raw_delta:.4e}",
                    "Rel Conv: {rel_conv:.6e}",
                    "Threshold {threshold_idx}/{total_thresholds} ({current_threshold:.6e})",
                    "Opt State Δ: {opt_state_delta:.4e}",
                    "Grad Dot Prod: {grad_dot_product:.4e}",
                    "LR: {learning_rate:.4e}",
                ]),
                step=step,
                n_steps=n_steps,
                current_loss=current_loss,
                ema_delta=ema_loss_delta if ema_loss_delta is not None else 0.0,
                raw_delta=raw_loss_delta,
                rel_conv=relative_convergence,
                opt_state_delta=opt_state_param_frameweight_delta,
                grad_dot_product=grad_dot_product,
                learning_rate=optimizer.lr_schedule(),
                threshold_idx=current_threshold_idx + 1,
                total_thresholds=len(convergence_thresholds),
                current_threshold=current_threshold,
            )

            if grad_dot_product < 0:
                print(f"Warning: Gradient dot product negative at step {step}, possible oscillation.")
                steps_since_threshold_start = 0

            if (current_loss < tolerance) or (current_loss == jnp.nan) or (current_loss == jnp.inf):
                print(f"Reached convergence tolerance/nan vals at step {step}")
                break

            if step == 0:
                optimizer = optimizer.update_history_compute_ema_loss(
                    optimizer=optimizer,
                    simulation=_simulation,
                    data_targets=tuple(data_to_fit),
                    indexes=tuple(indexes),
                    loss_functions=tuple(loss_functions),
                    state=save_state,
                    ema_params=ema_params,
                )

            if (
                steps_since_threshold_start >= min_steps_per_threshold
                and ema_loss_delta is not None
                and relative_convergence < current_threshold
                and step > optimizer.initial_steps
            ):
                print(
                    f"Relative threshold {current_threshold_idx + 1}/{len(convergence_thresholds)} met at step {step}"
                )
                print(f"Relative convergence: {relative_convergence:.8e}, threshold: {current_threshold:.2e}")
                optimizer = optimizer.update_history_compute_ema_loss(
                    optimizer=optimizer,
                    simulation=_simulation,
                    data_targets=tuple(data_to_fit),
                    indexes=tuple(indexes),
                    loss_functions=tuple(loss_functions),
                    state=save_state,
                    ema_params=ema_params,
                )
                ema_loss = optimizer.ema_history.states[-1].losses.total_train_loss
                print(f"Updated History and computed loss from EMA params. EMA param Loss: {ema_loss:.6e}")
                print(f"EMA Params: {ema_params}")
                current_threshold_idx += 1
                steps_since_threshold_start = 0

                if current_threshold_idx >= len(convergence_thresholds):
                    print(f"All relative thresholds completed at step {step}")
                    break
                else:
                    current_threshold = convergence_thresholds[current_threshold_idx]
                    print(
                        f"Moving to relative threshold {current_threshold_idx + 1}/{len(convergence_thresholds)}: {current_threshold:.2e}"
                    )
    except Exception as e:
        error_msg = f"Optimization failed due to an error: {e}. Returning best state from history."
        error_details = [
            "\n" * 10,
            "Simulation parameters at failure: ",
            str(_simulation.params),
            "\n" * 10,
        ]
        if save_state is not None:
            error_details.extend([
                "Latest save state at failure: ",
                str(save_state.params),
                "\n" * 10,
            ])
        error_details.extend([
            "Latest EMA params state at failure: ",
            str(ema_params),
            "\n" * 10,
            "Opt State parameters at failure: ",
            str(opt_state.params),
            "\n" * 10,
        ])
        raise RuntimeError(error_msg + "".join(str(d) for d in error_details))

    print(
        "\n" * 10,
        "Simulation parameters at end: ",
        _simulation.params,
        "\n" * 10,
        "Latest save state at end: ",
        save_state.params,
        "\n" * 10,
        "Latest EMA params state at end: ",
        ema_params,
        "\n" * 10,
        "Opt State parameters at end: ",
        opt_state.params,
        "\n" * 10,
    )

    best_state = optimizer.history.get_best_state()
    if best_state is not None:
        _simulation.params = optimizer.history.best_state.params

    return cast(InitialisedSimulation, _simulation), optimizer


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

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            parameter_partition_masks=partition_masks,
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
        )
        opt_state = optimizer.initialise(model=sim)

        # Run optimisation sweep
        sim, optimizer = optimise_sweep(
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
