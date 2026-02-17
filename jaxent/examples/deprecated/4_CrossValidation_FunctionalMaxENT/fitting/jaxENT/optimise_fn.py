import os
from typing import List, Sequence, Tuple, cast

import jax
import jax.numpy as jnp

from jaxent.src.custom_types.config import Optimisable_Parameters
import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_Model_Parameters
from jaxent.src.opt.base import InitialisedSimulation, JaxEnt_Loss
from jaxent.src.opt.losses import (
    maxent_convexKL_loss,hdx_uptake_MAE_loss_vectorized,hdx_uptake_eye_MSE_loss,hdx_uptake_sigma_MSE_loss
)
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationHistory, OptimizationState
from jaxent.src.utils.hdf import (
    save_optimization_history_to_file,
)
from jaxent.src.utils.jit_fn import jit_Guard

from jax import Array
def create_data_loaders(
    hdx_data: List[HDX_peptide],
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    feature_top: list[pt.Partial_Topology],
    cov_matrix: Array | None = None,
) -> ExpD_Dataloader:
    """
    Create data loaders for training and validation datasets.

    Args:
        train_data: List of HDX_peptide objects for training.
        val_data: List of HDX_peptide objects for validation.
        features: BV_input_features object containing input features.
        feature_top: List of Partial_Topology objects for topology features.

    Returns:
        ExpD_Dataloader object containing the data loaders.
    """

    # train_indices = set(hdx_data.top.fragment_index for hdx_data in train_data)
    # val_indices = set([hdx_data.top.fragment_index for hdx_data in val_data])

    # assert (
    #     len(train_indices.intersection(val_indices)) == 0
    # ), "Training and validation datasets have overlapping fragment indices."
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
    ema_alpha: float = 0.5,  # EMA smoothing factor
    min_steps_per_threshold: int = 2,  # Minimum steps before checking convergence
) -> Tuple[InitialisedSimulation, OptaxOptimizer]:
    """EMA-only approach with relative convergence thresholds."""
    convergence_thresholds = sorted(convergence, reverse=True)
    # divide convergence thresholds by optimiser.learning_rate
    convergence_thresholds = [ct * optimizer.learning_rate for ct in convergence_thresholds]
    current_threshold_idx = 0
    current_threshold = convergence_thresholds[current_threshold_idx]

    ema_loss_delta = None
    ema_params = None
    steps_since_threshold_start = 0
    optimizer.history = OptimizationHistory()

    try:
        previous_loss = None
        prev_opt_state = None  # replace this with opt_state
        for step in range(n_steps):
            prev_grads = (
                opt_state.gradients
                if opt_state.gradients is not None
                else jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), opt_state.params)
            )  # type: ignore

            opt_state, current_loss, save_state, _simulation = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=_simulation,
                data_targets=tuple(data_to_fit),
                loss_functions=tuple(loss_functions),
                indexes=tuple(indexes),
            )

            # Calculate delta only after first step
            if previous_loss is not None:
                raw_loss_delta = jnp.abs(previous_loss - current_loss)

                # Update EMA
                if (
                    ema_loss_delta is None or ema_params is None
                ):  # First real delta calculation - initialize with first value
                    ema_loss_delta = raw_loss_delta
                    ema_params = save_state.params

                else:
                    ema_loss_delta = ema_alpha * raw_loss_delta + (1 - ema_alpha) * ema_loss_delta
                    ema_params = ema_alpha * save_state.params + (1 - ema_alpha) * ema_params

            else:
                raw_loss_delta = 0.0  # For logging purposes
                # Keep ema_loss_delta as None until we have real data - don't set to 0.0!

            # Calculate relative convergence
            relative_convergence = (
                ema_loss_delta / current_loss
                if ema_loss_delta is not None and current_loss > 0
                else 0.0
            )

            # Store current loss for next iteration (BEFORE using it in calculations!)
            previous_loss = current_loss

            steps_since_threshold_start += 1
            _opt_state = OptimizationState(
                params=opt_state.params,
                opt_state=opt_state.opt_state,
                step=opt_state.step,
            )
            if prev_opt_state is None:
                # create a pytree of zeros with the same structure as opt_state.params
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
            # compute the dot product between the previous and current gradients (Simulation_Parameters object) to check for oscillations
            grad_dot_product = jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(
                    lambda a, b: jnp.vdot(a, b),
                    prev_grads,
                    opt_state.gradients,  # type: ignore
                ),
            )

            prev_opt_state = _opt_state

            jax.debug.print(
                fmt=" ".join(
                    [
                        "Step {step}/{n_steps}",
                        "Loss: {current_loss:.6e}",
                        "EMA Δ: {ema_delta:.4e}",
                        "Raw Δ: {raw_delta:.4e}",
                        "Rel Conv: {rel_conv:.6e}",
                        "Threshold {threshold_idx}/{total_thresholds} ({current_threshold:.6e})",
                        "Opt State Δ: {opt_state_delta:.4e}",
                        "Grad Dot Prod: {grad_dot_product:.4e}",
                        "LR: {learning_rate:.4e}",
                    ]
                ),
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
                print(
                    f"Warning: Gradient dot product negative at step {step}, possible oscillation."
                )
                # rescale learning rate by plateau_denominator
                # Get current learning rate from optimizer state
                steps_since_threshold_start = 0
                # _history.add_state(save_state)
                # break
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

            # Relative convergence check after minimum steps (and after we have a real EMA value)
            if (
                steps_since_threshold_start >= min_steps_per_threshold
                and ema_loss_delta is not None
                and relative_convergence < current_threshold
                and step > optimizer.initial_steps
            ):
                print(
                    f"Relative threshold {current_threshold_idx + 1}/{len(convergence_thresholds)} met at step {step}"
                )
                print(
                    f"Relative convergence: {relative_convergence:.8e}, threshold: {current_threshold:.2e}"
                )
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
                print(
                    f"Updated History and computed loss from EMA params. EMA param Loss: {ema_loss:.6e}"
                )
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
        raise RuntimeError(
            f"Optimization failed due to an error: {e}. Returning best state from history.",
            "\n" * 10,
            "Simulation parameters at failure: ",
            _simulation.params,
            "\n" * 10,
            "Latest save state at failure: ",
            save_state.params,
            "\n" * 10,
            "Latest EMA params state at failure: ",
            ema_params,
            "\n" * 10,
            "Opt State parameters at failure: ",
            opt_state.params,
            "\n" * 10,
        )

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


# @jit_Guard.clear_caches_after()
# def run_optimise_ISO_TRI_BI(
#     train_data: List[HDX_peptide],
#     val_data: List[HDX_peptide],
#     features: BV_input_features,
#     forward_model: BV_model,
#     model_parameters: BV_Model_Parameters,
#     feature_top: List[pt.Partial_Topology],
#     convergence: List[float],
#     loss_function: JaxEnt_Loss,
#     n_steps: int = 10,
#     name: str = "ISO_TRI_BI",
#     output_dir: str = "_optimise",
# ) -> None:
#     # create dataloader
#     data_to_fit = create_data_loaders(
#         hdx_data=train_data + val_data,
#         train_data=train_data,
#         val_data=val_data,
#         features=features,
#         feature_top=feature_top,
#     )

#     n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

#     parameters = Simulation_Parameters(
#         frame_weights=jnp.ones(n_frames) / n_frames,
#         frame_mask=jnp.ones(n_frames),
#         model_parameters=(model_parameters,),
#         forward_model_weights=jnp.array([1.0]),
#         normalise_loss_functions=jnp.ones(1),
#         forward_model_scaling=jnp.ones(1),
#     )

#     # create initialised simulation
#     sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
#     sim.initialise()

#     optimizer = OptaxOptimizer(
#         learning_rate=1e-4,
#         optimizer="adam",
#     )
#     opt_state = optimizer.initialise(
#         model=sim,
#         optimisable_funcs=None,
#     )
#     _, optimizer = optimise_sweep(
#         _simulation=sim,
#         data_to_fit=(data_to_fit,),
#         n_steps=n_steps,
#         tolerance=1e-10,
#         convergence=convergence,
#         indexes=[0],
#         loss_functions=[loss_function],
#         opt_state=opt_state,
#         optimizer=optimizer,
#     )

#     # Save the results
#     output_path = os.path.join(output_dir, f"{name}_results.hdf5")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     save_optimization_history_to_file(filename=output_path, history=optimizer.history)


# @jit_Guard.test_isolation()
# def run_optimise_ISO_TRI_BI_MAE(
#     train_data: List[HDX_peptide],
#     val_data: List[HDX_peptide],
#     prior_data: ExpD_Dataloader,
#     features: BV_input_features,
#     forward_model: BV_model,
#     model_parameters: BV_Model_Parameters,
#     feature_top: List[pt.Partial_Topology],
#     convergence: List[float],
#     loss_function: JaxEnt_Loss,
#     maxent_scaling: float = 1.0,
#     n_steps: int = 10,
#     name: str = "ISO_TRI_BI",
#     output_dir: str = "_optimise",
# ) -> None:
#     # create dataloader
#     data_to_fit = create_data_loaders(
#         hdx_data=train_data + val_data,
#         train_data=train_data,
#         val_data=val_data,
#         features=features,
#         feature_top=feature_top,
#     )

#     n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

#     parameters = Simulation_Parameters(
#         frame_weights=jnp.ones(n_frames) / n_frames,
#         frame_mask=jnp.ones(n_frames),
#         model_parameters=(model_parameters,),
#         forward_model_weights=jnp.array([maxent_scaling, 1.0]),
#         normalise_loss_functions=jnp.ones(2),
#         forward_model_scaling=jnp.ones(2),
#     )

#     # create initialised simulation
#     sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
#     with jit_Guard(sim, cleanup_on_exit=True) as guard:
#         sim.initialise()

#         optimizer = OptaxOptimizer(
#             learning_rate=1e-4,
#             optimizer="adam",
#         )
#         opt_state = optimizer.initialise(
#             model=sim,
#             optimisable_funcs=None,
#         )
#         sim = guard

#         # Run the optimisation sweep
#         sim, optimizer = optimise_sweep(
#             _simulation=sim,
#             data_to_fit=(data_to_fit, prior_data),
#             n_steps=n_steps,
#             tolerance=1e-10,
#             convergence=convergence,
#             indexes=[0, 0],
#             loss_functions=[loss_function, hdx_uptake_MAE_loss_vectorized],
#             opt_state=opt_state,
#             optimizer=optimizer,
#         )

#         # Save the results
#         output_path = os.path.join(output_dir, f"{name}_results.hdf5")
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         save_optimization_history_to_file(filename=output_path, history=optimizer.history)


@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)



@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT_BV(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            parameter_partition_masks={Optimisable_Parameters.frame_weights, Optimisable_Parameters.model_parameters},
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)



@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT_BV_objective(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    bv_reg_loss: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    bv_reg_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0, bv_reg_scaling]),
        normalise_loss_functions=jnp.array([1.0, 1.0, 0]),
        forward_model_scaling=jnp.ones(3) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            parameter_partition_masks={Optimisable_Parameters.frame_weights, Optimisable_Parameters.model_parameters},
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters, parameters),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss, bv_reg_loss],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)



@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_functional_MaxENT(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: list[BV_model],
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    bv_reg_loss: JaxEnt_Loss,
    maxent_loss: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    bv_reg_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,model_parameters),
        forward_model_weights=jnp.array([maxent_scaling, 1.0, bv_reg_scaling]),
        normalise_loss_functions=jnp.array([1.0, 1.0, 0]),
        forward_model_scaling=jnp.ones(3) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,features), forward_models=forward_model, params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            parameter_partition_masks={Optimisable_Parameters.frame_weights, Optimisable_Parameters.model_parameters},
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, prior_data, prior_data),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 1, 1],
            loss_functions=[loss_function, maxent_loss, bv_reg_loss],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)



@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT_MAE(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0, 1.0]),
        normalise_loss_functions=jnp.asarray([1.0, 0.0, 1.0]),
        forward_model_scaling=jnp.asarray([100.0, 1.0, 50.0]) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters, prior_data),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss, hdx_uptake_MAE_loss_vectorized],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)




@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT_MAE_Sigma(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0, 1.0]),
        normalise_loss_functions=jnp.asarray([1.0, 0.0, 1.0]),
        forward_model_scaling=jnp.asarray([100.0, 1.0, 50.0]) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters, prior_data),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss, hdx_uptake_eye_MSE_loss],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)



@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT_MAE_Sigma_Sigma(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    prior_data: ExpD_Dataloader,
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    maxent_scaling: float = 1.0,
    n_steps: int = 10,
    name: str = "CrossVal_MSAss_Filtered",
    output_dir: str = "_optimise",
    learning_rate: float = 1e-1,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 2,
    ema_alpha: float = 0.5,
    forward_model_scaling: float = 100.0,
    cov_matrix: Array | None = None,
    model_parameters_lr_scale: float = 1.0,
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
        cov_matrix=cov_matrix,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0, 1.0]),
        normalise_loss_functions=jnp.asarray([1.0, 0.0, 1.0]),
        forward_model_scaling=jnp.asarray([10.0, 1.0, 5.0]) * forward_model_scaling,
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as sim:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=learning_rate,
            clip_value=None,
            optimizer="adam",
            initial_learning_rate=initial_learning_rate,
            initial_steps=initial_steps,
            model_parameters_lr_scale=model_parameters_lr_scale,
        )
        opt_state = optimizer.initialise(
            model=sim,
        )
        # sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters, prior_data),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss, hdx_uptake_sigma_MSE_loss],
            opt_state=opt_state,
            optimizer=optimizer,
            ema_alpha=ema_alpha,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
        output_path = os.path.join(output_dir, f"{name}_results_EMA.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.ema_history)
