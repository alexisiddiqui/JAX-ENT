import copy
import os
from typing import List, Sequence, Tuple, cast

import jax
import jax.numpy as jnp

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
    hdx_uptake_MAE_loss_vectorized,
    maxent_convexKL_loss,
    maxent_JSD_loss,
    maxent_W1_loss,
    minent_ESS_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState
from jaxent.src.utils.hdf import (
    save_optimization_history_to_file,
)
from jaxent.src.utils.jit_fn import jit_Guard


def create_data_loaders(
    hdx_data: List[HDX_peptide],
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    feature_top: list[pt.Partial_Topology],
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

    loader = ExpD_Dataloader(data=hdx_data)
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
    convergence: list[float],  # List of convergence thresholds to iterate through
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    opt_state: OptimizationState,
    optimizer: OptaxOptimizer,
) -> Tuple[InitialisedSimulation, OptaxOptimizer]:
    """
    Optimisation method that sequentially iterates through convergence thresholds.
    Only advances to the next threshold after the current one is met.
    Stops only when all thresholds have been satisfied.

    Args:
        convergence: List of convergence thresholds to iterate through sequentially.
    """
    # Sort thresholds from largest to smallest (coarse to fine convergence)
    convergence_thresholds = sorted(convergence, reverse=True)
    current_threshold_idx = 0
    current_threshold = convergence_thresholds[current_threshold_idx]

    print(
        f"Starting optimization with {len(convergence_thresholds)} convergence thresholds: {convergence_thresholds}"
    )
    print(f"Current threshold: {current_threshold}")
    _history = copy.deepcopy(optimizer.history)  # Create a copy of the optimizer's history
    try:
        for step in range(n_steps):
            history = optimizer.history
            opt_state, current_loss, history = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=_simulation,
                data_targets=tuple(data_to_fit),
                loss_functions=tuple(loss_functions),
                indexes=tuple(indexes),
                history=history,
            )
            jax.debug.print(
                fmt=" ".join(
                    [
                        "Step {step}/{n_steps}",
                        "Training Loss: {train_loss:.8e}",
                        "Validation Loss: {val_loss:.4e}",
                        "Threshold {threshold_idx}/{total_thresholds} ({current_threshold:.2e})",
                    ]
                ),
                step=step,
                n_steps=n_steps,
                train_loss=opt_state.losses.total_train_loss,
                val_loss=opt_state.losses.total_val_loss,
                threshold_idx=current_threshold_idx + 1,
                total_thresholds=len(convergence_thresholds),
                current_threshold=current_threshold,
            )
            save_state = history.states[-1]
            # Check for tolerance, nan, or inf conditions (early termination)
            if (current_loss < tolerance) or (current_loss == jnp.nan) or (current_loss == jnp.inf):
                print(
                    f"Reached convergence tolerance/nan vals at step {step}, loss: {current_loss}"
                )
                break
            if step == 0:
                _history.add_state(save_state)

            # Check convergence for current threshold
            if step > 1:
                previous_loss = optimizer.history.states[-2].losses.total_train_loss
                loss_delta = abs(current_loss - previous_loss)

                # Check if current threshold is met
                if loss_delta < current_threshold:
                    print(
                        f"Threshold {current_threshold_idx + 1}/{len(convergence_thresholds)} met at step {step}"
                    )
                    print(f"Loss delta: {loss_delta:.8f}, threshold: {current_threshold}")
                    _history.add_state(save_state)
                    print(f"History updated with state at step {step}, loss: {current_loss:8f}")

                    # Move to next threshold
                    current_threshold_idx += 1

                    # Check if all thresholds have been satisfied
                    if current_threshold_idx >= len(convergence_thresholds):
                        print(f"All convergence thresholds completed at step {step}")
                        break
                    else:
                        # Update to next threshold
                        current_threshold = convergence_thresholds[current_threshold_idx]
                        print(
                            f"Moving to threshold {current_threshold_idx + 1}/{len(convergence_thresholds)}: {current_threshold}"
                        )
    except Exception as e:
        raise RuntimeError(
            f"Optimization failed due to an error: {e}. Returning best state from history.",
            "\n" * 10,
            "Simulation parameters at failure: ",
            _simulation.params,
            "\n" * 10,
            "Latest history state at failure: ",
            history.states[-1].params,
            "\n" * 10,
            "Opt State parameters at failure: ",
            opt_state.params,
            "\n" * 10,
        )

    optimizer.history = _history  # Update the optimizer's history with the copied history
    # Get best state from recorded history
    best_state = optimizer.history.get_best_state()
    if best_state is not None:
        _simulation.params = optimizer.history.best_state.params

    _simulation = cast(InitialisedSimulation, _simulation)
    return _simulation, optimizer


@jit_Guard.clear_caches_after()
def run_optimise_ISO_TRI_BI(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    forward_model: BV_model,
    model_parameters: BV_Model_Parameters,
    feature_top: List[pt.Partial_Topology],
    convergence: List[float],
    loss_function: JaxEnt_Loss,
    n_steps: int = 10,
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([1.0]),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    sim.initialise()

    optimizer = OptaxOptimizer(
        learning_rate=1e-4,
        optimizer="adam",
    )
    opt_state = optimizer.initialise(
        model=sim,
        optimisable_funcs=None,
    )
    _, optimizer = optimise_sweep(
        _simulation=sim,
        data_to_fit=(data_to_fit,),
        n_steps=n_steps,
        tolerance=1e-10,
        convergence=convergence,
        indexes=[0],
        loss_functions=[loss_function],
        opt_state=opt_state,
        optimizer=optimizer,
    )

    # Save the results
    output_path = os.path.join(output_dir, f"{name}_results.hdf5")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_optimization_history_to_file(filename=output_path, history=optimizer.history)


@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_maxENT_W1(
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
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 0.1, 1.0]),
        normalise_loss_functions=jnp.ones(3),
        forward_model_scaling=jnp.ones(3),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as guard:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=1e-4,
            optimizer="adam",
        )
        opt_state = optimizer.initialise(
            model=sim,
            optimisable_funcs=None,
        )
        sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(
                data_to_fit,
                parameters,
                None,
            ),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0, 0],
            loss_functions=[loss_function, maxent_convexKL_loss, maxent_W1_loss],
            opt_state=opt_state,
            optimizer=optimizer,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)


@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_W1(
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
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as guard:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=1e-4,
            optimizer="adam",
        )
        opt_state = optimizer.initialise(
            model=sim,
            optimisable_funcs=None,
        )
        sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0],
            loss_functions=[loss_function, maxent_W1_loss],
            opt_state=opt_state,
            optimizer=optimizer,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)


@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_JSD(
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
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as guard:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=1e-4,
            optimizer="adam",
        )
        opt_state = optimizer.initialise(
            model=sim,
            optimisable_funcs=None,
        )
        sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, parameters),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0],
            loss_functions=[loss_function, maxent_JSD_loss],
            opt_state=opt_state,
            optimizer=optimizer,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)


@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_MAE(
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
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as guard:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=1e-4,
            optimizer="adam",
        )
        opt_state = optimizer.initialise(
            model=sim,
            optimisable_funcs=None,
        )
        sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, prior_data),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0],
            loss_functions=[loss_function, hdx_uptake_MAE_loss_vectorized],
            opt_state=opt_state,
            optimizer=optimizer,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)


@jit_Guard.test_isolation()
def run_optimise_ISO_TRI_BI_minESS(
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
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as guard:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=1e-4,
            optimizer="adam",
        )
        opt_state = optimizer.initialise(
            model=sim,
            optimisable_funcs=None,
        )
        sim = guard

        # Run the optimisation sweep
        sim, optimizer = optimise_sweep(
            _simulation=sim,
            data_to_fit=(data_to_fit, prior_data),
            n_steps=n_steps,
            tolerance=1e-10,
            convergence=convergence,
            indexes=[0, 0],
            loss_functions=[loss_function, minent_ESS_loss],
            opt_state=opt_state,
            optimizer=optimizer,
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)


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
    name: str = "ISO_TRI_BI",
    output_dir: str = "_optimise",
) -> None:
    # create dataloader
    data_to_fit = create_data_loaders(
        hdx_data=train_data + val_data,
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_top=feature_top,
    )

    n_frames = features.features_shape[1]  # Assuming features.features_shape (n_residues, n_frames)

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
    )

    # create initialised simulation
    sim = Simulation(input_features=(features,), forward_models=(forward_model,), params=parameters)
    with jit_Guard(sim, cleanup_on_exit=True) as guard:
        sim.initialise()

        optimizer = OptaxOptimizer(
            learning_rate=1e-4,
            optimizer="adamw",
        )
        opt_state = optimizer.initialise(
            model=sim,
            optimisable_funcs=None,
        )
        sim = guard

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
        )

        # Save the results
        output_path = os.path.join(output_dir, f"{name}_results.hdf5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_optimization_history_to_file(filename=output_path, history=optimizer.history)
