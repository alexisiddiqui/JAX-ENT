from typing import Optional

import numpy as np

from jaxent.config.base import OptimiserSettings, Settings
from jaxent.datatypes import Experimental_Dataset, Simulation
from jaxent.forwardmodels.base import ForwardModel, Model_Parameters, Output_Features
from jaxent.lossfn.base import JaxEnt_Loss


def optimise(
    ensemble_paths: list[tuple[str, str]],
    features_dir: list[str],
    output_path: str,
    config_paths: list[str],
    name: str,
    batch_size: Optional[int],
    forward_models: list[str],
    loss_functions: list[str],
    log_path: Optional[str],
    overwrite: bool,
):
    # this function will be the input for the cli
    # this will take in paths and configurations and create the individual objects for analysis
    # TODO create reusable builder methods to generate objects from configuration
    pass


def run_optimise(
    simulation: Simulation,
    data_to_fit: tuple[Experimental_Dataset | Model_Parameters | Output_Features, ...],
    config: Settings | OptimiserSettings,
    forward_models: list[ForwardModel],
    loss_functions: list[JaxEnt_Loss],
    initialise: Optional[bool] = False,
) -> Simulation:
    # this function will take in the constructed objects and run the analysis

    if initialise:
        if not simulation.initialise():
            raise UserWarning("Failed to initialise simulation")

        assert len(data_to_fit) == len(loss_functions), (
            "Number of loss functions must be equal to number of data_to_fit"
        )
        assert len(loss_functions) == len(forward_models), (
            "Number of loss functions must be equal to number of forward models"
        )

    if isinstance(config, Settings):
        config = config.optimiser_config

    n_steps = config.n_steps
    tolerance = config.tolerance

    for i in range(n_steps):
        # run optimisation
        # collect loss over all data
        losses = []
        for loss_fn, data in zip(loss_functions, data_to_fit):
            losses.append(loss_fn(simulation, data))
            # update simulation

        # average losses
        average_loss = float(np.mean(losses))

        # update simulation
        simulation = simulation.update(average_loss)

        if average_loss < tolerance:
            break

    return simulation
