from typing import Sequence

from icecream import ic
from tqdm.auto import tqdm

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.utils.jit_fn import jit_Guard

ic.disable()


def run_forward(
    input_features: Sequence[Input_Features],
    forward_models: Sequence[ForwardModel],
    simulation_parameters: Sequence[Simulation_Parameters],
    raise_jit_failure: bool = False,
    validate: bool = True,
) -> Sequence[Sequence[Output_Features]]:
    """
    Run forward prediction on a sequence of simulation parameter objects.

    This function iterates through a sequence of Simulation_Parameters, applying the
    forward method for each parameter set using a tqdm progress bar.

    Args:
        input_features: Sequence of input features for each model.
        forward_models: Sequence of forward models.
        simulation_parameters: A sequence of simulation parameter objects.
        raise_jit_failure: Whether to raise an exception on JIT compilation failure.
        validate: Whether to validate the lengths of input sequences.

    Returns:
        A sequence of output features sequences, one for each simulation parameter set.

    Raises:
        ValueError: If input_features and forward_models sequences have different lengths.
        RuntimeError: If simulation initialization or forward prediction fails.
    """
    ic.configureOutput(prefix="RUN_FORWARD | ")
    ic("Starting run_forward")
    ic(f"Input features count: {len(input_features)}")
    ic(f"Forward models count: {len(forward_models)}")
    ic(f"Simulation parameters count: {len(simulation_parameters)}")
    ic(f"Raise JIT failure: {raise_jit_failure}, Validate: {validate}")

    input_features_list = list(input_features)
    forward_models_list = list(forward_models)

    if validate:
        ic("Validating input sequence lengths")
        if len(input_features_list) != len(forward_models_list):
            ic.format(
                "ERROR: Sequence length mismatch - Features: {}, Models: {}",
                len(input_features_list),
                len(forward_models_list),
            )
            raise ValueError(
                f"input_features and forward_models must have same length. "
                f"Got: input_features={len(input_features_list)}, "
                f"forward_models={len(forward_models_list)}"
            )
        ic("Input validation successful")

    try:
        ic("Creating Simulation object")
        simulation = Simulation(
            input_features=input_features_list,
            forward_models=forward_models_list,
            params=None,
            raise_jit_failure=raise_jit_failure,
        )
        ic("Simulation object created successfully")

        ic("Starting simulation with jit_Guard context manager")
        with jit_Guard(simulation, cleanup_on_exit=True) as sim:
            ic("Initializing simulation")
            initialization_success = sim.initialise()

            if not initialization_success:
                ic("ERROR: Simulation initialization failed")
                raise RuntimeError("Simulation initialization failed")

            ic("Simulation initialized successfully")

            ic("Running forward prediction loop")
            all_outputs = []
            for params in tqdm(simulation_parameters, desc="Forward prediction"):
                output_features = sim.forward(params)
                all_outputs.append(output_features)
            
            ic(f"Forward prediction loop finished, got {len(all_outputs)} sets of output features")
            return all_outputs

    except Exception as e:
        ic.format("ERROR during forward prediction: {} - {}", type(e).__name__, str(e))
        print(f"Failed to run forward prediction: {e}")
        raise e
