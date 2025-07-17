from typing import Sequence, Union

from icecream import ic

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.utils.jit_fn import jit_Guard

ic.disable()


def run_predict(
    input_features: Sequence[Input_Features],
    forward_models: Sequence[ForwardModel],
    model_parameters: Union[Sequence[Model_Parameters], Simulation_Parameters],
    forward_model_key: str | None = None,
    raise_jit_failure: bool = False,
    validate: bool = True,
) -> Sequence[Output_Features]:
    """
    Run prediction using sequences of Input_Features, Forward_Models and Model_Parameters.

    Args:
        input_features: Sequence of input features for each model
        forward_models: Sequence of forward models
        model_parameters: Either sequence of model parameters or simulation parameters
        raise_jit_failure: Whether to raise exception on JIT compilation failure
        validate: Whether to validate input lengths

    Returns:
        Sequence of output features from prediction

    Raises:
        ValueError: If sequences have different lengths
        RuntimeError: If simulation initialization or prediction fails
    """
    ic.configureOutput(prefix="RUN_PREDICT | ")
    ic("Starting run_predict")
    ic(f"Input features count: {len(input_features)}")
    ic(f"Forward models count: {len(forward_models)}")
    ic(f"Model parameters type: {type(model_parameters)}")
    ic(f"Raise JIT failure: {raise_jit_failure}, Validate: {validate}")

    # Convert input_features and forward_models to lists for consistency
    input_features_list = list(input_features)
    forward_models_list = list(forward_models)

    if validate:
        ic("Validating input sequence lengths")

        # Check if model_parameters is a sequence of Model_Parameters
        if isinstance(model_parameters, (list, tuple)) and not isinstance(
            model_parameters, Simulation_Parameters
        ):
            model_params_list = list(model_parameters)
            if not (len(input_features_list) == len(forward_models_list) == len(model_params_list)):
                ic.format(
                    "ERROR: Sequence length mismatch - Features: {}, Models: {}, Parameters: {}",
                    len(input_features_list),
                    len(forward_models_list),
                    len(model_params_list),
                )
                raise ValueError(
                    f"All sequences must have same length. "
                    f"Got: input_features={len(input_features_list)}, "
                    f"forward_models={len(forward_models_list)}, "
                    f"model_parameters={len(model_params_list)}"
                )
        else:
            # For Simulation_Parameters, just check input_features and forward_models match
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
            params=model_parameters
            if isinstance(model_parameters, Simulation_Parameters)
            else None,
            forward_model_key=forward_model_key,
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

            ic("Running prediction")
            output_features = sim.predict(model_parameters)
            ic(f"Prediction successful, got {len(output_features)} output features")

            return output_features

    except Exception as e:
        ic.format("ERROR during prediction: {} - {}", type(e).__name__, str(e))
        print(f"Failed to run prediction: {e}")
        raise e
