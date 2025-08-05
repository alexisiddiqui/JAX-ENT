import argparse
from pathlib import Path

import jax.numpy as jnp

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config, NetHDXConfig, linear_BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, linear_BV_model
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters
from jaxent.src.models.HDX.netHDX.forwardmodel import netHDX_model
from jaxent.src.models.HDX.netHDX.parameters import NetHDX_Model_Parameters
from jaxent.src.predict_forward import run_forward


def main():
    parser = argparse.ArgumentParser(
        description="Run forward prediction using featurised data and multiple sets of simulation parameters."
    )

    # General arguments
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the featurised data (.npz file).",
    )
    parser.add_argument(
        "--topology_path",
        type=str,
        required=True,
        help="Path to the topology data (.json file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="forward_results",
        help="Directory to save the forward prediction results.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="forward_predictions",
        help="Prefix for the output prediction files (e.g., 'forward_predictions_0.npz').",
    )
    parser.add_argument(
        "--raise_jit_failure",
        action="store_true",
        help="Raise an exception if JIT compilation fails.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1,
        help="Number of simulation parameter sets to run. Parameters provided as single values will be broadcast.",
    )

    subparsers = parser.add_subparsers(
        dest="model_type", required=True, help="Type of forward model to use for prediction."
    )

    # BV Model Subparser
    bv_parser = subparsers.add_parser("bv", help="Best-Vendruscolo (BV) model forward prediction.")
    bv_parser.add_argument(
        "--bv_bc",
        type=float,
        nargs="+",
        default=[0.35],
        help="BV model parameter bc (heavy atom contact scaling). Can be single float or list (length must match num_simulations or be 1).",
    )
    bv_parser.add_argument(
        "--bv_bh",
        type=float,
        nargs="+",
        default=[2.0],
        help="BV model parameter bh (H-bond acceptor contact scaling). Can be single float or list (length must match num_simulations or be 1).",
    )
    bv_parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[300.0],
        help="Temperature for BV model (K). Can be single float or list (length must match num_simulations or be 1).",
    )
    bv_parser.add_argument(
        "--num_timepoints",
        type=int,
        nargs="+",
        default=[0],
        help="Number of timepoints for the forward model. Affects key type. Can be single int or list (length must match num_simulations or be 1).",
    )
    bv_parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=None,
        help="List of timepoints for uptake calculation. Required if num_timepoints > 0.",
    )

    # Linear BV Model Subparser
    linear_bv_parser = subparsers.add_parser(
        "linear_bv", help="Linear Best-Vendruscolo (BV) model forward prediction."
    )
    linear_bv_parser.add_argument(
        "--bv_bc",
        type=float,
        nargs="+",
        default=[0.35],
        help="Linear BV model parameter bc (heavy atom contact scaling). Can be single float or list (length must match num_simulations or be 1).",
    )
    linear_bv_parser.add_argument(
        "--bv_bh",
        type=float,
        nargs="+",
        default=[2.0],
        help="Linear BV model parameter bh (H-bond acceptor contact scaling). Can be single float or list (length must match num_simulations or be 1).",
    )
    linear_bv_parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[300.0],
        help="Temperature for linear BV model (K). Can be single float or list (length must match num_simulations or be 1).",
    )
    linear_bv_parser.add_argument(
        "--num_timepoints",
        type=int,
        nargs="+",
        default=[0],
        help="Number of timepoints for the forward model. Affects key type. Can be single int or list (length must match num_simulations or be 1).",
    )
    linear_bv_parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=None,
        help="List of timepoints for uptake calculation. Required if num_timepoints > 0.",
    )

    # NetHDX Model Subparser
    nethdx_parser = subparsers.add_parser(
        "nethdx", help="Network-based HDX (netHDX) model forward prediction."
    )
    nethdx_parser.add_argument(
        "--shell_energy_scaling",
        type=float,
        nargs="+",
        default=[0.84],
        help="Shell energy scaling factor for netHDX model parameters. Can be single float or list (length must match num_simulations or be 1).",
    )
    nethdx_parser.add_argument(
        "--num_timepoints",
        type=int,
        nargs="+",
        default=[0],
        help="Number of timepoints for the forward model. Affects key type. Can be single int or list (length must match num_simulations or be 1).",
    )

    # Arguments for Simulation_Parameters (common to all models)
    for sub_parser in [bv_parser, linear_bv_parser, nethdx_parser]:
        sub_parser.add_argument(
            "--frame_weights",
            type=float,
            nargs="+",
            default=None,  # Will be set based on input features if None
            help="List of frame weights for Simulation_Parameters. Length must match number of frames. If not provided, uniform weights are assumed.",
        )
        sub_parser.add_argument(
            "--frame_mask",
            type=int,
            nargs="+",
            default=None,  # Will be set based on input features if None
            help="List of frame mask (0 or 1) for Simulation_Parameters. Length must match number of frames. If not provided, all frames are used.",
        )
        sub_parser.add_argument(
            "--forward_model_weights",
            type=float,
            nargs="+",
            default=[1.0],
            help="List of forward model weights for Simulation_Parameters. Can be single float or list (length must match num_simulations or be 1).",
        )
        sub_parser.add_argument(
            "--normalise_loss_functions",
            type=int,
            nargs="+",
            default=[1],
            help="List of normalise loss functions (0 or 1) for Simulation_Parameters. Can be single int or list (length must match num_simulations or be 1).",
        )
        sub_parser.add_argument(
            "--forward_model_scaling",
            type=float,
            nargs="+",
            default=[1.0],
            help="List of forward model scaling factors for Simulation_Parameters. Can be single float or list (length must match num_simulations or be 1).",
        )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper to broadcast single values or validate list lengths
    def _process_arg_list(arg_value, expected_len, arg_name, dtype=float):
        if arg_value is None:
            return None  # Handled later for frame_weights/mask

        # Special handling for 'timepoints' to allow multiple values per simulation
        if arg_name == "timepoints":
            # If timepoints is an empty list or a list containing only None, return None
            if not arg_value or (isinstance(arg_value, list) and all(x is None for x in arg_value)):
                return None
            # Otherwise, return the list of timepoints as floats
            return [float(val) for val in arg_value]

        if len(arg_value) == 1:
            return [dtype(arg_value[0])] * expected_len
        elif len(arg_value) == expected_len:
            return [dtype(val) for val in arg_value]
        else:
            raise ValueError(
                f"Length of --{arg_name} ({len(arg_value)}) must be 1 or match --num_simulations ({expected_len})."
            )

    # Load Input_Features (assuming a single set of features for all simulations)
    print(f"Loading features from {args.features_path}")
    try:
        input_features = Input_Features.load(args.features_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Features file not found: {args.features_path}")

    # Get number of frames
    n_frames = input_features.features_shape[-1]

    # Process frame_weights
    if args.frame_weights is None:
        frame_weights = jnp.ones(n_frames) / n_frames
    else:
        if len(args.frame_weights) != n_frames:
            raise ValueError(
                f"Length of --frame_weights ({len(args.frame_weights)}) must match number of frames ({n_frames})."
            )
        frame_weights = jnp.asarray(args.frame_weights)

    # Process frame_mask
    if args.frame_mask is None:
        frame_mask = jnp.ones(n_frames, dtype=jnp.bool_)
    else:
        if len(args.frame_mask) != n_frames:
            raise ValueError(
                f"Length of --frame_mask ({len(args.frame_mask)}) must match number of frames ({n_frames})."
            )
        frame_mask = jnp.asarray(args.frame_mask, dtype=jnp.bool_)

    # Load Partial_Topology (not directly used by run_forward, but good practice to load if featurised data is provided)
    print(f"Loading topology from {args.topology_path}")
    topology = pt.PTSerialiser.load_list_from_json(args.topology_path)
    print(f"Loaded {len(topology)} topology entries.")

    # Prepare sequences of ForwardModel and Simulation_Parameters
    forward_models_list: list[ForwardModel] = []
    simulation_parameters_list: list[Simulation_Parameters] = []

    for i in range(args.num_simulations):
        # Create Model_Parameters for current simulation
        current_model_parameters: Model_Parameters
        if args.model_type == "bv":
            timepoints_processed = _process_arg_list(
                args.timepoints, args.num_simulations, "timepoints"
            )
            current_model_parameters = BV_Model_Parameters(
                bv_bc=jnp.array(_process_arg_list(args.bv_bc, args.num_simulations, "bv_bc")[i]),
                bv_bh=jnp.array(_process_arg_list(args.bv_bh, args.num_simulations, "bv_bh")[i]),
                temperature=_process_arg_list(
                    args.temperature, args.num_simulations, "temperature"
                )[i],
                timepoints=(
                    jnp.asarray(timepoints_processed) if timepoints_processed is not None else None
                ),
            )
            config = BV_model_Config(
                num_timepoints=_process_arg_list(
                    args.num_timepoints, args.num_simulations, "num_timepoints", dtype=int
                )[i],
            )
            forward_models_list.append(BV_model(config=config))
        elif args.model_type == "linear_bv":
            current_model_parameters = linear_BV_Model_Parameters(
                bv_bc=jnp.array(_process_arg_list(args.bv_bc, args.num_simulations, "bv_bc")[i]),
                bv_bh=jnp.array(_process_arg_list(args.bv_bh, args.num_simulations, "bv_bh")[i]),
                temperature=_process_arg_list(
                    args.temperature, args.num_simulations, "temperature"
                )[i],
                timepoints=(
                    jnp.asarray(
                        _process_arg_list(args.timepoints, args.num_simulations, "timepoints")
                    )
                    if _process_arg_list(args.timepoints, args.num_simulations, "timepoints")
                    is not None
                    else None
                ),
            )
            config = linear_BV_model_Config(
                num_timepoints=_process_arg_list(
                    args.num_timepoints, args.num_simulations, "num_timepoints", dtype=int
                )[i],
            )
            forward_models_list.append(linear_BV_model(config=config))
        elif args.model_type == "nethdx":
            current_model_parameters = NetHDX_Model_Parameters(
                shell_energy_scaling=jnp.array(
                    _process_arg_list(
                        args.shell_energy_scaling, args.num_simulations, "shell_energy_scaling"
                    )[i]
                ),
                num_timepoints=_process_arg_list(
                    args.num_timepoints, args.num_simulations, "num_timepoints", dtype=int
                )[i],
            )
            config = NetHDXConfig(
                num_timepoints=_process_arg_list(
                    args.num_timepoints, args.num_simulations, "num_timepoints", dtype=int
                )[i],
            )
            forward_models_list.append(netHDX_model(config=config))
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        # Create Simulation_Parameters for current simulation
        current_forward_model_weights = jnp.asarray(
            _process_arg_list(
                args.forward_model_weights, args.num_simulations, "forward_model_weights"
            )[i]
        )
        current_normalise_loss_functions = jnp.asarray(
            _process_arg_list(
                args.normalise_loss_functions,
                args.num_simulations,
                "normalise_loss_functions",
                dtype=int,
            )[i],
            dtype=jnp.int32,
        )
        current_forward_model_scaling = jnp.asarray(
            _process_arg_list(
                args.forward_model_scaling, args.num_simulations, "forward_model_scaling"
            )[i]
        )

        simulation_parameters_list.append(
            Simulation_Parameters(
                frame_weights=frame_weights,
                frame_mask=frame_mask,
                model_parameters=[
                    current_model_parameters
                ],  # Wrap in list as Simulation_Parameters expects Sequence
                forward_model_weights=current_forward_model_weights,
                normalise_loss_functions=current_normalise_loss_functions,
                forward_model_scaling=current_forward_model_scaling,
            )
        )

    # Run forward prediction
    print("Running forward prediction...")
    all_output_features = run_forward(
        input_features=[input_features]
        * args.num_simulations,  # Repeat input features for each simulation
        forward_models=forward_models_list,
        simulation_parameters=simulation_parameters_list,
        raise_jit_failure=args.raise_jit_failure,
    )

    # Save output features for each simulation
    for i, output_features_for_sim in enumerate(all_output_features):
        output_file_path = output_path / f"{args.output_prefix}_{i}.npz"
        if output_features_for_sim:
            # Assuming each output_features_for_sim is a Sequence of Output_Features
            # and we want to save the y_pred of the first one.
            Output_Features.save(output_features_for_sim[0], str(output_file_path))
            print(f"Predictions for simulation {i} saved to {output_file_path}")
        else:
            print(f"No output features generated for simulation {i}.")

    print("Forward prediction complete.")


if __name__ == "__main__":
    main()
