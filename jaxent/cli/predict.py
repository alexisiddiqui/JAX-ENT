import argparse
from pathlib import Path

import jax.numpy as jnp

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config, NetHDXConfig, linear_BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, linear_BV_model
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters
from jaxent.src.models.HDX.netHDX.forwardmodel import netHDX_model
from jaxent.src.models.HDX.netHDX.parameters import NetHDX_Model_Parameters
from jaxent.src.predict import run_predict


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction using featurised data and model parameters."
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
        default="prediction_results",
        help="Directory to save the prediction results.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="predictions",
        help="Name for the output prediction file (without extension).",
    )
    parser.add_argument(
        "--raise_jit_failure",
        action="store_true",
        help="Raise an exception if JIT compilation fails.",
    )

    parser.add_argument(
        "--forward_model_key",
        type=str,
        default=None,
        help="Key for the specific forward model to use (e.g., 'HDX_peptide', 'HDX_resPF').",
    )

    subparsers = parser.add_subparsers(
        dest="model_type", required=True, help="Type of forward model to use for prediction."
    )

    # BV Model Subparser
    bv_parser = subparsers.add_parser("bv", help="Best-Vendruscolo (BV) model prediction.")
    bv_parser.add_argument(
        "--bv_bc",
        type=float,
        nargs="+",
        default=[0.35],
        help="BV model parameter bc (heavy atom contact scaling). Can be single float or list.",
    )
    bv_parser.add_argument(
        "--bv_bh",
        type=float,
        nargs="+",
        default=[2.0],
        help="BV model parameter bh (H-bond acceptor contact scaling). Can be single float or list.",
    )
    bv_parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for BV model (K).",
    )
    bv_parser.add_argument(
        "--num_timepoints",
        type=int,
        default=0,
        help="Number of timepoints for BV model. Affects key type.",
    )

    # Linear BV Model Subparser
    linear_bv_parser = subparsers.add_parser(
        "linear_bv", help="Linear Best-Vendruscolo (BV) model prediction."
    )
    linear_bv_parser.add_argument(
        "--bv_bc",
        type=float,
        nargs="+",
        default=[0.35],
        help="Linear BV model parameter bc (heavy atom contact scaling). Can be single float or list.",
    )
    linear_bv_parser.add_argument(
        "--bv_bh",
        type=float,
        nargs="+",
        default=[2.0],
        help="Linear BV model parameter bh (H-bond acceptor contact scaling). Can be single float or list.",
    )
    linear_bv_parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for linear BV model (K).",
    )
    linear_bv_parser.add_argument(
        "--num_timepoints",
        type=int,
        default=0,
        help="Number of timepoints for linear BV model. Affects key type.",
    )

    # NetHDX Model Subparser
    nethdx_parser = subparsers.add_parser(
        "nethdx", help="Network-based HDX (netHDX) model prediction."
    )
    nethdx_parser.add_argument(
        "--shell_energy_scaling",
        type=float,
        default=0.84,
        help="Shell energy scaling factor for netHDX model parameters.",
    )

    # Arguments for Simulation_Parameters (common to all models)
    for sub_parser in [bv_parser, linear_bv_parser, nethdx_parser]:
        sub_parser.add_argument(
            "--frame_weights",
            type=float,
            nargs="+",
            default=None,  # Will be set based on input features if None
            help="List of frame weights for Simulation_Parameters.",
        )
        sub_parser.add_argument(
            "--frame_mask",
            type=int,
            nargs="+",
            default=None,  # Will be set based on input features if None
            help="List of frame mask (0 or 1) for Simulation_Parameters.",
        )
        sub_parser.add_argument(
            "--forward_model_weights",
            type=float,
            nargs="+",
            default=[1.0],
            help="List of forward model weights for Simulation_Parameters.",
        )
        sub_parser.add_argument(
            "--normalise_loss_functions",
            type=int,
            nargs="+",
            default=[1],
            help="List of normalise loss functions (0 or 1) for Simulation_Parameters.",
        )
        sub_parser.add_argument(
            "--forward_model_scaling",
            type=float,
            nargs="+",
            default=[1.0],
            help="List of forward model scaling factors for Simulation_Parameters.",
        )
        sub_parser.add_argument(
            "--timepoints",
            type=float,
            nargs="+",
            default=[0.167, 1.0, 10.0],
            help="List of timepoints for BV model. Default is [0.167, 1.0, 10.0].",
        )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Input_Features
    print(f"Loading features from {args.features_path}")
    try:
        input_features = Input_Features.load(args.features_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Features file not found: {args.features_path}")

    # Load Partial_Topology
    print(f"Loading topology from {args.topology_path}")
    topology = pt.PTSerialiser.load_list_from_json(args.topology_path)
    print(f"Loaded {len(topology)} topology entries.")

    # Create ForwardModel and Model_Parameters
    forward_model: ForwardModel
    model_parameters: Model_Parameters

    if args.model_type == "bv":
        config = BV_model_Config(
            num_timepoints=args.num_timepoints,
        )
        config.timepoints = jnp.array(args.timepoints)
        forward_model = BV_model(config=config)
        model_parameters = BV_Model_Parameters(
            bv_bc=jnp.array(args.bv_bc),
            bv_bh=jnp.array(args.bv_bh),
            temperature=args.temperature,
            timepoints=jnp.array(args.timepoints),
        )
    elif args.model_type == "linear_bv":
        config = linear_BV_model_Config(
            num_timepoints=args.num_timepoints,
        )
        config.timepoints = jnp.array(args.timepoints)
        forward_model = linear_BV_model(config=config)
        model_parameters = linear_BV_Model_Parameters(
            bv_bc=jnp.array(args.bv_bc),
            bv_bh=jnp.array(args.bv_bh),
            temperature=args.temperature,
            num_timepoints=args.num_timepoints,
        )
    elif args.model_type == "nethdx":
        config = NetHDXConfig(
            num_timepoints=len(args.timepoints),
        )
        config.timepoints = jnp.array(args.timepoints)
        forward_model = netHDX_model(config=config)
        model_parameters = NetHDX_Model_Parameters(
            shell_energy_scaling=jnp.array(args.shell_energy_scaling),
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Create Simulation_Parameters
    n_frames = input_features.features_shape[-1]  # Assuming last dim is frames

    frame_weights = (
        jnp.asarray(args.frame_weights)
        if args.frame_weights is not None
        else jnp.ones(n_frames) / n_frames
    )
    frame_mask = (
        jnp.asarray(args.frame_mask, dtype=jnp.bool_)
        if args.frame_mask is not None
        else jnp.ones(n_frames, dtype=jnp.bool_)
    )
    forward_model_weights = jnp.asarray(args.forward_model_weights)
    normalise_loss_functions = jnp.asarray(args.normalise_loss_functions, dtype=jnp.int32)
    forward_model_scaling = jnp.asarray(args.forward_model_scaling)

    simulation_parameters = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=frame_mask,
        model_parameters=[
            model_parameters
        ],  # Wrap in list as Simulation_Parameters expects Sequence
        forward_model_weights=forward_model_weights,
        normalise_loss_functions=normalise_loss_functions,
        forward_model_scaling=forward_model_scaling,
    )

    # Run prediction
    print("Running prediction...")
    output_features = run_predict(
        input_features=[input_features],  # run_predict expects Sequence
        forward_models=[forward_model],  # run_predict expects Sequence
        model_parameters=simulation_parameters,
        raise_jit_failure=args.raise_jit_failure,
    )

    # Save output features
    output_file_path = output_path / f"{args.output_name}.npz"
    if output_features:
        first_output = output_features[0]
        jnp.savez(output_file_path, predictions=first_output.y_pred())
        print(f"Predictions saved to {output_file_path}")
    else:
        print("No output features generated.")

    print("Prediction complete.")


if __name__ == "__main__":
    main()
