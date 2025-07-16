import argparse
from pathlib import Path

import jax.numpy as jnp
import MDAnalysis as mda

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.config import BV_model_Config, NetHDXConfig, linear_BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, linear_BV_model
from jaxent.src.models.HDX.netHDX.forwardmodel import netHDX_model


def main():
    parser = argparse.ArgumentParser(
        description="Featurise molecular dynamics trajectories for HDX analysis."
    )

    # General arguments
    parser.add_argument(
        "--top_path",
        type=str,
        required=True,
        help="Path to the PDB file.",
    )

    parser.add_argument(
        "--trajectory_path",
        type=str,
        default=None,
        help="Path to the trajectory file (e.g., .xtc, .dcd). Optional.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="featurisation_results",
        help="Directory to save the featurised data and topology.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default_featurisation",
        help="Name for the featurisation run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for featurisation. Optional.",
    )

    subparsers = parser.add_subparsers(
        dest="model_type", required=True, help="Type of forward model to use."
    )

    # BV Model Subparser
    bv_parser = subparsers.add_parser("bv", help="Best-Vendruscolo (BV) model featurisation.")
    bv_parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for BV model (K).",
    )
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
        "--ph", type=float, default=7.0, help="pH for intrinsic rate calculation."
    )
    bv_parser.add_argument(
        "--heavy_radius",
        type=float,
        default=6.5,
        help="Heavy atom contact cutoff radius (Angstroms).",
    )
    bv_parser.add_argument(
        "--o_radius",
        type=float,
        default=2.4,
        help="Oxygen atom (H-bond acceptor) contact cutoff radius (Angstroms).",
    )
    bv_parser.add_argument(
        "--num_timepoints",
        type=int,
        default=1,
        help="Number of timepoints for BV model. Affects key type.",
    )
    bv_parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=[0.167, 1.0, 10.0],
        help="Timepoints for BV model (minutes).",
    )
    bv_parser.add_argument(
        "--residue_ignore",
        type=int,
        nargs=2,
        default=[-2, 2],
        help="Range of residues to ignore relative to donor (e.g., -2 2).",
    )
    bv_parser.add_argument(
        "--peptide_trim",
        type=int,
        default=2,
        help="Number of residues to trim from peptide ends for analysis.",
    )
    bv_parser.add_argument(
        "--peptide",
        action="store_true",
        help="Flag to indicate if the model is peptide-based.",
    )
    bv_parser.add_argument(
        "--mda_selection_exclusion",
        type=str,
        default="resname PRO or resid 1",
        help="MDAnalysis selection string for atoms to exclude.",
    )

    # Linear BV Model Subparser (inherits from BV, so similar args)
    linear_bv_parser = subparsers.add_parser(
        "linear_bv", help="Linear Best-Vendruscolo (BV) model featurisation."
    )
    linear_bv_parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for linear BV model (K).",
    )
    linear_bv_parser.add_argument(
        "--bv_bc",
        type=float,
        nargs="+",
        default=[0.35, 0.35, 0.35],
        help="Linear BV model parameter bc (heavy atom contact scaling). Can be single float or list.",
    )
    linear_bv_parser.add_argument(
        "--bv_bh",
        type=float,
        nargs="+",
        default=[2.0, 2.0, 2.0],
        help="Linear BV model parameter bh (H-bond acceptor contact scaling). Can be single float or list.",
    )
    linear_bv_parser.add_argument(
        "--ph", type=float, default=7.0, help="pH for intrinsic rate calculation."
    )
    linear_bv_parser.add_argument(
        "--heavy_radius",
        type=float,
        default=6.5,
        help="Heavy atom contact cutoff radius (Angstroms).",
    )
    linear_bv_parser.add_argument(
        "--o_radius",
        type=float,
        default=2.4,
        help="Oxygen atom (H-bond acceptor) contact cutoff radius (Angstroms).",
    )
    linear_bv_parser.add_argument(
        "--num_timepoints",
        type=int,
        default=3,
        help="Number of timepoints for linear BV model. Affects key type.",
    )
    linear_bv_parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=[0.167, 1.0, 10.0],
        help="Timepoints for linear BV model (minutes).",
    )
    linear_bv_parser.add_argument(
        "--residue_ignore",
        type=int,
        nargs=2,
        default=[-2, 2],
        help="Range of residues to ignore relative to donor (e.g., -2 2).",
    )
    linear_bv_parser.add_argument(
        "--peptide_trim",
        type=int,
        default=2,
        help="Number of residues to trim from peptide ends for analysis.",
    )
    linear_bv_parser.add_argument(
        "--peptide",
        action="store_true",
        help="Flag to indicate if the model is peptide-based.",
    )
    linear_bv_parser.add_argument(
        "--mda_selection_exclusion",
        type=str,
        default="resname PRO or resid 1",
        help="MDAnalysis selection string for atoms to exclude.",
    )

    # NetHDX Model Subparser
    nethdx_parser = subparsers.add_parser(
        "nethdx", help="Network-based HDX (netHDX) model featurisation."
    )
    nethdx_parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for netHDX model (K).",
    )
    nethdx_parser.add_argument(
        "--distance_cutoff",
        type=float,
        nargs="+",
        default=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5],
        help="Distance cutoff(s) for H-bond network calculation (Angstroms). Can be single float or list.",
    )
    nethdx_parser.add_argument(
        "--angle_cutoff",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="Angle cutoff(s) for H-bond network calculation (degrees). Can be single float or list.",
    )
    nethdx_parser.add_argument(
        "--residue_ignore",
        type=int,
        nargs=2,
        default=[-1, 1],
        help="Range of residues to ignore relative to donor (e.g., -1 1).",
    )
    nethdx_parser.add_argument(
        "--num_timepoints",
        type=int,
        default=1,
        help="Number of timepoints for netHDX model. Affects key type.",
    )
    nethdx_parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=[0.167, 1.0, 10.0],
        help="Timepoints for netHDX model (minutes).",
    )
    nethdx_parser.add_argument(
        "--shell_energy_scaling",
        type=float,
        default=0.84,
        help="Shell energy scaling factor for netHDX.",
    )
    nethdx_parser.add_argument(
        "--peptide_trim",
        type=int,
        default=2,
        help="Number of residues to trim from peptide ends for analysis.",
    )
    nethdx_parser.add_argument(
        "--peptide",
        action="store_true",
        help="Flag to indicate if the model is peptide-based.",
    )
    nethdx_parser.add_argument(
        "--mda_selection_exclusion",
        type=str,
        default="resname PRO or resid 1",
        help="MDAnalysis selection string for atoms to exclude.",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load MDAnalysis Universe
    print(f"Loading PDB: {args.pdb_path}")
    if args.trajectory_path:
        print(f"Loading trajectory: {args.trajectory_path}")
        universe = mda.Universe(args.pdb_path, args.trajectory_path)
    else:
        universe = mda.Universe(args.pdb_path)

    universes = [universe]

    # Create FeaturiserSettings
    featuriser_settings = FeaturiserSettings(name=args.name, batch_size=args.batch_size)

    # Create ForwardModel based on selected type
    forward_model: ForwardModel
    if args.model_type == "bv":
        config = BV_model_Config(
            temperature=args.temperature,
            bv_bc=jnp.array(args.bv_bc),
            bv_bh=jnp.array(args.bv_bh),
            ph=args.ph,
            heavy_radius=args.heavy_radius,
            o_radius=args.o_radius,
            num_timepoints=args.num_timepoints,
            timepoints=jnp.array(args.timepoints),
            residue_ignore=tuple(args.residue_ignore),
            peptide_trim=args.peptide_trim,
            peptide=args.peptide,
            mda_selection_exclusion=args.mda_selection_exclusion,
        )
        forward_model = BV_model(config=config)
    elif args.model_type == "linear_bv":
        config = linear_BV_model_Config(
            temperature=args.temperature,
            bv_bc=jnp.array(args.bv_bc),
            bv_bh=jnp.array(args.bv_bh),
            ph=args.ph,
            heavy_radius=args.heavy_radius,
            o_radius=args.o_radius,
            num_timepoints=args.num_timepoints,
            timepoints=jnp.array(args.timepoints),
            residue_ignore=tuple(args.residue_ignore),
            peptide_trim=args.peptide_trim,
            peptide=args.peptide,
            mda_selection_exclusion=args.mda_selection_exclusion,
        )
        forward_model = linear_BV_model(config=config)
    elif args.model_type == "nethdx":
        config = NetHDXConfig(
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            angle_cutoff=args.angle_cutoff,
            residue_ignore=tuple(args.residue_ignore),
            num_timepoints=args.num_timepoints,
            timepoints=jnp.array(args.timepoints),
            shell_energy_scaling=args.shell_energy_scaling,
            peptide_trim=args.peptide_trim,
            peptide=args.peptide,
            mda_selection_exclusion=args.mda_selection_exclusion,
        )
        forward_model = netHDX_model(config=config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Create Experiment_Builder
    ensemble_builder = Experiment_Builder(universes=universes, forward_models=[forward_model])

    # Run featurisation
    print("Running featurisation...")
    features, feature_topology = run_featurise(
        ensemble=ensemble_builder, config=featuriser_settings
    )

    # Save features and topology
    print("Saving featurised data and topology...")
    features_set = features[0]  # Assuming single model for now
    topology_set = feature_topology[0]  # Assuming single model for now

    # Save features using jnp.savez
    features_path = output_path / "features.npz"
    jnp.savez(
        features_path,
        heavy_contacts=features_set.heavy_contacts,
        acceptor_contacts=features_set.acceptor_contacts,
        k_ints=features_set.k_ints,
    )
    print(f"Features saved to {features_path}")

    # Save topology using Partial_Topology.save_list_to_json
    topology_file = output_path / "topology.json"
    Partial_Topology.save_list_to_json(topology_set, topology_file)
    print(f"Topology saved to {topology_file}")

    print("Featurisation complete.")


if __name__ == "__main__":
    main()
