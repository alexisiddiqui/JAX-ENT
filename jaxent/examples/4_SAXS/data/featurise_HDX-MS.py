"""
Featurise individual neuralplexer-generated PDB structures using the BV HDX-MS model.

This script treats each PDB as a separate single-frame universe and featurises them
individually. Per-structure features are combined in CSV order into a single
BV_input_features.npz for use in HDX-MS fitting.

Usage:
    python jaxent/examples/4_SAXS/data/featurise_HDX-MS.py \
        --input_dir jaxent/examples/4_SAXS/ensemble_generation/neuralplexer/collected_structures \
        --output_dir jaxent/examples/4_SAXS/data/_HDX_features \
        --ordering_csv jaxent/examples/4_SAXS/ensemble_generation/neuralplexer/collected_structures/frame_ordering.csv \
        [--kint_path path/to/kints.dat] \
        [--output_name BV_features]

CSV headers expected:
    index, Primary_cluster, Secondary_cluster, n_Ca, n_CDZ_ligands, n_Ca_final, n_CDZ_ligands_final, dir_prefix

Output:
    - {output_name}.npz: Combined BV_input_features
    - topology_{output_name}.json: Residue-level topology
"""

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import MDAnalysis as mda
import pandas as pd

from jaxent.examples.common.loading import load_HDXer_kints
import jaxent.src.interfaces.topology as pt
from jaxent.src.interfaces.topology import PTSerialiser
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Featurise PDB structures using BV HDX-MS model and combine into single .npz"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing PDB structures (mirrors collected_structures layout)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save combined features",
    )
    parser.add_argument(
        "--ordering_csv",
        required=True,
        help="Path to frame_ordering.csv (columns: index, ..., dir_prefix)",
    )
    parser.add_argument(
        "--kint_path",
        default=None,
        help="(optional) HDXer .dat file; if omitted, kints computed via HDXrate from sequence",
    )
    parser.add_argument(
        "--output_name",
        default="BV_features",
        help="(optional) Base name for output files (default: BV_features)",
    )
    parser.add_argument(
        "--timepoints",
        default=None,
        help="(optional) Comma-separated BV timepoints in minutes (e.g., '0.167,1.0,10.0,60.0,120.0')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve all paths
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ordering_csv = Path(args.ordering_csv).expanduser().resolve()

    if args.kint_path:
        kint_path = Path(args.kint_path).expanduser().resolve()
    else:
        kint_path = None

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ordering CSV: {ordering_csv}")
    print(f"Kint path: {kint_path if kint_path else 'None (will compute from sequence)'}")
    print(f"Output name: {args.output_name}")

    # Validate inputs
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not ordering_csv.exists():
        raise FileNotFoundError(f"Ordering CSV not found: {ordering_csv}")
    if kint_path and not kint_path.exists():
        raise FileNotFoundError(f"Kint path not found: {kint_path}")

    # Read CSV in order
    df = pd.read_csv(ordering_csv)
    print(f"Loaded ordering CSV with {len(df)} structures")

    # Build universe list in CSV order
    universes = []
    for _, row in df.iterrows():
        pdb_path = input_dir / f"{row['dir_prefix']}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        universe = mda.Universe(str(pdb_path))
        universes.append(universe)
        print(f"  Loaded: {pdb_path.name}")

    print(f"Loaded {len(universes)} universes in CSV order")

    # Configure BV model
    bv_config = BV_model_Config()

    # Parse timepoints if provided
    if args.timepoints:
        timepoints_list = [float(x.strip()) for x in args.timepoints.split(",")]
        bv_config.timepoints = jnp.array(timepoints_list)
        print(f"Set BV timepoints: {bv_config.timepoints}")

    # Featurise all structures at once using BV_model
    print("\nInitialising BV model...")
    bv_model = BV_model(bv_config)
    bv_model.initialise(universes)

    print("Featurising structures...")
    features, topology = bv_model.featurise(universes)

    print(f"Featurised {len(universes)} structures")
    print(f"  Heavy contacts shape: {features.heavy_contacts.shape}")
    print(f"  Acceptor contacts shape: {features.acceptor_contacts.shape}")
    print(f"  K_ints shape: {features.k_ints.shape if features.k_ints is not None else 'None'}")
    print(f"  Topology length: {len(topology)}")

    # Optionally replace kints with provided data
    if kint_path:
        print(f"\nLoading kints from: {kint_path}")
        _kints, _kint_topology = load_HDXer_kints(str(kint_path))
        _kint_top_merged = pt.TopologyFactory.merge(_kint_topology)

        # Validate topology compatibility
        for top in topology:
            if not pt.PairwiseTopologyComparisons.intersects(top, _kint_top_merged):
                raise ValueError(
                    f"Topology {top} does not intersect with kint topology {_kint_top_merged}. "
                    "Ensure that the kint topology matches the feature topology."
                )

        features = BV_input_features(
            heavy_contacts=features.heavy_contacts,
            acceptor_contacts=features.acceptor_contacts,
            k_ints=_kints,
        )
        print(f"Replaced k_ints from file (shape: {_kints.shape})")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined output
    features_output = output_dir / f"{args.output_name}.npz"
    features.save(str(features_output))
    print(f"\nSaved combined features to: {features_output}")

    topology_output = output_dir / f"topology_{args.output_name}.json"
    PTSerialiser.save_list_to_json(topology, str(topology_output))
    print(f"Saved topology to: {topology_output}")

    # Print summary
    n_frames = features.heavy_contacts.shape[1]
    n_residues = features.heavy_contacts.shape[0]
    print(f"\nFeaturisation summary:")
    print(f"  Total frames (structures): {n_frames}")
    print(f"  Total residues: {n_residues}")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nFeaturisation complete. Elapsed time: {end - start:.2f} seconds")
