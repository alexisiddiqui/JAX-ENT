from typing import Sequence

import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.interfaces.topology import Partial_Topology


def create_sparse_map(
    input_features: Input_Features,
    feature_topology: Sequence[Partial_Topology],
    output_features: Sequence[ExpD_Datapoint],
    check_trim: bool = True,
) -> sparse.BCOO:
    """
    Creates a sparse mapping matrix in BCOO format to map residue-wise features
    to experimental fragment features using robust topology intersection methods.

    Args:
        input_features: Input features with shape (..., n_residues)
        feature_topology: List of topology fragments matching input features
        output_features: List of experimental fragments to map to
        check_trim: If True, use peptide_residues for peptides when checking intersections

    Returns:
        BCOO sparse matrix that maps from residue-wise features to experimental fragments
        Matrix shape will be (n_experimental_fragments, n_residues)
    """
    print("Input shapes:")
    print(f"Input features shape: {input_features.features_shape}")
    print(f"Number of feature topology fragments: {len(feature_topology)}")
    print(f"Number of output features: {len(output_features)}")

    # Validate inputs
    assert all([isinstance(top, Partial_Topology) for top in feature_topology]), (
        "Feature topology must be a list of Partial_Topology objects"
    )
    assert all([isinstance(frag, ExpD_Datapoint) for frag in output_features]), (
        "Output features must be a list of ExpD_Datapoint objects"
    )
    assert input_features.features_shape[0] == len(feature_topology), (
        "Input features and topology do not match"
    )

    # Validate and organize feature topology indices
    try:
        assert all([top.fragment_index is not None for top in feature_topology]), (
            "Fragment indices are not present in feature topology"
        )
        assert len(set([top.fragment_index for top in feature_topology])) == len(
            feature_topology
        ), "Fragment indices are not unique in feature topology"
    except AssertionError as e:
        raise ValueError(f"Fragment indices are invalid in feature topology: {e}")

    # Sort feature topology by fragment index to ensure alignment with input features
    feature_topology_sorted = sorted(
        feature_topology, key=lambda x: x.fragment_index if x.fragment_index is not None else -1
    )

    assert all([top.fragment_index == i for i, top in enumerate(feature_topology_sorted)]), (
        "Topology fragments are not indexed from 0 - have they been matched with the input features?"
    )

    print("\nSample feature topology:")
    for i, top in enumerate(feature_topology_sorted[:3]):
        print(f"Fragment {i}: {top}")

    print("\nSample output features:")
    for i, frag in enumerate(output_features[:3]):
        print(f"Fragment {i}: {frag.top}")

    # Group topologies by chain for efficient processing
    feature_topologies_by_chain = Partial_Topology.group_set_by_chain(set(feature_topology_sorted))
    output_topologies_by_chain = {}
    for i, exp_frag in enumerate(output_features):
        chain = exp_frag.top.chain
        if chain not in output_topologies_by_chain:
            output_topologies_by_chain[chain] = []
        output_topologies_by_chain[chain].append((i, exp_frag.top))

    print(f"\nFeature topologies grouped by chain: {list(feature_topologies_by_chain.keys())}")
    print(f"Output topologies grouped by chain: {list(output_topologies_by_chain.keys())}")

    n_residues = len(feature_topology_sorted)
    n_fragments = len(output_features)

    # Initialize lists to store indices and values for BCOO matrix
    rows = []  # Fragment indices (output)
    cols = []  # Residue indices (input features)
    values = []  # Contribution weights

    # Process each output fragment using robust topology methods
    for frag_idx, exp_frag in enumerate(output_features):
        exp_topology = exp_frag.top
        chain = exp_topology.chain

        # Get active residues for this experimental fragment
        exp_active_residues = exp_topology._get_active_residues(check_trim=check_trim)
        exp_residue_count = len(exp_active_residues)

        if exp_residue_count == 0:
            print(f"Warning: Experimental fragment {frag_idx} has no active residues")
            continue

        # Only check feature topologies from the same chain
        if chain not in feature_topologies_by_chain:
            print(f"Warning: No feature topologies found for chain {chain}")
            continue

        same_chain_features = feature_topologies_by_chain[chain]

        # Find intersecting feature topologies using robust methods
        for feat_topology in same_chain_features:
            # Check if topologies intersect
            if exp_topology.intersects(feat_topology, check_trim=check_trim):
                # Get the overlapping residues
                overlap_residues = exp_topology.get_overlap(feat_topology, check_trim=check_trim)

                if overlap_residues:
                    # Get the feature index (this should be the position in the sorted list)
                    feat_idx = feat_topology.fragment_index

                    if feat_idx is None:
                        print(f"Warning: Feature topology {feat_topology} has no fragment_index")
                        continue

                    # Calculate contribution weight based on overlap
                    # Weight by the proportion of overlapping residues relative to experimental fragment
                    overlap_count = len(overlap_residues)
                    contribution_weight = overlap_count / exp_residue_count

                    # Add this mapping
                    rows.append(frag_idx)
                    cols.append(feat_idx)
                    values.append(contribution_weight)

                    print(
                        f"  Mapping: exp_frag[{frag_idx}] <- feat[{feat_idx}] "
                        f"(overlap: {overlap_count}/{exp_residue_count} residues, "
                        f"weight: {contribution_weight:.3f})"
                    )

    print("\nSparse matrix statistics:")
    print(f"Number of non-zero elements: {len(values)}")
    print(f"Matrix shape: ({n_fragments}, {n_residues})")
    if n_fragments * n_residues > 0:
        print(f"Sparsity: {len(values) / (n_fragments * n_residues):.4f}")
    else:
        print("Sparsity: N/A (zero dimension matrix)")

    if not values:
        raise ValueError(
            "No matching residues found - sparse matrix would be empty. "
            "Check that feature_topology and output_features have overlapping residues."
        )

    # Convert to JAX arrays
    indices = jnp.array([rows, cols], dtype=jnp.int32)
    values_array = jnp.array(values, dtype=jnp.float32)

    print(f"Final indices shape: {indices.shape}")
    print(f"Final values shape: {values_array.shape}")

    # Create dense matrix first, then convert to BCOO
    dense_matrix = jnp.zeros((n_fragments, n_residues), dtype=jnp.float32)
    dense_matrix = dense_matrix.at[indices[0], indices[1]].set(values_array)

    # Convert to BCOO format
    bcoo_matrix = sparse.bcoo_fromdense(dense_matrix)

    print(f"Created BCOO matrix with shape {bcoo_matrix.shape}")

    return bcoo_matrix


def apply_sparse_mapping(sparse_map: sparse.BCOO, features: Array) -> Array:
    """
    Applies the sparse mapping to input features.

    Args:
        sparse_map: BCOO sparse matrix mapping residues to fragments
        features: Input feature array with shape (..., n_residues)

    Returns:
        Mapped features with shape (..., n_fragments)
    """
    # Use matrix multiplication for proper broadcasting
    return sparse_map.todense() @ features
