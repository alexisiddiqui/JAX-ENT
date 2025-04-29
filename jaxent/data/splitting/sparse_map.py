import copy
from typing import Sequence, cast

import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

from jaxent.data.loader import ExpD_Datapoint
from jaxent.interfaces.topology import Partial_Topology
from jaxent.types.features import Input_Features


####################################################################################################
# TODO this needs to be modified to handle timepoints - variable sizing of experimental fragments
# The input features will be output for every residue regardless - so the sparse mapping should just expanded out to match the number of timepoints
def create_sparse_map(
    input_features: Input_Features,
    feature_topology: Sequence[Partial_Topology],
    output_features: Sequence[ExpD_Datapoint],
) -> sparse.BCOO:
    ####################################################################################################
    """
    Creates a sparse mapping matrix in BCOO format to map residue-wise features
    to experimental fragment features.

    Args:
        input_features: Input features with shape (..., n_residues)
        feature_topology: List of topology fragments matching input features
        output_features: List of experimental fragments to map to

    Returns:
        BCOO sparse matrix that maps from residue-wise features to experimental fragments
        Matrix shape will be (n_experimental_fragments, n_residues)
    """
    print("Input shapes:")
    print(f"Input features shape: {input_features.features_shape}")
    print(f"Number of feature topology fragments: {len(feature_topology)}")
    print(f"Number of output features: {len(output_features)}")

    assert all([isinstance(top, Partial_Topology) for top in feature_topology]), (
        "Feature topology must be a list of Partial_Topology objects"
    )
    assert all([isinstance(frag, ExpD_Datapoint) for frag in output_features]), (
        "Output features must be a list of Experimental_Fragment objects"  # noqa
    )

    # Print some example fragments
    print("\nSample feature topology:")
    for i, top in enumerate(feature_topology[:3]):  # First 3 fragments
        print(f"Fragment {i}: chain={top.chain}, start={top.residue_start}, end={top.residue_end}")

    print("\nSample output features:")
    for i, frag in enumerate(output_features[:3]):  # First 3 fragments
        print(
            f"Fragment {i}: chain={frag.top.chain}, start={frag.top.residue_start}, end={frag.top.residue_end}"
        )

    assert input_features.features_shape[0] == len(feature_topology), (
        "Input features and topology do not match"
    )
    try:
        # assert that fragment indices are present and unique
        assert all([top.fragment_index is not None for top in feature_topology]), (
            "Fragment indices are not present in feature topology"
        )
        assert len(set([top.fragment_index for top in feature_topology])) == len(
            feature_topology
        ), "Fragment indices are not unique in feature topology"

        assert all([out_feat.top.fragment_index is not None for out_feat in output_features]), (
            "Fragment indices are not present in output features"
        )
        assert all(feat_top.fragment_index is not None for feat_top in feature_topology), (
            "Fragment indices are not present in feature topology"
        )
    except AssertionError as e:
        UserWarning(f"Fragment indices are not present in feature topology: {e}")

    print(feature_topology)

    # sort feature_topology by fragment index
    feature_topology = sorted(
        feature_topology, key=lambda x: x.fragment_index if x.fragment_index is not None else -1
    )
    print(feature_topology)

    assert all([top.fragment_index == i for i, top in enumerate(feature_topology)]), (
        "Topology fragments are not indexed from 0 - have they been matched with the input features?"
    )

    stripped_feature_topology = copy.deepcopy(feature_topology)
    for top in stripped_feature_topology:
        top.fragment_index = None
    stripped_output_topology = copy.deepcopy(output_features)
    for frag in stripped_output_topology:
        frag.top.fragment_index = None

    # # assert that set of feature topology is contained within output features topologies
    # assert set([f.top for f in stripped_output_topology]).issubset(
    #     set(stripped_feature_topology)
    # ), f"""Feature topology is not contained within output features: ≈
    #         \n Output features: \n
    #         {set([f.top for f in output_features])}
    #         \n Stripped Feature topology: \n
    #         {set(stripped_feature_topology)}"""
    if not set([f.top for f in stripped_output_topology]).issubset(set(stripped_feature_topology)):
        UserWarning(
            f"Feature topology is not contained within output features: \n Output features: \n {set([f.top for f in output_features])} \n Stripped Feature topology: \n {set(stripped_feature_topology)}"
        )

    n_residues = len(feature_topology)
    n_fragments = len(output_features)

    # Initialize lists to store indices and values for BCOO matrix
    rows = []  # Fragment indices
    cols = []  # Residue indices
    values = []  # Contribution weights

    # For each experimental fragment
    for frag_idx, exp_frag in enumerate(output_features):
        # Get the residues covered by this fragment
        if cast(int, exp_frag.top.length) > 2:  # For peptides
            frag_residues = exp_frag.top.peptide_residues
        else:  # For single residues
            frag_residues = range(
                exp_frag.top.residue_start, cast(int, exp_frag.top.residue_end) + 1
            )

        # Find matching residues in feature topology
        for res_id in frag_residues:
            for res_idx, feat_top in enumerate(feature_topology):
                if (
                    feat_top.chain == exp_frag.top.chain
                    and feat_top.residue_start <= res_id <= cast(int, feat_top.residue_end)
                ):
                    # Add indices for this residue's contribution
                    rows.append(frag_idx)
                    cols.append(res_idx)
                    # Weight by 1/number of residues in fragment
                    values.append(1.0 / len(frag_residues))
    # Debug prints
    print(f"Number of non-zero elements: {len(values)}")
    print(f"Matrix shape: ({n_fragments}, {n_residues})")

    if not values:
        raise ValueError("No matching residues found - sparse matrix would be empty")

    # Convert to JAX arrays
    indices = jnp.array([rows, cols], dtype=jnp.int32)
    values = jnp.array(values)

    # More debug prints
    print(f"Indices shape: {indices.shape}")
    print(f"Values shape: {values.shape}")

    print("Debug info before BCOO creation:")
    print(f"indices array shape: {indices.shape}")
    print(f"values array shape: {values.shape}")
    print(f"target matrix shape: {(n_fragments, n_residues)}")
    print(
        f"Sample indices: {indices[:, :5] if indices.size > 0 else 'empty'}"
    )  # Show first 5 indices
    print(f"Sample values: {values[:5] if values.size > 0 else 'empty'}")  # Show first 5 values

    # Need to ensure indices and values are properly shaped for BCOO
    if indices.size > 0:
        # Ensure indices has correct shape (2, N) where N is number of non-zero elements
        indices = indices.reshape(2, -1)
        # Ensure values has shape (N,)
        values = values.reshape(-1)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float32)

    # Explicitly specify the n_batch and n_dense parameters
    dense = jnp.zeros((n_fragments, n_residues))

    # Convert indices and values to numpy for indexing
    indices_np = jnp.asarray(indices).astype(jnp.int32)
    values_np = jnp.asarray(values).astype(jnp.float32)

    # Create dense matrix with the sparse values
    dense = dense.at[indices_np[0], indices_np[1]].set(values_np)

    # Convert to BCOO format
    bcoo_mat = sparse.bcoo_fromdense(dense)

    return bcoo_mat


####################################################################################################


def apply_sparse_mapping(sparse_map: sparse.BCOO, features: Array) -> Array:
    """
    Applies the sparse mapping to input features using bcoo_multiply_dense.

    Args:
        sparse_map: BCOO sparse matrix mapping residues to fragments
        features: Input feature array with shape (..., n_residues)

    Returns:
        Mapped features with shape (..., n_fragments)
    """
    return sparse_map.todense() @ features
    return sparse.bcoo_multiply_dense(sparse_map, features)
