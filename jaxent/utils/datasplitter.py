import copy
import random
from typing import Sequence

import MDAnalysis as mda
import numpy as np
from jax import Array

# import kmeans from sklearn.cluster
from sklearn.cluster import KMeans

from jaxent.datatypes import (
    Experimental_Dataset,
    Experimental_Fragment,
    Input_Features,
    Topology_Fragment,
)
from jaxent.forwardmodels.functions import find_common_residues


def find_fragment_centrality(
    topology_fragments: list[Topology_Fragment], mode: str = "max", peptide: bool = True
) -> list[float]:
    """
    Find the overlap between Topology_Fragments in a list

    Parameters:
        topology_fragments: List of Topology_Fragment objects to compare
        mode: Either "max" or "mean" to determine how to calculate overlap
        peptide: Whether to use peptide residues or all residues in range

    Returns:
        A list of overlap counts for each fragment
    """
    overlaps = []

    for i, fragment in enumerate(topology_fragments):
        # Get the set of residues for current fragment
        if peptide and fragment.length > 2:
            current_residues = set(fragment.peptide_residues)
        else:
            current_residues = set(range(fragment.residue_start, fragment.residue_end + 1))

        fragment_overlaps = []

        # Compare with all other fragments
        for j, other in enumerate(topology_fragments):
            if i != j and fragment.chain == other.chain:
                # Get the set of residues for other fragment
                if peptide and other.length > 2:
                    other_residues = set(other.peptide_residues)
                else:
                    other_residues = set(range(other.residue_start, other.residue_end + 1))

                # Calculate overlap
                overlap = len(current_residues.intersection(other_residues))
                if overlap > 0:
                    fragment_overlaps.append(overlap)

        # Calculate final overlap value based on mode
        if fragment_overlaps:
            if mode == "max":
                overlaps.append(max(fragment_overlaps))
            elif mode == "mean":
                overlaps.append(sum(fragment_overlaps) / len(fragment_overlaps))
            else:
                raise ValueError("Mode must be either 'max' or 'mean'")
        else:
            overlaps.append(0)

    return overlaps


def filter_common_residues(
    dataset: Sequence[Experimental_Fragment],
    common_residues: set[Topology_Fragment],
    peptide: bool = True,
) -> list[Experimental_Fragment]:
    # filters the dataset to only include common residues
    new_data = []
    for data in dataset:
        assert data.top is not None, "Topology fragment is not defined in the experimental data."
        peptide_residues = data.top.extract_residues(peptide=peptide)

        if any([res in common_residues for res in peptide_residues]):
            new_data.append(data)

    return new_data


class DataSplitter:
    def __init__(
        self,
        dataset: Experimental_Dataset,
        random_seed: int = 42,
        ensemble: list[mda.Universe] | None = None,
        common_residues: set[Topology_Fragment] | None = None,
        peptide: bool = True,
        centrality: bool = True,
        train_size: float = 0.5,
    ):
        self.dataset = copy.deepcopy(dataset)

        self.random_seed = random_seed
        random.seed(random_seed)

        self.train_size = train_size
        self.centrality = centrality
        if common_residues is None and ensemble is not None:
            common_residues = find_common_residues(ensemble)[0]
        else:
            raise ValueError("Either common_residues or ensemble must be provided")

        self.common_residues = common_residues

        self.dataset.data = filter_common_residues(
            self.dataset.data, common_residues, peptide=peptide
        )

        self.fragment_centrality = find_fragment_centrality(
            self.dataset.top, mode="max", peptide=peptide
        )

    def sample_by_centrality(self, threshold: float = 0.9) -> list[Experimental_Fragment]:
        # this helps us select the most central fragments from the dataset - this is crucial for effective splitting with peptide data
        print("Sampling data by centrality")
        print(f"Threshold: {threshold}")
        print(f"Max Centrality: {max(self.fragment_centrality)}")
        max_centrality = max(self.fragment_centrality)
        if max_centrality == 0:
            # Assign uniform weights if all weights are zero
            centrality_weights = [1 / len(self.fragment_centrality)] * len(self.fragment_centrality)
        else:
            centrality_weights = [1 - (c / max_centrality) for c in self.fragment_centrality]
            total = sum(centrality_weights)
            # Normalize weights
            centrality_weights = [w / total for w in centrality_weights]

        # randomly sample data based on centrality as weights
        # TODO update this to use jax random seed
        train_data = random.choices(
            self.dataset.data, weights=centrality_weights, k=int(threshold * len(self.dataset.data))
        )

        # only need to sample the training data - rest is discarded
        return train_data

    def validate_split(
        self, train_data: list[Experimental_Fragment], val_data: list[Experimental_Fragment]
    ) -> bool:
        # this needs to check that the split is suitable for training - i.e. that datsets are not too small
        assert (len(train_data) and len(val_data)) > 0, (
            "No data found in training or validation set"
        )

        if len(train_data) < 5 or len(val_data) < 5:
            raise ValueError("Training or validation set is too small")

        # check they are a sufficient proportion of the dataset
        if (
            len(train_data) / len(self.dataset.data) < 0.1
            or len(val_data) / len(self.dataset.data) < 0.1
        ):
            raise ValueError(
                f"Training or validation set is too small: {len(train_data)} / {len(self.dataset.data)}, {len(val_data)} / {len(self.dataset.data)}"
            )
        print("Split is valid - continuing")
        return True

    def random_split(
        self, remove_overlap: bool = True
    ) -> tuple[list[Experimental_Fragment], list[Experimental_Fragment]]:
        # just performs a random split
        if self.centrality:
            dataset = self.sample_by_centrality()
        else:
            dataset = self.dataset.data

        train_data = random.sample(dataset, int(self.train_size * len(dataset)))

        val_data = [d for d in dataset if d not in train_data]

        # print some stats
        print(f"Number of training fragments: {len(train_data)}")
        print(f"Number of validation fragments: {len(val_data)}")

        train_fragments = set([d.top for d in train_data])
        val_fragments = set([d.top for d in val_data])

        intersecting_fragments = train_fragments.intersection(val_fragments)
        print(f"Number of intersecting fragments: {len(intersecting_fragments)}")

        if remove_overlap:
            # remove overlapping fragments from validation set
            val_data = [d for d in val_data if d.top not in intersecting_fragments]
            train_data = [d for d in train_data if d.top not in intersecting_fragments]

        if not self.validate_split(train_data, val_data):
            print("Split is not valid - trying again")

            # change random seed
            self.random_seed += 1
            random.seed(self.random_seed)

            return self.random_split(remove_overlap=remove_overlap)

        return train_data, val_data

    def stratified_split(
        self, n_strata: int = 5
    ) -> tuple[list[Experimental_Fragment], list[Experimental_Fragment]]:
        """
        Performs a stratified split across y values of the dataset.
        This ensures that the distribution of y values is similar in both train and validation sets.

        Args:
            n_strata: Number of strata to create from the y values

        Returns:
            Tuple of (train_data, val_data) lists containing Experimental_Fragment objects
        """
        # Get the y values for the dataset
        y_vals = self.dataset.y_true

        # Create strata by binning y values
        bin_edges = np.linspace(np.min(y_vals), np.max(y_vals), n_strata + 1)
        bin_indices = np.digitize(y_vals, bin_edges[:-1])

        train_data = []
        val_data = []

        # Split each stratum according to train_size
        for stratum in range(1, n_strata + 1):
            stratum_indices = np.where(bin_indices == stratum)[0]
            if len(stratum_indices) == 0:
                continue

            # Get fragments for this stratum
            stratum_fragments = [self.dataset.data[i] for i in stratum_indices]

            # Sample train data from this stratum
            n_train = int(self.train_size * len(stratum_fragments))
            train_fragments = random.sample(stratum_fragments, n_train)

            # Remaining fragments go to validation
            val_fragments = [f for f in stratum_fragments if f not in train_fragments]

            train_data.extend(train_fragments)
            val_data.extend(val_fragments)

        # Validate the split
        if not self.validate_split(train_data, val_data):
            print("Split is not valid - trying again with different random seed")
            self.random_seed += 1
            random.seed(self.random_seed)
            return self.stratified_split(n_strata)

        return train_data, val_data

    def spatial_split(
        self, ensemble: list[mda.Universe]
    ) -> tuple[list[Experimental_Fragment], list[Experimental_Fragment]]:
        """
        Performs a split along the average coordinates of the provided ensemble.
        Selects a residue at random and then selects the closest residues to it up to train_size.

        Args:
            ensemble: List of MDAnalysis Universe objects containing structure information

        Returns:
            Tuple of (train_data, val_data) lists containing Experimental_Fragment objects
        """
        ###############################################################################
        # TODO this needs to be modified to use common_resiues
        # Calculate average coordinates for each residue
        avg_coords = {}
        for residue in ensemble[0].residues:
            coords = []
            for universe in ensemble:
                res = universe.residues[residue.ix]
                coords.append(res.atoms.center_of_mass())
            avg_coords[residue.resid] = np.mean(coords, axis=0)

        # Select a random seed residue
        seed_resid = random.choice(list(avg_coords.keys()))
        seed_coords = avg_coords[seed_resid]

        # Calculate distances from seed to all other residues
        distances = {
            resid: np.linalg.norm(coords - seed_coords) for resid, coords in avg_coords.items()
        }

        # Sort residues by distance
        sorted_resids = sorted(distances.keys(), key=lambda x: float(distances[x]))
        ################################################################################
        # Take closest residues for training set up to train_size
        n_train = int(self.train_size * len(sorted_resids))
        train_resids = set(sorted_resids[:n_train])
        val_resids = set(sorted_resids[n_train:])

        # Split fragments based on residue assignments
        train_data = []
        val_data = []

        for fragment in self.dataset.data:
            if fragment.top.residue_start in train_resids:
                train_data.append(fragment)
            else:
                val_data.append(fragment)

        # Validate the split
        if not self.validate_split(train_data, val_data):
            print("Split is not valid - trying again with different seed residue")
            self.random_seed += 1
            random.seed(self.random_seed)
            return self.spatial_split(ensemble)

        return train_data, val_data

    def cluster_split(
        self, n_clusters: int = 10, peptide: bool = True, cluster_index: str = "residue_index"
    ) -> tuple[list[Experimental_Fragment], list[Experimental_Fragment]]:
        """
        Clusters data along the specified index and performs splitting based on clusters.

        Args:
            n_clusters: Number of clusters to create
            peptide: Whether to use peptide residues or all residues
            cluster_index: What to cluster on ("sequence", "residue_index", "fragment_index", "features")

        Returns:
            Tuple of (train_data, val_data) lists containing Experimental_Fragment objects
        """
        possible_indexes = {"sequence", "residue_index", "fragment_index", "features"}
        if cluster_index not in possible_indexes:
            raise ValueError(f"Cluster index must be one of {possible_indexes}")

        # Create feature vectors based on clustering index
        features = []
        for fragment in self.dataset.data:
            if cluster_index == "sequence":
                # Use one-hot encoding of amino acid sequence
                aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
                feature = [aa_dict.get(aa, -1) for aa in fragment.top.fragment_sequence]
            elif cluster_index == "residue_index":
                # Use start and end residue indices
                feature = [fragment.top.residue_start, fragment.top.residue_end]
            elif cluster_index == "fragment_index":
                # Use fragment index directly
                feature = [fragment.top.fragment_index]
            else:  # features
                # Use experimental values
                feature = fragment.extract_features()

            features.append(feature)

        # Convert to numpy array and normalize
        features = np.array(features)
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(features)

        train_data = []
        val_data = []

        # Split each cluster according to train_size
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) == 0:
                continue

            # Get fragments for this cluster
            cluster_fragments = [self.dataset.data[i] for i in cluster_indices]

            # Sample train data from this cluster
            n_train = int(self.train_size * len(cluster_fragments))
            train_fragments = random.sample(cluster_fragments, n_train)

            # Remaining fragments go to validation
            val_fragments = [f for f in cluster_fragments if f not in train_fragments]

            train_data.extend(train_fragments)
            val_data.extend(val_fragments)

        # Validate the split
        if not self.validate_split(train_data, val_data):
            print("Split is not valid - trying again with different random seed")
            self.random_seed += 1
            random.seed(self.random_seed)
            return self.cluster_split(n_clusters, peptide, cluster_index)

        return train_data, val_data


import jax.numpy as jnp
from jax.experimental import sparse


####################################################################################################
# TODO this needs to be modified to handle timepoints - variable sizing of experimental fragments
# The input features will be output for every residue regardless - so the sparse mapping should just expanded out to match the number of timepoints
def create_sparse_map(
    input_features: Input_Features,
    feature_topology: Sequence[Topology_Fragment],
    output_features: Sequence[Experimental_Fragment],
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

    # assert that fragment indices are present and unique
    assert all([top.fragment_index is not None for top in feature_topology]), (
        "Fragment indices are not present in feature topology"
    )
    assert len(set([top.fragment_index for top in feature_topology])) == len(feature_topology), (
        "Fragment indices are not unique in feature topology"
    )

    # sort feature_topology by fragment index
    feature_topology = sorted(feature_topology, key=lambda x: x.fragment_index)
    #

    assert all([top.fragment_index == i for i, top in enumerate(feature_topology)]), (
        "Topology fragments are not indexed from 0 - have they been matched with the input features?"
    )

    stripped_feature_topology = copy.deepcopy(feature_topology)
    for top in stripped_feature_topology:
        top.fragment_index = None

    # assert that set of feature topology is contained within output features topologies
    assert set([f.top for f in output_features]).issubset(set(stripped_feature_topology)), (
        "Feature topology is not contained within output features"
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
        if exp_frag.top.length > 2:  # For peptides
            frag_residues = exp_frag.top.peptide_residues
        else:  # For single residues
            frag_residues = range(exp_frag.top.residue_start, exp_frag.top.residue_end + 1)

        # Find matching residues in feature topology
        for res_id in frag_residues:
            for res_idx, feat_top in enumerate(feature_topology):
                if (
                    feat_top.chain == exp_frag.top.chain
                    and feat_top.residue_start <= res_id <= feat_top.residue_end
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
    return sparse.bcoo_multiply_dense(sparse_map, features)
