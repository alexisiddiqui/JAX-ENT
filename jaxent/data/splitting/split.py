import copy
import random
from typing import Sequence, cast

import MDAnalysis as mda
import numpy as np
from MDAnalysis.core.groups import ResidueGroup

# import kmeans from sklearn.cluster
from sklearn.cluster import KMeans

from jaxent.data.loading import Experimental_Dataset, Experimental_Fragment
from jaxent.models.common import find_common_residues
from jaxent.types.topology import Partial_Topology


def find_fragment_centrality(
    Partial_Topologys: list[Partial_Topology], mode: str = "max", peptide: bool = True
) -> list[float]:
    """
    Find the overlap between Partial_Topologys in a list

    Parameters:
        Partial_Topologys: List of Partial_Topology objects to compare
        mode: Either "max" or "mean" to determine how to calculate overlap
        peptide: Whether to use peptide residues or all residues in range

    Returns:
        A list of overlap counts for each fragment
    """
    overlaps = []

    for i, fragment in enumerate(Partial_Topologys):
        # Get the set of residues for current fragment
        if peptide and cast(int, fragment.length) > 2:
            current_residues = set(fragment.peptide_residues)
        else:
            current_residues = set(
                range(fragment.residue_start, cast(int, fragment.residue_end) + 1)
            )

        fragment_overlaps = []

        # Compare with all other fragments
        for j, other in enumerate(Partial_Topologys):
            if i != j and fragment.chain == other.chain:
                # Get the set of residues for other fragment
                if peptide and cast(int, other.length) > 2:
                    other_residues = set(other.peptide_residues)
                else:
                    other_residues = set(
                        range(other.residue_start, cast(int, other.residue_end) + 1)
                    )

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
    common_residues: set[Partial_Topology],
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
        common_residues: set[Partial_Topology] | None = None,
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
        for residue in cast(ResidueGroup, ensemble[0].residues):
            coords = []
            for universe in ensemble:
                res = cast(ResidueGroup, universe.residues)[residue.ix]
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
        # val_resids = set(sorted_resids[n_train:])

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
