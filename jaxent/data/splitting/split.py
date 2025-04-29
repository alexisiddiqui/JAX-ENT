import copy
import random
from typing import Sequence, cast

import MDAnalysis as mda
import numpy as np
from icecream import ic  # Import icecream for debugging
from MDAnalysis.core.groups import ResidueGroup

# import kmeans from sklearn.cluster
from sklearn.cluster import KMeans  # type: ignore

from jaxent.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.interfaces.topology import Partial_Topology
from jaxent.models.func.common import find_common_residues

# ic.disable()


def find_fragment_centrality(
    Partial_Topologys: list[Partial_Topology], mode: str = "mean", peptide: bool = True
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
    ic.configureOutput(prefix="CENTRALITY | ")
    ic(f"Finding fragment centrality with mode={mode}, peptide={peptide}")
    ic(f"Number of fragments: {len(Partial_Topologys)}")

    overlaps = []

    for i, fragment in enumerate(Partial_Topologys):
        # Get the set of residues for current fragment
        if peptide and cast(int, fragment.length) > 2:
            current_residues = set(fragment.peptide_residues)
            ic(f"Fragment {i}: Using peptide residues, count={len(current_residues)}")
        else:
            current_residues = set(
                range(fragment.residue_start, cast(int, fragment.residue_end) + 1)
            )
            ic(f"Fragment {i}: Using all residues, count={len(current_residues)}")

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
                    ic(f"Fragment {i} overlaps with {j}: {overlap} residues")

        # Calculate final overlap value based on mode
        if fragment_overlaps:
            if mode == "max":
                overlaps.append(max(fragment_overlaps))
                ic(f"Fragment {i} max overlap: {max(fragment_overlaps)}")
            elif mode == "mean":
                mean_overlap = sum(fragment_overlaps) / len(fragment_overlaps)
                overlaps.append(mean_overlap)
                ic(f"Fragment {i} mean overlap: {mean_overlap:.2f}")
            else:
                raise ValueError("Mode must be either 'max' or 'mean'")
        else:
            overlaps.append(0)
            ic(f"Fragment {i} has no overlaps")

    if overlaps:
        ic(
            f"Centrality results: min={min(overlaps)}, max={max(overlaps)}, avg={sum(overlaps) / len(overlaps):.2f}"
        )
    else:
        ic("Centrality results: No overlaps found")
        overlaps.append(0)  # Ensure the list is not empty
    return overlaps


def filter_common_residues(
    dataset: Sequence[ExpD_Datapoint],
    common_residues: set[Partial_Topology],
    peptide: bool = True,
) -> list[ExpD_Datapoint]:
    # filters the dataset to only include common residues
    # ic.configureOutput(prefix="FILTER | ")
    # ic(f"Filtering dataset with {len(dataset)} fragments")
    # ic(f"Common residues: {len(common_residues)}")
    # ic(f"Using peptide residues: {peptide}")
    print([str(data.top) for data in dataset])
    print([str(res) for res in common_residues])

    new_data = []
    for i, data in enumerate(dataset):
        assert data.top is not None, "Topology is not defined in the experimental data."
        peptide_residues = data.top.extract_residues(peptide=peptide)
        print(peptide_residues)
        ic(f"Fragment {i}: Found {len(peptide_residues)} residues")

        in_common = any([res in common_residues for res in peptide_residues])
        ic(f"Fragment {i}: Has common residues: {in_common}")

        if in_common:
            new_data.append(data)

    ic(f"After filtering: {len(new_data)} fragments remain")
    if not new_data:
        raise ValueError("Filtered dataset is empty. No common residues found.")
    return new_data


class DataSplitter:
    def __init__(
        self,
        dataset: ExpD_Dataloader,
        random_seed: int = 42,
        ensemble: list[mda.Universe] | None = None,
        common_residues: set[Partial_Topology] | None = None,
        peptide: bool = True,
        peptide_trim: int = 1,
        centrality: bool = True,
        train_size: float = 0.5,
        align_sequence: bool = False,
        mda_selection_ignore: str = "resname PRO or resid 1",
    ):
        ic.configureOutput(prefix="SPLITTER | ")
        ic(f"Initializing DataSplitter with random_seed={random_seed}")
        ic(f"Train size: {train_size}, Use centrality: {centrality}")
        ic(f"Dataset size: {len(dataset.data)}")

        self.dataset = copy.deepcopy(dataset)

        self.random_seed = random_seed
        random.seed(random_seed)
        ic(f"Random seed set to {random_seed}")

        self.train_size = train_size
        self.centrality = centrality
        if common_residues is None and ensemble is not None:
            ic("Finding common residues from ensemble")
            common_residues = find_common_residues(ensemble, mda_selection_ignore)[0]
            ic(f"Found {len(common_residues)} common residues")
        elif (common_residues and ensemble) is None:
            ic.format("ERROR: Both common_residues and ensemble are None")
        elif common_residues and ensemble:
            ic(f"Using provided common residues: {len(common_residues)}")
        else:
            raise ValueError("Either common_residues or ensemble must be provided")

        assert isinstance(common_residues, set), (
            f"Common residues must be a set of Partial_Topology objects {common_residues}"
        )

        self.common_residues = common_residues

        print(f"Common residues: {len(common_residues)}")
        print(f"Dataset size: {len(dataset.data)}")

        ic("Filtering dataset to common residues")
        self.dataset.data = filter_common_residues(
            self.dataset.data, common_residues, peptide=peptide
        )
        ic(f"After filtering: {len(self.dataset.data)} fragments")

        self.dataset.top = [data.top for data in self.dataset.data]

        ic("Calculating fragment centrality")
        self.fragment_centrality = find_fragment_centrality(
            self.dataset.top, mode="mean", peptide=peptide
        )
        ic(
            f"Fragment centrality stats: min={min(self.fragment_centrality)}, max={max(self.fragment_centrality)}"
        )

    def sample_by_centrality(self, threshold: float = 0.9) -> list[ExpD_Datapoint]:
        # this helps us select the most central fragments from the dataset - this is crucial for effective splitting with peptide data
        ic.configureOutput(prefix="CENTRALITY_SAMPLE | ")
        ic("Sampling data by centrality")
        ic(f"Threshold: {threshold}")
        ic(f"Max Centrality: {max(self.fragment_centrality)}")

        max_centrality = max(self.fragment_centrality)
        if max_centrality == 0:
            # Assign uniform weights if all weights are zero
            ic("All centrality values are zero, using uniform weights")
            centrality_weights = [1 / len(self.fragment_centrality)] * len(self.fragment_centrality)
        else:
            ic("Calculating weights based on centrality")
            centrality_weights = [1 - (c / max_centrality) for c in self.fragment_centrality]
            total = sum(centrality_weights)
            # Normalize weights
            centrality_weights = [w / total for w in centrality_weights]
            ic(
                f"Weight range: min={min(centrality_weights):.5f}, max={max(centrality_weights):.5f}"
            )

        # randomly sample data based on centrality as weights
        sample_size = int(threshold * len(self.dataset.data))
        ic(f"Sampling {sample_size} fragments")
        train_data = random.choices(self.dataset.data, weights=centrality_weights, k=sample_size)
        ic(f"Sampled {len(train_data)} fragments")

        # only need to sample the training data - rest is discarded
        return train_data

    def validate_split(
        self, train_data: list[ExpD_Datapoint], val_data: list[ExpD_Datapoint]
    ) -> bool:
        # this needs to check that the split is suitable for training - i.e. that datsets are not too small
        ic.configureOutput(prefix="VALIDATE_SPLIT | ")
        ic(f"Validating split: train={len(train_data)}, val={len(val_data)}")

        assert (len(train_data) and len(val_data)) > 0, (
            "No data found in training or validation set"
        )
        ic("Both sets contain data")

        if len(train_data) < 5 or len(val_data) < 5:
            ic.format("ERROR: Set too small - train={}, val={}", len(train_data), len(val_data))
            raise ValueError("Training or validation set is too small")

        # check they are a sufficient proportion of the dataset
        train_ratio = len(train_data) / len(self.dataset.data)
        val_ratio = len(val_data) / len(self.dataset.data)
        ic(f"Train ratio: {train_ratio:.3f}, Val ratio: {val_ratio:.3f}")

        if train_ratio < 0.1 or val_ratio < 0.1:
            ic.format(
                "ERROR: Set ratio too small - train={:.3f}, val={:.3f}", train_ratio, val_ratio
            )
            raise ValueError(
                f"Training or validation set is too small: {len(train_data)} / {len(self.dataset.data)}, {len(val_data)} / {len(self.dataset.data)}"
            )
        ic("Split is valid - continuing")
        return True

    def random_split(
        self, remove_overlap: bool = False
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        # just performs a random split
        ic.configureOutput(prefix="RANDOM_SPLIT | ")
        ic(f"Performing random split with remove_overlap={remove_overlap}")

        if self.centrality:
            ic("Using centrality-based sampling")
            dataset = self.sample_by_centrality()
        else:
            ic("Using full dataset for sampling")
            dataset = self.dataset.data
        ic(f"Dataset size for splitting: {len(dataset)}")

        train_size = int(self.train_size * len(dataset))
        ic(f"Train size: {train_size}")
        train_data = random.sample(dataset, train_size)

        val_data = [d for d in dataset if d not in train_data]
        ic(f"Validation size: {len(val_data)}")

        # print some stats
        ic(f"Number of training fragments: {len(train_data)}")
        ic(f"Number of validation fragments: {len(val_data)}")

        train_fragments = set([d.top for d in train_data])
        val_fragments = set([d.top for d in val_data])

        intersecting_fragments = train_fragments.intersection(val_fragments)
        ic(f"Number of intersecting fragments: {len(intersecting_fragments)}")

        if remove_overlap:
            ic("Removing overlapping fragments")
            # remove overlapping fragments from validation set
            val_data = [d for d in val_data if d.top not in intersecting_fragments]
            train_data = [d for d in train_data if d.top not in intersecting_fragments]
            ic(f"After removal: train={len(train_data)}, val={len(val_data)}")

        try:
            is_valid = self.validate_split(train_data, val_data)
            ic(f"Split validation result: {is_valid}")
            return train_data, val_data
        except Exception as e:
            ic.format("Split validation failed: {}", str(e))
            print(f"Split is not valid - trying again {self.random_seed}")

            # change random seed
            self.random_seed += 1
            print(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)

            return self.random_split(remove_overlap=remove_overlap)

    def stratified_split(
        self, n_strata: int = 5
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Performs a stratified split across y values of the dataset.
        This ensures that the distribution of y values is similar in both train and validation sets.

        Args:
            n_strata: Number of strata to create from the y values

        Returns:
            Tuple of (train_data, val_data) lists containing Experimental_Fragment objects
        """
        ic.configureOutput(prefix="STRATIFIED_SPLIT | ")
        ic(f"Performing stratified split with {n_strata} strata")

        # Get the y values for the dataset
        y_vals = self.dataset.y_true
        ic(
            f"Y values statistics: min={np.min(y_vals):.3f}, max={np.max(y_vals):.3f}, mean={np.mean(y_vals):.3f}"
        )

        # Create strata by binning y values
        bin_edges = np.linspace(np.min(y_vals), np.max(y_vals), n_strata + 1)
        ic(f"Bin edges: {bin_edges}")
        bin_indices = np.digitize(y_vals, bin_edges[:-1])

        # Count samples in each stratum
        stratum_counts = [np.sum(bin_indices == i) for i in range(1, n_strata + 1)]
        ic(f"Samples per stratum: {stratum_counts}")

        train_data = []
        val_data = []

        # Split each stratum according to train_size
        for stratum in range(1, n_strata + 1):
            stratum_indices = np.where(bin_indices == stratum)[0]
            ic(f"Stratum {stratum}: {len(stratum_indices)} samples")

            if len(stratum_indices) == 0:
                ic(f"Stratum {stratum} is empty, skipping")
                continue

            # Get fragments for this stratum
            stratum_fragments = [self.dataset.data[i] for i in stratum_indices]

            # Sample train data from this stratum
            n_train = int(self.train_size * len(stratum_fragments))
            ic(f"Stratum {stratum}: Selecting {n_train} samples for training")
            train_fragments = random.sample(stratum_fragments, n_train)

            # Remaining fragments go to validation
            val_fragments = [f for f in stratum_fragments if f not in train_fragments]
            ic(f"Stratum {stratum}: {len(val_fragments)} samples for validation")

            train_data.extend(train_fragments)
            val_data.extend(val_fragments)

        ic(f"Final split: {len(train_data)} training, {len(val_data)} validation")

        # Validate the split
        try:
            is_valid = self.validate_split(train_data, val_data)
            ic(f"Split validation result: {is_valid}")
            return train_data, val_data
        except Exception as e:
            ic.format("Split validation failed: {}", str(e))
            print("Split is not valid - trying again with different random seed")
            self.random_seed += 1
            ic(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.stratified_split(n_strata)

    def spatial_split(
        self, ensemble: list[mda.Universe]
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Performs a split along the average coordinates of the provided ensemble.
        Selects a residue at random and then selects the closest residues to it up to train_size.

        Args:
            ensemble: List of MDAnalysis Universe objects containing structure information

        Returns:
            Tuple of (train_data, val_data) lists containing Experimental_Fragment objects
        """
        ic.configureOutput(prefix="SPATIAL_SPLIT | ")
        ic(f"Performing spatial split with {len(ensemble)} ensemble members")

        ###############################################################################
        # TODO this needs to be modified to use common_resiues
        # Calculate average coordinates for each residue
        ic("Calculating average coordinates for residues")
        avg_coords = {}
        for residue in cast(ResidueGroup, ensemble[0].residues):
            ic(f"Processing residue {residue.resid}")
            coords = []
            for universe in ensemble:
                res = cast(ResidueGroup, universe.residues)[residue.ix]
                coords.append(res.atoms.center_of_mass())
            avg_coords[residue.resid] = np.mean(coords, axis=0)

        ic(f"Calculated coordinates for {len(avg_coords)} residues")

        # Select a random seed residue
        seed_resid = random.choice(list(avg_coords.keys()))
        seed_coords = avg_coords[seed_resid]
        ic(f"Selected seed residue {seed_resid} at coordinates {seed_coords}")

        # Calculate distances from seed to all other residues
        ic("Calculating distances from seed residue")
        distances = {
            resid: np.linalg.norm(coords - seed_coords) for resid, coords in avg_coords.items()
        }
        ic(f"Distance range: min={min(distances.values()):.3f}, max={max(distances.values()):.3f}")

        # Sort residues by distance
        sorted_resids = sorted(distances.keys(), key=lambda x: float(distances[x]))
        ic(f"Sorted {len(sorted_resids)} residues by distance")
        ################################################################################
        # Take closest residues for training set up to train_size
        n_train = int(self.train_size * len(sorted_resids))
        train_resids = set(sorted_resids[:n_train])
        ic(f"Selected {len(train_resids)} residues for training")
        # val_resids = set(sorted_resids[n_train:])

        # Split fragments based on residue assignments
        train_data = []
        val_data = []

        ic("Assigning fragments to train/val sets")
        for i, fragment in enumerate(self.dataset.data):
            if fragment.top.residue_start in train_resids:
                train_data.append(fragment)
                ic(f"Fragment {i} assigned to training")
            else:
                val_data.append(fragment)
                ic(f"Fragment {i} assigned to validation")

        ic(f"Initial split: {len(train_data)} training, {len(val_data)} validation")

        # Validate the split
        try:
            is_valid = self.validate_split(train_data, val_data)
            ic(f"Split validation result: {is_valid}")
            return train_data, val_data
        except Exception as e:
            ic.format("Split validation failed: {}", str(e))
            print("Split is not valid - trying again with different seed residue")
            self.random_seed += 1
            ic(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.spatial_split(ensemble)

    def cluster_split(
        self, n_clusters: int = 10, peptide: bool = True, cluster_index: str = "residue_index"
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Clusters data along the specified index and performs splitting based on clusters.

        Args:
            n_clusters: Number of clusters to create
            peptide: Whether to use peptide residues or all residues
            cluster_index: What to cluster on ("sequence", "residue_index", "fragment_index", "features")

        Returns:
            Tuple of (train_data, val_data) lists containing Experimental_Fragment objects
        """
        ic.configureOutput(prefix="CLUSTER_SPLIT | ")
        ic(f"Performing cluster split with {n_clusters} clusters")
        ic(f"Peptide: {peptide}, Cluster index: {cluster_index}")

        possible_indexes = {"sequence", "residue_index", "fragment_index", "features"}
        if cluster_index not in possible_indexes:
            ic.format("Invalid cluster index: {}", cluster_index)
            raise ValueError(f"Cluster index must be one of {possible_indexes}")

        # Create feature vectors based on clustering index
        ic("Creating feature vectors for clustering")
        features = []
        for i, fragment in enumerate(self.dataset.data):
            if cluster_index == "sequence":
                # Use one-hot encoding of amino acid sequence
                aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
                feature = [aa_dict.get(aa, -1) for aa in fragment.top.fragment_sequence]
                ic(f"Fragment {i}: sequence feature vector length {len(feature)}")
            elif cluster_index == "residue_index":
                # Use start and end residue indices
                feature = [fragment.top.residue_start, fragment.top.residue_end]
                ic(f"Fragment {i}: residue indices {feature}")
            elif cluster_index == "fragment_index":
                # Use fragment index directly
                feature = [fragment.top.fragment_index]
                ic(f"Fragment {i}: fragment index {feature}")
            else:  # features
                # Use experimental values
                feature = fragment.extract_features()
                ic(f"Fragment {i}: extracted features length {len(feature)}")

            features.append(feature)

        # Convert to numpy array and normalize
        features = np.array(features)
        ic(f"Features array shape: {features.shape}")

        # Check for NaN values
        if np.isnan(features).any():
            ic.format("WARNING: Features contain NaN values")

        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        ic("Features normalized")

        # Perform k-means clustering
        ic(f"Performing k-means clustering with {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(features)

        # Count samples in each cluster
        cluster_counts = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        ic(f"Samples per cluster: {cluster_counts}")

        train_data = []
        val_data = []

        # Split each cluster according to train_size
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            ic(f"Cluster {cluster}: {len(cluster_indices)} samples")

            if len(cluster_indices) == 0:
                ic(f"Cluster {cluster} is empty, skipping")
                continue

            # Get fragments for this cluster
            cluster_fragments = [self.dataset.data[i] for i in cluster_indices]

            # Sample train data from this cluster
            n_train = int(self.train_size * len(cluster_fragments))
            ic(f"Cluster {cluster}: Selecting {n_train} samples for training")
            train_fragments = random.sample(cluster_fragments, n_train)

            # Remaining fragments go to validation
            val_fragments = [f for f in cluster_fragments if f not in train_fragments]
            ic(f"Cluster {cluster}: {len(val_fragments)} samples for validation")

            train_data.extend(train_fragments)
            val_data.extend(val_fragments)

        ic(f"Final split: {len(train_data)} training, {len(val_data)} validation")

        # Validate the split
        try:
            is_valid = self.validate_split(train_data, val_data)
            ic(f"Split validation result: {is_valid}")
            return train_data, val_data
        except Exception as e:
            ic.format("Split validation failed: {}", str(e))
            print("Split is not valid - trying again with different random seed")
            self.random_seed += 1
            ic(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.cluster_split(n_clusters, peptide, cluster_index)
