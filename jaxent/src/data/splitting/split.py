import copy
import random
from typing import Sequence, cast

import MDAnalysis as mda
import numpy as np
from MDAnalysis.core.groups import ResidueGroup

# import kmeans from sklearn.cluster
from sklearn.cluster import KMeans  # type: ignore

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.interfaces.topology import Partial_Topology


def filter_common_residues(
    dataset: Sequence[ExpD_Datapoint],
    common_residues: set[Partial_Topology],
    check_trim: bool = False,
) -> list[ExpD_Datapoint]:
    """
    Filter dataset to only include datapoints that have residues in common with the provided set.
    Uses topology-aware methods for robust comparison including chain awareness and peptide trimming.

    Args:
        dataset: Sequence of experimental datapoints to filter
        common_residues: Set of Partial_Topology objects representing residues to match against
        check_trim: If True, use peptide_residues for peptides; if False, use all residues

    Returns:
        List of datapoints that have overlapping residues with common_residues

    Raises:
        ValueError: If filtered dataset is empty
    """
    from tqdm import tqdm

    print(f"Dataset topologies ({len(dataset)}):")
    print([str(data.top) for data in dataset])
    print(f"Common residues ({len(common_residues)}):")
    print([str(res) for res in common_residues])

    new_data = []

    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Filtering fragments"):
        assert data.top is not None, "Topology is not defined in the experimental data."

        # Use the robust topology intersection method to check for overlap
        # This automatically handles chain matching, peptide trimming, etc.
        has_common_residues = any(
            data.top.intersects(common_topo, check_trim=check_trim)
            for common_topo in common_residues
        )

        # Only print detailed debugging information if no intersections found
        if not has_common_residues:
            pass

        if has_common_residues:
            new_data.append(data)

    print(f"After filtering: {len(new_data)}/{len(dataset)} fragments remain")

    if not new_data:
        raise ValueError("Filtered dataset is empty. No common residues found.")

    return new_data


class DataSplitter:
    """
    This class handles data splitting for experimental datasets.
    It supports various splitting strategies including random, stratified, spatial, and clustering.
    It also allows for centrality-based sampling to prioritize more central fragments in the dataset.

    The process requries either an ensemble of MDAnalysis Universes or a set of precomputed common residues to ensure that fragments
    share a common set of residues, which is crucial for effective splitting, especially with peptide data. Partial_Topology objects only cover an individual chain and so comparisons must be made for each chain separately.

    If no common residues are provided, it will attempt to compute them from the ensemble.
    - This uses the 'find_common_residues' method from the Partial_Topology class to identify residues that are shared across all fragments in the ensemble.
    - find_common_residues produces a set of Partial_Topology objects representing these common residues. These are then grouped by chain and are then merged to create Partial_Topologies that cover entire chains.

    The Experimental Dataset is then filtered to only include fragments that contain these common residues for each chain.
    - To save computation this is performed against the chain partial topologies, which are then used to filter the dataset.

    The overlap between experimental data points is calculated using the 'calculate_fragment_redundancy' method from the Partial_Topology class.

    Splitting:
    The overall process of splitting the dataset involves selecting residue parital_topologies using the splitting algorithm specified and then builing merged Partial_Topology objects from these residues for the training and validation sets.
    The datapoints are then filtered to only include those that match the selected training/validation partial_topologies for each chain.
    Overlapping fragments are removed by checking against the merged Partial_Topology objects for each chain of the opposite training/validation set.

    - Random Split:
    This randomly selects the datapoints - merges them into Partial_Topology objects and then filters the dataset to only include those that match the selected training/validation partial_topologies for each chain.

    - Stratified Split:
    This extracts the y values from the dataset and applies z scoring to each dimension (ex: timepoints in HDX-uptake) across the datapoints. This ensures that y values can be binned into strata by averaging across z scores.
    The dataset is binned into strata based on these z scores and then an equal number of datapoints are sampled from each stratum to create the training and validation sets. Each training/validation set is merged across chains to create Partial_Topology objects that cover the entire chain for training and validation set.

    - Spatial Split:
    This performs a split based on the proximal residues in the ensemble. This is achieved by calculating the pairwise-residue distances from the averaged coordiantes of residudes across ensembles. This will require an additional method.
    From these pairwise distacnes a seed residue is selected at random from a selected chain (most common if not provided) and the closest residues to it are selected up to the specified train_size. This is then used to create Partial_Topology objects for the training and validation sets.

    - Cluster Split:
    This performs a clustering of the dataset based on the specified clustering type (sequence, residue_index, fragment_index, or features (z-score normalised by dimension/timepoint)). Clustering is performed using k-means clustering based on the data. Merged Partial_Topology objects are created for the training and validation sets for each chain based on the cluster labels assigned to each datapoint.
    """

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
        check_chains: bool = True,
        mda_selection_ignore: str = "resname PRO or resid 1",
    ):
        self.dataset = copy.deepcopy(dataset)

        self.random_seed = random_seed
        random.seed(random_seed)

        self.train_size = train_size
        self.centrality = centrality
        if common_residues is None and ensemble is not None:
            common_residues, excluded_residues = Partial_Topology.find_common_residues(
                ensemble, mda_selection_ignore
            )
        elif (common_residues and ensemble) is None:
            pass
        elif common_residues and ensemble:
            pass
        else:
            raise ValueError("Either common_residues or ensemble must be provided")

        assert isinstance(common_residues, set), (
            f"Common residues must be a set of Partial_Topology objects {common_residues}"
        )

        self.common_residues = common_residues

        print(f"Common residues: {len(common_residues)}")
        print(f"Dataset size: {len(dataset.data)}")

        self.dataset.data = filter_common_residues(
            self.dataset.data, common_residues, peptide=peptide
        )

        self.dataset.top = [data.top for data in self.dataset.data]

        self.fragment_overlaps = Partial_Topology.calculate_fragment_redundancy(
            self.dataset.top, mode="mean", peptide=peptide
        )

    def sample_by_centrality(self, threshold: float = 0.9) -> list[ExpD_Datapoint]:
        # this helps us select the most central fragments from the dataset - this is crucial for effective splitting with peptide data
        max_centrality = max(self.fragment_overlaps)
        if max_centrality == 0:
            # Assign uniform weights if all weights are zero
            centrality_weights = [1 / len(self.fragment_overlaps)] * len(self.fragment_overlaps)
        else:
            centrality_weights = [1 - (c / max_centrality) for c in self.fragment_overlaps]
            total = sum(centrality_weights)
            # Normalize weights
            centrality_weights = [w / total for w in centrality_weights]

        # randomly sample data based on centrality as weights
        sample_size = int(threshold * len(self.dataset.data))
        train_data = random.choices(self.dataset.data, weights=centrality_weights, k=sample_size)

        # only need to sample the training data - rest is discarded
        return train_data

    def validate_split(
        self, train_data: list[ExpD_Datapoint], val_data: list[ExpD_Datapoint]
    ) -> bool:
        # this needs to check that the split is suitable for training - i.e. that datsets are not too small
        assert (len(train_data) and len(val_data)) > 0, (
            "No data found in training or validation set"
        )

        if len(train_data) < 5 or len(val_data) < 5:
            raise ValueError("Training or validation set is too small")

        # check they are a sufficient proportion of the dataset
        train_ratio = len(train_data) / len(self.dataset.data)
        val_ratio = len(val_data) / len(self.dataset.data)

        if train_ratio < 0.1 or val_ratio < 0.1:
            raise ValueError(
                f"Training or validation set is too small: {len(train_data)} / {len(self.dataset.data)}, {len(val_data)} / {len(self.dataset.data)}"
            )
        return True

    def random_split(
        self, remove_overlap: bool = False
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        # just performs a random split
        if self.centrality:
            dataset = self.sample_by_centrality()
        else:
            dataset = self.dataset.data

        train_size = int(self.train_size * len(dataset))
        train_data = random.sample(dataset, train_size)

        val_data = [d for d in dataset if d not in train_data]

        train_fragments = set([d.top for d in train_data])
        val_fragments = set([d.top for d in val_data])

        intersecting_fragments = train_fragments.intersection(val_fragments)

        if remove_overlap:
            # remove overlapping fragments from validation set
            val_data = [d for d in val_data if d.top not in intersecting_fragments]
            train_data = [d for d in train_data if d.top not in intersecting_fragments]

        try:
            is_valid = self.validate_split(train_data, val_data)
            return train_data, val_data
        except Exception:
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
        # Get the y values for the dataset
        y_vals = self.dataset.y_true

        # Create strata by binning y values
        bin_edges = np.linspace(np.min(y_vals), np.max(y_vals), n_strata + 1)
        bin_indices = np.digitize(y_vals, bin_edges[:-1])

        # Count samples in each stratum
        stratum_counts = [np.sum(bin_indices == i) for i in range(1, n_strata + 1)]

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
        try:
            is_valid = self.validate_split(train_data, val_data)
            return train_data, val_data
        except Exception:
            print("Split is not valid - trying again with different random seed")
            self.random_seed += 1
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

        for i, fragment in enumerate(self.dataset.data):
            if fragment.top.residue_start in train_resids:
                train_data.append(fragment)
            else:
                val_data.append(fragment)

        # Validate the split
        try:
            is_valid = self.validate_split(train_data, val_data)
            return train_data, val_data
        except Exception:
            print("Split is not valid - trying again with different seed residue")
            self.random_seed += 1
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
        possible_indexes = {"sequence", "residue_index", "fragment_index", "features"}
        if cluster_index not in possible_indexes:
            raise ValueError(f"Cluster index must be one of {possible_indexes}")

        # Create feature vectors based on clustering index
        features = []
        for i, fragment in enumerate(self.dataset.data):
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

        # Check for NaN values
        if np.isnan(features).any():
            pass

        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(features)

        # Count samples in each cluster
        cluster_counts = [np.sum(cluster_labels == i) for i in range(n_clusters)]

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
        try:
            is_valid = self.validate_split(train_data, val_data)
            return train_data, val_data
        except Exception:
            print("Split is not valid - trying again with different random seed")
            self.random_seed += 1
            random.seed(self.random_seed)
            return self.cluster_split(n_clusters, peptide, cluster_index)
