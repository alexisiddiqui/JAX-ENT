import copy
import random
from typing import Optional, Sequence

import MDAnalysis as mda

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

    The process requires either an ensemble of MDAnalysis Universes or a set of precomputed common residues to ensure that fragments
    share a common set of residues, which is crucial for effective splitting, especially with peptide data.

    Key improvements:
    - Simplified topology merging with reusable class method
    - Pre-computed splittable_residues for efficient splitting
    - Streamlined splitting logic using filter_common_residues
    """

    def __init__(
        self,
        dataset: ExpD_Dataloader,
        random_seed: int = 42,
        ensemble: list[mda.Universe] | None = None,
        common_residues: set[Partial_Topology] | None = None,
        check_trim: bool = False,
        peptide_trim: int = 2,
        centrality: bool = True,
        train_size: float = 0.5,
        include_selection: str | list[str] = "protein",
        exclude_selection: str | list[str] = "resname PRO or resid 1",
        max_retry_depth: int = 10,
        min_split_size: int = 5,
    ):
        self.dataset = copy.deepcopy(dataset)
        self.random_seed = random_seed
        self.original_random_seed = random_seed  # Store original for retry logic
        random.seed(random_seed)
        self.train_size = train_size
        self.centrality = centrality
        self.check_trim = check_trim
        self.check_trim_trim = peptide_trim
        self.max_retry_depth = max_retry_depth
        self.current_retry_count = 0
        self.min_split_size = min_split_size

        # Process selection strings - convert to lists and ensure same length
        include_selection = self._process_selection_strings(include_selection)
        exclude_selection = self._process_selection_strings(exclude_selection)

        # Ensure both lists are the same length
        if len(include_selection) != len(exclude_selection):
            # If one is length 1, repeat it to match the other
            if len(include_selection) == 1 and len(exclude_selection) > 1:
                include_selection = include_selection * len(exclude_selection)
            elif len(exclude_selection) == 1 and len(include_selection) > 1:
                exclude_selection = exclude_selection * len(include_selection)
            else:
                raise ValueError(
                    f"include_selection and exclude_selection must be the same length or one must be length 1. "
                    f"Got lengths {len(include_selection)} and {len(exclude_selection)}"
                )

        self.include_selection = include_selection
        self.exclude_selection = exclude_selection

        print(
            f"Using selection strings: include_selection={include_selection}, exclude_selection={exclude_selection}"
        )

        # Handle common_residues - either provided or computed from ensemble
        if common_residues is None and ensemble is not None:
            # For now, use the first selection string - could be enhanced to use all
            common_residues, excluded_residues = Partial_Topology.find_common_residues(
                ensemble,
                include_selection=self.include_selection,
                exclude_selection=self.exclude_selection,
            )
        elif common_residues is None and ensemble is None:
            raise ValueError("Either common_residues or ensemble must be provided")

        assert isinstance(common_residues, set), (
            f"Common residues must be a set of Partial_Topology objects {common_residues}"
        )

        if len(common_residues) < 2:
            raise ValueError(
                f"Common residues must contain at least 2 residues for splitting, got {len(common_residues)}"
            )

        # Store original common_residues
        self.common_residues = common_residues

        print(f"Common residues: {len(common_residues)}")
        print(f"Dataset size: {len(dataset.data)}")

        # Filter dataset using common residues
        self.dataset.data = filter_common_residues(
            self.dataset.data, common_residues, check_trim=check_trim
        )

        # If dataset is empty after filtering, raise an error
        if not self.dataset.data:
            raise ValueError("Filtered dataset is empty. No common residues found in the dataset.")

        # Create splittable_residues - these are the merged topology objects per chain
        # that define the splittable space for data splitting
        self.splittable_residues = self._create_splittable_residues()

        print(f"Created splittable residues for {len(self.splittable_residues)} chains:")
        for chain, topo in self.splittable_residues.items():
            print(f"  Chain {chain}: {topo}")

        # Calculate fragment overlaps for centrality sampling
        if self.dataset.data:  # Only calculate if we have data
            self.fragment_overlaps = Partial_Topology.calculate_fragment_redundancy(
                [data.top for data in self.dataset.data], mode="mean", check_trim=check_trim
            )
        else:
            # Handle empty dataset case
            self.fragment_overlaps = []

        # Initialize storage for last split results
        self.last_split_train_topologies_by_chain = {}
        self.last_split_val_topologies_by_chain = {}

    def _process_selection_strings(self, selection: str | list[str]) -> list[str]:
        """
        Process selection strings to ensure they are lists.

        Args:
            selection: Either a string or list of strings

        Returns:
            List of selection strings
        """
        if isinstance(selection, str):
            return [selection]
        elif isinstance(selection, list):
            return selection
        else:
            raise ValueError(f"Selection must be string or list of strings, got {type(selection)}")

    def _create_splittable_residues(self) -> dict[str, Partial_Topology]:
        """
        Create merged topology objects per chain from common_residues.
        These define the splittable space for each chain.

        Returns:
            Dictionary mapping chain -> merged Partial_Topology for that chain
        """
        common_residues_by_chain = Partial_Topology.group_set_by_chain(self.common_residues)
        splittable_residues = {}

        for chain, chain_residues in common_residues_by_chain.items():
            if chain_residues:
                try:
                    merged_topo = Partial_Topology.merge(
                        list(chain_residues),
                        trim=self.check_trim,
                        merged_name=f"splittable_chain_{chain}",
                    )
                    splittable_residues[chain] = merged_topo
                except ValueError as e:
                    print(f"Warning: Could not merge common residues for chain {chain}: {e}")

        return splittable_residues

    def _merge_topologies_by_chain(
        self,
        topologies: list[Partial_Topology],
        check_trim: bool = False,
        name_prefix: str = "merged",
    ) -> dict[str, Partial_Topology]:
        """
        Group topologies by chain and merge them.

        Args:
            topologies: List of Partial_Topology objects to group and merge
            check_trim: Whether to apply trimming during merge
            name_prefix: Prefix for naming merged topologies

        Returns:
            Dictionary mapping chain -> merged Partial_Topology for that chain
        """
        topologies_by_chain = Partial_Topology.group_set_by_chain(set(topologies))
        merged_topologies = {}

        for chain, chain_topologies in topologies_by_chain.items():
            if chain_topologies:
                try:
                    merged_topo = Partial_Topology.merge(
                        list(chain_topologies),
                        trim=check_trim,
                        merged_name=f"{name_prefix}_chain_{chain}",
                    )
                    merged_topologies[chain] = merged_topo
                    print(
                        f"Merged {len(chain_topologies)} topologies for chain {chain}: {merged_topo}"
                    )
                except ValueError as e:
                    print(f"Warning: Could not merge topologies for chain {chain}: {e}")

        return merged_topologies

    def _remove_overlaps(
        self,
        merged_train_topologies: dict[str, Partial_Topology],
        merged_val_topologies: dict[str, Partial_Topology],
    ) -> dict[str, Partial_Topology]:
        """
        Removes overlapping residues between merged train and validation topologies.

        Args:
            merged_train_topologies: Dictionary of chain -> merged Partial_Topology for training data.
            merged_val_topologies: Dictionary of chain -> merged Partial_Topology for validation data.

        Returns:
            Updated dictionary of chain -> merged Partial_Topology for validation data, with overlaps removed.

        Raises:
            ValueError: If no validation topologies remain after overlap removal.
        """
        print("Removing overlaps between merged train/val topologies...")

        updated_val_topologies = {}

        for chain in merged_val_topologies.keys():
            val_topo = merged_val_topologies[chain]

            if chain in merged_train_topologies:
                train_topo = merged_train_topologies[chain]

                # Check if there's overlap between train and val topologies for this chain
                if val_topo.intersects(train_topo, check_trim=self.check_trim):
                    overlap_residues = val_topo.get_overlap(train_topo, check_trim=self.check_trim)
                    print(f"Chain {chain}: Found {len(overlap_residues)} overlapping residues")

                    # Create a temporary topology with overlapping residues to remove
                    overlap_topo = Partial_Topology.from_residues(
                        chain=chain, residues=overlap_residues, fragment_name="overlap_temp"
                    )

                    try:
                        # Remove overlapping residues from validation topology
                        updated_val_topo = val_topo.remove_residues([overlap_topo])
                        updated_val_topologies[chain] = updated_val_topo
                        print(
                            f"Chain {chain}: Removed overlap, val topology now has {len(updated_val_topo.residues)} residues"
                        )
                    except ValueError as e:
                        # If no residues remain after removal, skip this chain for validation
                        print(
                            f"Chain {chain}: Skipping validation topology - no residues remain after overlap removal: {e}"
                        )
                        continue
                else:
                    # No overlap, keep original topology
                    updated_val_topologies[chain] = val_topo
            else:
                # No corresponding train topology for this chain, keep original
                updated_val_topologies[chain] = val_topo

        # Update merged topologies
        merged_val_topologies = updated_val_topologies

        # Check if we still have validation topologies after overlap removal
        if not merged_val_topologies:
            raise ValueError("No validation topologies remain after overlap removal")

        return merged_val_topologies

    def sample_by_centrality(self, threshold: float = 0.9) -> list[ExpD_Datapoint]:
        """
        Sample fragments by centrality to select the most central fragments from the dataset.
        This is crucial for effective splitting with peptide data.
        """
        if not self.fragment_overlaps:
            # If no fragment overlaps calculated, return the entire dataset
            print("No fragment overlaps available, returning entire dataset")
            return self.dataset.data

        max_centrality = max(self.fragment_overlaps)
        if max_centrality == 0:
            # Assign uniform weights if all weights are zero
            centrality_weights = [1 / len(self.fragment_overlaps)] * len(self.fragment_overlaps)
        else:
            centrality_weights = [1 - (c / max_centrality) for c in self.fragment_overlaps]
            total = sum(centrality_weights)
            if total == 0:
                # If total is zero, all weights are zero. Assign uniform weights.
                centrality_weights = [1 / len(self.fragment_overlaps)] * len(self.fragment_overlaps)
            else:
                # Normalize weights
                centrality_weights = [w / total for w in centrality_weights]

        # randomly sample data based on centrality as weights
        sample_size = int(threshold * len(self.dataset.data))
        train_data = random.choices(self.dataset.data, weights=centrality_weights, k=sample_size)

        print(f"Sampled {len(train_data)} fragments by centrality (threshold={threshold})")
        return train_data

    def validate_split(
        self, train_data: list[ExpD_Datapoint], val_data: list[ExpD_Datapoint]
    ) -> bool:
        """
        Validate that the split is suitable for training - i.e. that datasets are not too small
        """
        assert len(train_data) > 0 and len(val_data) > 0, (
            "No data found in training or validation set"
        )

        if len(train_data) < self.min_split_size or len(val_data) < self.min_split_size:
            raise ValueError(
                f"Training ({len(train_data)}) or validation ({len(val_data)}) set is too small"
            )

        # check they are a sufficient proportion of the dataset
        train_ratio = len(train_data) / len(self.dataset.data)
        val_ratio = len(val_data) / len(self.dataset.data)

        if train_ratio < 0.1 or val_ratio < 0.1:
            raise ValueError(
                f"Training or validation set is too small: "
                f"train={len(train_data)}/{len(self.dataset.data)} ({train_ratio:.2%}), "
                f"val={len(val_data)}/{len(self.dataset.data)} ({val_ratio:.2%})"
            )
        return True

    def _reset_retry_counter(self):
        """Reset retry counter for new split attempts."""
        self.current_retry_count = 0
        self.random_seed = self.original_random_seed
        random.seed(self.random_seed)

    def random_split(
        self, remove_overlap: bool = False
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Perform random split using simplified approach with pre-computed splittable residues.

        Process:
        0. Optional: Remove highly redundant fragments by centrality sampling
        1. Randomly select training datapoints
        2. Create merged topologies for train/val sets using class method
        3. Remove overlaps between merged topologies if requested (using Partial_Topology methods)
        4. Use filter_common_residues to assign datapoints to train/val sets

        Args:
            remove_overlap: If True, remove overlaps between train and val merged topologies

        Returns:
            Tuple of (training_datapoints, validation_datapoints)
        """

        print(
            f"Starting random split with train_size={self.train_size}, remove_overlap={remove_overlap}"
        )

        # Step 1: Apply centrality sampling if requested, then randomly select training datapoints
        if self.centrality:
            source_dataset = self.sample_by_centrality()
        else:
            source_dataset = self.dataset.data

        if len(source_dataset) < 2:
            raise ValueError(
                f"Source dataset is too small to split ({len(source_dataset)} datapoints)."
            )

        train_size = int(self.train_size * len(source_dataset))
        selected_train_data = random.sample(source_dataset, train_size)
        selected_val_data = [d for d in source_dataset if d not in selected_train_data]

        print(
            f"Randomly selected {len(selected_train_data)} training and {len(selected_val_data)} validation datapoints"
        )

        # Step 2: Create merged topologies using the class method
        train_topologies = [d.top for d in selected_train_data]
        val_topologies = [d.top for d in selected_val_data]

        merged_train_topologies = self._merge_topologies_by_chain(
            train_topologies, self.check_trim, "train"
        )
        merged_val_topologies = self._merge_topologies_by_chain(
            val_topologies, self.check_trim, "val"
        )

        # Step 3: Remove overlaps between merged topologies if requested
        if remove_overlap:
            merged_val_topologies = self._remove_overlaps(
                merged_train_topologies, merged_val_topologies
            )

        # Step 4: Use filter_common_residues to create train/val sets
        # Convert merged topologies back to sets for filtering
        train_topology_set = set(merged_train_topologies.values())
        val_topology_set = set(merged_val_topologies.values())

        # Filter entire dataset using the merged topologies
        final_train_data = filter_common_residues(
            self.dataset.data, train_topology_set, check_trim=self.check_trim
        )
        final_val_data = filter_common_residues(
            self.dataset.data, val_topology_set, check_trim=self.check_trim
        )

        # Store merged topologies for reference
        self.last_split_train_topologies_by_chain = merged_train_topologies
        self.last_split_val_topologies_by_chain = merged_val_topologies

        print(f"Final split: {len(final_train_data)} training, {len(final_val_data)} validation")

        try:
            self.validate_split(final_train_data, final_val_data)
            return final_train_data, final_val_data
        except Exception as e:
            if self.current_retry_count >= self.max_retry_depth:
                raise ValueError(
                    f"Failed to create valid split after {self.max_retry_depth} attempts. "
                    f"Last error: {e}. Consider adjusting train_size or other parameters."
                )

            print(
                f"Split is not valid - trying again (attempt {self.current_retry_count + 1}/{self.max_retry_depth}, seed {self.random_seed}): {e}"
            )
            # Increment retry count and random seed
            self.current_retry_count += 1
            self.random_seed += 1
            print(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.random_split(remove_overlap=remove_overlap)

    def sequence_cluster_split(
        self, n_clusters: Optional[int] = None, remove_overlap: bool = False
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Perform sequence (k_means) clustering split using k-means clustering on peptide start/end positions.

        Process:
        0. Optional: Remove highly redundant fragments by centrality sampling
        1. Extract start and end positions of each peptide as features
        2. Perform k-means clustering using 1/10th of dataset size as clusters (or specified n_clusters)
        3. Randomly assign clusters to train/val according to train_size
        4. Create merged topologies for train/val sets
        5. Remove overlaps between merged topologies if requested
        6. Filter dataset and validate split

        Args:
            n_clusters: Number of clusters (default uses 1/10th of dataset size)
            remove_overlap: If True, remove overlaps between train and val merged topologies

        Returns:
            Tuple of (training_datapoints, validation_datapoints)
        """
        try:
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError as e:
            if "numpy" in str(e):
                raise ImportError(
                    "NumPy is required for clustering. Install with: pip install numpy"
                )
            else:
                raise ImportError(
                    "scikit-learn is required for clustering. Install with: pip install scikit-learn"
                )

        print(
            f"Starting sequence cluster split with n_clusters={n_clusters}, remove_overlap={remove_overlap}"
        )

        # Step 0: Apply centrality sampling if requested
        if self.centrality:
            source_dataset = self.sample_by_centrality()
        else:
            source_dataset = self.dataset.data

        if len(source_dataset) < 2:
            raise ValueError(
                f"Source dataset is too small to split ({len(source_dataset)} datapoints)."
            )

        # Step 1: Extract start and end positions as features
        features = []
        for datapoint in source_dataset:
            active_residues = datapoint.top._get_active_residues(self.check_trim)
            if not active_residues:
                raise ValueError(f"No active residues found for datapoint {datapoint.top}")

            start_pos = min(active_residues)
            end_pos = max(active_residues)
            features.append([start_pos, end_pos])

        features = np.array(features)
        print(f"Extracted start/end position features for {len(features)} fragments")

        # Step 2: Determine number of clusters (use 1/10th of dataset if default)
        if n_clusters is None or n_clusters > len(source_dataset):
            n_clusters = max(2, len(source_dataset) // 10)

        # Ensure we don't have more clusters than datapoints
        n_clusters = min(n_clusters, len(source_dataset))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(features)

        print(f"Clustered {len(source_dataset)} fragments into {n_clusters} clusters")

        # Step 3: Randomly assign clusters to train/val according to train_size
        unique_clusters = np.unique(cluster_labels)
        n_train_clusters = int(self.train_size * len(unique_clusters))

        random.seed(self.random_seed)
        train_clusters = set(random.sample(list(unique_clusters), n_train_clusters))
        val_clusters = set(unique_clusters) - train_clusters

        print(
            f"Assigned {len(train_clusters)} clusters to training, {len(val_clusters)} to validation"
        )

        # Step 4: Create train/val datasets based on cluster assignments
        selected_train_data = []
        selected_val_data = []

        for i, datapoint in enumerate(source_dataset):
            cluster = cluster_labels[i]
            if cluster in train_clusters:
                selected_train_data.append(datapoint)
            else:
                selected_val_data.append(datapoint)

        print(
            f"Cluster assignment: {len(selected_train_data)} training, {len(selected_val_data)} validation datapoints"
        )

        # Step 5: Create merged topologies using the class method
        train_topologies = [d.top for d in selected_train_data]
        val_topologies = [d.top for d in selected_val_data]

        merged_train_topologies = self._merge_topologies_by_chain(
            train_topologies, self.check_trim, "train_cluster"
        )
        merged_val_topologies = self._merge_topologies_by_chain(
            val_topologies, self.check_trim, "val_cluster"
        )

        # Step 6: Remove overlaps between merged topologies if requested
        if remove_overlap:
            merged_val_topologies = self._remove_overlaps(
                merged_train_topologies, merged_val_topologies
            )

        # Step 7: Use filter_common_residues to create final train/val sets
        # Convert merged topologies back to sets for filtering
        train_topology_set = set(merged_train_topologies.values())
        val_topology_set = set(merged_val_topologies.values())

        # Filter entire dataset using the merged topologies
        final_train_data = filter_common_residues(
            self.dataset.data, train_topology_set, check_trim=self.check_trim
        )
        final_val_data = filter_common_residues(
            self.dataset.data, val_topology_set, check_trim=self.check_trim
        )

        # Store merged topologies for reference
        self.last_split_train_topologies_by_chain = merged_train_topologies
        self.last_split_val_topologies_by_chain = merged_val_topologies

        print(f"Final split: {len(final_train_data)} training, {len(final_val_data)} validation")

        try:
            self.validate_split(final_train_data, final_val_data)
            return final_train_data, final_val_data
        except Exception as e:
            if self.current_retry_count >= self.max_retry_depth:
                raise ValueError(
                    f"Failed to create valid split after {self.max_retry_depth} attempts. "
                    f"Last error: {e}. Consider adjusting train_size or other parameters."
                )

            print(
                f"Split is not valid - trying again (attempt {self.current_retry_count + 1}/{self.max_retry_depth}, seed {self.random_seed}): {e}"
            )
            # Increment retry count and random seed
            self.current_retry_count += 1
            self.random_seed += 1
            print(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.sequence_cluster_split(n_clusters=n_clusters, remove_overlap=remove_overlap)

    def sequence_split(
        self, remove_overlap: bool = False
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Splits the dataset along the sequence of the data points.

        Process:
        0. Optional: Remove highly redundant fragments by centrality sampling
        1. Sort datapoints by sequence position (chain, then start position)
        2. Take first train_size fraction for training, remainder for validation
        3. Create merged topologies for train/val sets using class method
        4. Remove overlaps between merged topologies if requested
        5. Filter dataset and validate split

        Args:
            remove_overlap: If True, remove overlaps between train and val merged topologies

        Returns:
            Tuple of (training_datapoints, validation_datapoints)
        """
        print(
            f"Starting sequence split with train_size={self.train_size}, remove_overlap={remove_overlap}"
        )

        # Step 0: Apply centrality sampling if requested
        if self.centrality:
            source_dataset = self.sample_by_centrality()
        else:
            source_dataset = self.dataset.data

        if len(source_dataset) < 2:
            raise ValueError(
                f"Source dataset is too small to split ({len(source_dataset)} datapoints)."
            )

        # Step 1: Sort datapoints by sequence position
        def get_sequence_sort_key(datapoint):
            """Generate sort key for sequence ordering"""
            active_residues = datapoint.top._get_active_residues(self.check_trim)
            if not active_residues:
                raise ValueError(f"No active residues found for datapoint {datapoint.top}")

            # Sort by chain first, then by start position
            chain_score = datapoint.top._get_chain_score()
            start_pos = min(active_residues)

            return (chain_score, start_pos)

        # Sort source dataset by sequence position
        sorted_dataset = sorted(source_dataset, key=get_sequence_sort_key)

        print(f"Sorted {len(sorted_dataset)} datapoints by sequence position")

        # Step 2: Split by sequence order
        train_size = int(self.train_size * len(sorted_dataset))
        selected_train_data = sorted_dataset[:train_size]
        selected_val_data = sorted_dataset[train_size:]

        print(
            f"Sequence split: {len(selected_train_data)} training, {len(selected_val_data)} validation datapoints"
        )

        # Step 3: Create merged topologies using the class method
        train_topologies = [d.top for d in selected_train_data]
        val_topologies = [d.top for d in selected_val_data]

        merged_train_topologies = self._merge_topologies_by_chain(
            train_topologies, self.check_trim, "train_sequence"
        )
        merged_val_topologies = self._merge_topologies_by_chain(
            val_topologies, self.check_trim, "val_sequence"
        )

        # Step 4: Remove overlaps between merged topologies if requested
        if remove_overlap:
            merged_val_topologies = self._remove_overlaps(
                merged_train_topologies, merged_val_topologies
            )

        # Step 5: Use filter_common_residues to create final train/val sets
        # Convert merged topologies back to sets for filtering
        train_topology_set = set(merged_train_topologies.values())
        val_topology_set = set(merged_val_topologies.values())

        # Filter entire dataset using the merged topologies
        final_train_data = filter_common_residues(
            self.dataset.data, train_topology_set, check_trim=self.check_trim
        )
        final_val_data = filter_common_residues(
            self.dataset.data, val_topology_set, check_trim=self.check_trim
        )

        # Store merged topologies for reference
        self.last_split_train_topologies_by_chain = merged_train_topologies
        self.last_split_val_topologies_by_chain = merged_val_topologies

        print(f"Final split: {len(final_train_data)} training, {len(final_val_data)} validation")

        try:
            self.validate_split(final_train_data, final_val_data)
            self._reset_retry_counter()  # Reset counter on successful split
            return final_train_data, final_val_data
        except Exception as e:
            if self.current_retry_count >= self.max_retry_depth:
                raise ValueError(
                    f"Failed to create valid split after {self.max_retry_depth} attempts. "
                    f"Last error: {e}. Consider adjusting train_size or other parameters."
                )

            print(
                f"Split is not valid - trying again (attempt {self.current_retry_count + 1}/{self.max_retry_depth}, seed {self.random_seed}): {e}"
            )
            # Increment retry count and random seed
            self.current_retry_count += 1
            self.random_seed += 1
            print(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.sequence_split(remove_overlap=remove_overlap)

    def stratified_split(
        self, remove_overlap: bool = False, n_strata: Optional[int] = 10
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Perform stratified split based on residue types or other criteria.

        Process:
        0. Optional: Remove highly redundant fragments by centrality sampling
        1. Group datapoints by the average of their respective y_true values
        2. Create strata based on the number of unique y_true values - ensure at least 2 values per stratum
        3. Randomly assign strata to train/val according to train_size
        4. Create merged topologies for train/val sets using class method
        5. Remove overlaps between merged topologies if requested
        6. Filter dataset and validate split

        Args:
            remove_overlap: If True, remove overlaps between train and val merged topologies
            n_strata: Number of strata to create (default: 10)

        Returns:
            Tuple of (training_datapoints, validation_datapoints)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required for stratified splitting. Install with: pip install numpy"
            )

        print(
            f"Starting stratified split with n_strata={n_strata}, remove_overlap={remove_overlap}"
        )

        # Step 0: Apply centrality sampling if requested
        if self.centrality:
            source_dataset = self.sample_by_centrality()
        else:
            source_dataset = self.dataset.data

        if len(source_dataset) < 2:
            raise ValueError(
                f"Source dataset is too small to split ({len(source_dataset)} datapoints)."
            )

        # Step 1: Extract y_true values and compute averages for each datapoint
        datapoint_values = []
        for i, datapoint in enumerate(source_dataset):
            # Extract y_true values - assuming datapoint has y_true attribute
            # This could be a single value or array of values
            if hasattr(datapoint, "y_true") and datapoint.y_true is not None:
                y_true = datapoint.y_true
                if isinstance(y_true, (list, np.ndarray)):
                    if len(y_true) > 0:
                        avg_value = np.mean(y_true)
                    else:
                        avg_value = 0.0
                else:
                    avg_value = float(y_true)
            elif hasattr(datapoint, "target") and datapoint.target is not None:
                # Alternative attribute name
                target = datapoint.target
                if isinstance(target, (list, np.ndarray)):
                    if len(target) > 0:
                        avg_value = np.mean(target)
                    else:
                        avg_value = 0.0
                else:
                    avg_value = float(target)
            else:
                # Fallback: use a hash of the topology as a pseudo-random stratification value
                avg_value = hash(str(datapoint.top)) % 1000

            datapoint_values.append((avg_value, i, datapoint))

        print(f"Extracted target values for {len(datapoint_values)} datapoints")

        # Step 2: Create strata based on value ranges
        # Sort by values to create meaningful strata
        datapoint_values.sort(key=lambda x: x[0])

        # Determine number of strata - ensure we have at least 2 datapoints per stratum
        min_strata_size = 2
        max_possible_strata = len(source_dataset) // min_strata_size
        if n_strata is None or n_strata > max_possible_strata:
            n_strata = max(2, max_possible_strata)

        n_strata = min(n_strata, len(source_dataset))

        print(f"Creating {n_strata} strata from {len(source_dataset)} datapoints")

        # Assign datapoints to strata based on value quantiles
        strata = [[] for _ in range(n_strata)]
        items_per_stratum = len(datapoint_values) // n_strata
        remainder = len(datapoint_values) % n_strata

        start_idx = 0
        for stratum_idx in range(n_strata):
            # Add extra item to first 'remainder' strata to distribute remainder evenly
            stratum_size = items_per_stratum + (1 if stratum_idx < remainder else 0)
            end_idx = start_idx + stratum_size

            stratum_datapoints = [dp for _, _, dp in datapoint_values[start_idx:end_idx]]
            strata[stratum_idx] = stratum_datapoints
            start_idx = end_idx

        # Verify all strata have at least 2 datapoints
        for i, stratum in enumerate(strata):
            if len(stratum) < 2:
                raise ValueError(f"Stratum {i} has only {len(stratum)} datapoints, need at least 2")

        print(f"Created strata with sizes: {[len(s) for s in strata]}")

        # Step 3: Randomly assign strata to train/val according to train_size
        n_train_strata = max(1, int(self.train_size * n_strata))
        n_val_strata = n_strata - n_train_strata

        # Ensure both train and val have at least one stratum
        if n_train_strata == 0:
            n_train_strata = 1
            n_val_strata = n_strata - 1
        elif n_val_strata == 0:
            n_val_strata = 1
            n_train_strata = n_strata - 1

        random.seed(self.random_seed)
        stratum_indices = list(range(n_strata))
        train_stratum_indices = set(random.sample(stratum_indices, n_train_strata))
        val_stratum_indices = set(stratum_indices) - train_stratum_indices

        print(
            f"Assigned {len(train_stratum_indices)} strata to training, {len(val_stratum_indices)} to validation"
        )

        # Step 4: Create train/val datasets based on stratum assignments
        selected_train_data = []
        selected_val_data = []

        for stratum_idx, stratum in enumerate(strata):
            if stratum_idx in train_stratum_indices:
                selected_train_data.extend(stratum)
            else:
                selected_val_data.extend(stratum)

        print(
            f"Stratified assignment: {len(selected_train_data)} training, {len(selected_val_data)} validation datapoints"
        )

        # Step 5: Create merged topologies using the class method
        train_topologies = [d.top for d in selected_train_data]
        val_topologies = [d.top for d in selected_val_data]

        merged_train_topologies = self._merge_topologies_by_chain(
            train_topologies, self.check_trim, "train_stratified"
        )
        merged_val_topologies = self._merge_topologies_by_chain(
            val_topologies, self.check_trim, "val_stratified"
        )

        # Step 6: Remove overlaps between merged topologies if requested
        if remove_overlap:
            merged_val_topologies = self._remove_overlaps(
                merged_train_topologies, merged_val_topologies
            )

        # Step 7: Use filter_common_residues to create final train/val sets
        # Convert merged topologies back to sets for filtering
        train_topology_set = set(merged_train_topologies.values())
        val_topology_set = set(merged_val_topologies.values())

        # Filter entire dataset using the merged topologies
        final_train_data = filter_common_residues(
            self.dataset.data, train_topology_set, check_trim=self.check_trim
        )
        final_val_data = filter_common_residues(
            self.dataset.data, val_topology_set, check_trim=self.check_trim
        )

        # Store merged topologies for reference
        self.last_split_train_topologies_by_chain = merged_train_topologies
        self.last_split_val_topologies_by_chain = merged_val_topologies

        print(f"Final split: {len(final_train_data)} training, {len(final_val_data)} validation")

        try:
            self.validate_split(final_train_data, final_val_data)
            self._reset_retry_counter()  # Reset counter on successful split
            return final_train_data, final_val_data
        except Exception as e:
            if self.current_retry_count >= self.max_retry_depth:
                raise ValueError(
                    f"Failed to create valid split after {self.max_retry_depth} attempts. "
                    f"Last error: {e}. Consider adjusting train_size, n_strata, or other parameters."
                )

            print(
                f"Split is not valid - trying again (attempt {self.current_retry_count + 1}/{self.max_retry_depth}, seed {self.random_seed}): {e}"
            )
            # Increment retry count and random seed
            self.current_retry_count += 1
            self.random_seed += 1
            print(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.stratified_split(remove_overlap=remove_overlap, n_strata=n_strata)

    def spatial_split(
        self,
        universe: mda.Universe,
        remove_overlap: bool = False,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> tuple[list[ExpD_Datapoint], list[ExpD_Datapoint]]:
        """
        Perform spatial split based on residue positions in 3D space.

        Process:
        0. Optional: Remove highly redundant fragments by centrality sampling
        1. Select a random residue as the center of the training set
        2. Compute distances between datapoints using Partial_Topology methods
        3. Fill training set in order of proximity (closest first)
        4. Create merged topologies for train/val sets using class method
        5. Remove overlaps between merged topologies if requested
        6. Filter dataset and validate split

        Args:
            universe: MDAnalysis Universe containing trajectory data for distance calculations
            remove_overlap: If True, remove overlaps between train and val merged topologies
            include_selection: MDAnalysis selection string for atoms to include in distance calculations
            exclude_selection: Optional MDAnalysis selection string for atoms to exclude
            start: Start frame for trajectory analysis
            stop: Stop frame for trajectory analysis
            step: Step size for frame iteration

        Returns:
            Tuple of (training_datapoints, validation_datapoints)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required for spatial splitting. Install with: pip install numpy"
            )

        print(
            f"Starting spatial split with train_size={self.train_size}, remove_overlap={remove_overlap}"
        )

        # Step 0: Apply centrality sampling if requested
        if self.centrality:
            source_dataset = self.sample_by_centrality()
        else:
            source_dataset = self.dataset.data

        if len(source_dataset) < 2:
            raise ValueError(
                f"Source dataset is too small to split ({len(source_dataset)} datapoints)."
            )

        # Step 1: Extract topologies from source dataset
        source_topologies = [d.top for d in source_dataset]

        print(f"Computing spatial distances for {len(source_topologies)} topologies...")

        # Step 2: Compute pairwise distances using trajectory data
        try:
            distance_matrix, distance_std = Partial_Topology.partial_topology_pairwise_distances(
                topologies=source_topologies,
                universe=universe,
                include_selection=include_selection,
                exclude_selection=exclude_selection,
                exclude_termini=True,
                termini_chain_selection=include_selection,
                renumber_residues=True,
                start=start,
                stop=stop,
                step=step,
                compound="group",
                pbc=True,
                verbose=True,
                check_trim=self.check_trim,
            )
        except Exception as e:
            raise ValueError(f"Failed to compute spatial distances: {e}")

        # Step 3: Select a random center topology
        random.seed(self.random_seed)
        center_idx = random.randint(0, len(source_topologies) - 1)
        center_topology = source_topologies[center_idx]

        print(f"Selected center topology: {center_topology}")

        # Step 4: Get distances from center to all other topologies
        center_distances = distance_matrix[center_idx, :]

        # Create (distance, index, datapoint) tuples for sorting
        distance_data = []
        for i, (dist, datapoint) in enumerate(zip(center_distances, source_dataset)):
            distance_data.append((dist, i, datapoint))

        # Sort by distance (closest first)
        distance_data.sort(key=lambda x: x[0])

        print(f"Distance range: {center_distances.min():.2f} - {center_distances.max():.2f} Å")

        # Step 5: Split by spatial proximity
        train_size = int(self.train_size * len(source_dataset))

        # Take closest topologies for training
        selected_train_data = [item[2] for item in distance_data[:train_size]]
        selected_val_data = [item[2] for item in distance_data[train_size:]]

        avg_train_dist = np.mean([item[0] for item in distance_data[:train_size]])
        avg_val_dist = np.mean([item[0] for item in distance_data[train_size:]])

        print(
            f"Spatial split: {len(selected_train_data)} training (avg dist: {avg_train_dist:.2f} Å), "
            f"{len(selected_val_data)} validation (avg dist: {avg_val_dist:.2f} Å)"
        )

        # Step 6: Create merged topologies using the class method
        train_topologies = [d.top for d in selected_train_data]
        val_topologies = [d.top for d in selected_val_data]

        merged_train_topologies = self._merge_topologies_by_chain(
            train_topologies, self.check_trim, "train_spatial"
        )
        merged_val_topologies = self._merge_topologies_by_chain(
            val_topologies, self.check_trim, "val_spatial"
        )

        # Step 7: Remove overlaps between merged topologies if requested
        if remove_overlap:
            merged_val_topologies = self._remove_overlaps(
                merged_train_topologies, merged_val_topologies
            )

        # Step 8: Use filter_common_residues to create final train/val sets
        # Convert merged topologies back to sets for filtering
        train_topology_set = set(merged_train_topologies.values())
        val_topology_set = set(merged_val_topologies.values())

        # Filter entire dataset using the merged topologies
        final_train_data = filter_common_residues(
            self.dataset.data, train_topology_set, check_trim=self.check_trim
        )
        final_val_data = filter_common_residues(
            self.dataset.data, val_topology_set, check_trim=self.check_trim
        )

        # Store merged topologies for reference
        self.last_split_train_topologies_by_chain = merged_train_topologies
        self.last_split_val_topologies_by_chain = merged_val_topologies

        print(f"Final split: {len(final_train_data)} training, {len(final_val_data)} validation")

        try:
            self.validate_split(final_train_data, final_val_data)
            self._reset_retry_counter()  # Reset counter on successful split
            return final_train_data, final_val_data
        except Exception as e:
            if self.current_retry_count >= self.max_retry_depth:
                raise ValueError(
                    f"Failed to create valid split after {self.max_retry_depth} attempts. "
                    f"Last error: {e}. Consider adjusting train_size or other parameters."
                )

            print(
                f"Split is not valid - trying again (attempt {self.current_retry_count + 1}/{self.max_retry_depth}, seed {self.random_seed}): {e}"
            )
            # Increment retry count and random seed
            self.current_retry_count += 1
            self.random_seed += 1
            print(f"Incrementing random seed to {self.random_seed}")
            random.seed(self.random_seed)
            return self.spatial_split(
                universe=universe,
                remove_overlap=remove_overlap,
                include_selection=include_selection,
                exclude_selection=exclude_selection,
                start=start,
                stop=stop,
                step=step,
            )
