# this file is for temporary storage of split methods
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
