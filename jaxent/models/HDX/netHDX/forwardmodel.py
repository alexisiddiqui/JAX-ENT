import MDAnalysis as mda

from jaxent.features.netHDX import build_hbond_network
from jaxent.models.common import find_common_residues
from jaxent.models.config import NetHDXConfig
from jaxent.models.HDX.netHDX.features import NetHDX_input_features
from jaxent.models.HDX.netHDX.netHDX_functions import NetHDX_ForwardPass
from jaxent.models.HDX.netHDX.parameters import NetHDX_Model_Parameters
from jaxent.types.base import ForwardModel, ForwardPass, m_key
from jaxent.types.topology import Partial_Topology


class netHDX_model(ForwardModel[NetHDX_Model_Parameters, NetHDX_input_features, NetHDXConfig]):
    """
    Network-based HDX model that uses hydrogen bond networks to predict protection factors.
    Inherits from BV_model for compatibility but overrides featurization to use H-bond networks.
    """

    def __init__(self, config: NetHDXConfig):
        super().__init__(config=config)
        self.forward: dict[m_key, ForwardPass] = {m_key("HDX_resPF"): NetHDX_ForwardPass()}

    def initialise(self, ensemble: list[mda.Universe]) -> bool:
        ########################################################################
        # TODO needs to ensure that there are backbone protons
        # also need to check whether hydrogen bonds analysis requires Hbonds - maybe could be skippable if angles are not required
        self.common_topology: set[Partial_Topology] = find_common_residues(
            ensemble, ignore_mda_selection="(resname PRO or resid 1)"
        )[0]
        # Calculate total number of frames in the ensemble
        self.n_frames = sum([u.trajectory.n_frames for u in ensemble])

        # Get residue indices for each universe
        self.common_residue_indices = [frag.residue_start for frag in self.common_topology]

        return True

    def featurise(
        self, ensemble: list[mda.Universe]
    ) -> tuple[NetHDX_input_features, list[Partial_Topology]]:
        """
        Featurize the ensemble using hydrogen bond networks,
        only including residues in the common topology.

        Args:
            ensemble: list of MDAnalysis Universe objects

        Returns:
            tuple of network features and feature topology

        Raises:
            ValueError: If ensemble validation fails
        """
        # First validate the ensemble
        if not self.initialise(ensemble):
            raise ValueError("Ensemble validation failed")

        # Calculate H-bond network features, filtering to only include common residues
        network_features = build_hbond_network(
            ensemble,
            config=self.config,
            common_residue_ids=self.common_residue_indices,  # Pass the common residue IDs
        )

        # Verify feature shape matches number of frames
        assert network_features.contact_matrices[0].shape[0] == len(self.common_residue_indices), (
            "Mismatch between number of residues and network features"
        )

        # Create feature topology from common topology
        feature_topology = list(self.common_topology)

        # Sort by residue start
        feature_topology.sort(key=lambda x: x.residue_start)

        # Update fragment indices
        for idx, frag in enumerate(feature_topology):
            frag.fragment_index = idx

        return network_features, feature_topology
        ########################################################################
