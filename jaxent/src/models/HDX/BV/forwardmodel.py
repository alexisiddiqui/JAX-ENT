from typing import Sequence

import jax.numpy as jnp
from MDAnalysis import Universe

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import ExpD_Datapoint
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.config import BV_model_Config, linear_BV_model_Config
from jaxent.src.models.func.common import get_residue_atom_pairs
from jaxent.src.models.func.contacts import calc_BV_contacts_universe
from jaxent.src.models.func.uptake import calculate_intrinsic_rates
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.models.HDX.forward import (
    BV_ForwardPass,
    BV_uptake_ForwardPass,
    linear_BV_ForwardPass,
)


class BV_model(ForwardModel[BV_Model_Parameters, BV_input_features, BV_model_Config]):
    """
    The BV or Best-Vendruscolo model for HDX-MS data.
    This computes protection factors using heavy and h bond acceptor (O) contacts.
    The list of universes must contain compatible residue sequences.
    """

    def __init__(self, config: BV_model_Config) -> None:
        super().__init__(config=config)
        self.common_k_ints: list[float]
        self.forward: dict[m_key, ForwardPass] = {
            m_key("HDX_resPF"): BV_ForwardPass(),
            m_key("HDX_peptide"): BV_uptake_ForwardPass(),
        }
        self.compatability: dict[m_key, ExpD_Datapoint]

    def initialise(
        self,
        ensemble: list[Universe],
        include_selection: str | list[str] = "protein",
        exclude_selection: str | list[str] = "resname PRO or resid 1",
    ) -> bool:
        """
        Initialize the BV model with an ensemble of structures.

        Args:
            ensemble: List of MDAnalysis Universe objects
            include_selection: MDAnalysis selection string(s) for atoms to include
            exclude_selection: MDAnalysis selection string(s) for atoms to exclude

        Returns:
            bool: True if initialization was successful
        """
        # Process selection strings - expand single elements to match ensemble length
        if isinstance(include_selection, str):
            include_selection = [include_selection]
        if isinstance(exclude_selection, str):
            exclude_selection = [exclude_selection]

        if len(include_selection) == 1 and len(ensemble) > 1:
            include_selection = include_selection * len(ensemble)
        if len(exclude_selection) == 1 and len(ensemble) > 1:
            exclude_selection = exclude_selection * len(ensemble)

        # Find common residues across the ensemble using provided selections
        self.common_topology: set[Partial_Topology] = Partial_Topology.find_common_residues(
            ensemble, include_selection=include_selection, exclude_selection=exclude_selection
        )[0]

        # Calculate total number of frames in the ensemble
        self.n_frames = sum([u.trajectory.n_frames for u in ensemble])

        # Calculate intrinsic rates for each universe and average them
        kint_values_by_topology = {}  # Maps Partial_Topology -> list of kint values from each universe

        for idx, universe in enumerate(ensemble):
            # Calculate intrinsic rates for the entire universe

            # Convert common topology to MDA residue group for mapping
            common_residue_group = Partial_Topology.to_mda_group(
                self.common_topology,
                universe,
                include_selection=include_selection[idx],
                exclude_selection=exclude_selection[idx],
                exclude_termini=True,
                renumber_residues=True,
            )
            k_ints_res_dict = calculate_intrinsic_rates(common_residue_group)

            # Map results back to Partial_Topology objects
            for topo in self.common_topology:
                # Get the corresponding residue group for this topology from this universe
                topo_residue_group = Partial_Topology.to_mda_group(
                    {topo},
                    universe,
                    include_selection=include_selection[idx],
                    exclude_selection=exclude_selection[idx],
                    exclude_termini=True,
                    renumber_residues=True,
                )

                # Look up kint values for each residue in this topology's residue group
                topo_kint_values = []
                for residue in topo_residue_group:
                    if residue in k_ints_res_dict:
                        topo_kint_values.append(k_ints_res_dict[residue])

                # Average kint values within this topology (if multiple residues)
                if topo_kint_values:
                    avg_kint_for_topo = sum(topo_kint_values) / len(topo_kint_values)

                    if topo not in kint_values_by_topology:
                        kint_values_by_topology[topo] = []
                    kint_values_by_topology[topo].append(avg_kint_for_topo)

        # Average kint values across ensembles for each topology
        self.common_k_ints_map = {}
        for topo, kint_values in kint_values_by_topology.items():
            if kint_values:  # Only include topologies that have kint values
                averaged_kint = sum(kint_values) / len(kint_values)
                self.common_k_ints_map[topo] = averaged_kint

        # Convert to list format for compatibility with existing code
        # Sort by topology to ensure consistent ordering
        sorted_topologies = sorted(
            self.common_k_ints_map.keys(), key=lambda x: (x.chain, x.residue_start)
        )
        self.common_k_ints = [self.common_k_ints_map[topo] for topo in sorted_topologies]
        self.topology_order = sorted_topologies  # Store ordering for reference

        print(f"Calculated intrinsic rates for {len(self.common_k_ints_map)} common residues")
        print(f"Averaged over {len(ensemble)} universes")

        return True

    ########################################################################
    # TODO fix typing
    def featurise(
        self, ensemble: list[Universe]
    ) -> tuple[BV_input_features, list[Partial_Topology]]:
        """
        Calculate BV model features (heavy atom contacts and H-bond acceptor contacts)
        for each residue in the ensemble.

        Args:
            ensemble: List of MDAnalysis Universe objects

        Returns:
            features across universes, per residue, per frame:

        Notes:
            - Heavy atom cutoff: 0.65 nm
            - O atom cutoff: 0.24 nm
            - Excludes residues within ±2 positions
        """
        # concatenate all the features

        heavy_contacts: list[Sequence[Sequence[float]]] = []
        acceptor_contacts: list[Sequence[Sequence[float]]] = []
        # Constants
        ########################################################################
        # TODO: fix the typing here - to do with the composition of generics in the parent class
        HEAVY_RADIUS = self.config.heavy_radius  # 0.65 nm in Angstroms
        O_RADIUS = self.config.o_radius  # 0.24 nm in Angstroms
        ########################################################################

        ########################################################################
        # TODO aw man in the future we gotta change the structure in order to be able to support heterogenous ensembles
        common_residue_group = Partial_Topology.to_mda_group(self.common_topology, ensemble[0])
        ########################################################################

        feature_topology = [
            frag for frag in self.common_topology if "PRO" not in frag.fragment_sequence
        ]
        # skip the first residue
        feature_topology = [frag for frag in feature_topology if frag.residue_start != 1]

        # sort the features by residue start
        feature_topology.sort(key=lambda x: x.residue_start)
        # reset the fragment indices
        for idx, frag in enumerate(feature_topology):
            frag.fragment_index = idx

        for universe in ensemble:
            # Get residue indices and atom indices for amide N and H atoms
            NH_residue_atom_pairs = get_residue_atom_pairs(common_residue_group, "N")
            HN_residue_atom_pairs = get_residue_atom_pairs(common_residue_group, "H | HN")

            # Calculate heavy atom contacts
            _heavy_contacts = calc_BV_contacts_universe(
                universe=universe,
                residue_atom_index=NH_residue_atom_pairs,
                contact_selection="heavy",
                radius=HEAVY_RADIUS,
            )

            # Calculate O atom contacts (H-bond acceptors)
            _o_contacts = calc_BV_contacts_universe(
                universe=universe,
                residue_atom_index=HN_residue_atom_pairs,
                contact_selection="oxygen",
                radius=O_RADIUS,
            )

            heavy_contacts.append(_heavy_contacts)
            acceptor_contacts.append(_o_contacts)

        assert len(feature_topology) > 1, "Feature topology is None"

        _heavy_contacts = [jnp.asarray(contact) for contact in heavy_contacts]
        _acceptor_contacts = [jnp.asarray(contact) for contact in acceptor_contacts]

        # Ensure all contact arrays are 2D for concatenation
        _heavy_contacts = [c.reshape(-1, 1) if c.ndim == 1 else c for c in _heavy_contacts]
        _acceptor_contacts = [c.reshape(-1, 1) if c.ndim == 1 else c for c in _acceptor_contacts]

        # stack the contacts along a new axis
        heavy_contacts_array = jnp.stack(_heavy_contacts, axis=1)
        acceptor_contacts_array = jnp.stack(_acceptor_contacts, axis=1)

        return BV_input_features(
            heavy_contacts=heavy_contacts_array,
            acceptor_contacts=acceptor_contacts_array,
            k_ints=jnp.array(self.common_k_ints),
        ), feature_topology

    # TODO this function needs to output some kind of topology information so that these can be aligned with the experimental data - not sure how best to do this
    # options are to ouput a seperate object or to tie this into input features

    ########################################################################

    # def forward(self) -> list[Output_Features]:
    #     """
    #     This function applies the forward models to the input features
    #     need to find a way to do this efficiently in jax
    #     """
    #     # first averages the input parameters using the frame weights
    #     average_features = map(
    #         frame_average_features,
    #         self.input_features,
    #         [self.params.frame_weights] * len(self.input_features),
    #     )
    #     # map the single_pass function
    #     output_features = map(
    #         single_pass, self.forwardpass, average_features, self.params.model_parameters
    #     )
    #     # update this to use externally defined optimisers - perhaps update should just update the parameters
    #     # change argnum to use enums

    #     return list(output_features)

    ########################################################################


class linear_BV_model(BV_model):
    """
    Linear BV model that uses a linear combination of bc and bh to predict protection factors.
    Inherits from BV_model for compatibility but overrides featurization to use H-bond networks.
    """

    def __init__(self, config: linear_BV_model_Config):
        super().__init__(config=config)
        self.forward: dict[m_key, ForwardPass] = {
            m_key("HDX_resPF"): linear_BV_ForwardPass(),
        }
        self.compatability: dict[m_key, ExpD_Datapoint]
