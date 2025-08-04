from typing import Sequence, cast

import jax.numpy as jnp
import MDAnalysis as mda

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import ExpD_Datapoint
from jaxent.src.interfaces.topology import Partial_Topology, mda_TopologyAdapter, rank_and_index
from jaxent.src.models.config import BV_model_Config, linear_BV_model_Config
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

    base_include_selection: str = "protein"
    base_exclude_selection: str = "resname PRO"
    exclude_termini: bool = True

    def __init__(self, config: BV_model_Config) -> None:
        super().__init__(config=config)
        self.common_k_ints: list[float]
        self.forward: dict[m_key, ForwardPass] = {
            m_key("HDX_resPF"): BV_ForwardPass(),
            m_key("HDX_peptide"): BV_uptake_ForwardPass(),
        }
        self.compatability: dict[m_key, ExpD_Datapoint]

    def _combine_selections(self, base_selection: str, additional_selection: str | None) -> str:
        """Combine base selection with additional selection using 'and' logic."""
        if additional_selection is None:
            return base_selection
        return f"({base_selection}) and ({additional_selection})"

    def _prepare_selection_lists(
        self,
        ensemble: list[mda.Universe],
        include_selection: str | list[str] | None | list[None],
        exclude_selection: str | list[str] | None | list[None],
    ) -> tuple[list[str], list[str]]:
        """Prepare and validate selection lists for the ensemble."""
        # Handle None cases
        if include_selection is None:
            include_selection = []
        elif isinstance(include_selection, str):
            include_selection = [include_selection]

        if exclude_selection is None:
            exclude_selection = []
        elif isinstance(exclude_selection, str):
            exclude_selection = [exclude_selection]

        # Expand single elements to match ensemble length
        if len(include_selection) == 1 and len(ensemble) > 1:
            include_selection = include_selection * len(ensemble)
        elif len(include_selection) == 0:
            include_selection = [None] * len(ensemble)

        if len(exclude_selection) == 1 and len(ensemble) > 1:
            exclude_selection = exclude_selection * len(ensemble)
        elif len(exclude_selection) == 0:
            exclude_selection = [None] * len(ensemble)

        # Combine with base selections
        final_include_selection = []
        final_exclude_selection = []

        for i in range(len(ensemble)):
            # Combine include selections
            combined_include = self._combine_selections(
                self.base_include_selection,
                include_selection[i] if i < len(include_selection) else None,
            )
            final_include_selection.append(combined_include)

            # Combine exclude selections
            base_exclude = self.base_exclude_selection
            additional_exclude = exclude_selection[i] if i < len(exclude_selection) else None

            if additional_exclude is None:
                combined_exclude = base_exclude
            else:
                combined_exclude = f"({base_exclude}) or ({additional_exclude})"
            final_exclude_selection.append(combined_exclude)

        return final_include_selection, final_exclude_selection

    def initialise(
        self,
        ensemble: list[mda.Universe],
        include_selection: str | list[str] | None = None,
        exclude_selection: str | list[str] | None = None,
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
        # Process and combine selections with base selections
        self.final_include_selection, self.final_exclude_selection = self._prepare_selection_lists(
            ensemble, include_selection, exclude_selection
        )

        # Find common residues across the ensemble using combined selections
        self.common_topology: set[Partial_Topology] = mda_TopologyAdapter.find_common_residues(
            ensemble,
            include_selection=self.final_include_selection,
            exclude_selection=self.final_exclude_selection,
            renumber_residues=True,
        )[0]

        # Calculate total number of frames in the ensemble
        self.n_frames = sum([u.trajectory.n_frames for u in ensemble])

        # Calculate intrinsic rates for each universe and average them
        kint_values_by_topology = {}  # Maps Partial_Topology -> list of kint values from each universe

        for idx, universe in enumerate(ensemble):
            # Calculate intrinsic rates for the entire universe

            # Convert common topology to MDA residue group for mapping
            common_residue_group = mda_TopologyAdapter.to_mda_group(
                self.common_topology,
                universe,
                include_selection=self.final_include_selection[idx],
                exclude_selection=self.final_exclude_selection[idx],
                exclude_termini=self.exclude_termini,
                renumber_residues=True,
            )
            common_residue_group = cast(mda.ResidueGroup, common_residue_group)
            k_ints_res_dict = calculate_intrinsic_rates(universe, common_residue_group)

            # Map results back to Partial_Topology objects
            for topo in self.common_topology:
                # Get the corresponding residue group for this topology from this universe
                topo_residue_group = mda_TopologyAdapter.to_mda_group(
                    {topo},
                    universe,
                    include_selection=self.final_include_selection[idx],
                    exclude_selection=self.final_exclude_selection[idx],
                    exclude_termini=self.exclude_termini,
                    renumber_residues=True,
                )

                # Each topology should represent a single residue
                # Look up kint value for this single residue
                if len(topo_residue_group) == 1:
                    residue = topo_residue_group[0]
                    if residue in k_ints_res_dict:
                        kint_value = k_ints_res_dict[residue]

                        if topo not in kint_values_by_topology:
                            kint_values_by_topology[topo] = []
                        kint_values_by_topology[topo].append(kint_value)
                else:
                    print(
                        f"Warning: Topology {topo} contains {len(topo_residue_group)} residues, expected 1"
                    )

        # Average kint values across ensembles for each topology
        self.common_k_ints_map = {}
        for topo, kint_values in kint_values_by_topology.items():
            if kint_values:  # Only include topologies that have kint values
                averaged_kint = sum(kint_values) / len(kint_values)
                self.common_k_ints_map[topo] = averaged_kint

        # Convert to list format for compatibility with existing code
        # Sort by topology to ensure consistent ordering
        sorted_topologies = rank_and_index(list(self.common_k_ints_map.keys()), check_trim=False)
        self.common_k_ints = [self.common_k_ints_map[topo] for topo in sorted_topologies]
        self.topology_order = sorted_topologies  # Store ordering for reference

        print(f"Calculated intrinsic rates for {len(self.common_k_ints_map)} common residues")
        print(f"Averaged over {len(ensemble)} universes")

        return True

    ########################################################################
    # TODO fix typing

    def featurise(
        self, ensemble: list[mda.Universe]
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
        heavy_contacts: list[Sequence[Sequence[float]]] = []
        acceptor_contacts: list[Sequence[Sequence[float]]] = []

        # Constants from config
        HEAVY_RADIUS = self.config.heavy_radius  # 0.65 nm in Angstroms
        O_RADIUS = self.config.o_radius  # 0.24 nm in Angstroms

        for idx, universe in enumerate(ensemble):
            # Get N atoms (for heavy atom contacts)
            n_atoms = mda_TopologyAdapter.to_mda_group(
                topologies=self.common_topology,
                universe=universe,
                include_selection=self.final_include_selection[idx],
                exclude_selection=self.final_exclude_selection[idx],
                exclude_termini=self.exclude_termini,
                termini_chain_selection="protein",
                renumber_residues=True,
                mda_atom_filtering="name N",
            )
            n_atoms = cast(mda.AtomGroup, n_atoms)

            # Get H atoms (for H-bond acceptor contacts)
            h_atoms = mda_TopologyAdapter.to_mda_group(
                topologies=self.common_topology,
                universe=universe,
                include_selection=self.final_include_selection[idx],
                exclude_selection=self.final_exclude_selection[idx],
                exclude_termini=self.exclude_termini,
                termini_chain_selection="protein",
                renumber_residues=True,
                mda_atom_filtering="name H or name HN",
            )
            h_atoms = cast(mda.AtomGroup, h_atoms)

            # Calculate heavy atom contacts using N atoms
            _heavy_contacts = calc_BV_contacts_universe(
                universe=universe,
                target_atoms=n_atoms,
                contact_selection="heavy",
                radius=HEAVY_RADIUS,
            )

            # Calculate O atom contacts (H-bond acceptors) using H atoms
            _o_contacts = calc_BV_contacts_universe(
                universe=universe,
                target_atoms=h_atoms,
                contact_selection="oxygen",
                radius=O_RADIUS,
            )

            # --- Ensure ordering matches self.topology_order ---
            # Get reordering indices for this universe
            reorder_indices = mda_TopologyAdapter.get_residuegroup_ranking_indices(n_atoms)
            # Reorder contacts accordingly
            _heavy_contacts = [_heavy_contacts[i] for i in reorder_indices]
            _o_contacts = [_o_contacts[i] for i in reorder_indices]
            # ---------------------------------------------------

            heavy_contacts.append(_heavy_contacts)
            acceptor_contacts.append(_o_contacts)

        assert len(self.topology_order) > 0, "Feature topology is empty"

        # Convert to JAX arrays and ensure proper 2D shape (residues, frames)
        _heavy_contacts = [jnp.asarray(contact) for contact in heavy_contacts]
        _acceptor_contacts = [jnp.asarray(contact) for contact in acceptor_contacts]

        # Ensure all contact arrays are 2D (residues, frames)
        _heavy_contacts = [c.reshape(c.shape[0], -1) if c.ndim > 2 else c for c in _heavy_contacts]
        _acceptor_contacts = [
            c.reshape(c.shape[0], -1) if c.ndim > 2 else c for c in _acceptor_contacts
        ]

        # Concatenate along the frames axis (axis=1) to combine frames from all universes
        # This flattens all frames from all universes into a single frames dimension
        heavy_contacts_array = jnp.concatenate(_heavy_contacts, axis=1)
        acceptor_contacts_array = jnp.concatenate(_acceptor_contacts, axis=1)

        return BV_input_features(
            heavy_contacts=heavy_contacts_array,
            acceptor_contacts=acceptor_contacts_array,
            k_ints=jnp.array(self.common_k_ints),
        ), self.topology_order

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
