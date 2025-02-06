from dataclasses import dataclass
from typing import ClassVar, List, Optional, Sequence

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node
from MDAnalysis import Universe

from jaxent.config.base import BaseConfig
from jaxent.datatypes import HDX_peptide, HDX_protection_factor
from jaxent.forwardmodels.base import (
    ForwardModel,
    ForwardPass,
    Input_Features,
    Model_Parameters,
)
from jaxent.forwardmodels.functions import (
    calc_BV_contacts_universe,
    calculate_intrinsic_rates,
    find_common_residues,
    get_residue_atom_pairs,
)
from jaxent.forwardmodels.netHDX_functions import (
    NetHDX_ForwardPass,
    NetHDXConfig,
    build_hbond_network,
)

# import register_pytree_node


@dataclass(frozen=True)
class BV_input_features:
    __slots__ = ["heavy_contacts", "acceptor_contacts"]

    heavy_contacts: Sequence[Sequence[float]]  # (frames, residues)
    acceptor_contacts: Sequence[Sequence[float]]  # (frames, residues)

    ########################################################################
    # update the features shape to have a fixed/more consistent structure
    @property
    def features_shape(self) -> tuple[int, ...]:
        length = len(self.heavy_contacts[0])
        heavy_shape = len(self.heavy_contacts)
        acceptor_shape = len(self.acceptor_contacts)
        return (heavy_shape, acceptor_shape, length)


########################################################################
# fix the typing to use numpy arrays
@dataclass(frozen=True)
class BV_output_features:
    __slots__ = ["log_Pf", "k_ints"]

    log_Pf: list | Sequence[float] | Array  # (1, residues)]
    k_ints: Optional[list] | Optional[Array]

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.log_Pf))


########################################################################


@dataclass(frozen=True, slots=True)
class BV_Model_Parameters(Model_Parameters):
    bv_bc: float = 0.35
    bv_bh: float = 2.0

    temperature: float = 300
    static_params: ClassVar[set[str]] = {"temperature"}

    def __mul__(self, scalar: float) -> "BV_Model_Parameters":
        return BV_Model_Parameters(
            bv_bc=self.bv_bc * scalar,
            bv_bh=self.bv_bh * scalar,
            temperature=self.temperature,
        )

    __rmul__ = __mul__

    def __sub__(self, other: "BV_Model_Parameters") -> "BV_Model_Parameters":
        return BV_Model_Parameters(
            bv_bc=self.bv_bc - other.bv_bc,
            bv_bh=self.bv_bh - other.bv_bh,
            temperature=self.temperature,
        )

    def update_parameters(self, new_params: "BV_Model_Parameters") -> "BV_Model_Parameters":
        """
        Creates a new instance with updated parameters, preserving static parameters.

        Args:
            new_params: Tuple of new parameter values in the order bv_bc, bv_bh
        Returns:
            A new BV_Model_Parameters instance with updated non-static parameters
        """

        # Update non-static parameters
        bv_bc, bv_bh = new_params.bv_bc, new_params.bv_bh

        return BV_Model_Parameters(bv_bc=bv_bc, bv_bh=bv_bh, temperature=self.temperature)


register_pytree_node(
    BV_Model_Parameters, BV_Model_Parameters.tree_flatten, BV_Model_Parameters.tree_unflatten
)


@dataclass(frozen=True)
class BV_model_Config(BaseConfig):
    # __slots__ = (
    #     "temperature",
    #     "bv_bc",
    #     "bv_bh",
    #     "ph",
    #     "heavy_radius",
    #     "o_radius",
    #     "forward_parameters",
    # )

    temperature: float = 300
    bv_bc: float = 0.35
    bv_bh: float = 2.0
    ph: float = 7
    heavy_radius: float = 6.5
    o_radius: float = 2.4

    @property
    def forward_parameters(self) -> Model_Parameters:
        return BV_Model_Parameters(bv_bc=self.bv_bc, bv_bh=self.bv_bh, temperature=self.temperature)


class NetHDX_Model_Parameters(Model_Parameters):
    shell_energy_scaling: float = 0.84  # Energy scaling factor for each shell contact (-0.5 kcal/mol per shell (-2.1 kj/mol)), using R=8.31/1000 and T=300K


########################################################################
# fix the typing to use jax arrays
class BV_ForwardPass(ForwardPass[BV_input_features, BV_output_features, BV_Model_Parameters]):
    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> BV_output_features:
        bc, bh = parameters.bv_bc, parameters.bv_bh

        # Convert lists to numpy arrays for computation
        heavy_contacts = jnp.array(input_features.heavy_contacts)
        acceptor_contacts = jnp.array(input_features.acceptor_contacts)

        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)

        # Convert back to list for output
        log_pf_list = log_pf

        return BV_output_features(log_Pf=log_pf_list, k_ints=None)


########################################################################


class BV_model(ForwardModel[BV_Model_Parameters]):
    """
    The BV or Best-Vendruscolo model for HDX-MS data.
    This computes protection factors using heavy and h bond acceptor (O) contacts.
    The list of universes must contain compatible residue sequences.
    """

    def __init__(self, config: BV_model_Config) -> None:
        super().__init__(config=config)
        self.common_k_ints: list[float]
        self.forward: ForwardPass = BV_ForwardPass()
        self.compatability: HDX_peptide | HDX_protection_factor

    def initialise(self, ensemble: list[Universe]) -> bool:
        """
        Initialize the BV model with an ensemble of structures.

        Args:
            ensemble: List of MDAnalysis Universe objects

        Returns:
            bool: True if initialization was successful
        """
        # Find common residues across the ensemble
        self.common_residues, _ = find_common_residues(ensemble)

        # Calculate total number of frames in the ensemble
        self.n_frames = sum([u.trajectory.n_frames for u in ensemble])

        # Get residue indices for each universe
        common_residue_indices = [frag.residue_start for frag in self.common_residues]

        # Calculate intrinsic rates for the common residues

        kint_list = []
        for idx, u in enumerate(ensemble):
            k_ints_res_idx_dict = calculate_intrinsic_rates(u)

            # filter by keys in common_residue_indices
            k_ints_res_idx_tuple = [
                k_ints_res_idx_dict[res_idx] for res_idx in common_residue_indices
            ]

            kint_list.append(k_ints_res_idx_tuple)

        # assert that the lists in kint_list are the same length
        assert all(len(kint_list[0]) == len(kint_list[i]) for i in range(1, len(kint_list)))

        self.common_k_ints = kint_list[0]

        return True

    ########################################################################
    # TODO fix typing
    def featurise(self, ensemble: list[Universe]) -> Input_Features:
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

        heavy_contacts: list[Sequence[float]] = []
        acceptor_contacts: list[Sequence[float]] = []
        # Constants
        HEAVY_RADIUS = self.config.heavy_radius  # 0.65 nm in Angstroms
        O_RADIUS = self.config.o_radius  # 0.24 nm in Angstroms

        common_residues = {
            (frag.fragment_sequence, frag.residue_start) for frag in self.common_residues
        }

        for universe in ensemble:
            # Get residue indices and atom indices for amide N and H atoms
            NH_residue_atom_pairs = get_residue_atom_pairs(universe, common_residues, "N")
            HN_residue_atom_pairs = get_residue_atom_pairs(universe, common_residues, "H")

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

            heavy_contacts.extend(_heavy_contacts)
            acceptor_contacts.extend(_o_contacts)

        return BV_input_features(heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts)

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


class netHDX_model(ForwardModel[NetHDX_Model_Parameters]):
    """
    Network-based HDX model that uses hydrogen bond networks to predict protection factors.
    Inherits from BV_model for compatibility but overrides featurization to use H-bond networks.
    """

    def __init__(self, config: NetHDXConfig):
        super().__init__(config=config)
        self.forward: ForwardPass = NetHDX_ForwardPass()

    def initialise(self, ensemble: List[Universe]) -> bool:
        ########################################################################
        # TODO needs to ensure that there are backbone protons
        # also need to check whether hydrogen bonds analysis requires Hbonds - maybe could be skippable if angles are not required

        return True

    def featurise(self, ensemble: List[Universe]) -> Input_Features:
        """
        Featurize the ensemble using hydrogen bond networks.

        Args:
            ensemble: List of MDAnalysis Universe objects

        Returns:
            List of contact matrices for each frame

        Raises:
            ValueError: If ensemble validation fails
        """
        # First validate the ensemble
        if not self.initialise(ensemble):
            raise ValueError("Ensemble validation failed")
        ########################################################################

        # Calculate H-bond network features using functional approach?
        # TODO: fix typing for jax
        network_features = build_hbond_network(ensemble, self.config)

        return network_features
        ########################################################################
