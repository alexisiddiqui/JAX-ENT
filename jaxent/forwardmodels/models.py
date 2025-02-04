from dataclasses import dataclass
from typing import List, Optional

from MDAnalysis import Universe

from jaxent.config.base import BaseConfig
from jaxent.datatypes import HDX_peptide, HDX_protection_factor
from jaxent.forwardmodels.base import ForwardModel, ForwardPass, Input_Features, Output_Features
from jaxent.forwardmodels.functions import (
    calc_BV_contacts_universe,
    calculate_intrinsic_rates,
    find_common_residues,
    get_residue_atom_pairs,
    get_residue_indices,
)
from jaxent.forwardmodels.netHDX_functions import (
    NetHDX_ForwardPass,
    NetHDXConfig,
    build_hbond_network,
)


@dataclass(frozen=True)
class BV_model_Config(BaseConfig):
    temperature: float = 300
    bv_bc: float = 0.35
    bv_bh: float = 2.0
    ph: float = 7
    heavy_radius: float = 6.5
    o_radius: float = 2.4

    @property
    def forward_parameters(self) -> tuple[float, float]:
        return (self.bv_bc, self.bv_bh)


@dataclass(frozen=True)
class BV_input_features:
    heavy_contacts: list  # (frames, residues)
    acceptor_contacts: list  # (frames, residues)

    @property
    def features_shape(self) -> tuple[int, ...]:
        heavy_shape = len(self.heavy_contacts)
        acceptor_shape = len(self.heavy_contacts)
        return (heavy_shape, acceptor_shape)


@dataclass(frozen=True)
class BV_output_features:
    log_Pf: list  # (1, residues)
    k_ints: Optional[list]

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.log_Pf))


class BV_ForwardPass(ForwardPass):
    def __call__(
        self, avg_input_features: BV_input_features, parameters: BV_model_Config
    ) -> Output_Features:
        bc, bh = parameters.forward_parameters

        log_pf = bc * avg_input_features.heavy_contacts + bh * avg_input_features.acceptor_contacts

        return BV_output_features(
            log_Pf=log_pf,
            k_ints=None,  # Optional, can be set later if needed
        )


# T_In = BV_input_features  # set proper alias if applicable


class BV_model(ForwardModel):
    """
    The BV or Best-Vendruscolo model for HDX-MS data.
    This computes protection factors using heavy and h bond acceptor (O) contacts.
    The list of universes must contain compatible residue sequences.
    """

    def __init__(self, config: BV_model_Config) -> None:
        super().__init__()
        self.config = config
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
        common_residue_indices = [get_residue_indices(u, self.common_residues) for u in ensemble]

        # Calculate intrinsic rates for the common residues

        ########################################################################
        # this definitely requires fixing to handle multiple universes
        k_ints_res_idx = []
        for idx, u in enumerate(ensemble):
            k_ints_res_idx_dict = calculate_intrinsic_rates(u)

            self.common_k_ints = list(k_ints_res_idx_dict.values())
            break

        return True

        ########################################################################

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

        heavy_contacts = []
        acceptor_contacts = []
        # Constants
        HEAVY_RADIUS = self.config.heavy_radius  # 0.65 nm in Angstroms
        O_RADIUS = self.config.o_radius  # 0.24 nm in Angstroms

        for universe in ensemble:
            # Get residue indices and atom indices for amide N and H atoms
            NH_residue_atom_pairs = get_residue_atom_pairs(universe, self.common_residues, "N")
            HN_residue_atom_pairs = get_residue_atom_pairs(universe, self.common_residues, "H")

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

        return BV_input_features(heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts)


class netHDX_model(BV_model):
    """
    Network-based HDX model that uses hydrogen bond networks to predict protection factors.
    Inherits from BV_model for compatibility but overrides featurization to use H-bond networks.
    """

    def __init__(self, config: Optional[NetHDXConfig] = None):
        super().__init__()
        self.config = config if config is not None else NetHDXConfig()
        self.forward: ForwardPass = NetHDX_ForwardPass()

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

        # Calculate H-bond network features using functional approach
        network_features = build_hbond_network(ensemble, self.config)

        return network_features
