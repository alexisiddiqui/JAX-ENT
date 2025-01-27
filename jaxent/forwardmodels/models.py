from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from MDAnalysis import Universe

from jaxent.config.base import BaseConfig
from jaxent.datatypes import HDX_peptide, HDX_protection_factor
from jaxent.forwardmodels.base import (
    ForwardModel,
    ForwardPass,
    Input_Features,
    Output_Features,
)
from jaxent.forwardmodels.functions import (
    calc_BV_contacts_universe,
    calculate_intrinsic_rates,
    find_common_residues,
    get_residue_atom_pairs,
    get_rssidue_indices,
)
from jaxent.forwardmodels.netHDX_functions import HBondNetwork


@dataclass(frozen=True)
class BV_model_Config(BaseConfig):
    """Configuration class for the Best-Vendruscolo (BV) model.
    
    Attributes:
        temperature: float = 300
            Temperature in Kelvin used for calculations
        bv_bc: float = 0.35
            Coefficient for heavy atom contacts contribution
        bv_bh: float = 2.0
            Coefficient for hydrogen bond acceptor contacts contribution 
        ph: float = 7
            pH value for intrinsic rate calculations
        heavy_radius: float = 6.5
            Radius in Angstroms for heavy atom contact detection
        o_radius: float = 2.4
            Radius in Angstroms for oxygen atom (H-bond acceptor) contact detection
    
    Properties:
        forward_parameters: tuple[float, float]
            Returns the model coefficients (bv_bc, bv_bh)
    """
    
    temperature: float = 300
    bv_bc: float = 0.35
    bv_bh: float = 2.0
    ph: float = 7
    heavy_radius: float = 6.5
    o_radius: float = 2.4

    @property
    def forward_parameters(self) -> tuple[float, float]:
        """Get the model coefficients for forward pass calculations"""
        return (self.bv_bc, self.bv_bh)


@dataclass(frozen=True)
class BV_input_features(Input_Features):
    """Input features for the Best-Vendruscolo (BV) model.
    
    Attributes:
        heavy_contacts: list
            Heavy atom contact data across frames and residues
        acceptor_contacts: list
            Hydrogen bond acceptor contact data across frames and residues
    
    Properties:
        features_shape: tuple[int, ...]
            Returns the shape of the input features (frames, residues)
    """
    
    heavy_contacts: list   # (frames, residues)
    acceptor_contacts: list   # (frames, residues)

    @property
    def features_shape(self) -> tuple[int, ...]:
        """Get the shape of the input features"""
        heavy_shape = len(self.heavy_contacts)
        acceptor_shape = len(self.heavy_contacts)
        return (heavy_shape, acceptor_shape)


@dataclass(frozen=True)
class BV_output_features(Output_Features):
    """Output features from the Best-Vendruscolo (BV) model.
    
    Attributes:
        log_Pf: list
            Log protection factors for each residue
        k_ints: Optional[list]
            Intrinsic rates for each residue (optional)
    
    Properties:
        output_shape: tuple[int, ...]
            Returns the shape of the output features (1, residues)
    """
    
    log_Pf: list   # (1, residues)
    k_ints: Optional[list]

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Get the shape of the model output"""
        return (1, len(self.log_Pf))


class BV_ForwardPass(ForwardPass):
    """Forward pass implementation for the Best-Vendruscolo (BV) model.
    
    This class performs the forward calculation using input features and model parameters
    to produce protection factors based on heavy atom and hydrogen bond acceptor contacts.
    """
    
    def __call__(
        self, 
        avg_input_features: BV_input_features,
        parameters: BV_model_Config
    ) -> BV_output_features:
        """Perform the forward pass calculation
        
        Args:
            avg_input_features: BV_input_features
                Input features containing heavy and acceptor contact data
            parameters: BV_model_Config
                Model configuration parameters
                
        Returns:
            BV_output_features containing calculated log protection factors
            
        Notes:
            The calculation uses the formula:
                log(Pf) = bc * heavy_contacts + bh * acceptor_contacts
            where bc and bh are model coefficients from the configuration
        """
        bc, bh = parameters.forward_parameters

        # Calculate log protection factors using linear combination
        log_pf = bc * avg_input_features.heavy_contacts + bh * avg_input_features.acceptor_contacts

        return BV_output_features(
            log_Pf=log_pf,
            k_ints=None   # Optional, can be set later if needed
        )


class BV_model(ForwardModel):
    """The Best-Vendruscolo (BV) model for HDX-MS data analysis.
    
    This model computes protection factors using heavy atom and hydrogen bond acceptor contacts.
    It implements the methodology described in Best & Vendruscolo (2018).
    
    Attributes:
        common_residues: set[tuple[str, int]]
            Set of common residues across all structures in the ensemble
        common_k_ints: list[float]
            Mean intrinsic rates for each residue
        forward: ForwardPass
            Forward pass implementation for model calculations
        compatability: HDX_ peptide | HDX_protection_factor
            Model compatibility with input data types
    
    Methods:
        initialise(ensemble): Initializes the model with an ensemble of structures
        featurise(ensemble): Calculates input features from the ensemble
    """
    
    def __init__(self) -> None:
        """Initialize the Best-Vendruscolo (BV) model"""
        super().__init__()
        self.common_residues: set[tuple[str, int]] = set()
        self.common_k_ints: list[float] = []
        self.forward: ForwardPass = BV_ForwardPass()
        self.compatability: HDX_ peptide | HDX_protection_factor = HDX_protection_factor

    def initialise(self, ensemble: list[Universe]) -> bool:
        """Initialize the model with an ensemble of structures
        
        Args:
            ensemble: List of MDAnalysis Universe objects
            
        Returns:
            bool indicating success of initialization
            
        Notes:
            This method performs several key steps:
                1. Identifies common residues across all structures
                2. Calculates total number of frames in the ensemble
                3. Determines residue indices for each universe
                4. Computes mean intrinsic rates for each residue
                
            The model is compatible with ensembles containing multiple 
            trajectories or structures as long as they share a common sequence
        """
        # Find common residues across the ensemble
        self.common_residues, _ = find_common_residues(ensemble)

        # Calculate total number of frames in the ensemble
        self.n_frames = sum([u.trajectory.n_frames for u in ensemble])

        # Get residue indices for each universe
        common_residue_indices = [
            get_ressidue_indices(u, self.common_residues) 
            for u in ensemble
        ]

        # Calculate intrinsic rates for the common residues
        k_ints = []
        for u in ensemble:
            k_ints.append(calculate_intrinsic_rates(u))

        self.common_k_ints = np.mean(k_ints, axis=0)[common_residue_indices]

        return True

    def featurise(self, ensemble: list[Universe]) -> list[list[list[float]]]:
        """Calculate BV model features from the ensemble
        
        Args:
            ensemble: List of MDAnalysis Universe objects
            
        Returns:
            List of features per universe, residue and frame in format:
                [[[heavy_contacts, o_contacts], ...], ...]
                
        Notes:
            - Heavy atom cutoff: 0.65 nm (6.5 Angstroms)
            - O atom cutoff: 0.24 nm (2.4 Angstroms)
            - Excludes residues within ±2 positions
            
            This method calculates two types of contacts:
                1. Heavy atom contacts using amide nitrogen atoms
                2. Hydrogen bond acceptor contacts using amide hydrogen atoms
        """
        features = []

        # Constants for contact detection
        HEAVY_RADIUS = 0.65   # nm in Angstroms
        O_RADIUS = 2.4   # nm in Angstroms

        for universe in ensemble:
            universe_features = []

            # Get residue indices and atom indices for amide N and H atoms
            NH_ressidue_atom_pairs = get_ressidue_atom_pairs(
                universe, self.common_residues, "N"
            )
            HN_ressidue_atom_pairs = get_ressidue_atom_pairs(
                universe, self.common_residues, "H"
            )

            # Calculate heavy atom contacts
            heavy_contacts = calc_BV_contacts_universe(
                universe=universe,
                residue_atom_index=NH_ressidue_atom_pairs,
                contact_selection="heavy",
                radius=HEAVY_RADIUS,
            )

            # Calculate O atom contacts (H-bond acceptors)
            o_contacts = calc_BV_contacts_universe(
                universe=universe,
                residue_atom_index=HN_ressidue_atom_pairs,
                contact_selection="oxygen",
                radius=O_RADIUS,
            )

            # Combine features for each residue
            for h_contacts, o_contacts in zip(heavy_contacts, o_contacts):
                universe_features.append([float(h_contacts), float(o_contacts)])

            features.append(universe_features)

        return features


class netHDX_model(BV_model):
    """Hydrogen bond network model extending the Best-Vendruscolo (BV) framework.
    
    This model uses hydrogen bond networks to calculate protection factors,
    building on the original BV model by incorporating additional network features.
    """
    
    def __init__(self):
        """Initialize the netHDX model"""
        super().__init__()
        self.hbond_network = HBondNetwork()

    def featurise(self, ensemble: List[Universe]) -> List[List[List[float]]]:
        """Featurize the ensemble using hydrogen bond networks
        
        Args:
            ensemble: List of MDAnalysis Universe objects
            
        Returns:
            List of contact matrices for each frame in format:
                [[[contact_1, contact_2, ...], ...], ...]
                
        Notes:
            This method first validates and initializes the model with the ensemble,
            then calculates hydrogen bond network features.
            
            The output is a list of contact matrices where each matrix represents
            H-bond contacts between residues for a given frame.
        """
        # First validate the ensemble
        if not self.initialise(ensemble):
            raise ValueError("Ensemble validation failed")

        # Calculate H-bond network features
        network_features = self.hbond_network.build_network(ensemble)

        # Convert to required output format
        return [matrix.tolist() for matrix in network_features.contact_matrices]
