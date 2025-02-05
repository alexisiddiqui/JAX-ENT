from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array  # This is the correct import path
from jax.tree_util import register_pytree_node
from MDAnalysis import Universe

from jaxent.forwardmodels.base import (
    ForwardModel,
    ForwardPass,
    Input_Features,
    Model_Parameters,
    Output_Features,
)
from jaxent.utils.jax import frame_average_features, single_pass


@dataclass()
class Topology_Fragment:
    """
    Dataclass that holds the information of a single topology fragment.
    This is usually a residue but may also be a group of residues such as peptides.
    """

    chain: str
    fragment_sequence: str  # resname (residue) or single letter codes (peptide) or atom name (atom)
    residue_start: int  # inclusive
    residue_end: int | None = None  # inclusive - if None, then this is a single residue
    fragment_index: Optional[int] = (
        None  # atom or peptide index - optional for residues - required for peptides and atoms
    )

    def __post_init__(self):
        if self.residue_end is None:
            self.residue_end = self.residue_start
        self.length = self.residue_end - self.residue_start + 1

    def __eq__(self, other) -> bool:
        """Equal comparison - checks if all attributes are the same"""
        if not isinstance(other, Topology_Fragment):
            return NotImplemented
        return (
            self.chain == other.chain
            and self.fragment_index == other.fragment_index
            and self.fragment_sequence == other.fragment_sequence
            and self.residue_start == other.residue_start
            and self.residue_end == other.residue_end
        )

    def __hash__(self) -> int:
        """Hash function - uses the hash of chain, index, sequence, residue start and end"""
        return hash(
            (
                self.chain,
                self.fragment_index,
                self.fragment_sequence,
                self.residue_start,
                self.residue_end,
            )
        )


@dataclass()
class Experimental_Fragment:
    """
    Base class for experimental data - grouped into subdomain fragments
    Limtation is that it only covers a single chain - which should be fine in most cases.
    """

    top: Topology_Fragment | None

    @abstractmethod
    def extract_features(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the child class.")


@dataclass()
class HDX_peptide(Experimental_Fragment):
    """
    dataclass that holds the information of a single peptide produced by HDX-MS experiments
    """

    charge: int | None
    retention_time: float
    intensity: float
    dfrac: list[float]

    def extract_features(self) -> np.ndarray:
        return np.array(self.dfrac)


@dataclass()
class HDX_protection_factor(Experimental_Fragment):
    """
    Dataclass that holds the information of a single protection factor produced by REX experiments
    or NMR derived protection factors. May also be used to describe protection factors output from forward models.
    """

    protection_factor: float

    def extract_features(self) -> np.ndarray:
        return np.array([self.protection_factor])


########################################################################\
# TODO - use generics/typevar to abstractly define the datatypes
class Experimental_Dataset:
    """
    Class to hold the information of the experimental data.
    This is created from a list of HDX_peptide objects or list of HDX_protection_factor objects.
    Once loaded, the dataset then extracts the information into a optimisable format.
    """

    def __init__(self, data: Sequence[Experimental_Fragment]):
        self.data = data
        self.y_true = self.extract_data()

    def extract_data(self) -> np.ndarray:
        """
        Map across every eleemtn in data and stack the features into a single array
        """
        return np.hstack([_exp_data.extract_features() for _exp_data in self.data])


########################################################################\


@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Array
    model_parameters: list[Model_Parameters]
    forward_model_weights: Array

    def tree_flatten(self):
        # Flatten into (arrays to differentiate, static metadata)
        # Keep model_parameters as static since we only diff forward_parameters
        arrays = (
            self.frame_weights,
            [m for m in self.model_parameters],  # List of arrays to diff
            self.forward_model_weights,
        )
        static = (self.model_parameters,)  # Store full model params as static
        return arrays, static

    @classmethod
    def tree_unflatten(cls, static, arrays):
        model_params = static[0]  # Original model params
        frame_weights, forward_params, forward_weights = arrays

        # Update each model param with new forward params
        new_model_params = [
            mp.update_parameters(fp) for mp, fp in zip(model_params, forward_params)
        ]

        return cls(frame_weights, new_model_params, forward_weights)


register_pytree_node(
    Simulation_Parameters, Simulation_Parameters.tree_flatten, Simulation_Parameters.tree_unflatten
)


class Optimisable_Parameters(Enum):
    frame_weights = 0
    model_parameters = 1
    forward_model_weights = 2


class Simulation:
    """
    This is the core object that is used during optimisation
    """

    def __init__(
        self,
        input_features: list[Input_Features],
        forward_models: Sequence[ForwardModel],
        params: Optional[Simulation_Parameters],
    ) -> None:
        self.input_features: list[Input_Features] = input_features
        self.forward_models: Sequence[ForwardModel] = forward_models

        self.params: Simulation_Parameters = params

        self.forwardpass: list[ForwardPass] = [model.forward for model in self.forward_models]

        self.output_features: Optional[list[Output_Features]] = None

    # def __post_init__(self):
    #     self.forwardpass =
    def initialise(self) -> bool:
        # assert that input features have the same first dimension of "features_shape"
        lengths = [len(feature.features_shape) for feature in self.input_features]
        assert len(set(lengths)) == 1, "Input features have different shapes. Exiting."
        self.length = lengths[0]

        if self.params is None:
            raise ValueError("No simulation parameters were provided. Exiting.")

        print("Simulation initialised successfully.")
        return True

    def forward(self) -> list[Output_Features]:
        """
        This function applies the forward models to the input features
        need to find a way to do this efficiently in jax
        """

        # first averages the input parameters using the frame weights
        average_features = map(
            frame_average_features,
            self.input_features,
            [np.array(self.params.frame_weights)] * len(self.input_features),
        )
        # map the single_pass function
        output_features = map(
            single_pass, self.forwardpass, average_features, self.params.model_parameters
        )
        ########################################################################\
        # update this to use externally defined optimisers - perhaps update should just update the parameters
        # change argnum to use enums

        return list(output_features)

    def update(
        self, params: Simulation_Parameters, argnums: tuple[int, ...] = (0, 1, 2)
    ) -> "Simulation":
        """
        Updates simulation parameters using proposed parameters and argnums to decide which parameters to update.
        As Simulation parameters are immutable, this function returns a new Simulation instance with updated parameters.

        Args:
            params: The new parameters
            argnums: Tuple indicating which parameters to optimize
                    (0: frame_weights, 1: model_parameters, 2: forward_model_weights)

        Returns:
            Updated Simulation instance
        """
        # Create lists for the new parameter values
        new_frame_weights = self.params.frame_weights
        new_model_parameters = self.params.model_parameters
        new_forward_weights = self.params.forward_model_weights

        # Update only the parameters specified in argnums
        if Optimisable_Parameters.frame_weights.value in argnums:
            new_frame_weights = params.frame_weights

        if Optimisable_Parameters.model_parameters.value in argnums:
            new_model_parameters = params.model_parameters

        if Optimisable_Parameters.forward_model_weights.value in argnums:
            new_forward_weights = params.forward_model_weights

        # Create new parameters object
        new_params = Simulation_Parameters(
            frame_weights=new_frame_weights,
            model_parameters=new_model_parameters,
            forward_model_weights=new_forward_weights,
        )

        # Create new simulation with updated parameters
        return Simulation(
            input_features=self.input_features,
            forward_models=self.forward_models,
            params=new_params,
        )


class Experiment_Ensemble:
    """
    Class to hold the information of a simulation ensemble and validate whether the forward model is compatible with the ensemble.
    This is created from a list of MDA Universe objects, as well as a forward model object.
    """

    def __init__(
        self,
        universes: list[Universe],
        forward_models: Sequence[ForwardModel],
        experimental_data: Optional[list[Experimental_Dataset]] = None,
        features: Optional[list[Input_Features]] = None,
    ):
        self.ensembles = universes
        self.experimental_data = experimental_data
        self.forward_models = forward_models
        self.features = features

    def create_model(
        self,
        features: Optional[list[Input_Features]],
        experimental_data: Optional[list[Experimental_Dataset]],
        simulation_params: Optional[Simulation_Parameters],
    ) -> Simulation:
        if features is not None:
            self.features = features

        if experimental_data is not None:
            self.experimental_data = experimental_data

        if self.experimental_data is None:
            raise ValueError("No experimental data was provided. Exiting.")

        if simulation_params is not None:
            self.params = simulation_params

        if len(self.forward_models) == 0:
            raise ValueError("No forward models were successfully initialised. Exiting.")
        if len(self.experimental_data) == 0:
            raise ValueError("No experimental data was successfully initialised. Exiting.")
        if self.features is None:
            raise ValueError("No input features were successfully initialised. Exiting.")

        assert len(self.experimental_data) == len(self.forward_models), (
            "Number of forward models must be equal to number of experimental datasets"
        )
        assert len(self.experimental_data) == len(self.features), (
            "Number of input features must be equal to number of experimental datasets"
        )
        if self.params is None:
            print("No simulation parameters were provided. Loading default parameters.")
            self.params = self.load_default_parameters()

        assert len(self.params.model_parameters) == len(self.forward_models), (
            "Number of forward models must be equal to number of forward model parameters settings"
        )
        # only add forward models that have been successfully initialised
        valid_models = self.validate_forward_models()
        # remove entities that align with none
        valid_data = [
            data[0] for data in zip(self.experimental_data, valid_models) if all(data) is not None
        ]

        self.experimental_data = valid_data
        self.forward_models = [model for model in valid_models if model is not None]

        simulation = Simulation(
            forward_models=self.forward_models, input_features=self.features, params=self.params
        )

        return simulation

    def load_default_parameters(self) -> Simulation_Parameters:
        # load default parameters

        assert self.features is not None, "No input features were provided. Exiting."

        frame_weights = jnp.array([1 / len(self.features)] * len(self.features))
        forward_model_weights = jnp.array([1 / len(self.forward_models)] * len(self.forward_models))
        model_parameters = [model.params for model in self.forward_models]

        return Simulation_Parameters(
            frame_weights=frame_weights,
            forward_model_weights=forward_model_weights,
            model_parameters=model_parameters,
        )

    def validate_forward_models(self) -> list[ForwardModel | None]:
        validated_models: list[ForwardModel | None] = []
        for model in self.forward_models:
            print(f"Initialising {model}")

            if model.initialise(self.ensembles):
                validated_models.append(model)
            else:
                validated_models.append(None)
                UserWarning(f"Model {model} failed to initialise. Skipping this model.")

        return validated_models
