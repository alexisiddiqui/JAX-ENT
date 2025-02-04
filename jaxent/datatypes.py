from dataclasses import dataclass
from typing import Optional

import jax
import numpy as np
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
class HDX_peptide:
    """
    dataclass that holds the information of a single peptide produced by HDX-MS experiments
    """

    chain: str
    peptide_number: int
    peptide_sequence: str
    residue_start: int
    residue_end: int
    charge: int | None
    retention_time: float
    intensity: float
    dfrac: float


@dataclass()
class HDX_protection_factor:
    """
    Dataclass that holds the information of a single protection factor produced by REX experiments
    or NMR derived protection factors. May also be used to describe protection factors output from forward models.
    """

    chain: str
    residue_number: int
    residue: int
    protection_factor: float


@dataclass()
class Topology_Fragment:
    """
    Dataclass that holds the information of a single topology fragment.
    This is usually a residue but may also be a group of residues such as peptides.
    """

    chain: str
    fragment_index: int
    fragment_sequence: str | list[str]
    residue_start: int
    residue_end: int

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


class Experimental_Dataset:
    """
    Class to hold the information of the experimental data.
    This is created from a list of HDX_peptide objects or list of HDX_protection_factor objects.
    """

    def __init__(self, data: list[HDX_peptide] | list[HDX_protection_factor]):
        self.data = data


class Simulation:
    """
    This is the core object that is used during optimisation
    """

    def __init__(
        self, forward_models: list[ForwardModel], input_features: list[Input_Features]
    ) -> None:
        self.input_features: list[Input_Features] = input_features
        self.frame_weights: list[float]
        self.parameters: list[Model_Parameters]
        self.forward_model_weights: list[float]
        self.forward_models: list[ForwardModel] = forward_models
        self.output_features: Optional[list[Output_Features]] = None
        self.forwardpass: list[ForwardPass]

    def __post_init__(self):
        self.forwardpass = [model.forward for model in self.forward_models]
        self.forward_parameters = [p.forward_parameters for p in self.parameters]

    def initialise(self) -> bool:
        # assert that input features have the same first dimension of "features_shape"
        lengths = [len(feature.features_shape) for feature in self.input_features]
        assert len(set(lengths)) == 1, "Input features have different shapes. Exiting."
        self.length = lengths[0]
        if self.frame_weights is None:
            self.frame_weights = [1 / self.length] * self.length
        if self.frame_weights:
            assert np.sum(self.frame_weights) == 1, "Frame weights do not sum to 1. Exiting."
        if self.forward_model_weights is None:
            self.forward_model_weights = [1 for _ in range(len(self.forward_models))]

        return True

    def forward(self):
        """
        This function applies the forward models to the input features
        need to find a way to do this efficiently in jax
        """

        # first averages the input parameters using the frame weights
        average_features = map(
            frame_average_features,
            self.input_features,
            [self.frame_weights] * len(self.input_features),
        )
        # map the single_pass function
        return map(single_pass, self.forwardpass, average_features, self.parameters)
        ########################################################################\
        # update this to use externally defined optimisers - perhaps update should just update the parameters
        # change argnum to use enums

    def update(
        self, loss: float, argnums: tuple[int, ...] = (0, 1, 2), learning_rate=0.01
    ) -> "Simulation":
        """
        Updates simulation parameters based on loss and specified parameters to optimize.

        Args:
            loss: The scalar loss value
            argnums: Tuple indicating which parameters to optimize
                    (0: frame_weights, 1: parameters, 2: forward_model_weights)

        Returns:
            Updated Simulation instance
        """
        # Create new simulation instance
        new_simulation = Simulation(
            forward_models=self.forward_models, input_features=self.input_features
        )

        # Define parameters tuple in same order as argnums reference
        params_tuple = (self.frame_weights, self.forward_parameters, self.forward_model_weights)

        # Get gradients only for specified parameters
        grads = jax.grad(lambda *args: loss, argnums=argnums)(*params_tuple)

        # Convert single gradient to tuple if only one parameter being optimized
        if isinstance(grads, jax.Array):
            grads = (grads,)

        # Create dictionary mapping param index to its gradient
        grad_dict = dict(zip(argnums, grads))

        # Could be moved to config

        # Update frame weights if specified
        if 0 in argnums:
            new_frame_weights = [
                w - learning_rate * g for w, g in zip(self.frame_weights, grad_dict[0])
            ]
            # Normalize
            frame_sum = sum(new_frame_weights)
            new_frame_weights = [w / frame_sum for w in new_frame_weights]
        else:
            new_frame_weights = self.frame_weights

        # Update parameters if specified
        if 1 in argnums:
            new_parameters = [
                param - learning_rate * grad
                for param, grad in zip(self.forward_parameters, grad_dict[1])
            ]
        else:
            new_parameters = self.forward_parameters

        # Update forward model weights if specified
        if 2 in argnums:
            new_model_weights = [
                w - learning_rate * g for w, g in zip(self.forward_model_weights, grad_dict[2])
            ]
        else:
            new_model_weights = self.forward_model_weights

        # Set all parameters in new simulation
        new_simulation.frame_weights = new_frame_weights
        new_simulation.forward_parameters = new_parameters
        new_simulation.forward_model_weights = new_model_weights

        return new_simulation

        ########################################################################


class Experiment_Ensemble:
    """
    Class to hold the information of a simulation ensemble and validate whether the forward model is compatible with the ensemble.
    This is created from a list of MDA Universe objects, as well as a forward model object.
    """

    def __init__(
        self,
        universes: list[Universe],
        forward_models: list[ForwardModel],
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
    ) -> Simulation:
        if features is not None:
            self.features = features

        if experimental_data is not None:
            self.experimental_data = experimental_data

        if self.experimental_data is None:
            raise ValueError("No experimental data was provided. Exiting.")

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
        # only add forward models that have been successfully initialised
        valid_models = self.validate_forward_models()
        # remove entities that align with none
        valid_data = [
            data[0] for data in zip(self.experimental_data, valid_models) if all(data) is not None
        ]

        self.experimental_data = valid_data
        self.forward_models = [model for model in valid_models if model is not None]

        simulation = Simulation(forward_models=self.forward_models, input_features=self.features)

        return simulation

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
