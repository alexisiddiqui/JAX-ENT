from dataclasses import dataclass
from typing import Optional

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

    def initialise(self):
        # assert that input features have the same first dimension of "features_shape"
        lengths = [len(feature.features_shape) for feature in self.input_features]
        assert len(set(lengths)) == 1, "Input features have different shapes. Exiting."
        self.length = lengths[0]
        if self.frame_weights is None:
            self.frame_weights = [1 / self.length for _ in range(self.length)]
        if self.forward_model_weights is None:
            self.forward_model_weights = [1 for _ in range(len(self.forward_models))]

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
    ) -> Simulation:
        if features is not None:
            self.features = features

        # only add forward models that have been successfully initialised
        valid_models = self.validate_forward_models()
        # remove entities that align with none
        valid_data = [data for data in self.experimental_data if data is not None]

        self.experimental_data = valid_data
        self.forward_models = [model for model in valid_models if model is not None]

        if len(self.forward_models) == 0:
            raise ValueError("No forward models were successfully initialised. Exiting.")
        if len(self.experimental_data) == 0:
            raise ValueError("No experimental data was successfully initialised. Exiting.")
        if self.features is None:
            raise ValueError("No input features were successfully initialised. Exiting.")

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
