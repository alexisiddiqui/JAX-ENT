from dataclasses import dataclass

from MDAnalysis import Universe

from jaxent.forwardmodels.base import ForwardModel


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

    def __init__(self) -> None:
        self.input_features: list
        self.frame_weights: list
        self.forward_model_weights: list
        self.forward_models: list
        self.output_features: list

    def forward(self):
        """
        This function applies the forward models to the input features
        need to find a way to do this efficiently in jax
        """
        pass


class Experiment_Ensemble:
    """
    Class to hold the information of a simulation ensemble and validate whether the forward model is compatible with the ensemble.
    This is created from a list of MDA Universe objects, as well as a forward model object.
    """

    def __init__(
        self,
        universes: list[Universe],
        forward_models: list[ForwardModel],
        experimental_data: list[Experimental_Dataset],
    ):
        self.ensembles = universes
        self.model = forward_models

    def create_model(self) -> Simulation:
        # only add forward models that have been successfully initialised
        self.model = self.validate_forward_models()

    def validate_forward_models(self) -> list[ForwardModel]:
        validated_models = []
        for model in self.model:
            print(f"Initialising {model}")

            if model.initialise(self.ensembles):
                validated_models.append(model)
            else:
                UserWarning(f"Model {model} failed to initialise. Skipping this model.")

        return validated_models
