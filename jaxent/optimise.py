from jaxent.datatypes import Experiment_Ensemble, Simulation, Experimental_Dataset
from jaxent.config.base import Settings, OptimiserSettings
from jaxent.forwardmodels.base import Input_Features, ForwardModel
from jaxent.lossfn.base import JaxEnt_Loss
from typing import Optional

def optimise(
        ensemble_paths: list[tuple[str,str]],
        features_dir: list[str],
        output_path: str,
        config_paths: list[str],
        name: str,
        batch_size: Optional[int],
        forward_models: list[str],
        loss_functions: list[str],
        log_path: Optional[str],
        overwrite: bool,
):
# this function will be the input for the cli
# this will take in paths and configurations and create the individual objects for analysis
# TODO create reusable builder methods to generate objects from configuration









def run_optimise(
        ensemble: Experiment_Ensemble,
        input_features: list[Input_Features],
        experimental_data: list[Experimental_Dataset],
        config: Settings | OptimiserSettings,
        forward_models: list[ForwardModel],
        loss_functions: list[JaxEnt_Loss],
        name: Optional[str],
)->Simulation:
# this function will take in the constructed objects and run the analysis

    blah = Simulation()

    return blah