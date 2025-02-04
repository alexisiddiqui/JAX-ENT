from jaxent.datatypes import Experiment_Ensemble, Simulation, Experimental_Dataset
from jaxent.config.base import Settings, FeaturiserSettings
from jaxent.forwardmodels.base import Input_Features, ForwardModel
from typing import Optional

def featurise(
        ensemble_paths: list[tuple[str,str]],
        output_path: str,
        config_paths: list[str],
        name: str,
        batch_size: Optional[int],
        forward_models: list[str],
        log_path: Optional[str],
        overwrite: bool,
):
# this function will be the input for the cli
# this will take in paths and configurations and create the individual objects for analysis
# TODO create reusable builder methods to generate objects from configuration









def run_featurise(
        ensemble: Experiment_Ensemble,
        config: Settings | FeaturiserSettings,
        forward_models: list[ForwardModel],
        name: Optional[str],
)->Input_Features:
# this function will take in the constructed objects and run the analysis

    blah = Input_Features()

    return blah