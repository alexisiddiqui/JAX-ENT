from dataclasses import dataclass


@dataclass(from_dict=True)
class Settings:
    n_replicates: int = 3,








class PathManager():
    '''
    Path manager class to componse and manage paths for an optimisation experiment.
    This needs to include the following:
    - Path(s) to the experimental data
    - Path(s) to the Simulation trajectories
    - Path(s) to the model checkpoints/parameter
    - Path(s) to the optimisation results
    - Path(s) to the logs 
    - Path(s) for the analysis outputs
    '''
    def __init__(self, settings):
        self.settings = settings


