"""
This script is used to calculate the uptake for a given trajectory using the bv model.
It first featurises the trajectory, and then calculates the residue-level uptake using the bv model using the predict method.

The outputs are saved

Args:
    - 'topology': The topology path of the trajectory.
    - 'trajectory': The trajectory path.
    - 'output': The output path where the results will be saved.
    - 'name': The prefix given to the files.
    - 'bv_bc': bv model bc constant.
    - 'bv_bh: bv model bh constant.
    - 'timepoints': The timepoints at which the uptake is calculated - if none provided then predicts protection factors


"""
