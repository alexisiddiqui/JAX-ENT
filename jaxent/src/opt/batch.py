# in this script we will define the batch_optimise function which takes in a sequence of run_optimise inputs
# TODO to make this work we need to 1) redefine _optimise method to return a nested pytree of simulation parameters
# # - we then unpack these to generate OptimisationHistory objects, essentially we are going to stack Simulation_Parameters and LossComponents in two additional dimensions - (steps, batch)
# 1b) I think we need to define a new type for the output of _batch_optimise that registers the simualtion parameters and loss components for each step - LossLandscape
# 2) we need to find way to handle the batch remainder
from typing import Callable, Sequence

from jax import vmap

from jaxent.src.models.core import Simulation
from jaxent.src.opt.base import OptimizationHistory
from jaxent.src.opt.run import OptimiseFnInputs, _optimise


def batch_optimise(
    inputs: Sequence[OptimiseFnInputs],
    batch_size: int = 4,
    _opt_fn: Callable = _optimise,
) -> Sequence[tuple[Simulation, OptimizationHistory]]:
    """Runs the optimisation process in batches"""
    # this function will take in a sequence of optimise inputs and run them in batches
    # it will return the simulation and history for each optimisation
    # the batch size will determine how many optimisations are run in parallel using vmap

    fn_input_len = len(inputs)

    # create batches of inputs - make sure we cover every input, how do we handle the last batch?
    batched_inputs = []
    for i in range(0, fn_input_len, batch_size):
        batched_inputs.append(inputs[i : i + batch_size])

    assert sum(len(batch) for batch in batched_inputs) == fn_input_len, (
        "Batched inputs do not cover all inputs"
    )

    # run the optimisations
    results = []

    for batch in batched_inputs:
        # run the optimisations in parallel
        results.extend(vmap(_opt_fn)(batch))

    return results
