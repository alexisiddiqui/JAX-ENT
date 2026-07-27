import pytest

from jaxent.src.models.core import Simulation


@pytest.mark.parametrize("raise_jit_failure", [False, True])
def test_simulation_roundtrip_preserves_raise_jit_failure(raise_jit_failure):
    simulation = Simulation([], [], None, raise_jit_failure=raise_jit_failure)
    simulation.length = 0
    simulation._input_features = tuple()

    dynamic_values, aux_data = simulation.tree_flatten()
    reconstructed = Simulation.tree_unflatten(aux_data, dynamic_values)

    assert reconstructed.raise_jit_failure is raise_jit_failure
