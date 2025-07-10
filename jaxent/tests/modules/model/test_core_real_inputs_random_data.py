
import pytest
import jax
import jax.numpy as jnp
from typing import Sequence

from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_input_features, BV_model

# --- Test Fixtures ---

@pytest.fixture
def real_inputs_random_data():
    """
    Provides a set of real input objects populated with random data.
    This avoids file I/O while still testing the integration of real classes.
    """
    key = jax.random.PRNGKey(42)
    num_residues = 20
    num_frames = 100
    
    # 1. Create a real model instance
    bv_config = BV_model_Config()
    forward_models = [BV_model(bv_config)]

    # 2. Create real input features with random data
    key, *subkeys = jax.random.split(key, 4)
    input_features = [
        BV_input_features(
            heavy_contacts=jax.random.uniform(subkeys[0], (num_residues, num_frames)),
            acceptor_contacts=jax.random.uniform(subkeys[1], (num_residues, num_frames)),
            k_ints=jax.random.uniform(subkeys[2], (num_residues,))
        )
    ]

    # 3. Create real simulation parameters with random weights
    key, subkey = jax.random.split(key)
    frame_weights = jax.random.uniform(subkey, (num_frames,))
    frame_weights /= jnp.sum(frame_weights)  # Normalize

    params = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=jnp.ones(num_frames),
        model_parameters=[model.config.forward_parameters for model in forward_models],
        forward_model_weights=jnp.array([1.0]),
        forward_model_scaling=jnp.array([1.0]),
        normalise_loss_functions=jnp.array([0.0]),
    )
    
    return input_features, forward_models, params

# --- Test Cases ---

def test_simulation_initialise_real_inputs(real_inputs_random_data):
    """Tests that Simulation initializes correctly with real input classes."""
    input_features, forward_models, params = real_inputs_random_data
    
    simulation = Simulation(
        input_features=input_features,
        forward_models=forward_models,
        params=params
    )
    
    assert simulation.initialise(), "Initialisation should return True."
    assert simulation.length == 100
    assert hasattr(simulation, '_jit_forward_pure')

def test_simulation_forward_jit_real_inputs(real_inputs_random_data):
    """Tests the JIT-compiled forward pass with real input classes."""
    input_features, forward_models, params = real_inputs_random_data
    
    simulation = Simulation(input_features, forward_models, params)
    simulation.initialise()
    
    simulation.forward(params)
    
    assert simulation.outputs is not None
    assert len(simulation.outputs) == 1
    # The output of BV_model is log_Pf, which should have a shape equal to num_residues
    assert simulation.outputs[0].log_Pf.shape == (20,)

def test_simulation_as_pytree_real_inputs(real_inputs_random_data):
    """Tests that Simulation works as a PyTree with real input classes."""
    input_features, forward_models, params = real_inputs_random_data
    
    simulation = Simulation(input_features, forward_models, params)
    simulation.initialise()

    @jax.jit
    def run_sim_in_jit(sim: Simulation, p: Simulation_Parameters):
        sim.forward(p)
        return sim.outputs

    # Run twice to ensure no re-compilation hangs
    outputs1 = run_sim_in_jit(simulation, params)
    outputs2 = run_sim_in_jit(simulation, params)

    assert len(outputs1) == 1
    assert outputs1[0].log_Pf.shape == (20,)
    assert jnp.allclose(outputs1[0].log_Pf, outputs2[0].log_Pf)

def test_simulation_multiple_models_real_inputs(real_inputs_random_data):
    """Tests Simulation with multiple real models and inputs."""
    _, _, params = real_inputs_random_data
    key = jax.random.PRNGKey(0)
    num_residues = 20
    num_frames = 100

    # Create two models and two sets of random input features
    bv_config1 = BV_model_Config()
    bv_config2 = BV_model_Config()
    # Manually set different parameters to test multi-model logic
    bv_config2.bv_bh = jnp.array([3.0])
    forward_models = [BV_model(bv_config1), BV_model(bv_config2)]
    
    key, *subkeys = jax.random.split(key, 7)
    input_features = [
        BV_input_features(
            heavy_contacts=jax.random.uniform(subkeys[0], (num_residues, num_frames)),
            acceptor_contacts=jax.random.uniform(subkeys[1], (num_residues, num_frames)),
            k_ints=jax.random.uniform(subkeys[2], (num_residues,))
        ),
        BV_input_features(
            heavy_contacts=jax.random.uniform(subkeys[3], (num_residues, num_frames)),
            acceptor_contacts=jax.random.uniform(subkeys[4], (num_residues, num_frames)),
            k_ints=jax.random.uniform(subkeys[5], (num_residues,))
        )
    ]
    
    new_model_parameters = [model.config.forward_parameters for model in forward_models]
    params = Simulation_Parameters(
        frame_weights=params.frame_weights,
        frame_mask=params.frame_mask,
        model_parameters=new_model_parameters,
        forward_model_weights=jnp.array([1.0, 1.0]),
        forward_model_scaling=jnp.array([1.0, 1.0]),
        normalise_loss_functions=jnp.array([0.0, 0.0])
    )

    simulation = Simulation(input_features, forward_models, params)
    simulation.initialise()
    simulation.forward(params)

    assert len(simulation.outputs) == 2
    assert simulation.outputs[0].log_Pf.shape == (20,)
    assert simulation.outputs[1].log_Pf.shape == (20,)
    # The outputs should be different because the model parameters (k_int) are different
    assert not jnp.allclose(simulation.outputs[0].log_Pf, simulation.outputs[1].log_Pf)
