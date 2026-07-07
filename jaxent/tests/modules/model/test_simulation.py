from dataclasses import dataclass, field
from typing import Any, ClassVar, FrozenSet, Sequence

import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model


# Mock implementations for dependencies
@dataclass(frozen=True)
class MockModelParameters(Model_Parameters):
    key: FrozenSet[m_key] = field(default_factory=lambda: frozenset({m_key("mock_model")}))
    param1: jax.Array = field(default_factory=lambda: jnp.array(1.0))
    param2: jax.Array = field(default_factory=lambda: jnp.array(2.0))

    def tree_flatten(self):
        """Flatten the MockModelParameters for PyTree compatibility."""
        return (self.param1, self.param2), self.key

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct MockModelParameters from flattened data."""
        param1, param2 = children
        key = aux_data
        return cls(key=key, param1=param1, param2=param2)


# Register MockModelParameters as a PyTree
register_pytree_node(
    MockModelParameters,
    MockModelParameters.tree_flatten,
    MockModelParameters.tree_unflatten,
)


# Non-dataclass mock for Input_Features to avoid __slots__ conflict
class MockInputFeatures(Input_Features[Any]):
    __features__: ClassVar[set[str]] = {"data"}
    key: ClassVar[set[m_key]] = {m_key("mock_input")}

    def __init__(self, data: jax.Array = jnp.ones((10, 5))):
        self.data = data

    @property
    def features_shape(self) -> tuple[float | int, ...]:
        return self.data.shape

    def cast_to_jax(self) -> "MockInputFeatures":
        return MockInputFeatures(data=jnp.asarray(self.data))

    # Implement tree_flatten and tree_unflatten for PyTree compatibility
    def tree_flatten(self):
        return (self.data,), (self.key,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(data=children[0])

    def __eq__(self, other):  # For comparison in tests
        if not isinstance(other, MockInputFeatures):
            return NotImplemented
        return jnp.array_equal(self.data, other.data)

    def feat_pred(self) -> jax.Array:
        return self.data


# Register MockInputFeatures as a PyTree
register_pytree_node(
    MockInputFeatures,
    MockInputFeatures.tree_flatten,
    MockInputFeatures.tree_unflatten,
)


# Non-dataclass mock for Output_Features to avoid __slots__ conflict
class MockOutputFeatures(Output_Features):
    __features__: ClassVar[set[str]] = {"output_data"}
    key: ClassVar[m_key] = m_key("mock_output")

    def __init__(self, output_data: jax.Array = jnp.zeros((10, 1))):
        self.output_data = output_data

    @property
    def output_shape(self) -> tuple[float, ...]:
        return self.output_data.shape

    def y_pred(self) -> jax.Array:
        return self.output_data

    # Implement tree_flatten and tree_unflatten for PyTree compatibility
    def tree_flatten(self):
        return (self.output_data,), (self.key,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(output_data=children[0])

    def __eq__(self, other):  # For comparison in tests
        if not isinstance(other, MockOutputFeatures):
            return NotImplemented
        return jnp.array_equal(self.output_data, other.output_data)


# Register MockOutputFeatures as a PyTree
register_pytree_node(
    MockOutputFeatures,
    MockOutputFeatures.tree_flatten,
    MockOutputFeatures.tree_unflatten,
)


class MockForwardPass(ForwardPass[MockInputFeatures, MockOutputFeatures, MockModelParameters]):
    def __call__(
        self, input_features: MockInputFeatures, parameters: MockModelParameters
    ) -> MockOutputFeatures:
        # Simple operation: sum input data and parameters
        output_val = jnp.sum(input_features.data) + parameters.param1 + parameters.param2
        return MockOutputFeatures(
            output_data=jnp.full((input_features.data.shape[0], 1), output_val)
        )


@dataclass(frozen=True)
class MockForwardModelConfig:
    key: m_key = m_key("mock_model_config")
    forward_parameters: MockModelParameters = field(default_factory=MockModelParameters)


class MockForwardModel(
    ForwardModel[MockModelParameters, MockInputFeatures, MockForwardModelConfig]
):
    def __init__(self, config: MockForwardModelConfig):
        self.config = config
        self.params = config.forward_parameters
        self._forwardpass = MockForwardPass()  # Store an instance of the callable

    def initialise(self, ensemble: list[Any]) -> bool:
        return True

    def featurise(self, ensemble: list[Any]) -> tuple[MockInputFeatures, list[Any]]:
        return MockInputFeatures(), []

    @property
    def forwardpass(self) -> ForwardPass:
        return self._forwardpass


# Helper function to create a default Simulation_Parameters instance
def create_default_simulation_params(num_models: int = 1) -> Simulation_Parameters:
    return Simulation_Parameters(
        frame_weights=jnp.ones(
            5
        ),  
        frame_mask=jnp.ones(5, dtype=jnp.int32),
        model_parameters=[MockModelParameters() for _ in range(num_models)],
        forward_model_weights=jnp.ones(num_models),
        normalise_loss_functions=jnp.ones(num_models, dtype=jnp.int32),
        forward_model_scaling=jnp.ones(num_models),
    )


def create_bv_simulation(
    heavy_contacts: jax.Array,
    acceptor_contacts: jax.Array,
    k_ints: jax.Array,
) -> Simulation:
    config = BV_model_Config()
    features = [
        BV_input_features(
            heavy_contacts=heavy_contacts,
            acceptor_contacts=acceptor_contacts,
            k_ints=k_ints,
        )
    ]
    models = [BV_model(config)]
    n_frames = heavy_contacts.shape[-1]
    params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames, dtype=jnp.float32) / n_frames,
        frame_mask=jnp.ones(n_frames, dtype=jnp.float32),
        model_parameters=[config.forward_parameters],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.float32),
    )

    simulation = Simulation(input_features=features, forward_models=models, params=params)
    simulation.initialise()
    return simulation


def _contains_array_equal(tree, expected: jax.Array) -> bool:
    return any(
        isinstance(leaf, jax.Array)
        and leaf.shape == expected.shape
        and bool(jnp.array_equal(leaf, expected))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


# Test cases
@pytest.mark.parametrize("raise_jit_failure", [True, False])
class TestSimulation:
    def test_simulation_initialization(self, raise_jit_failure):
        input_features = [MockInputFeatures(data=jnp.ones((10, 5)))]
        forward_models = [MockForwardModel(MockForwardModelConfig())]
        params = create_default_simulation_params(num_models=1)

        simulation = Simulation(input_features, forward_models, params, raise_jit_failure)
        assert simulation.input_features == input_features
        assert simulation.forward_models == forward_models
        assert simulation.params == params
        assert isinstance(simulation.forwardpass, tuple)
        assert len(simulation.forwardpass) == 1
        assert isinstance(simulation.forwardpass[0], MockForwardPass)

        # Test initialise method
        assert simulation.initialise() is True
        assert simulation.length == 5  # From MockInputFeatures(data=jnp.ones((10, 5)))
        assert isinstance(simulation._input_features, tuple)
        assert jnp.array_equal(simulation._input_features[0].data, input_features[0].data)
        assert callable(simulation._jit_forward_pure)  # Should be jitted or fallback

    def test_simulation_initialization_no_params_raises_error(self, raise_jit_failure):
        input_features = [MockInputFeatures()]
        forward_models = [MockForwardModel(MockForwardModelConfig())]

        simulation = Simulation(input_features, forward_models, None, raise_jit_failure)
        with pytest.raises(ValueError, match="No simulation parameters were provided. Exiting."):
            simulation.initialise()

    def test_simulation_initialization_mismatched_model_counts_raises_error(
        self, raise_jit_failure
    ):
        input_features = [MockInputFeatures()]
        forward_models = [MockForwardModel(MockForwardModelConfig())]
        params = create_default_simulation_params(num_models=2)  # Mismatched

        simulation = Simulation(input_features, forward_models, params, raise_jit_failure)
        with pytest.raises(
            AssertionError,
            match="Number of forward models must equal number of model parameters",
        ):
            simulation.initialise()

    def test_simulation_forward_pure(self, raise_jit_failure):
        params = create_default_simulation_params(num_models=1)
        # Normalize weights before calling forward_pure
        params = Simulation_Parameters.normalize_weights(params)
        input_features = [MockInputFeatures(data=jnp.ones((10, 5)))]
        forwardpass = (MockForwardPass(),)

        # Manually call the static method
        outputs = Simulation.forward_pure(params, input_features, forwardpass)

        assert isinstance(outputs, Sequence)
        assert len(outputs) == 1
        assert isinstance(outputs[0], MockOutputFeatures)
        # The actual calculation after frame_average_features produces 13
        # Original: sum(jnp.ones((10,5))) + param1 + param2 = ? + 1 + 2
        # After frame averaging: 10 + 1 + 2 = 13 (frame averaging reduces the sum)
        assert jnp.allclose(outputs[0].output_data, jnp.full((10, 1), 13.0))

    def test_simulation_forward_jit_compilation(self, raise_jit_failure):
        input_features = [MockInputFeatures(data=jnp.ones((10, 5)))]
        forward_models = [MockForwardModel(MockForwardModelConfig())]
        params = create_default_simulation_params(num_models=1)

        simulation = Simulation(input_features, forward_models, params, raise_jit_failure)
        simulation.initialise()  # This will attempt JIT compilation

        # Call forward, which uses the JIT-compiled function
        simulation.forward(simulation, params)

        assert isinstance(simulation.outputs, Sequence)
        assert len(simulation.outputs) == 1
        assert isinstance(simulation.outputs[0], MockOutputFeatures)
        assert jnp.allclose(simulation.outputs[0].output_data, jnp.full((10, 1), 13.0))

        # Verify that it's actually JIT compiled (this is an indirect check)
        # A direct check would involve inspecting JAX's internal tracing, which is harder.
        # The fact that it runs without error after initialise implies JIT success.
        # We can also check if the _jit_forward_pure is indeed a jitted function
        assert hasattr(simulation._jit_forward_pure, "__wrapped__")  # JIT wraps the function

    def test_simulation_pytree_registration(self, raise_jit_failure):
        input_features = [MockInputFeatures(data=jnp.ones((10, 5)))]
        forward_models = [MockForwardModel(MockForwardModelConfig())]
        params = create_default_simulation_params(num_models=1)

        original_simulation = Simulation(input_features, forward_models, params, raise_jit_failure)
        original_simulation.initialise()
        original_simulation.forward(original_simulation,params)  # Populate outputs

        # Flatten the original simulation object
        flat_simulation, tree_def = jax.tree_util.tree_flatten(original_simulation)

        # Unflatten to reconstruct a new simulation object
        reconstructed_simulation = jax.tree_util.tree_unflatten(tree_def, flat_simulation)

        # Assert that the reconstructed object is equivalent to the original
        # Note: Direct equality check might fail for JAX arrays, compare contents
        assert isinstance(reconstructed_simulation, Simulation)
        assert reconstructed_simulation.length == original_simulation.length

        # Compare parameters (dynamic part)
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, reconstructed_simulation.params, original_simulation.params
            )
        )

        # Compare static parts (aux_data) - this is where the previous issue was
        # We need to manually compare the components that were in aux_data
        # input_features, forward_models, forwardpass, length, outputs
        assert len(reconstructed_simulation.input_features) == len(
            original_simulation.input_features
        )
        assert jnp.array_equal(
            reconstructed_simulation.input_features[0].data,
            original_simulation.input_features[0].data,
        )

        assert len(reconstructed_simulation.forward_models) == len(
            original_simulation.forward_models
        )
        # Cannot directly compare MockForwardModel instances, check their configs/params if possible
        assert (
            reconstructed_simulation.forward_models[0].config.key
            == original_simulation.forward_models[0].config.key
        )

        assert len(reconstructed_simulation.forwardpass) == len(original_simulation.forwardpass)
        # Cannot directly compare MockForwardPass instances, just check type
        assert isinstance(reconstructed_simulation.forwardpass[0], MockForwardPass)

        # Outputs should not be part of aux_data if the previous fix was applied.
        # If it was, then reconstructed_simulation.outputs would be None or uninitialized.
        # If the fix was NOT applied, then outputs would be compared here.
        # For now, assuming the fix is applied, so outputs are not part of the pytree.
        # The forward method will populate outputs after unflattening.
        # So, we should call forward on the reconstructed simulation and then compare outputs.
        reconstructed_simulation.forward(reconstructed_simulation,reconstructed_simulation.params)
        assert jnp.allclose(
            reconstructed_simulation.outputs[0].output_data,
            original_simulation.outputs[0].output_data,
        )


def test_bv_contact_features_are_dynamic_simulation_leaves() -> None:
    heavy_1 = jnp.asarray([[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], dtype=jnp.float32)
    acceptor_1 = jnp.asarray([[1.1, 1.0, 0.9], [0.7, 0.5, 0.3]], dtype=jnp.float32)
    k_ints_1 = jnp.asarray([0.05, 0.15], dtype=jnp.float32)

    heavy_2 = heavy_1 + 2.0
    acceptor_2 = acceptor_1 + 3.0
    k_ints_2 = k_ints_1 + 0.4

    simulation_1 = create_bv_simulation(heavy_1, acceptor_1, k_ints_1)
    simulation_2 = create_bv_simulation(heavy_2, acceptor_2, k_ints_2)

    dynamic_values_1, aux_data_1 = simulation_1.tree_flatten()
    dynamic_values_2, _ = simulation_2.tree_flatten()
    simulation_2_same_aux = Simulation.tree_unflatten(aux_data_1, dynamic_values_2)

    assert _contains_array_equal(dynamic_values_1, heavy_1)
    assert _contains_array_equal(dynamic_values_1, acceptor_1)
    assert _contains_array_equal(dynamic_values_1, k_ints_1)
    assert not _contains_array_equal(aux_data_1, heavy_1)
    assert not _contains_array_equal(aux_data_1, acceptor_1)
    assert not _contains_array_equal(aux_data_1, k_ints_1)

    _, treedef_1 = jax.tree_util.tree_flatten(simulation_1)
    _, treedef_2 = jax.tree_util.tree_flatten(simulation_2_same_aux)
    assert treedef_1 == treedef_2

    @jax.jit
    def contact_total(simulation: Simulation) -> jax.Array:
        features = simulation._input_features[0]
        return (
            jnp.sum(features.heavy_contacts)
            + jnp.sum(features.acceptor_contacts)
            + jnp.sum(features.k_ints)
        )

    total_1 = contact_total(simulation_1)
    total_2 = contact_total(simulation_2_same_aux)

    assert not jnp.allclose(total_1, total_2)
    assert contact_total._cache_size() == 1
