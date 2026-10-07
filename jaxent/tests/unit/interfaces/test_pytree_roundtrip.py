import pytest
import jax.numpy as jnp

from jaxent.src.custom_types.features import AbstractFeatures
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters
from jaxent.src.models.HDX.netHDX.parameters import NetHDX_Model_Parameters
from jaxent.src.models.SAXS.parameters import SAXS_Debye_Parameters, SAXS_direct_Model_Parameters
from jaxent.src.models.XLMS.parameters import XLMS_Model_Parameters


@pytest.mark.parametrize("raise_jit_failure", [False, True])
def test_simulation_roundtrip_preserves_raise_jit_failure(raise_jit_failure):
    simulation = Simulation([], [], None, raise_jit_failure=raise_jit_failure)
    simulation.length = 0
    simulation._input_features = tuple()

    dynamic_values, aux_data = simulation.tree_flatten()
    reconstructed = Simulation.tree_unflatten(aux_data, dynamic_values)

    assert reconstructed.raise_jit_failure is raise_jit_failure


def test_simulation_roundtrip_preserves_all_fields_and_can_initialise():
    class Feature:
        features_shape = (1, 1)

        def cast_to_jax(self):
            return self

    feature = Feature()
    params = Simulation_Parameters.from_frame_weights(
        jnp.array([1.0]),
        model_parameters=[],
        forward_model_weights=jnp.array([]),
        normalise_loss_functions=jnp.array([]),
        forward_model_scaling=jnp.array([]),
    )
    simulation = Simulation([feature], [], params, raise_jit_failure=False)
    simulation.initialise()

    dynamic_values, aux_data = simulation.tree_flatten()
    reconstructed = Simulation.tree_unflatten(aux_data, dynamic_values)

    assert reconstructed.params is simulation.params
    assert reconstructed.outputs == simulation.outputs
    assert reconstructed.forward_models is simulation.forward_models
    assert reconstructed.forwardpass == simulation.forwardpass
    assert reconstructed.length == simulation.length
    assert reconstructed._jit_forward_pure is simulation._jit_forward_pure
    assert reconstructed.raise_jit_failure is False
    assert reconstructed.input_features is reconstructed._input_features
    assert reconstructed._input_features == simulation._input_features
    assert reconstructed.initialise()


def _mro_slots_reference(cls):
    slots = []
    for parent in cls.__mro__:
        if hasattr(parent, "__slots__"):
            slots.extend(parent.__slots__)
    return tuple(dict.fromkeys(slots))


def test_model_parameter_slot_scanners_are_cached_and_keyed_by_class():
    classes = [
        BV_Model_Parameters,
        linear_BV_Model_Parameters,
        NetHDX_Model_Parameters,
        XLMS_Model_Parameters,
        SAXS_direct_Model_Parameters,
        SAXS_Debye_Parameters,
    ]
    Model_Parameters._get_ordered_slots.cache_clear()
    Model_Parameters._get_grouped_slots.cache_clear()

    expected = {
        cls: (
            _mro_slots_reference(cls),
            tuple(slot for slot in _mro_slots_reference(cls) if slot not in cls.static_params),
            tuple(slot for slot in _mro_slots_reference(cls) if slot in cls.static_params),
        )
        for cls in classes
    }
    for cls in classes:
        assert cls._get_ordered_slots() == expected[cls][0]
        assert cls._get_grouped_slots() == (expected[cls][1], expected[cls][2])
    for cls in classes:
        assert cls._get_ordered_slots() is cls._get_ordered_slots()
        assert cls._get_grouped_slots() is cls._get_grouped_slots()

    assert Model_Parameters._get_ordered_slots.cache_info().misses == len(classes)
    assert Model_Parameters._get_grouped_slots.cache_info().misses == len(classes)
    assert Model_Parameters._get_ordered_slots.cache_info().hits >= len(classes)
    assert Model_Parameters._get_grouped_slots.cache_info().hits >= len(classes)
    assert expected[BV_Model_Parameters] != expected[SAXS_Debye_Parameters]


def test_abstract_features_roundtrip_bypasses_constructor_validation():
    class TestFeatures(AbstractFeatures):
        __slots__ = ("dynamic", "static")
        __features__ = {"dynamic"}

        def __init__(self, dynamic, static):
            self.dynamic = dynamic
            self.static = static

    TestFeatures._get_ordered_slots.cache_clear()
    TestFeatures._get_grouped_slots.cache_clear()
    original = TestFeatures(jnp.array([1.0]), "metadata")
    arrays, static_data = original.tree_flatten()
    reconstructed = TestFeatures.tree_unflatten(static_data, arrays)

    assert reconstructed.dynamic is arrays[0]
    assert reconstructed.static == "metadata"
