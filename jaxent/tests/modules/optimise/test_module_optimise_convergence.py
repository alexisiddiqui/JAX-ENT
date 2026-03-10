from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt.run import run_optimise


# These test doubles keep the convergence check independent from MDAnalysis-heavy fixtures.
@dataclass(frozen=True)
class SyntheticModelParameters(Model_Parameters):
    key: frozenset[m_key] = field(default_factory=lambda: frozenset({m_key("synthetic_model")}))
    bias: jax.Array = field(default_factory=lambda: jnp.array(0.0, dtype=jnp.float32))

    def tree_flatten(self):
        return (self.bias,), self.key

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (bias,) = children
        return cls(key=aux_data, bias=bias)


register_pytree_node(
    SyntheticModelParameters,
    SyntheticModelParameters.tree_flatten,
    SyntheticModelParameters.tree_unflatten,
)


class SyntheticInputFeatures(Input_Features[Any]):
    __features__: ClassVar[set[str]] = {"data"}
    key: ClassVar[set[m_key]] = {m_key("synthetic_input")}

    def __init__(self, data: jax.Array):
        self.data = data

    @property
    def features_shape(self) -> tuple[int, ...]:
        return self.data.shape

    def cast_to_jax(self) -> "SyntheticInputFeatures":
        return type(self)(jnp.asarray(self.data))

    def tree_flatten(self):
        return (self.data,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

    def feat_pred(self):
        return self.data


register_pytree_node(
    SyntheticInputFeatures,
    SyntheticInputFeatures.tree_flatten,
    SyntheticInputFeatures.tree_unflatten,
)


class SyntheticOutputFeatures(Output_Features):
    __features__: ClassVar[set[str]] = {"output_data"}
    key: ClassVar[m_key] = m_key("synthetic_output")

    def __init__(self, output_data: jax.Array):
        self.output_data = output_data

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.output_data.shape

    def y_pred(self):
        return self.output_data

    def tree_flatten(self):
        return (self.output_data,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])


register_pytree_node(
    SyntheticOutputFeatures,
    SyntheticOutputFeatures.tree_flatten,
    SyntheticOutputFeatures.tree_unflatten,
)


class SyntheticForwardPass(
    ForwardPass[SyntheticInputFeatures, SyntheticOutputFeatures, SyntheticModelParameters]
):
    def __call__(
        self,
        input_features: SyntheticInputFeatures,
        parameters: SyntheticModelParameters,
    ) -> SyntheticOutputFeatures:
        prediction = jnp.sum(input_features.data) + parameters.bias
        return SyntheticOutputFeatures(output_data=jnp.asarray([prediction]))


@dataclass(frozen=True)
class SyntheticForwardModelConfig:
    key: m_key = m_key("synthetic_forward")
    forward_parameters: SyntheticModelParameters = field(default_factory=SyntheticModelParameters)


class SyntheticForwardModel(
    ForwardModel[SyntheticModelParameters, SyntheticInputFeatures, SyntheticForwardModelConfig]
):
    def __init__(self, config: SyntheticForwardModelConfig):
        self.config = config
        self.params = config.forward_parameters
        self._forwardpass = SyntheticForwardPass()

    def initialise(self, ensemble: list[Any]) -> bool:
        return True

    def featurise(self, ensemble: list[Any]) -> tuple[SyntheticInputFeatures, list[Any]]:
        return SyntheticInputFeatures(jnp.zeros((1, 5), dtype=jnp.float32)), []

    @property
    def forwardpass(self) -> ForwardPass:
        return self._forwardpass


def _create_synthetic_simulation() -> tuple[Simulation, list[SyntheticForwardModel]]:
    models = [SyntheticForwardModel(SyntheticForwardModelConfig())]
    params = Simulation_Parameters(
        frame_weights=jnp.ones(5, dtype=jnp.float32),
        frame_mask=jnp.ones(5, dtype=jnp.float32),
        model_parameters=[SyntheticModelParameters()],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.float32),
    )
    simulation = Simulation(
        input_features=[
            SyntheticInputFeatures(
                jnp.asarray([[0.0, 1.0, 2.0, 3.0, 10.0]], dtype=jnp.float32)
            )
        ],
        forward_models=models,
        params=params,
    )
    simulation.initialise()
    return simulation, models


def synthetic_output_l2_loss(model, target: jax.Array, prediction_index: int):
    prediction = model.outputs[prediction_index].output_data
    loss = jnp.mean(jnp.square(prediction - target))
    return loss, loss


def test_module_optimise_tracks_train_loss_improvement_for_synthetic_example():
    simulation, models = _create_synthetic_simulation()
    config = OptimiserSettings(
        name="synthetic_convergence",
        n_steps=25,
        tolerance=1e-8,
        convergence=200.0,
        learning_rate=0.5,
    )

    _, history = run_optimise(
        simulation,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        forward_models=models,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
    )

    train_losses = jnp.asarray([state.losses.total_train_loss for state in history.states])

    assert len(history.states) >= 2
    assert history.states[-1].step > history.states[0].step
    assert bool(jnp.all(jnp.isfinite(train_losses)))
    assert bool(train_losses[-1] < train_losses[0])
