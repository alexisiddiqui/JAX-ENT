import jax.numpy as jnp
import pytest

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import run_optimise


class DummySimulation:
    def __init__(self, params: Simulation_Parameters):
        self.params = params

    @staticmethod
    def forward(sim: "DummySimulation", params: Simulation_Parameters) -> "DummySimulation":
        sim.params = params
        return sim


def dummy_loss(model: DummySimulation, dataset: jnp.ndarray, prediction_index: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ = prediction_index
    train_loss = jnp.sum((model.params.frame_weights - dataset) ** 2)
    # Intentionally mirror train and validation here; this test only verifies the
    # jit_update_step=True execution path and signature compatibility.
    val_loss = train_loss
    return train_loss, val_loss


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_optimise_with_jit_enabled():
    model_config = BV_model_Config()
    params = Simulation_Parameters(
        frame_weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
        frame_mask=jnp.array([1.0, 1.0], dtype=jnp.float32),
        model_parameters=[model_config.forward_parameters],
        forward_model_weights=jnp.array([1.0], dtype=jnp.float32),
        normalise_loss_functions=jnp.array([1.0], dtype=jnp.float32),
        forward_model_scaling=jnp.array([1.0], dtype=jnp.float32),
    )
    simulation = DummySimulation(params)
    dataset = jnp.array([0.6, 0.4], dtype=jnp.float32)
    config = OptimiserSettings(name="jit-step-smoke", n_steps=2, tolerance=0.0, convergence=1e-10)
    optimizer = OptaxOptimizer(initial_steps=10)

    _, history = run_optimise(
        simulation=simulation,  # type: ignore[arg-type]
        data_to_fit=(dataset,),
        config=config,
        forward_models=[],
        indexes=[0],
        loss_functions=[dummy_loss],  # type: ignore[list-item]
        optimizer=optimizer,
        jit_update_step=True,
    )

    assert callable(optimizer.step)
    assert len(history.states) >= 1
