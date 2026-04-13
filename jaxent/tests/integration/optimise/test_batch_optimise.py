from pathlib import Path

import jax.numpy as jnp
from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings, OptimiserSettings
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.data.splitting.mapping import SparseFragmentMapping
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.opt.base import HParamBatch
from jaxent.src.opt.batch import _replace_hparams, batch_optimise
from jaxent.src.opt.losses import hdx_pf_l2_loss
from jaxent.src.opt.run import run_optimise
from jaxent.tests.test_utils import get_inst_path


def _make_config(name: str, learning_rate: float) -> OptimiserSettings:
    return OptimiserSettings(
        name=name,
        n_steps=4,
        tolerance=1e-8,
        convergence=200.0,
        learning_rate=learning_rate,
    )


def _build_quick_hdx_fixture() -> tuple[Simulation, list[BV_model], ExpD_Dataloader]:
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    trajectory_path = inst_path / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

    universe = Universe(str(topology_path), str(trajectory_path))
    models = [BV_model(bv_config)]
    ensemble = Experiment_Builder([universe], models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    trajectory_length = features[0].features_shape[1]
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length) / 2.0,
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )
    simulation = Simulation(forward_models=models, input_features=features, params=params)
    simulation.initialise()
    simulation.forward(simulation, params)

    exp_data = [
        HDX_protection_factor(protection_factor=10.0, top=top)
        for top in feature_topology[0]
    ]
    dataset = ExpD_Dataloader(data=exp_data)
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=[universe],
        common_residues=set(feature_topology[0]),
    )
    train_data, val_data = splitter.random_split()

    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)
    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)
    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    dataset.train = Dataset(
        data=train_data,
        y_true=jnp.array([point.extract_features() for point in train_data]),
        data_mapping=SparseFragmentMapping(sparse_map=train_sparse_map),
    )
    dataset.val = Dataset(
        data=val_data,
        y_true=jnp.array([point.extract_features() for point in val_data]),
        data_mapping=SparseFragmentMapping(sparse_map=val_sparse_map),
    )
    dataset.test = Dataset(
        data=exp_data,
        y_true=jnp.array([point.extract_features() for point in exp_data]),
        data_mapping=SparseFragmentMapping(sparse_map=test_sparse_map),
    )
    return simulation, models, dataset


def _clone_simulation_with_params(
    simulation: Simulation,
    params: Simulation_Parameters,
) -> Simulation:
    _, aux_data = simulation.tree_flatten()
    return Simulation.tree_unflatten(aux_data, (params, tuple()))


def test_batch_optimise_real_fixture_matches_sequential_final_states() -> None:
    simulation, models, dataset = _build_quick_hdx_fixture()
    learning_rate = 0.3
    config = _make_config("integration_batch", learning_rate=learning_rate)

    hparam_batch = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.asarray([[1.0], [0.8], [1.2]], dtype=jnp.float32),
        learning_rate=jnp.asarray([learning_rate, learning_rate, learning_rate], dtype=jnp.float32),
    )
    batch_result = batch_optimise(
        simulation=simulation,
        hparam_batch=hparam_batch,
        batch_size=2,
        data_to_fit=(dataset,),
        config=config,
        indexes=[0],
        loss_functions=[hdx_pf_l2_loss],
    )

    assert len(batch_result.best_states) == 3
    assert batch_result.convergence_steps.shape == (3,)

    base_params = simulation.params
    sequential_losses = []
    sequential_weights = []
    for run_idx in range(3):
        run_params = _replace_hparams(
            base_params,
            forward_model_weights=hparam_batch.forward_model_weights[run_idx],
            forward_model_scaling=hparam_batch.forward_model_scaling[run_idx],
        )
        run_simulation = _clone_simulation_with_params(simulation, run_params)
        run_simulation.forward(run_simulation, run_params)
        _, history = run_optimise(
            run_simulation,
            data_to_fit=(dataset,),
            config=_make_config(f"integration_seq_{run_idx}", learning_rate),
            forward_models=models,
            indexes=[0],
            loss_functions=[hdx_pf_l2_loss],
            jit_update_step=False,
        )
        sequential_losses.append(history.best_state.losses.total_train_loss)
        sequential_weights.append(history.best_state.params.frame_weights)

    for batch_best, seq_loss, seq_weights in zip(
        batch_result.best_states,
        sequential_losses,
        sequential_weights,
    ):
        assert jnp.allclose(batch_best.losses.total_train_loss, seq_loss, rtol=1e-4)
        assert jnp.allclose(batch_best.params.frame_weights, seq_weights, atol=1e-4)
