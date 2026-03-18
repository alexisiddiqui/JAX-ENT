from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import sparse

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.key import m_key
from jaxent.src.data.splitting.mapping import PairIndexMapping, QSubsetMapping, SparseFragmentMapping
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.SAXS.config import SAXS_direct_Config
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.models.SAXS.forwardmodel import SAXS_direct_model
from jaxent.src.models.XLMS.config import XLMS_Config
from jaxent.src.models.XLMS.features import XLMS_input_features
from jaxent.src.models.XLMS.forwardmodel import XLMS_distance_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.opt.base import OptimizationHistory


def make_test_optimiser_settings(name: str, n_steps: int = 12) -> OptimiserSettings:
    return OptimiserSettings(
        name=name,
        n_steps=n_steps,
        tolerance=1e-10,
        convergence=200.0,
        learning_rate=0.3,
    )


def _build_hdx_features(n_res: int, n_frames: int) -> BV_input_features:
    residue_axis = np.linspace(0.6, 1.6, n_res, dtype=np.float32)[:, None]
    frame_axis = np.linspace(0.75, 1.25, n_frames, dtype=np.float32)[None, :]
    heavy_contacts = jnp.asarray(residue_axis * frame_axis + 0.2, dtype=jnp.float32)
    acceptor_contacts = jnp.asarray(
        residue_axis * np.flip(frame_axis, axis=1) + 0.1, dtype=jnp.float32
    )
    k_ints = jnp.asarray(np.linspace(0.3, 1.1, n_res, dtype=np.float32))
    return BV_input_features(
        heavy_contacts=heavy_contacts,
        acceptor_contacts=acceptor_contacts,
        k_ints=k_ints,
    )


def _build_saxs_features(n_q: int, n_frames: int, seed: int) -> SAXS_curve_input_features:
    rng = np.random.default_rng(seed)
    q_axis = np.linspace(0.1, 2.0, n_q, dtype=np.float32)[:, None]
    frame_axis = np.linspace(0.7, 1.3, n_frames, dtype=np.float32)[None, :]
    noise = rng.normal(loc=0.0, scale=0.01, size=(n_q, n_frames)).astype(np.float32)
    intensities = jnp.asarray(q_axis * frame_axis + 0.5 + noise, dtype=jnp.float32)
    return SAXS_curve_input_features(intensities=intensities)


def _build_xlms_features(n_res: int, n_frames: int, seed: int) -> XLMS_input_features:
    rng = np.random.default_rng(seed)
    residue_indices = np.arange(n_res, dtype=np.float32)
    base_matrix = np.abs(residue_indices[:, None] - residue_indices[None, :]) + 1.0
    frame_scales = np.linspace(0.8, 1.2, n_frames, dtype=np.float32)

    frames = []
    for scale in frame_scales:
        noise = rng.normal(loc=0.0, scale=0.02, size=(n_res, n_res)).astype(np.float32)
        sym_noise = 0.5 * (noise + noise.T)
        frame = (base_matrix * scale) + sym_noise
        np.fill_diagonal(frame, 0.0)
        frames.append(frame)
    distances = jnp.asarray(np.stack(frames, axis=-1), dtype=jnp.float32)
    return XLMS_input_features(distances=distances)


def build_mixed_simulation(
    *,
    include_saxs: bool,
    include_xlms: bool,
    seed: int = 0,
    n_frames: int = 4,
    n_res: int = 6,
    n_q: int = 12,
    target_frame: int = 0,
) -> tuple[Simulation, list, list[Array]]:
    models = []
    features = []
    model_parameters = []
    targets = []

    hdx_config = BV_model_Config()
    hdx_feature = _build_hdx_features(n_res=n_res, n_frames=n_frames)
    models.append(BV_model(hdx_config))
    features.append(hdx_feature)
    model_parameters.append(hdx_config.forward_parameters)
    hdx_params = hdx_config.forward_parameters
    hdx_target = (
        (jnp.asarray(hdx_params.bv_bc) * jnp.asarray(hdx_feature.heavy_contacts)[:, target_frame])
        + (jnp.asarray(hdx_params.bv_bh) * jnp.asarray(hdx_feature.acceptor_contacts)[:, target_frame])
    )
    targets.append(jnp.asarray(hdx_target))

    if include_saxs:
        saxs_config = SAXS_direct_Config()
        saxs_feature = _build_saxs_features(n_q=n_q, n_frames=n_frames, seed=seed + 11)
        models.append(SAXS_direct_model(saxs_config))
        features.append(saxs_feature)
        model_parameters.append(saxs_config.forward_parameters)
        targets.append(jnp.asarray(saxs_feature.intensities)[:, target_frame])

    if include_xlms:
        xlms_config = XLMS_Config()
        xlms_feature = _build_xlms_features(n_res=n_res, n_frames=n_frames, seed=seed + 23)
        models.append(XLMS_distance_model(xlms_config))
        features.append(xlms_feature)
        model_parameters.append(xlms_config.forward_parameters)
        targets.append(jnp.asarray(xlms_feature.distances)[:, :, target_frame])

    n_models = len(models)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames, dtype=jnp.float32) / n_frames,
        frame_mask=jnp.ones(n_frames, dtype=jnp.float32),
        model_parameters=model_parameters,
        forward_model_weights=jnp.ones(n_models, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(n_models, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(n_models, dtype=jnp.float32),
    )

    simulation = Simulation(input_features=features, forward_models=models, params=params)
    simulation.initialise()
    simulation = Simulation.forward(simulation, simulation.params)
    return simulation, models, targets


def output_l2_loss(model: Simulation, target: Array, prediction_index: int) -> tuple[Array, Array]:
    prediction = jnp.asarray(model.outputs[prediction_index].y_pred())
    target = jnp.asarray(target)
    loss = jnp.mean(jnp.square(prediction - target))
    return loss, loss


def mapped_output_l2_loss(
    model: Simulation,
    mapped_target: tuple[SparseFragmentMapping | QSubsetMapping | PairIndexMapping, Array],
    prediction_index: int,
) -> tuple[Array, Array]:
    data_mapping, target = mapped_target
    prediction = jnp.asarray(model.outputs[prediction_index].y_pred())
    mapped_prediction = jnp.asarray(data_mapping.apply(prediction))
    target = jnp.asarray(target)
    loss = jnp.mean(jnp.square(mapped_prediction - target))
    return loss, loss


def assert_history_has_measurable_loss_drop(
    history: OptimizationHistory, expected_components: int
) -> None:
    train_losses = jnp.asarray([state.losses.total_train_loss for state in history.states])
    assert len(history.states) >= 1
    assert bool(jnp.all(jnp.isfinite(train_losses)))
    assert history.states[0].losses.train_losses.shape[0] == expected_components


def compute_total_train_loss(
    simulation: Simulation,
    data_targets: Sequence,
    indexes: Sequence[int],
    loss_functions: Sequence,
) -> Array:
    simulation = Simulation.forward(simulation, simulation.params)
    train_components = [
        loss_fn(simulation, data_target, index)[0]
        for loss_fn, data_target, index in zip(loss_functions, data_targets, indexes)
    ]
    return jnp.sum(jnp.asarray(train_components))


def assert_measurable_loss_decrease(
    initial_loss: Array,
    final_loss: Array,
    min_delta: float = 1e-6,
) -> None:
    initial_value = float(jnp.asarray(initial_loss))
    final_value = float(jnp.asarray(final_loss))
    assert final_value < (initial_value - min_delta)


def assert_output_keys(simulation: Simulation, expected_keys: Sequence[m_key]) -> None:
    outputs_by_key = simulation.outputs_by_key
    for key in expected_keys:
        assert key in outputs_by_key


def build_mapping_targets_for_three_models(
    hdx_target: Array,
    saxs_target: Array,
    xlms_target: Array,
) -> tuple[
    tuple[SparseFragmentMapping | QSubsetMapping | PairIndexMapping, ...],
    tuple[Array, Array, Array],
]:
    n_res = int(hdx_target.shape[0])
    hdx_mapping = SparseFragmentMapping(
        sparse_map=sparse.bcoo_fromdense(jnp.eye(n_res, dtype=jnp.float32))
    )

    n_q = int(saxs_target.shape[0])
    saxs_indices = jnp.asarray([0, n_q // 2, n_q - 1], dtype=jnp.int32)
    saxs_mapping = QSubsetMapping(indices=saxs_indices)

    n_xlms = int(xlms_target.shape[0])
    pair_i = jnp.asarray([0, n_xlms // 2, n_xlms - 2], dtype=jnp.int32)
    pair_j = jnp.asarray([2, n_xlms // 2 + 1, n_xlms - 1], dtype=jnp.int32)
    xlms_mapping = PairIndexMapping(indices_i=pair_i, indices_j=pair_j)

    mapped_hdx_target = hdx_mapping.apply(jnp.asarray(hdx_target))
    mapped_saxs_target = saxs_mapping.apply(jnp.asarray(saxs_target))
    mapped_xlms_target = xlms_mapping.apply(jnp.asarray(xlms_target))

    mappings = (hdx_mapping, saxs_mapping, xlms_mapping)
    mapped_targets = (mapped_hdx_target, mapped_saxs_target, mapped_xlms_target)
    return mappings, mapped_targets
