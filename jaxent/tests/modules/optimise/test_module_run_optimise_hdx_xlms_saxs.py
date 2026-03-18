import jax.numpy as jnp

from jaxent.src.custom_types.key import m_key
from jaxent.src.opt.run import run_optimise

from ._mixed_model_optimise_helpers import (
    assert_history_has_measurable_loss_drop,
    assert_measurable_loss_decrease,
    assert_output_keys,
    build_mapping_targets_for_three_models,
    build_mixed_simulation,
    compute_total_train_loss,
    make_test_optimiser_settings,
    output_l2_loss,
)


def test_module_run_optimise_hdx_xlms_saxs_reduces_loss_and_exposes_keyed_outputs():
    simulation, models, targets = build_mixed_simulation(
        include_saxs=True,
        include_xlms=True,
        seed=19,
    )
    data_targets = tuple(targets)
    indexes = [0, 1, 2]
    loss_functions = [
        output_l2_loss,
        output_l2_loss,
        output_l2_loss,
    ]
    initial_loss = compute_total_train_loss(
        simulation=simulation,
        data_targets=data_targets,
        indexes=indexes,
        loss_functions=loss_functions,
    )

    optimised_simulation, history = run_optimise(
        simulation=simulation,
        data_to_fit=data_targets,
        config=make_test_optimiser_settings("module_hdx_xlms_saxs"),
        forward_models=models,
        indexes=indexes,
        loss_functions=loss_functions,
    )
    final_loss = compute_total_train_loss(
        simulation=optimised_simulation,
        data_targets=data_targets,
        indexes=indexes,
        loss_functions=loss_functions,
    )

    assert_output_keys(
        optimised_simulation,
        expected_keys=[m_key("HDX_resPF"), m_key("SAXS_Iq"), m_key("XLMS_distance")],
    )
    assert len(optimised_simulation.outputs) == 3
    assert_history_has_measurable_loss_drop(history, expected_components=3)
    assert_measurable_loss_decrease(initial_loss, final_loss)


def test_module_mapping_smoke_extracts_expected_values_for_hdx_saxs_xlms_outputs():
    simulation, _, targets = build_mixed_simulation(
        include_saxs=True,
        include_xlms=True,
        seed=23,
    )
    mappings, _ = build_mapping_targets_for_three_models(*targets)
    hdx_mapping, saxs_mapping, xlms_mapping = mappings

    outputs_by_key = simulation.outputs_by_key
    hdx_prediction = outputs_by_key[m_key("HDX_resPF")].y_pred()
    saxs_prediction = outputs_by_key[m_key("SAXS_Iq")].y_pred()
    xlms_prediction = outputs_by_key[m_key("XLMS_distance")].y_pred()

    mapped_hdx = hdx_mapping.apply(hdx_prediction)
    mapped_saxs = saxs_mapping.apply(saxs_prediction)
    mapped_xlms = xlms_mapping.apply(xlms_prediction)

    expected_saxs = saxs_prediction[saxs_mapping.indices]
    expected_xlms = xlms_prediction[xlms_mapping.indices_i, xlms_mapping.indices_j]

    assert mapped_hdx.shape == hdx_prediction.shape
    assert mapped_saxs.shape[0] == saxs_mapping.indices.shape[0]
    assert mapped_xlms.shape[0] == xlms_mapping.indices_i.shape[0]
    assert bool(jnp.allclose(mapped_saxs, expected_saxs))
    assert bool(jnp.allclose(mapped_xlms, expected_xlms))
