from jaxent.src.custom_types.key import m_key
from jaxent.src.opt.run import run_optimise

from ._mixed_model_optimise_helpers import (
    assert_history_has_measurable_loss_drop,
    assert_measurable_loss_decrease,
    assert_output_keys,
    build_mixed_simulation,
    compute_total_train_loss,
    make_test_optimiser_settings,
    output_l2_loss,
)


def test_module_run_optimise_hdx_xlms_reduces_loss_and_exposes_keyed_outputs():
    simulation, models, targets = build_mixed_simulation(
        include_saxs=False,
        include_xlms=True,
        seed=11,
    )
    data_targets = tuple(targets)
    indexes = [0, 1]
    loss_functions = [output_l2_loss, output_l2_loss]
    initial_loss = compute_total_train_loss(
        simulation=simulation,
        data_targets=data_targets,
        indexes=indexes,
        loss_functions=loss_functions,
    )

    optimised_simulation, history = run_optimise(
        simulation=simulation,
        data_to_fit=data_targets,
        config=make_test_optimiser_settings("module_hdx_xlms"),
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
        expected_keys=[m_key("HDX_resPF"), m_key("XLMS_distance")],
    )
    assert len(optimised_simulation.outputs) == 2
    assert_history_has_measurable_loss_drop(history, expected_components=2)
    assert_measurable_loss_decrease(initial_loss, final_loss)
