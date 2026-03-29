import jax
import jax.numpy as jnp
from typing import Any
from jax import Array
from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters

def get_previous_grads(opt_state: Any) -> Any:
    return (
        opt_state.gradients
        if opt_state.gradients is not None
        else jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), opt_state.params)
    )

def check_gradient_oscillation(prev_grads: Any, current_grads: Any) -> Any:
    if current_grads is None:
        return 0.0
    grad_dot_product = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(
            lambda a, b: jnp.vdot(a, b),
            prev_grads,
            current_grads,
        ),
    )
    return grad_dot_product

def create_gradient_masks(
    parameter_partition_masks: set[Optimisable_Parameters],
    params: Simulation_Parameters,
    optimisable_funcs: Array | None,
) -> Simulation_Parameters:
    """Creates gradient masks as a Simulation_Parameters instance with integer values.

    Args:
        params: The simulation parameters to create masks for

    Returns:
        A Simulation_Parameters instance containing integer masks (0 or 1) for each parameter
    """
    if optimisable_funcs is None:
        print(
            DeprecationWarning(
                "The optimisable_funcs argument is deprecated and will be removed in future versions"
            )
        )
        optimisable_funcs = jnp.zeros_like(params.forward_model_weights, dtype=jnp.float32)

    # Create masks based on which parameters are enabled for optimization
    frame_mask = (
        1.0 if Optimisable_Parameters.frame_weights in parameter_partition_masks else 0.0
    )
    model_mask = (
        1.0 if Optimisable_Parameters.model_parameters in parameter_partition_masks else 0.0
    )

    mask_mask = 1.0 if Optimisable_Parameters.frame_mask in parameter_partition_masks else 0.0

    if mask_mask == 1.0:
        raise NotImplementedError(
            "Frame mask optimization not fully implemented - while gradients can flow, "
            "frame masking is not applied during weights normalisation before the forward step"
        )

    print(parameter_partition_masks)
    print(f"Masks: frame={frame_mask}, model={model_mask}, frame_mask={mask_mask}")

    # Create frame weights mask
    frame_weights_mask = jax.tree_map(
        lambda x: jnp.full_like(x, frame_mask, dtype=jnp.float32), params.frame_weights
    )

    # Create frame mask mask
    frame_mask_mask = jax.tree_map(
        lambda x: jnp.full_like(x, mask_mask, dtype=jnp.float32), params.frame_mask
    )
    print("Frame mask mask:", frame_mask_mask)

    # Create model parameters mask - handle each Model_Parameters instance separately
    model_parameters_mask = []
    for model_param in params.model_parameters:
        # For each Model_Parameters instance, create a mask of the same structure
        masked_model_param = jax.tree_map(
            lambda x: jnp.full_like(x, model_mask, dtype=jnp.float32), model_param
        )
        model_parameters_mask.append(masked_model_param)

    # In create_parameter_partition_masks
    param_mask = Simulation_Parameters(
        frame_weights=frame_weights_mask,
        frame_mask=frame_mask_mask,
        model_parameters=model_parameters_mask,
        normalise_loss_functions=jnp.zeros_like(
            params.normalise_loss_functions, dtype=jnp.float32
        ),
        forward_model_weights=jnp.zeros_like(params.forward_model_weights, dtype=jnp.float32),
        forward_model_scaling=jnp.zeros_like(params.forward_model_scaling, dtype=jnp.float32),
    )
    print("Original params structure:", jax.tree_util.tree_structure(params))

    print(
        "Mask structure:",
        jax.tree_util.tree_structure(param_mask),
    )
    # raise ValueError("Stop here")
    return param_mask


def mask_gradients(
    grads: Simulation_Parameters, masks: Simulation_Parameters
) -> Simulation_Parameters:
    """Apply masks to gradients. Uses integer masks (0 or 1) instead of booleans.

    Args:
        grads: The gradients to mask
        masks: The integer masks to apply (0 or 1)

    Returns:
        A new Simulation_Parameters instance with masked gradients
    """
    # Mask frame weights - directly multiply by the integer mask
    masked_frame_weights = jax.tree_map(
        lambda g, m: g * m, grads.frame_weights, masks.frame_weights
    )

    # Mask frame mask
    masked_frame_mask = jax.tree_map(lambda g, m: g * m, grads.frame_mask, masks.frame_mask)

    # Mask model parameters - handle each Model_Parameters instance separately
    masked_model_parameters = []
    for grad_param, mask_param in zip(grads.model_parameters, masks.model_parameters):
        # For each pair of Model_Parameters instances, apply the mask
        masked_param = jax.tree_map(lambda g, m: g * m, grad_param, mask_param)
        masked_model_parameters.append(masked_param)

    masked_grads = Simulation_Parameters(
        frame_weights=masked_frame_weights,
        frame_mask=masked_frame_mask,
        model_parameters=masked_model_parameters,
        normalise_loss_functions=masks.normalise_loss_functions,
        forward_model_weights=masks.forward_model_weights,
        forward_model_scaling=masks.forward_model_scaling,
    )
    return masked_grads
