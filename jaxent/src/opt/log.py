import logging
from typing import Any, Optional

import jax.numpy as jnp

from jaxent.src.opt.track import ConvergenceTracker

LOGGER = logging.getLogger("jaxent.opt")


def _logger_or_default(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else LOGGER


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def log_optimization_step(
    step: int,
    n_steps: int,
    current_loss: Any,
    raw_delta: Any,
    prev_params: Optional[Any],
    opt_state: Any,
    grad_dot_product: Any,
    tracker: ConvergenceTracker,
    optimizer: Any,
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> None:
    if silent:
        return

    _logger = _logger_or_default(logger)
    if prev_params is None:
        prev_params = opt_state.params

    opt_state_param_frameweight_delta = jnp.linalg.norm(
        opt_state.params.frame_weights - prev_params.frame_weights
    )

    ema_delta = tracker.ema_loss_delta if tracker.ema_loss_delta is not None else 0.0
    rel_conv = tracker.get_relative_convergence(current_loss)

    if hasattr(optimizer, "current_learning_rate"):
        learning_rate = optimizer.current_learning_rate
    else:
        learning_rate = optimizer.learning_rate

    _logger.info(
        "Step %s/%s Loss %.6e EMAΔ %.4e RawΔ %.4e RelConv %.6e "
        "Threshold %s/%s (%.6e) OptΔ %.4e GradDot %.4e LR %.4e",
        step,
        n_steps,
        _as_float(current_loss),
        _as_float(ema_delta),
        _as_float(raw_delta),
        _as_float(rel_conv),
        tracker.current_threshold_idx + 1,
        len(tracker.convergence_thresholds),
        tracker.current_threshold,
        _as_float(opt_state_param_frameweight_delta),
        _as_float(grad_dot_product),
        float(learning_rate),
    )


def log_oscillation_warning(
    step: int,
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> None:
    if silent:
        return
    _logger_or_default(logger).warning(
        "Gradient dot product negative at step %s; possible oscillation.", step
    )


def print_optimization_summary(
    step: int,
    total_time: float,
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> None:
    if silent:
        return
    _logger = _logger_or_default(logger)
    avg_iteration_time = total_time / (step + 1) if step >= 0 else 0
    iterations_per_second = (step + 1) / total_time if total_time > 0 else 0

    _logger.info(
        "Optimization loop complete: iterations=%s total_time=%.2fs avg_iter=%.4fs iter_per_sec=%.2f",
        step + 1,
        total_time,
        avg_iteration_time,
        iterations_per_second,
    )


def format_optimization_error(
    e: Exception,
    simulation: Any,
    save_state: Optional[Any],
    ema_params: Optional[Any],
    opt_state: Any,
) -> str:
    error_msg = f"Optimization failed due to an error: {e}. Returning best state from history."
    error_details = [
        "\n" * 2,
        "Simulation parameters at failure: ",
        str(simulation.params),
        "\n" * 2,
    ]
    if save_state is not None:
        error_details.extend(
            [
                "Latest save state at failure: ",
                str(save_state.params),
                "\n" * 2,
            ]
        )
    error_details.extend(
        [
            "Latest EMA params state at failure: ",
            str(ema_params),
            "\n" * 2,
            "Opt State parameters at failure: ",
            str(opt_state.params),
            "\n" * 2,
        ]
    )
    return error_msg + "".join(str(d) for d in error_details)


def log_final_states(
    simulation: Any,
    save_state: Optional[Any],
    ema_params: Optional[Any],
    opt_state: Any,
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> None:
    if silent:
        return
    _logger = _logger_or_default(logger)
    _logger.info("Simulation parameters at end: %s", simulation.params)
    _logger.info(
        "Latest save state at end: %s",
        save_state.params if save_state is not None else "None",
    )
    _logger.info("Latest EMA params state at end: %s", ema_params)
    _logger.info("Opt State parameters at end: %s", opt_state.params)


def log_threshold_met(
    step: int,
    current_loss: Any,
    tracker: ConvergenceTracker,
    optimizer: Any,
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> None:
    if silent:
        return
    _logger = _logger_or_default(logger)
    _logger.info(
        "Relative threshold %s/%s met at step %s",
        tracker.current_threshold_idx + 1,
        len(tracker.convergence_thresholds),
        step,
    )
    _logger.info(
        "Relative convergence %.8e threshold %.2e",
        tracker.get_relative_convergence(current_loss),
        tracker.current_threshold,
    )
    if optimizer.ema_history is not None and optimizer.ema_history.states:
        ema_loss = optimizer.ema_history.states[-1].losses.total_train_loss
        _logger.info("Updated EMA history loss %.6e", _as_float(ema_loss))
    _logger.info("EMA params: %s", tracker.ema_params)
