import jax
import jax.numpy as jnp
from typing import Optional, Any
from jaxent.src.opt.track import ConvergenceTracker

def log_optimization_step(
    step: int,
    n_steps: int,
    current_loss: Any,
    raw_delta: Any,
    prev_params: Optional[Any],
    opt_state: Any,
    grad_dot_product: Any,
    tracker: ConvergenceTracker,
    optimizer: Any
) -> None:
    if prev_params is None:
        prev_params = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e0), opt_state.params)

    opt_state_param_frameweight_delta = jnp.linalg.norm(
        opt_state.params.frame_weights - prev_params.frame_weights
    )

    ema_delta = tracker.ema_loss_delta if tracker.ema_loss_delta is not None else 0.0
    rel_conv = tracker.get_relative_convergence(current_loss)
    
    jax.debug.print(
        fmt=" ".join(
            [
                "Step {step}/{n_steps}",
                "Loss: {current_loss:.6e}",
                "EMA Δ: {ema_delta:.4e}",
                "Raw Δ: {raw_delta:.4e}",
                "Rel Conv: {rel_conv:.6e}",
                "Threshold {threshold_idx}/{total_thresholds} ({current_threshold:.6e})",
                "Opt State Δ: {opt_state_delta:.4e}",
                "Grad Dot Prod: {grad_dot_product:.4e}",
                "LR: {learning_rate:.4e}",
            ]
        ),
        step=step,
        n_steps=n_steps,
        current_loss=current_loss,
        ema_delta=ema_delta,
        raw_delta=raw_delta,
        rel_conv=rel_conv,
        opt_state_delta=opt_state_param_frameweight_delta,
        grad_dot_product=grad_dot_product,
        learning_rate=optimizer.lr_schedule(),
        threshold_idx=tracker.current_threshold_idx + 1,
        total_thresholds=len(tracker.convergence_thresholds),
        current_threshold=tracker.current_threshold,
    )

def log_oscillation_warning(step: int) -> None:
    print(f"Warning: Gradient dot product negative at step {step}, possible oscillation.")
    
def print_optimization_summary(step: int, total_time: float) -> None:
    avg_iteration_time = total_time / (step + 1) if step >= 0 else 0
    iterations_per_second = (step + 1) / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("Optimization Loop Performance:")
    print(f"  Total iterations completed: {step + 1}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per iteration: {avg_iteration_time:.4f}s")
    print(f"  Iterations per second: {iterations_per_second:.2f}")
    print("=" * 50 + "\n")

def format_optimization_error(
    e: Exception, 
    simulation: Any, 
    save_state: Optional[Any], 
    ema_params: Optional[Any], 
    opt_state: Any
) -> str:
    error_msg = f"Optimization failed due to an error: {e}. Returning best state from history."
    error_details = [
        "\n" * 10,
        "Simulation parameters at failure: ",
        str(simulation.params),
        "\n" * 10,
    ]
    if save_state is not None:
        error_details.extend([
            "Latest save state at failure: ",
            str(save_state.params),
            "\n" * 10,
        ])
    error_details.extend([
        "Latest EMA params state at failure: ",
        str(ema_params),
        "\n" * 10,
        "Opt State parameters at failure: ",
        str(opt_state.params),
        "\n" * 10,
    ])
    return error_msg + "".join(str(d) for d in error_details)

def log_final_states(
    simulation: Any,
    save_state: Optional[Any],
    ema_params: Optional[Any],
    opt_state: Any
) -> None:
    print(
        "\\n" * 10,
        "Simulation parameters at end: ",
        simulation.params,
        "\\n" * 10,
        "Latest save state at end: ",
        save_state.params if save_state is not None else "None",
        "\\n" * 10,
        "Latest EMA params state at end: ",
        ema_params,
        "\\n" * 10,
        "Opt State parameters at end: ",
        opt_state.params,
        "\n" * 10,
    )

def log_threshold_met(
    step: int,
    current_loss: Any,
    tracker: ConvergenceTracker,
    optimizer: Any
) -> None:
    print(f"Relative threshold {tracker.current_threshold_idx + 1}/{len(tracker.convergence_thresholds)} met at step {step}")
    print(f"Relative convergence: {tracker.get_relative_convergence(current_loss):.8e}, threshold: {tracker.current_threshold:.2e}")
    ema_loss = optimizer.ema_history.states[-1].losses.total_train_loss
    print(f"Updated History and computed loss from EMA params. EMA param Loss: {ema_loss:.6e}")
    print(f"EMA Params: {tracker.ema_params}")
