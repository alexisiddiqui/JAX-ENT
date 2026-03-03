from __future__ import annotations

import numpy as np


def find_best_convergence_threshold(history) -> tuple:
    """Find the convergence step with the lowest validation loss.

    Parameters
    ----------
    history:
        An ``OptimizationHistory`` object with a ``.states`` attribute.

    Returns
    -------
    ``(best_step_idx, best_val_loss, best_state)``
    """
    if history is None or not getattr(history, "states", None):
        return None, np.inf, None

    best_val_loss = np.inf
    best_step_idx = 0
    best_state = None

    for step_idx, state in enumerate(history.states):
        if hasattr(state, "losses") and hasattr(state.losses, "val_losses"):
            val_losses = state.losses.val_losses
            if val_losses is not None and len(val_losses) > 0:
                val_loss = float(val_losses[0])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step_idx = step_idx
                    best_state = state

    return best_step_idx, best_val_loss, best_state
