from __future__ import annotations

import numpy as np
from .convergence_labels import find_best_labeled_state


def find_best_convergence_threshold(history) -> tuple:
    """Find the actual convergence threshold with the lowest validation loss.

    Parameters
    ----------
    history:
        An ``OptimizationHistory`` object with a ``.states`` attribute.

    Returns
    -------
    ``(best_step_idx, best_val_loss, best_state)``
    """
    if history is None or not getattr(history, "convergence_states", None):
        return None, np.inf, None
    threshold, state = find_best_labeled_state(history)
    if state is None:
        return None, np.inf, None
    return threshold, float(state.losses.val_losses[0]), state
