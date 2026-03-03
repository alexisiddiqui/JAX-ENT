from __future__ import annotations

import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence KL(p||q) between two distributions.

    Distributions are normalised internally. *eps* prevents ``log(0)``.

    .. note:: This is the **single most duplicated function** in the examples
       codebase — 11 active copies.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / np.sum(p)
    q = q / np.sum(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective sample size ``1 / sum(w²)`` for normalised *weights*.

    6 active copies across example scripts.
    """
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return float(1.0 / np.sum(w ** 2))
