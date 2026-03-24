"""Shared utilities for CaM pulldown fitting scripts."""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from jaxent.src.opt.loss.base import create_functional_loss


def load_dat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAXS .dat file, handling SASBDB multi-line text headers.

    Returns:
        (q_values, intensities, errors) as numpy arrays of shape (n_q,)
    """
    with open(path, "r") as f:
        lines = f.readlines()

    skiprows = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            try:
                float(parts[0])
                skiprows = i
                break
            except ValueError:
                continue

    dat = np.loadtxt(path, skiprows=skiprows)
    return dat[:, 0], dat[:, 1], dat[:, 2]


def find_q_indices(full_q: np.ndarray, subset_q: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Find integer indices of subset_q values in full_q array."""
    return np.array([np.where(np.abs(full_q - q) < tol)[0][0] for q in subset_q])


def create_mse_loss():
    """Create MSE loss function."""
    return create_functional_loss(
        lambda p, t: jnp.mean((p - t) ** 2),
        post_mean=False,
        flatten=True,
    )


def create_chi2_loss():
    """Create Chi-squared loss function (variance-normalised MSE)."""
    def chi2_fn(p, t):
        var_est = jnp.var(t) + 1e-10
        return jnp.mean((p - t) ** 2 / var_est)

    return create_functional_loss(chi2_fn, post_mean=False, flatten=True)
