"""Physical observation-covariance containers.

The legacy ``Dataset.covariance_matrix`` is an inverse covariance and is kept
unchanged for reproducibility.  This module deliberately stores covariance
information in its physical form and performs marginalisation before any
factorisation.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["sigma_diag", "chol", "log_det", "cross_block", "conditioner_chol"],
    meta_fields=["kind", "trace_normalised", "n_timepoints"],
)
@dataclass(frozen=True, slots=True)
class ObservationCovariance:
    """A physical covariance, either diagonal by timepoint or stacked.

    ``sigma_diag`` contains variances (the public ``from_diagonal`` constructor
    accepts standard deviations, matching ``sigma_diag_noise``).  Stacked
    matrices use fragment-major, timepoint-minor ordering.
    """

    kind: str
    sigma_diag: Array | None
    chol: Array | None
    log_det: Array
    cross_block: Array | None = None
    trace_normalised: bool = False
    n_timepoints: int | None = None
    conditioner_chol: Array | None = None

    def __post_init__(self):
        if self.kind not in ("diag", "stacked"):
            raise ValueError(f"Unknown covariance kind: {self.kind!r}")
        if self.kind == "diag" and self.sigma_diag is None:
            raise ValueError("Diagonal covariance requires sigma_diag")
        if self.kind == "stacked" and self.chol is None:
            raise ValueError("Stacked covariance requires chol")

    @classmethod
    def from_diagonal(cls, sigma_pt: Array) -> "ObservationCovariance":
        sigma_pt = jnp.asarray(sigma_pt)
        if sigma_pt.ndim != 2:
            raise ValueError("sigma_pt must have shape (n_fragments, n_timepoints)")
        if not bool(jnp.all(jnp.isfinite(sigma_pt))) or not bool(jnp.all(sigma_pt > 0)):
            raise ValueError("sigma_pt must be finite and strictly positive")
        variances = sigma_pt**2
        return cls(
            kind="diag", sigma_diag=variances, chol=None,
            log_det=jnp.sum(jnp.log(variances)), n_timepoints=sigma_pt.shape[1],
        )

    @classmethod
    def from_stacked(
        cls, Sigma: Array, nugget: float = 0.0, *, cross_block: Array | None = None,
        n_timepoints: int | None = None,
    ) -> "ObservationCovariance":
        Sigma = jnp.asarray(Sigma)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must be square")
        Sigma = (Sigma + Sigma.T) / 2 + nugget * jnp.eye(Sigma.shape[0], dtype=Sigma.dtype)
        chol = jnp.linalg.cholesky(Sigma)
        if not bool(jnp.all(jnp.isfinite(chol))):
            raise ValueError("Covariance factorization failed: Sigma is not positive definite")
        log_det = 2 * jnp.sum(jnp.log(jnp.diag(chol)))
        if not bool(jnp.isfinite(log_det)):
            raise ValueError("Covariance factorization produced a non-finite log determinant")
        return cls(
            kind="stacked", sigma_diag=None, chol=chol, log_det=log_det,
            cross_block=cross_block, n_timepoints=n_timepoints,
        )

    @property
    def covariance(self) -> Array:
        if self.kind == "diag":
            return jnp.diag(self.sigma_diag.reshape(-1))
        return self.chol @ self.chol.T

    @property
    def eigenvalues(self) -> Array:
        return jnp.linalg.eigvalsh(self.covariance)

    @property
    def condition_number(self) -> Array:
        eig = self.eigenvalues
        return jnp.max(eig) / jnp.min(eig)

    @property
    def effective_rank(self) -> Array:
        eig = self.eigenvalues
        return jnp.sum(eig > jnp.max(eig) * 1e-12)

    def _fragment_time_indices(self, indices: Array) -> Array:
        if self.n_timepoints is None:
            # A covariance constructed without temporal metadata is treated as
            # one observation per fragment.  Dataset integration supplies the
            # temporal block size explicitly for real stacked HDX matrices.
            return jnp.asarray(indices, dtype=jnp.int32).ravel()
        indices = jnp.asarray(indices, dtype=jnp.int32).ravel()
        return (indices[:, None] * self.n_timepoints + jnp.arange(self.n_timepoints)).reshape(-1)

    def subset(self, train_indices: Array, val_indices: Array):
        """Return marginal train/validation covariances.

        For stacked covariances this selects ``Sigma_SS`` and factorizes it;
        it never selects a block of the inverse precision.
        """
        if self.kind == "diag":
            train = ObservationCovariance(
                kind="diag", sigma_diag=self.sigma_diag[jnp.asarray(train_indices)],
                chol=None, log_det=jnp.sum(jnp.log(self.sigma_diag[jnp.asarray(train_indices)])),
                n_timepoints=self.sigma_diag.shape[1],
            )
            val = ObservationCovariance(
                kind="diag", sigma_diag=self.sigma_diag[jnp.asarray(val_indices)],
                chol=None, log_det=jnp.sum(jnp.log(self.sigma_diag[jnp.asarray(val_indices)])),
                n_timepoints=self.sigma_diag.shape[1],
            )
            return train, val

        ti = self._fragment_time_indices(train_indices)
        vi = self._fragment_time_indices(val_indices)
        cov = self.covariance
        train = ObservationCovariance.from_stacked(cov[jnp.ix_(ti, ti)], n_timepoints=self.n_timepoints)
        val = ObservationCovariance.from_stacked(
            cov[jnp.ix_(vi, vi)], n_timepoints=self.n_timepoints,
            cross_block=cov[jnp.ix_(vi, ti)],
        )
        object.__setattr__(val, "conditioner_chol", train.chol)
        return train, val

    def conditional(self, y_train_residual: Array):
        """Return conditional validation mean and covariance given train residuals."""
        if self.cross_block is None or self.conditioner_chol is None:
            raise ValueError("Conditional scoring requires a covariance returned by subset()")
        cross = self.cross_block
        solve = jax.scipy.linalg.cho_solve((self.conditioner_chol, True), y_train_residual)
        mean = cross @ solve
        covariance = self.covariance - cross @ jax.scipy.linalg.cho_solve(
            (self.conditioner_chol, True), cross.T
        )
        return mean, ObservationCovariance.from_stacked(
            covariance, n_timepoints=self.n_timepoints
        )
