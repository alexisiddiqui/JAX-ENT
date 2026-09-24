import jax
import jax.numpy as jnp
import numpy as np

from jaxent.src.custom_types.base import FrameAveragingMode, ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import (
    BV_Model_Parameters,
    BVRateDistributionParameters,
    linear_BV_Model_Parameters,
)


def _time_scale(kint_unit: str, time_unit: str) -> float:
    """Scale configured times into the reciprocal unit used by k_ints."""
    if (kint_unit, time_unit) in {("s^-1", "s"), ("min^-1", "min")}:
        return 1.0
    if (kint_unit, time_unit) == ("s^-1", "min"):
        return 60.0
    if (kint_unit, time_unit) == ("min^-1", "s"):
        return 1.0 / 60.0
    raise ValueError(f"incompatible rate/time units: {kint_unit!r}, {time_unit!r}")


def _stable_uptake(exposure):
    return -jnp.expm1(-jnp.maximum(exposure, 0.0))


# fix the typing to use jax arrays
class BV_ForwardPass(ForwardPass[BV_input_features, BV_output_features, BV_Model_Parameters]):
    frame_averaging_mode: FrameAveragingMode = "log_pf"

    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> BV_output_features:
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # print("Model parameters bc, bh:", bc, bh)

        # Convert lists to numpy arrays for computation
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        # print("Contact shapes:", heavy_contacts.shape, acceptor_contacts.shape)
        # print("Sample contacts:", heavy_contacts[0, :5], acceptor_contacts[0, :5])

        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)

        # Convert back to list for output
        log_pf_list = log_pf
        # print("Calculated log_pf:", log_pf[:5])

        return BV_output_features(log_Pf=log_pf_list, k_ints=None)


class BV_uptake_ForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, BV_Model_Parameters]
):
    frame_averaging_mode: FrameAveragingMode = "log_pf"

    def __init__(
        self,
        frame_averaging_mode: FrameAveragingMode = "log_pf",
        frame_groups=None,
    ) -> None:
        self.frame_averaging_mode = frame_averaging_mode
        self.frame_group_masks: tuple[np.ndarray, ...] | None = None
        if frame_groups is not None:
            self.set_frame_groups(frame_groups)

    def set_frame_groups(self, assignments) -> None:
        assignments = np.asarray(assignments)
        if assignments.ndim != 1:
            raise ValueError("frame-group assignments must be one-dimensional")
        self.frame_group_masks = tuple(
            assignments == label for label in np.unique(assignments)
        )

    def average_frames(
        self,
        input_features: BV_input_features,
        parameters: BV_Model_Parameters,
        frame_weights,
        implementation: str = "tensordot",
    ) -> uptake_BV_output_features:
        del implementation  # both specialised reductions are explicit dot products
        bc, bh = parameters.bv_bc, parameters.bv_bh
        log_pf = (
            bc * jnp.asarray(input_features.heavy_contacts)
            + bh * jnp.asarray(input_features.acceptor_contacts)
        )
        k_ints = jnp.asarray(input_features.k_ints)
        timepoints = jnp.asarray(parameters.timepoints).reshape(-1)
        rates = k_ints[:, None] * jnp.exp(-log_pf)

        if self.frame_averaging_mode == "rate":
            mean_rate = rates @ frame_weights
            uptake = 1.0 - jnp.exp(-timepoints[:, None] * mean_rate[None, :])
            return uptake_BV_output_features(uptake)
        if self.frame_averaging_mode == "frame_uptake":
            uptake = jax.vmap(
                lambda timepoint: (1.0 - jnp.exp(-timepoint * rates))
                @ frame_weights
            )(timepoints)
            return uptake_BV_output_features(uptake)
        if self.frame_averaging_mode != "uptake":
            raise ValueError(
                f"average_frames cannot handle {self.frame_averaging_mode!r}"
            )
        if self.frame_group_masks is None:
            raise ValueError("uptake mode requires configured frame-group assignments")

        uptake = jnp.zeros((timepoints.shape[0], rates.shape[0]), dtype=rates.dtype)
        for mask_array in self.frame_group_masks:
            mask = jnp.asarray(mask_array)
            group_weights = jnp.where(mask, frame_weights, 0.0)
            mass = jnp.sum(group_weights)
            safe_mass = jnp.where(mass > 0, mass, 1.0)
            group_rate = rates @ (group_weights / safe_mass)
            group_uptake = 1.0 - jnp.exp(
                -timepoints[:, None] * group_rate[None, :]
            )
            uptake = uptake + jnp.where(mass > 0, mass * group_uptake, 0.0)
        return uptake_BV_output_features(uptake)

    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        # Extract model parameters
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # Convert inputs to JAX arrays
        # heavy_contacts and acceptor_contacts are (n_residues, n_frames)
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        kints = jnp.asarray(input_features.k_ints)  # (n_residues,)
        time_points = jnp.asarray(parameters.timepoints).reshape(-1)  # (n_timepoints,)

        # Compute protection factors per frame: (n_residues, n_frames)
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        pf = jnp.exp(log_pf)  # (n_residues, n_frames)

        # Select kints shape based on pf dimensionality:
        #   pf 1-D (n_residues,) → features were pre-averaged in log-PF mode.
        if pf.ndim == 1:
            kints_for_uptake = kints                            # (n_residues,)
        else:
            kints_for_uptake = jnp.expand_dims(kints, axis=-1)  # (n_residues, 1)

        # Reshape time_points to broadcast over residue (and optional frame) dims without vmap.
        # (n_timepoints,) → (n_timepoints, 1) or (n_timepoints, 1, 1)
        time_reshaped = time_points[(slice(None),) + (None,) * pf.ndim]

        # uptake_per_timepoint: (n_timepoints, n_residues) or (n_timepoints, n_residues, n_frames)
        uptake_per_timepoint = 1 - jnp.exp(-kints_for_uptake * time_reshaped / pf)

        return uptake_BV_output_features(uptake_per_timepoint)


class linear_BV_ForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, linear_BV_Model_Parameters]
):
    """Additive conditional uptake with one global pair of BV slopes.

    Contacts are averaged once across frames during optimisation. Each interval
    then adds a positive conditional hazard, which makes uptake monotone, smooth,
    and bounded without fitting a separate contact coefficient per timepoint.
    """
    frame_averaging_mode: FrameAveragingMode = "linear_uptake"
    key = m_key("HDX_peptide")

    def average_frames(
        self, input_features, parameters, frame_weights, implementation="tensordot"
    ) -> uptake_BV_output_features:
        del implementation
        averaged = BV_input_features(
            heavy_contacts=jnp.asarray(input_features.heavy_contacts) @ frame_weights,
            acceptor_contacts=jnp.asarray(input_features.acceptor_contacts) @ frame_weights,
            k_ints=input_features.k_ints,
        )
        return self(averaged, parameters)

    def __call__(
        self, input_features: BV_input_features, parameters: linear_BV_Model_Parameters
    ) -> uptake_BV_output_features:
        heavy = jnp.asarray(input_features.heavy_contacts)
        acceptor = jnp.asarray(input_features.acceptor_contacts)
        if input_features.k_ints is None:
            raise ValueError("linear BV uptake requires intrinsic rates")
        k_ints = jnp.asarray(input_features.k_ints)
        if k_ints.ndim != 1:
            raise ValueError("k_ints must be a one-dimensional residue array")

        z = parameters.bv_bc * heavy + parameters.bv_bh * acceptor
        times = jnp.asarray(parameters.timepoints) * _time_scale(
            parameters.kint_unit, parameters.time_unit
        )
        intervals = jnp.diff(jnp.concatenate((jnp.zeros(1, dtype=times.dtype), times)))
        kint_shape = (k_ints.shape[0],) + (1,) * (z.ndim - 1)
        base_hazard = k_ints.reshape(kint_shape) * jnp.exp(-z)
        # Leading time axis, followed by residue and optional frame axes.
        interval_shape = (len(parameters.timepoints),) + (1,) * base_hazard.ndim
        multiplier = intervals * jnp.exp(parameters.interval_offsets)
        interval_hazard = multiplier.reshape(interval_shape) * base_hazard[None, ...]
        uptake = _stable_uptake(jnp.cumsum(interval_hazard, axis=0))
        return uptake_BV_output_features(uptake=uptake)


class BVRateDistributionForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, BVRateDistributionParameters]
):
    """Frame-coupled soft-mixture or Gamma moment closure for BV rates."""

    frame_averaging_mode: FrameAveragingMode = "rate_distribution"
    key = m_key("HDX_peptide")

    @staticmethod
    def _rates(input_features, parameters):
        heavy = jnp.asarray(input_features.heavy_contacts)
        acceptor = jnp.asarray(input_features.acceptor_contacts)
        if input_features.k_ints is None:
            raise ValueError("BV rate-distribution uptake requires intrinsic rates")
        z = parameters.bv_bc * heavy + parameters.bv_bh * acceptor
        k_ints = jnp.asarray(input_features.k_ints)
        kint_shape = (k_ints.shape[0],) + (1,) * (z.ndim - 1)
        rates = k_ints.reshape(kint_shape) * jnp.exp(-z)
        return z, rates

    def __call__(self, input_features, parameters) -> uptake_BV_output_features:
        """Return exact per-frame uptake for diagnostic prediction."""
        _, rates = self._rates(input_features, parameters)
        times = jnp.asarray(parameters.timepoints) * _time_scale(
            parameters.kint_unit, parameters.time_unit
        )
        uptake = _stable_uptake(
            times.reshape((-1,) + (1,) * rates.ndim) * rates[None, ...]
        )
        return uptake_BV_output_features(uptake)

    def average_frames(
        self, input_features, parameters, frame_weights, implementation="tensordot"
    ) -> uptake_BV_output_features:
        del implementation
        z, rates = self._rates(input_features, parameters)
        weights = jnp.asarray(frame_weights)
        times = jnp.asarray(parameters.timepoints) * _time_scale(
            parameters.kint_unit, parameters.time_unit
        )

        if parameters.backend == "soft_mixture":
            support = parameters.support_points
            tau = parameters.assignment_bandwidth
            logits = -0.5 * jnp.square((z[..., None] - support) / tau)
            assignments = jax.nn.softmax(logits, axis=-1)
            masses = jnp.einsum("rfq,f->rq", assignments, weights)
            component_rates = jnp.asarray(input_features.k_ints)[:, None] * jnp.exp(-support)
            component_uptake = _stable_uptake(
                times[:, None, None] * component_rates[None, :, :]
            )
            uptake = jnp.einsum("rq,trq->tr", masses, component_uptake)
            return uptake_BV_output_features(uptake)

        mean_rate = rates @ weights
        centered = rates - mean_rate[:, None]
        variance = jnp.sum(weights[None, :] * jnp.square(centered), axis=1)
        mean_safe = jnp.maximum(mean_rate, jnp.finfo(mean_rate.dtype).tiny)
        relative_variance = variance / jnp.square(mean_safe)
        variance_safe = jnp.maximum(
            variance, jnp.finfo(variance.dtype).eps * jnp.square(mean_safe)
        )
        gamma_survival = jnp.exp(
            -(jnp.square(mean_safe) / variance_safe)[None, :]
            * jnp.log1p(times[:, None] * variance_safe[None, :] / mean_safe[None, :])
        )
        exponential_survival = jnp.exp(-times[:, None] * mean_safe[None, :])
        survival = jnp.where(
            relative_variance[None, :] <= jnp.sqrt(jnp.finfo(variance.dtype).eps),
            exponential_survival,
            gamma_survival,
        )
        return uptake_BV_output_features(1.0 - survival)
