from collections.abc import Sequence
from dataclasses import dataclass, field
from beartype.typing import ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Model_Parameters

_POSITIVE_EPS = 1e-6


def _inverse_softplus(value: Array | float) -> Array:
    value = jnp.maximum(jnp.asarray(value), _POSITIVE_EPS)
    return value + jnp.log(-jnp.expm1(-value))


def _validated_times_and_units(
    timepoints, kint_unit: str, time_unit: str
) -> tuple[float, ...]:
    if kint_unit not in {"s^-1", "min^-1"}:
        raise ValueError("kint_unit must be 's^-1' or 'min^-1'")
    if time_unit not in {"s", "min"}:
        raise ValueError("time_unit must be 's' or 'min'")
    times = tuple(float(value) for value in timepoints)
    if not times or any(value <= 0 for value in times):
        raise ValueError("timepoints must be non-empty and strictly positive")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timepoints must be strictly increasing")
    return times


@dataclass(frozen=True, slots=True)
class BV_Model_Parameters(Model_Parameters):
    """Parameters for the original BV model."""

    bv_bc: Array = field(default_factory=lambda: jnp.array([0.35]))
    bv_bh: Array = field(default_factory=lambda: jnp.array([2.0]))
    key = frozenset({m_key("HDX_resPF"), m_key("HDX_peptide")})
    temperature: float = 300.0
    timepoints: Sequence[float] | Array | None = field(
        default_factory=lambda: (0.167, 1.0, 10.0)
    )
    static_params: ClassVar[set[str]] = {"temperature", "key", "timepoints"}

    def __post_init__(self):
        if self.timepoints is not None and not isinstance(self.timepoints, tuple):
            object.__setattr__(
                self, "timepoints", tuple(float(value) for value in self.timepoints)
            )

    def __mul__(self, scalar: float | Array) -> "BV_Model_Parameters":
        scalar = jnp.asarray(scalar)
        return BV_Model_Parameters(
            bv_bc=self.bv_bc * scalar,
            bv_bh=self.bv_bh * scalar,
            timepoints=self.timepoints,
            temperature=self.temperature,
        )

    __rmul__ = __mul__

    def __sub__(self, other: "BV_Model_Parameters") -> "BV_Model_Parameters":
        return BV_Model_Parameters(
            bv_bc=self.bv_bc - other.bv_bc,
            bv_bh=self.bv_bh - other.bv_bh,
            timepoints=self.timepoints,
            temperature=self.temperature,
        )

    def update_parameters(
        self, new_params: "BV_Model_Parameters"
    ) -> "BV_Model_Parameters":
        return BV_Model_Parameters(
            bv_bc=new_params.bv_bc,
            bv_bh=new_params.bv_bh,
            temperature=self.temperature,
            timepoints=self.timepoints,
        )


register_pytree_node(
    BV_Model_Parameters,
    BV_Model_Parameters.tree_flatten,
    BV_Model_Parameters.tree_unflatten,
)


@dataclass(frozen=True, slots=True, init=False)
class linear_BV_Model_Parameters(Model_Parameters):
    """Unconstrained parameters for the additive interval-hazard BV model.

    The legacy ``bv_bc=`` and ``bv_bh=`` arguments are accepted as physical
    values. Optimizers see ``raw_bv_bc`` and ``raw_bv_bh``; the corresponding
    properties are strictly positive.
    """

    raw_bv_bc: Array
    raw_bv_bh: Array
    interval_offsets: Array
    key = frozenset({m_key("HDX_peptide")})
    temperature: float = 300.0
    timepoints: tuple[float, ...] = (0.167, 1.0, 10.0)
    kint_unit: str = "s^-1"
    time_unit: str = "min"
    static_params: ClassVar[set[str]] = {
        "temperature",
        "timepoints",
        "kint_unit",
        "time_unit",
        "key",
    }

    def __init__(
        self,
        raw_bv_bc=None,
        raw_bv_bh=None,
        interval_offsets=None,
        *,
        bv_bc=None,
        bv_bh=None,
        temperature: float = 300.0,
        timepoints=(0.167, 1.0, 10.0),
        kint_unit: str = "s^-1",
        time_unit: str = "min",
    ):
        if raw_bv_bc is not None and bv_bc is not None:
            raise ValueError("Specify either raw_bv_bc or physical bv_bc, not both")
        if raw_bv_bh is not None and bv_bh is not None:
            raise ValueError("Specify either raw_bv_bh or physical bv_bh, not both")
        times = _validated_times_and_units(timepoints, kint_unit, time_unit)
        raw_bc = (
            _inverse_softplus(0.35 if bv_bc is None else bv_bc)
            if raw_bv_bc is None
            else jnp.asarray(raw_bv_bc)
        )
        raw_bh = (
            _inverse_softplus(2.0 if bv_bh is None else bv_bh)
            if raw_bv_bh is None
            else jnp.asarray(raw_bv_bh)
        )
        offsets = (
            jnp.zeros(len(times))
            if interval_offsets is None
            else jnp.asarray(interval_offsets)
        )
        if offsets.ndim != 1 or offsets.shape[0] != len(times):
            raise ValueError("interval_offsets must contain one value per timepoint")
        object.__setattr__(self, "raw_bv_bc", raw_bc)
        object.__setattr__(self, "raw_bv_bh", raw_bh)
        object.__setattr__(self, "interval_offsets", offsets)
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "timepoints", times)
        object.__setattr__(self, "kint_unit", kint_unit)
        object.__setattr__(self, "time_unit", time_unit)

    @property
    def bv_bc(self) -> Array:
        return jax.nn.softplus(self.raw_bv_bc)

    @property
    def bv_bh(self) -> Array:
        return jax.nn.softplus(self.raw_bv_bh)

    def regularization_loss(self, beta_reference=(0.35, 2.0)) -> Array:
        beta = jnp.stack((self.bv_bc, self.bv_bh))
        reference = jnp.asarray(beta_reference, dtype=beta.dtype)
        return jnp.mean(jnp.square(jnp.log(beta / reference))) + jnp.mean(
            jnp.square(self.interval_offsets)
        )


register_pytree_node(
    linear_BV_Model_Parameters,
    linear_BV_Model_Parameters.tree_flatten,
    linear_BV_Model_Parameters.tree_unflatten,
)


@dataclass(frozen=True, slots=True, init=False)
class BVRateDistributionParameters(Model_Parameters):
    """Unconstrained parameters for a frame-coupled BV rate distribution."""

    raw_bv_bc: Array
    raw_bv_bh: Array
    raw_support_gaps: Array
    key = frozenset({m_key("HDX_peptide")})
    temperature: float = 300.0
    timepoints: tuple[float, ...] = (0.167, 1.0, 10.0)
    backend: str = ""
    support_reference: tuple[float, ...] = ()
    bandwidth_floor: float = 0.05
    kint_unit: str = "s^-1"
    time_unit: str = "min"
    static_params: ClassVar[set[str]] = {
        "temperature",
        "timepoints",
        "backend",
        "support_reference",
        "bandwidth_floor",
        "kint_unit",
        "time_unit",
        "key",
    }

    def __init__(
        self,
        *,
        backend: str,
        raw_bv_bc=None,
        raw_bv_bh=None,
        raw_support_gaps=None,
        bv_bc=None,
        bv_bh=None,
        support_points=None,
        support_reference=None,
        bandwidth_floor: float = 0.05,
        temperature: float = 300.0,
        timepoints=(0.167, 1.0, 10.0),
        kint_unit: str = "s^-1",
        time_unit: str = "min",
    ):
        if backend not in {"soft_mixture", "gamma_moments"}:
            raise ValueError("backend must be 'soft_mixture' or 'gamma_moments'")
        if raw_bv_bc is not None and bv_bc is not None:
            raise ValueError("Specify either raw_bv_bc or physical bv_bc, not both")
        if raw_bv_bh is not None and bv_bh is not None:
            raise ValueError("Specify either raw_bv_bh or physical bv_bh, not both")
        times = _validated_times_and_units(timepoints, kint_unit, time_unit)
        raw_bc = (
            _inverse_softplus(0.35 if bv_bc is None else bv_bc)
            if raw_bv_bc is None
            else jnp.asarray(raw_bv_bc)
        )
        raw_bh = (
            _inverse_softplus(2.0 if bv_bh is None else bv_bh)
            if raw_bv_bh is None
            else jnp.asarray(raw_bv_bh)
        )

        if backend == "gamma_moments":
            raw_gaps = (
                jnp.asarray([])
                if raw_support_gaps is None
                else jnp.asarray(raw_support_gaps)
            )
            reference = ()
        else:
            if raw_support_gaps is not None and support_points is not None:
                raise ValueError("Specify raw_support_gaps or support_points, not both")
            if raw_support_gaps is None:
                points = jnp.asarray(
                    (0.0, 8.0) if support_points is None else support_points
                )
                if points.ndim != 1 or points.shape[0] not in {2, 4, 8}:
                    raise ValueError("soft_mixture requires 2, 4, or 8 support points")
                raw_gaps = _inverse_softplus(
                    jnp.concatenate((points[:1], jnp.diff(points)))
                )
                default_reference = points
            else:
                raw_gaps = jnp.asarray(raw_support_gaps)
                if raw_gaps.ndim != 1 or raw_gaps.shape[0] not in {2, 4, 8}:
                    raise ValueError(
                        "soft_mixture requires 2, 4, or 8 support coordinates"
                    )
                default_reference = jnp.cumsum(jax.nn.softplus(raw_gaps))
            reference_values = (
                default_reference
                if support_reference is None
                else jnp.asarray(support_reference)
            )
            if reference_values.shape != raw_gaps.shape:
                raise ValueError("support_reference must match the support count")
            reference = tuple(float(value) for value in reference_values)
        if bandwidth_floor <= 0:
            raise ValueError("bandwidth_floor must be positive")
        object.__setattr__(self, "raw_bv_bc", raw_bc)
        object.__setattr__(self, "raw_bv_bh", raw_bh)
        object.__setattr__(self, "raw_support_gaps", raw_gaps)
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "timepoints", times)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "support_reference", reference)
        object.__setattr__(self, "bandwidth_floor", float(bandwidth_floor))
        object.__setattr__(self, "kint_unit", kint_unit)
        object.__setattr__(self, "time_unit", time_unit)

    @property
    def bv_bc(self) -> Array:
        return jax.nn.softplus(self.raw_bv_bc)

    @property
    def bv_bh(self) -> Array:
        return jax.nn.softplus(self.raw_bv_bh)

    @property
    def support_points(self) -> Array:
        return jnp.cumsum(jax.nn.softplus(self.raw_support_gaps))

    @property
    def assignment_bandwidth(self) -> Array:
        return jnp.maximum(
            0.5 * jnp.median(jnp.diff(self.support_points)), self.bandwidth_floor
        )

    def regularization_loss(self, beta_reference=(0.35, 2.0)) -> Array:
        beta = jnp.stack((self.bv_bc, self.bv_bh))
        reference = jnp.asarray(beta_reference, dtype=beta.dtype)
        beta_loss = jnp.mean(jnp.square(jnp.log(beta / reference)))
        if self.backend == "gamma_moments":
            return beta_loss
        scale = jnp.maximum(self.assignment_bandwidth, self.bandwidth_floor)
        return beta_loss + jnp.mean(
            jnp.square(
                (self.support_points - jnp.asarray(self.support_reference)) / scale
            )
        )

    @classmethod
    def from_features(
        cls,
        heavy_contacts,
        acceptor_contacts,
        *,
        n_components: int,
        bv_bc: float = 0.35,
        bv_bh: float = 2.0,
        **kwargs,
    ) -> "BVRateDistributionParameters":
        """Initialise support anchors from empirical BV log-PF quantiles."""
        if n_components not in {2, 4, 8}:
            raise ValueError("n_components must be one of 2, 4, or 8")
        z = bv_bc * jnp.asarray(heavy_contacts) + bv_bh * jnp.asarray(acceptor_contacts)
        points = jnp.maximum(
            jnp.quantile(z.reshape(-1), jnp.linspace(0.0, 1.0, n_components)), 0.0
        )
        ordered = [points[0]]
        for index in range(1, n_components):
            ordered.append(jnp.maximum(points[index], ordered[-1] + _POSITIVE_EPS))
        points = jnp.stack(ordered)
        return cls(
            backend="soft_mixture",
            bv_bc=bv_bc,
            bv_bh=bv_bh,
            support_points=points,
            support_reference=points,
            **kwargs,
        )


register_pytree_node(
    BVRateDistributionParameters,
    BVRateDistributionParameters.tree_flatten,
    BVRateDistributionParameters.tree_unflatten,
)
