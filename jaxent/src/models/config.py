from dataclasses import field
from typing import Literal

import chex

import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types.config import BaseConfig
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.models.HDX.BV.parameters import (
    BV_Model_Parameters,
    BVRateDistributionParameters,
    linear_BV_Model_Parameters,
)
from jaxent.src.models.HDX.netHDX.parameters import NetHDX_Model_Parameters


class BV_model_Config(BaseConfig):
    temperature: float = 300.0
    bv_bc: Array = jnp.array([0.35])
    bv_bh: Array = jnp.array([2.0])
    ph: float = 7.0
    kint_unit: Literal["s^-1", "min^-1"] = "s^-1"
    heavy_radius: float = 6.5
    o_radius: float = 2.4
    num_timepoints: int = 0
    timepoints: Array = jnp.array([0.167, 1.0, 10.0])
    residue_ignore: tuple[int, int] = (-2, 2)  # Range of residues to ignore relative to donor
    peptide_trim: int = 1  # HDXer by defualt uses 1 residue trim but this should be 2
    peptide: bool = False
    switch: bool = False
    contact_mode: Literal["hard", "legacy_switch", "bradshaw_switch", "smooth_cutoff"] = "hard"
    switch_scale_nc: float = 10.0
    switch_scale_nh: float = 10.0
    # Protein N termini are handled chain-wise by BV_model's terminal policy.
    mda_selection_exclusion: str = "resname PRO"
    # General featurisation may include explicitly modelled solvent/cosolutes.
    # Protein-only parity analyses must opt in as protocol metadata.
    mda_contact_environment: str = "all"

    def __init__(
        self,
        num_timepoints: int | None = None,
        timepoints: Array | None = None,
        switch: bool | None = None,
        contact_mode: Literal["hard", "legacy_switch", "bradshaw_switch", "smooth_cutoff"] | None = None,
        switch_scale_nc: float = 10.0,
        switch_scale_nh: float = 10.0,
        kint_unit: Literal["s^-1", "min^-1"] = "s^-1",
    ) -> None:
        super().__init__()
        if contact_mode is not None and switch is not None:
            raise ValueError("contact_mode and the legacy switch argument are mutually exclusive")
        if contact_mode is None:
            contact_mode = "legacy_switch" if switch else "hard"
        if contact_mode not in {"hard", "legacy_switch", "bradshaw_switch", "smooth_cutoff"}:
            raise ValueError(f"unknown contact_mode: {contact_mode!r}")
        if switch_scale_nc <= 0 or switch_scale_nh <= 0:
            raise ValueError("Contact switch scales must be positive")
        if kint_unit not in {"s^-1", "min^-1"}:
            raise ValueError("kint_unit must be 's^-1' or 'min^-1'")
        self.contact_mode = contact_mode
        self.switch = contact_mode == "legacy_switch"
        self.switch_scale_nc = float(switch_scale_nc)
        self.switch_scale_nh = float(switch_scale_nh)
        self.kint_unit = kint_unit
        if timepoints is not None:
            self.timepoints = timepoints
            if num_timepoints is not None and num_timepoints != len(timepoints):
                raise ValueError(
                    f"num_timepoints ({num_timepoints}) and length of timepoints array ({len(timepoints)}) do not match."
                )
            if num_timepoints is None:
                num_timepoints = len(timepoints)

        if num_timepoints is None or num_timepoints == 0:
            self.key = m_key("HDX_resPF")
            if num_timepoints is None:
                self.num_timepoints = 0
            else:
                self.num_timepoints = num_timepoints
        else:
            if num_timepoints > 0:
                self.key = m_key("HDX_peptide")
            elif len(self.timepoints) != num_timepoints:
                self.key = m_key("HDX_resPF")
            else:
                raise ValueError("Please make sure your timepoint/prior parameters make sense")
            self.num_timepoints = num_timepoints

    @property
    def forward_parameters(self) -> Model_Parameters:
        return BV_Model_Parameters(
            bv_bc=jnp.asarray(self.bv_bc),
            bv_bh=jnp.asarray(self.bv_bh),
            timepoints=self.timepoints,
            temperature=self.temperature,
        )


class linear_BV_model_Config(BV_model_Config):
    """Configuration for the additive interval-hazard uptake model."""

    bv_bc: float = 0.35
    bv_bh: float = 2.0
    interval_offsets: Array = jnp.zeros(3)
    time_unit: Literal["s", "min"] = "min"
    prior_strength: float = 1e-3

    def __init__(
        self,
        num_timepoints: int | None = None,
        timepoints: Array | None = None,
        *,
        kint_unit: Literal["s^-1", "min^-1"] = "s^-1",
        time_unit: Literal["s", "min"] = "min",
        interval_offsets: Array | None = None,
    ):
        if timepoints is None:
            timepoints = jnp.array([0.167, 1.0, 10.0])
        if num_timepoints in {None, 0}:
            num_timepoints = len(timepoints)
        super().__init__(
            num_timepoints=num_timepoints, timepoints=timepoints, kint_unit=kint_unit
        )
        if time_unit not in {"s", "min"}:
            raise ValueError("time_unit must be 's' or 'min'")
        self.time_unit = time_unit
        self.key = m_key("HDX_peptide")
        self.interval_offsets = (
            jnp.zeros(num_timepoints) if interval_offsets is None else jnp.asarray(interval_offsets)
        )
        if self.interval_offsets.shape != (num_timepoints,):
            raise ValueError("interval_offsets must contain one value per timepoint")

    @property
    def forward_parameters(self) -> linear_BV_Model_Parameters:
        return linear_BV_Model_Parameters(
            bv_bc=jnp.asarray(self.bv_bc),
            bv_bh=jnp.asarray(self.bv_bh),
            interval_offsets=self.interval_offsets,
            temperature=self.temperature,
            timepoints=self.timepoints,
            kint_unit=self.kint_unit,
            time_unit=self.time_unit,
        )

    def regularization_loss(self, parameters: linear_BV_Model_Parameters) -> Array:
        return self.prior_strength * parameters.regularization_loss()


class BVRateDistributionConfig(BV_model_Config):
    """Configuration for an explicitly selected experimental rate backend."""

    time_unit: Literal["s", "min"] = "min"
    n_components: int = 4
    bandwidth_floor: float = 0.05
    prior_strength: float = 1e-3
    bv_bc: float = 0.35
    bv_bh: float = 2.0

    def __init__(
        self,
        *,
        backend: Literal["soft_mixture", "gamma_moments"],
        n_components: int = 4,
        timepoints: Array | None = None,
        kint_unit: Literal["s^-1", "min^-1"] = "s^-1",
        time_unit: Literal["s", "min"] = "min",
        support_points: Array | None = None,
        bandwidth_floor: float = 0.05,
    ) -> None:
        if backend not in {"soft_mixture", "gamma_moments"}:
            raise ValueError("backend must be 'soft_mixture' or 'gamma_moments'")
        if timepoints is None:
            timepoints = jnp.array([0.167, 1.0, 10.0])
        if backend == "soft_mixture" and n_components not in {2, 4, 8}:
            raise ValueError("soft_mixture n_components must be 2, 4, or 8")
        if backend == "gamma_moments" and support_points is not None:
            raise ValueError("gamma_moments does not use support points")
        super().__init__(
            num_timepoints=len(timepoints), timepoints=timepoints, kint_unit=kint_unit
        )
        if time_unit not in {"s", "min"}:
            raise ValueError("time_unit must be 's' or 'min'")
        self.backend = backend
        self.n_components = n_components
        self.time_unit = time_unit
        self.bandwidth_floor = float(bandwidth_floor)
        self.support_points = (
            jnp.linspace(0.0, 8.0, n_components)
            if support_points is None and backend == "soft_mixture"
            else support_points
        )
        if self.support_points is not None and len(self.support_points) != n_components:
            raise ValueError("support_points length must equal n_components")
        self.key = m_key("HDX_peptide")

    @property
    def forward_parameters(self) -> BVRateDistributionParameters:
        return BVRateDistributionParameters(
            backend=self.backend,
            bv_bc=jnp.asarray(self.bv_bc),
            bv_bh=jnp.asarray(self.bv_bh),
            support_points=self.support_points,
            bandwidth_floor=self.bandwidth_floor,
            temperature=self.temperature,
            timepoints=self.timepoints,
            kint_unit=self.kint_unit,
            time_unit=self.time_unit,
        )

    def regularization_loss(self, parameters: BVRateDistributionParameters) -> Array:
        return self.prior_strength * parameters.regularization_loss()


class NetHDXConfig(BaseConfig):
    """Configuration for netHDX calculations"""

    temperature: float = 300.0
    distance_cutoff: float | list[float] = field(
        default_factory=lambda: [2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5]
    )
    angle_cutoff: float | list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    residue_ignore: tuple[int, int] = (-1, 1)  # Range of residues to ignore relative to donor
    num_timepoints: int = 1
    timepoints: Array = jnp.array([0.167, 1.0, 10.0])
    shell_energy_scaling: float = 0.84
    peptide_trim: int = 2  # HDXer by defualt uses 1 residue trim but this should be 2
    peptide: bool = True
    mda_selection_exclusion: str = "resname PRO or resid 1"

    def __init__(
        self,
        distance_cutoff: float | list[float] = [2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5],
        angle_cutoff: float | list[float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        num_timepoints: int = 1,
    ) -> None:
        super().__init__()
        if self.distance_cutoff is not None:
            if isinstance(distance_cutoff, float):
                distance_cutoff = [distance_cutoff]
            self.distance_cutoff = distance_cutoff
        if self.angle_cutoff is not None:
            if isinstance(angle_cutoff, float):
                angle_cutoff = [angle_cutoff]
            self.angle_cutoff = angle_cutoff

        chex.assert_equal(
            len(list(self.distance_cutoff)), 
            len(list(self.angle_cutoff)),
            custom_message="Distance and angle cutoffs must be the same length"
        )
        if num_timepoints > 1:
            self.key = m_key("HDX_peptide")
        elif num_timepoints == 1:
            self.key = m_key("HDX_resPF")
        else:
            raise ValueError("Please make sure your timepoint/prior parameters make sense")
        self.num_timepoints = num_timepoints

    @property
    def forward_parameters(self) -> Model_Parameters:
        return NetHDX_Model_Parameters(
            temperature=self.temperature,
            shell_energy_scaling=self.shell_energy_scaling,
            timepoints=self.timepoints,
        )
