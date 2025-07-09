from dataclasses import field
from typing import Protocol

import jax.numpy as jnp
from jax import Array

from jaxent.src.interfaces.simulation import Model_Parameters
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters
from jaxent.src.models.HDX.netHDX.parameters import NetHDX_Model_Parameters
from jaxent.src.types.config import BaseConfig
from jaxent.src.types.key import m_key


class Model_Config(Protocol):
    key: m_key

    @property
    def forward_parameters(self) -> Model_Parameters: ...


class BV_model_Config(BaseConfig):
    temperature: float = 300
    bv_bc: Array = jnp.array([0.35])
    bv_bh: Array = jnp.array([2.0])
    ph: float = 7
    heavy_radius: float = 6.5
    o_radius: float = 2.4
    num_timepoints: int = 1
    timepoints: Array = jnp.array([0.167, 1.0, 10.0])
    residue_ignore: tuple[int, int] = (-2, 2)  # Range of residues to ignore relative to donor
    peptide_trim: int = 1  # HDXer by defualt uses 1 residue trim but this should be 2
    peptide: bool = False
    mda_selection_exclusion: str = "resname PRO or resid 1"

    def __init__(self, num_timepoints: int | None = None) -> None:
        super().__init__()
        if num_timepoints is None:
            self.key = m_key("HDX_resPF")
        else:
            if num_timepoints > 1:
                self.key = m_key("HDX_peptide")
            elif num_timepoints == 1:
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
    bv_bc: Array = field(default_factory=lambda: jnp.array([0.35, 0.35, 0.35]))
    bv_bh: Array = field(default_factory=lambda: jnp.array([2.0, 2.0, 2.0]))
    num_timepoints: int = 3

    def __init__(self, num_timepoints: int = 3):
        super().__init__(num_timepoints)

    def __post_init__(self):
        if (self.num_timepoints > 1) and ((len(self.bv_bh) or len(self.bv_bc)) == 1):
            object.__setattr__(self, "bv_bc", self.bv_bc * self.num_timepoints)
            object.__setattr__(self, "bv_bh", self.bv_bh * self.num_timepoints)

        assert self.num_timepoints == len(self.bv_bc) and self.num_timepoints == len(self.bv_bh), (
            ValueError("Please make sure your timepoint/prior parameters make sense")
        )

    @property
    def forward_parameters(self) -> linear_BV_Model_Parameters:
        return linear_BV_Model_Parameters(
            bv_bc=self.bv_bc,
            bv_bh=self.bv_bh,
            temperature=self.temperature,
            num_timepoints=self.num_timepoints,
        )


class NetHDXConfig(BaseConfig):
    """Configuration for netHDX calculations"""

    temperature: float = 300
    distance_cutoff: list[float] = field(
        default_factory=lambda: [2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5]
    )
    angle_cutoff: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    residue_ignore: tuple[int, int] = (-1, 1)  # Range of residues to ignore relative to donor
    num_timepoints: int = 1
    timepoints: Array = jnp.array([0.167, 1.0, 10.0])
    shell_energy_scaling: float = 0.84
    peptide_trim: int = 2  # HDXer by defualt uses 1 residue trim but this should be 2
    peptide: bool = True
    mda_selection_exclusion: str = "resname PRO or resid 1"

    def __init__(
        self,
        distance_cutoff: list[float] = [2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5],
        angle_cutoff: list[float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        num_timepoints: int = 1,
    ) -> None:
        # super().__init__()
        if self.distance_cutoff is not None:
            self.distance_cutoff = distance_cutoff
        if self.angle_cutoff is not None:
            self.angle_cutoff = angle_cutoff

        assert len(list(self.distance_cutoff)) == len(list(self.angle_cutoff)), (
            "Distance and angle cutoffs must be the same length"
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
