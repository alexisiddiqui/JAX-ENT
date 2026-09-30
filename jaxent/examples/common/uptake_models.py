"""Factories shared by example fitting and post-processing pipelines."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from jaxent.src.models.config import BV_model_Config, linear_BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, linear_BV_model


UptakeModel = Literal["standard", "linear"]


def build_uptake_model(
    model_type: UptakeModel,
    timepoints,
    *,
    kint_unit: Literal["s^-1", "min^-1"] = "s^-1",
    time_unit: Literal["s", "min"] = "min",
):
    """Construct the requested BV uptake model on an explicit time grid."""
    times = jnp.asarray(timepoints)
    if model_type == "standard":
        return BV_model(
            config=BV_model_Config(
                num_timepoints=len(times),
                timepoints=times,
                kint_unit=kint_unit,
            )
        )
    if model_type == "linear":
        return linear_BV_model(
            config=linear_BV_model_Config(
                num_timepoints=len(times),
                timepoints=times,
                kint_unit=kint_unit,
                time_unit=time_unit,
            )
        )
    raise ValueError(f"unknown uptake model: {model_type!r}")
