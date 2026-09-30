"""Disposable batch-optimizer driver for the Example 1--3 validation sweeps.

This intentionally batches hyperparameters within one data split.  Split-specific
data are not vmapped because the current batch API broadcasts ``data_to_fit`` to
all lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import jax.numpy as jnp

import jaxent.src.interfaces.topology as pt
from jaxent.examples.common import loading
from jaxent.examples.common.optimization import create_data_loaders
from jaxent.examples.common.losses import (
    get_loss_function_by_name,
    maxent_convexKL_loss,
)
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.custom_types.config import Optimisable_Parameters, OptimiserSettings
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.opt.base import HParamBatch
from jaxent.src.opt.batch import batch_optimise
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.utils.hdf import save_optimization_history_to_file
from jaxent.src.utils.jit_fn import jit_Guard


@dataclass(frozen=True)
class BatchTestSpec:
    experiment: Literal[1, 2, 3]
    fitting_dir: Path
    ensembles: tuple[str, ...]
    feature_names: dict[str, str]
    timepoints: tuple[float, ...]
    covariance_path: Path
    primary_loss: str
    optimize_bv_params: bool = False
    bv_values: tuple[float, ...] = ()
    bv_reg_function: str = "L1"
    kint_unit: str = "min^-1"


def _load_splits(datasplit_dir: Path, split_type: str, n_splits: int):
    splits = []
    for split_idx in range(n_splits):
        split_dir = datasplit_dir / split_type / f"split_{split_idx:03d}"
        train = HDX_peptide.load_list_from_files(
            json_path=str(split_dir / "train_topology.json"),
            csv_path=str(split_dir / "train_dfrac.csv"),
        )
        val = HDX_peptide.load_list_from_files(
            json_path=str(split_dir / "val_topology.json"),
            csv_path=str(split_dir / "val_dfrac.csv"),
        )
        splits.append((train, val))
    return splits


def _run_name(
    spec: BatchTestSpec,
    ensemble: str,
    split_type: str,
    split_idx: int,
    maxent: float,
    bv_value: float | None,
) -> str:
    if spec.experiment == 1:
        token = f"{maxent:g}".replace(".", "p")
        return f"{ensemble}_MSE_{split_type}_split{split_idx:03d}_maxent{token}"
    base = f"{ensemble}_MSE_{split_type}_split{split_idx:03d}_maxent{maxent:.1f}"
    if bv_value is not None:
        base += (
            f"_bvreg{bv_value:.1f}_bvregfn{spec.bv_reg_function}"
        )
    return base


def run_batch_test(
    spec: BatchTestSpec,
    output_dir: Path,
    *,
    maxent_values: tuple[float, ...] = (1, 5, 10, 50, 100, 500, 1000),
    split_types: tuple[str, ...] = ("sequence_cluster", "spatial"),
    n_splits: int = 3,
    n_steps: int = 5000,
    batch_size: int = 10,
    learning_rate: float = 1.0,
    forward_model_scaling: float = 1000.0,
    ema_alpha: float = 0.5,
    step_chunk_size: int = 100,
) -> None:
    fitting_dir = spec.fitting_dir.resolve()
    features_dir = fitting_dir / "_featurise"
    datasplit_dir = fitting_dir / "_datasplits"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    covariance = jnp.load(spec.covariance_path)["Sigma_inv"]
    covariance = covariance / jnp.linalg.norm(covariance)
    convergence = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    combos = list(product(maxent_values, spec.bv_values or (None,)))
    n_slots = 3 if spec.optimize_bv_params else 2
    loss_functions = [
        get_loss_function_by_name(spec.primary_loss),
        maxent_convexKL_loss,
    ]
    indexes = [0, 0]
    normalise = jnp.ones(n_slots)
    if spec.optimize_bv_params:
        loss_functions.append(
            get_loss_function_by_name(
                f"model_params_{spec.bv_reg_function}_loss"
            )
        )
        indexes.append(0)
        normalise = normalise.at[-1].set(0.0)

    raw_lane_weights = jnp.asarray(
        [
            [1.0, 1.0 / maxent]
            + ([float(bv)] if bv is not None else [])
            for maxent, bv in combos
        ]
    )
    # Simulation.__init__ applies this masked normalization in scalar runs.
    # HParamBatch values are injected after initialization, so reproduce it
    # explicitly or the batched objective (especially the BV term) changes.
    masked_weights = raw_lane_weights * normalise
    lane_weights = raw_lane_weights * (1.0 - normalise) + masked_weights / jnp.sum(
        masked_weights, axis=1, keepdims=True
    )
    hparams = HParamBatch(
        forward_model_weights=lane_weights,
        forward_model_scaling=jnp.full(
            (len(combos), n_slots), forward_model_scaling
        ),
        learning_rate=jnp.full((len(combos),), learning_rate),
    )

    partitions = {Optimisable_Parameters.frame_weights}
    if spec.optimize_bv_params:
        partitions.add(Optimisable_Parameters.model_parameters)
    partition_set = frozenset(partitions)
    settings = OptimiserSettings(
        name=f"example_{spec.experiment}_batch_test",
        n_steps=n_steps,
        tolerance=1e-10,
        convergence=convergence,
        learning_rate=learning_rate,
        optimiser_type="adam",
        ema_alpha=ema_alpha,
        step_chunk_size=step_chunk_size,
        reset_threshold_cooldown_on_oscillation=True,
        execution_mode="compiled",
        parameter_partitions=partition_set,
    )

    print(
        f"Batch test: example={spec.experiment}, lanes={len(combos)}, "
        f"batch_size={batch_size}, steps={n_steps}, averaging=rate"
    )
    for ensemble in spec.ensembles:
        feature_path = features_dir / f"features_{spec.feature_names[ensemble]}.npz"
        topology_path = feature_path.with_name(
            feature_path.name.replace("features_", "topology_")
        ).with_suffix(".json")
        features = BV_input_features.load(str(feature_path))
        feature_top = pt.PTSerialiser.load_list_from_json(str(topology_path))
        model = BV_model(
            BV_model_Config(
                num_timepoints=len(spec.timepoints),
                timepoints=jnp.asarray(spec.timepoints),
                kint_unit=spec.kint_unit,
            )
        )
        model.forward[m_key("HDX_peptide")].frame_averaging_mode = "rate"
        model_parameters = model.params
        n_frames = features.features_shape[1]

        for split_type in split_types:
            split_output = output_dir / split_type
            split_output.mkdir(parents=True, exist_ok=True)
            for split_idx, (train_data, val_data) in enumerate(
                _load_splits(datasplit_dir, split_type, n_splits)
            ):
                expected_outputs = [
                    split_output
                    / f"{_run_name(spec, ensemble, split_type, split_idx, maxent, bv_value)}_results.hdf5"
                    for maxent, bv_value in combos
                ]
                if all(path.exists() for path in expected_outputs):
                    print(
                        f"Skipping {ensemble}/{split_type}/split{split_idx:03d}: "
                        "all batch results already exist"
                    )
                    continue
                loading.validate_hdx_timepoint_count(
                    train_data,
                    spec.timepoints,
                    label=f"{split_type} split {split_idx} train",
                )
                loading.validate_hdx_timepoint_count(
                    val_data,
                    spec.timepoints,
                    label=f"{split_type} split {split_idx} validation",
                )
                loader = create_data_loaders(
                    hdx_data=train_data + val_data,
                    train_data=train_data,
                    val_data=val_data,
                    features=features,
                    feature_top=feature_top,
                    cov_matrix=covariance,
                )
                initial_params = Simulation_Parameters.from_frame_weights(
                    jnp.ones(n_frames) / n_frames,
                    model_parameters=(model_parameters,),
                    forward_model_weights=lane_weights[0],
                    normalise_loss_functions=normalise,
                    forward_model_scaling=jnp.full(
                        (n_slots,), forward_model_scaling
                    ),
                )
                data_to_fit = [loader, initial_params]
                if spec.optimize_bv_params:
                    data_to_fit.append(initial_params)
                simulation = Simulation(
                    input_features=(features,),
                    forward_models=(model,),
                    params=initial_params,
                    frame_average_impl="tensordot",
                )
                optimizer = OptaxOptimizer(
                    learning_rate=learning_rate,
                    parameter_partition_masks=partitions,
                    clip_value=None,
                    optimizer="adam",
                    lr_adjustment=True,
                    model_parameters_lr_scale=1.0,
                )
                print(
                    f"Running {ensemble}/{split_type}/split{split_idx:03d}: "
                    f"{len(combos)} lanes"
                )
                with jit_Guard(simulation, cleanup_on_exit=True) as simulation:
                    simulation.initialise()
                    result = batch_optimise(
                        simulation=simulation,
                        hparam_batch=hparams,
                        batch_size=batch_size,
                        data_to_fit=tuple(data_to_fit),
                        config=settings,
                        indexes=tuple(indexes),
                        loss_functions=tuple(loss_functions),
                        optimizer=optimizer,
                    )
                print(
                    "  Executed steps: "
                    + ", ".join(str(int(step)) for step in result.convergence_steps)
                )
                for history, (maxent, bv_value) in zip(
                    result.histories, combos, strict=True
                ):
                    name = _run_name(
                        spec,
                        ensemble,
                        split_type,
                        split_idx,
                        maxent,
                        bv_value,
                    )
                    save_optimization_history_to_file(
                        filename=str(split_output / f"{name}_results.hdf5"),
                        history=history,
                    )
