#!/usr/bin/env python3
"""Attribute eager-loop RSS growth to JAX buffers, caching, or async dispatch."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.opt import optimiser as optimiser_module
from jaxent.src.opt.base import OptimizationState
from jaxent.src.opt.chunk import _make_record, evaluate_convergence, optimisation_step
from jaxent.src.opt.run import _build_chunk_state


SCHEMA_VERSION = 1
CHECKPOINTS = (0, 1, 10, 100, 1000)
VARIANTS = (
    "retained_checkpoint_sync",
    "retained_per_step_sync",
    "discarded_checkpoint_sync",
)
WORKER_VARIANTS = (*VARIANTS, "preserved_sim_checkpoint_sync")
WORKLOADS = ("synthetic", "sigma")
ROOT = Path(__file__).resolve().parents[1]
SIGMA_BASE = (
    ROOT / "jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT"
)


@dataclass(frozen=True)
class Fixture:
    simulation: Any
    data: tuple[Any, ...]
    indexes: tuple[int, ...]
    losses: tuple[Any, ...]
    optimizer: Any
    initial_state: OptimizationState


@dataclass(frozen=True)
class MemorySample:
    phase: str
    step: int
    rss_bytes: int
    python_heap_bytes: int
    python_heap_peak_bytes: int
    live_array_count: int
    live_array_bytes: int
    elapsed_s: float
    compilations: int | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--workloads", default="synthetic,sigma")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("profiling-output/eager-native-memory")
    )
    parser.add_argument("--max-worker-rss-gb", type=float, default=6.0)
    parser.add_argument("--reclassify-json", type=Path)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--residues", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=WORKER_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    return parser


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


@contextlib.contextmanager
def _count_compiles() -> Iterator[dict[str, float | int]]:
    import jax._src.compiler as compiler

    stats: dict[str, float | int] = {"count": 0, "seconds": 0.0}
    original = compiler.backend_compile

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stats["count"] = int(stats["count"]) + 1
            stats["seconds"] = float(stats["seconds"]) + time.perf_counter() - start

    compiler.backend_compile = wrapped
    try:
        yield stats
    finally:
        compiler.backend_compile = original


def _synthetic_loss(model, target, index):
    prediction = jnp.asarray(model.outputs[index].y_pred())
    value = jnp.mean(jnp.square(prediction - target))
    return value, value


def _synthetic_fixture(args: argparse.Namespace) -> Fixture:
    rng = np.random.default_rng(args.seed)
    residues, frames = args.residues, args.frames
    residue_axis = np.linspace(0.6, 1.6, residues, dtype=np.float32)[:, None]
    frame_axis = np.linspace(0.75, 1.25, frames, dtype=np.float32)[None, :]
    noise = rng.normal(0.0, 0.01, size=(residues, frames)).astype(np.float32)
    heavy = jnp.asarray(residue_axis * frame_axis + 0.2 + noise)
    acceptor = jnp.asarray(
        residue_axis * np.flip(frame_axis, axis=1) + 0.1 + noise[::-1]
    )
    k_ints = jnp.asarray(np.linspace(0.3, 1.1, residues, dtype=np.float32))
    features = BV_input_features(
        heavy_contacts=heavy, acceptor_contacts=acceptor, k_ints=k_ints
    )
    timepoints = jnp.asarray([0.167, 1.0, 10.0, 60.0, 120.0])
    config = BV_model_Config(num_timepoints=5, timepoints=timepoints)
    model = BV_model(config)
    parameters = Simulation_Parameters(
        frame_weight_logits=jnp.zeros(frames, dtype=jnp.float32),
        model_parameters=[config.forward_parameters],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.float32),
    )
    simulation = Simulation(
        input_features=[features],
        forward_models=[model],
        params=parameters,
        raise_jit_failure=True,
    )
    simulation.initialise()
    simulation = Simulation.forward(simulation, simulation.params)
    target_log_pf = (
        jnp.asarray(config.forward_parameters.bv_bc) * heavy[:, 0]
        + jnp.asarray(config.forward_parameters.bv_bh) * acceptor[:, 0]
    )
    target = 1.0 - jnp.exp(
        -timepoints[:, None] * k_ints[None, :] / jnp.exp(target_log_pf)[None, :]
    )
    optimizer = optimiser_module.OptaxOptimizer(
        learning_rate=1e-4, optimizer="adam"
    )
    return Fixture(
        simulation=simulation,
        data=(target,),
        indexes=(0,),
        losses=(_synthetic_loss,),
        optimizer=optimizer,
        initial_state=optimizer.initialise(simulation, None),
    )


def _sigma_fixture(_args: argparse.Namespace) -> Fixture:
    import jaxent.src.interfaces.topology as pt
    from jaxent.examples.common.losses import (
        get_loss_function_by_name,
        maxent_convexKL_loss,
    )

    features = BV_input_features.load(SIGMA_BASE / "_featurise/features_iso_bi.npz")
    feature_top = pt.PTSerialiser.load_list_from_json(
        SIGMA_BASE / "_featurise/topology_iso_bi.json"
    )
    split = SIGMA_BASE / "_datasplits/random/split_000"
    train_data = HDX_peptide.load_list_from_files(
        json_path=split / "train_topology.json",
        csv_path=split / "train_dfrac.csv",
    )
    val_data = HDX_peptide.load_list_from_files(
        json_path=split / "val_topology.json",
        csv_path=split / "val_dfrac.csv",
    )
    covariance = np.load(
        SIGMA_BASE / "_covariance_matrices_sigma/ISO_BI_Sigma_weighted.npz"
    )["Sigma_inv"]
    covariance = jnp.asarray(covariance) / jnp.linalg.norm(jnp.asarray(covariance))
    loader = ExpD_Dataloader(data=train_data + val_data, covariance_matrix=covariance)
    loader.create_datasets(
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_topology=feature_top,
    )
    config = BV_model_Config(num_timepoints=5)
    config.timepoints = jnp.asarray([0.167, 1.0, 10.0, 60.0, 120.0])
    model = BV_model(config=config)
    frames = features.features_shape[1]
    parameters = Simulation_Parameters.from_frame_weights(
        jnp.ones(frames) / frames,
        model_parameters=(model.params,),
        forward_model_weights=jnp.asarray([1.0, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * 100.0,
    )
    simulation = Simulation(
        input_features=(features,), forward_models=(model,), params=parameters
    )
    simulation.initialise()
    optimizer = optimiser_module.OptaxOptimizer(
        learning_rate=0.1,
        parameter_partition_masks={Optimisable_Parameters.frame_weights},
        clip_value=None,
        optimizer="adam",
    )
    losses = (
        get_loss_function_by_name("hdx_uptake_sigma_MSE_loss"),
        maxent_convexKL_loss,
    )
    return Fixture(
        simulation=simulation,
        data=(loader, parameters),
        indexes=(0, 0),
        losses=losses,
        optimizer=optimizer,
        initial_state=optimizer.initialise(model=simulation),
    )


def _snapshot_parameters(state: OptimizationState) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(leaf).copy()
        for leaf in jax.tree_util.tree_leaves(state.params)
        if hasattr(leaf, "shape")
    )


def _fingerprint(snapshot: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for leaf in snapshot:
        value = np.ascontiguousarray(leaf)
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _live_arrays() -> tuple[int, int]:
    arrays = jax.live_arrays()
    result = len(arrays), sum(int(array.nbytes) for array in arrays)
    del arrays
    return result


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant is None or args.workload is None or args.worker_json is None:
        raise ValueError("worker mode requires workload, variant, and worker-json")
    if args.workload == "synthetic":
        fixture = _synthetic_fixture(args)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            fixture = _sigma_fixture(args)
    carry, inputs, losses, indexes = _build_chunk_state(
        fixture.simulation,
        fixture.data,
        0.0,
        0.0,
        fixture.indexes,
        fixture.losses,
        fixture.initial_state,
        fixture.optimizer,
    )
    inputs = inputs._replace(ema_alpha=jnp.asarray(0.5, dtype=jnp.float32))
    checkpoints = frozenset(
        [step for step in CHECKPOINTS if step <= args.steps] + [args.steps]
    )
    retained_metrics: list[Any] | None = []
    records: list[Any] | None = []
    samples: list[MemorySample] = []
    process = psutil.Process(os.getpid())
    started = time.perf_counter()
    compile_stats: dict[str, float | int] = {"count": 0}

    def sample(phase: str, step: int, sync_tree: Any | None = None) -> None:
        if sync_tree is not None:
            _block_tree(sync_tree)
        current, peak = tracemalloc.get_traced_memory()
        live_count, live_bytes = _live_arrays()
        samples.append(
            MemorySample(
                phase=phase,
                step=step,
                rss_bytes=process.memory_info().rss,
                python_heap_bytes=current,
                python_heap_peak_bytes=peak,
                live_array_count=live_count,
                live_array_bytes=live_bytes,
                elapsed_s=time.perf_counter() - started,
                compilations=int(compile_stats["count"]),
            )
        )

    tracemalloc.start()
    with _count_compiles() as observed:
        compile_stats = observed
        sample("checkpoint", 0, carry)
        old_active = carry.active
        for step in range(1, args.steps + 1):
            previous_sim = carry.sim
            carry, metrics = optimisation_step(
                carry,
                inputs,
                fixture.optimizer,
                losses,
                indexes,
                args.steps + 1,
            )
            if args.variant == "preserved_sim_checkpoint_sync":
                carry = carry._replace(sim=previous_sim)
            if args.variant != "discarded_checkpoint_sync":
                assert retained_metrics is not None
                retained_metrics.append(metrics)
            if args.variant == "retained_per_step_sync":
                _block_tree((carry, metrics))
            if step % 100 == 0 or step == args.steps:
                carry, event = evaluate_convergence(carry, inputs, args.steps + 1)
                if args.variant != "discarded_checkpoint_sync":
                    assert records is not None
                    records.append(
                        _make_record(carry, event, old_active, None, True)
                    )
                old_active = carry.active
            if step in checkpoints:
                sample("checkpoint", step, (carry, retained_metrics, records))

    terminal_state = carry.opt_state
    _block_tree(terminal_state)
    snapshot = _snapshot_parameters(terminal_state)
    losses_value = terminal_state.losses
    final_loss = (
        float(np.asarray(losses_value.total_train_loss))
        if losses_value is not None
        else float("nan")
    )
    fingerprint = _fingerprint(snapshot)

    retained_metrics = None
    records = None
    gc.collect()
    sample("post_gc", args.steps, terminal_state)
    jax.clear_caches()
    gc.collect()
    sample("post_clear_caches", args.steps, terminal_state)
    tracemalloc.stop()

    report = {
        "workload": args.workload,
        "variant": args.variant,
        "execution_path": "eager_optimisation_step",
        "compiled_scan_used": False,
        "steps": args.steps,
        "samples": [asdict(item) for item in samples],
        "final_loss": final_loss,
        "parameter_fingerprint": fingerprint,
        "compile_seconds": float(compile_stats.get("seconds", 0.0)),
    }
    args.worker_json.parent.mkdir(parents=True, exist_ok=True)
    args.worker_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _slope(report: dict[str, Any], key: str) -> float:
    samples = [
        sample
        for sample in report["samples"]
        if sample["phase"] == "checkpoint" and sample["step"] >= 10
    ]
    if len(samples) < 2:
        return 0.0
    return float(
        np.polyfit(
            [sample["step"] for sample in samples],
            [sample[key] for sample in samples],
            1,
        )[0]
    )


def classify(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    slopes = {
        name: {
            key: _slope(report, key)
            for key in (
                "rss_bytes",
                "python_heap_bytes",
                "live_array_count",
                "live_array_bytes",
            )
        }
        for name, report in reports.items()
    }
    retained = slopes["retained_checkpoint_sync"]
    per_step = slopes["retained_per_step_sync"]
    discarded = slopes["discarded_checkpoint_sync"]
    rss = retained["rss_bytes"]
    meaningful = rss > 1024.0
    async_removed = meaningful and per_step["rss_bytes"] <= rss * 0.25
    retention_removed = meaningful and discarded["rss_bytes"] <= rss * 0.25
    terminal = reports["retained_checkpoint_sync"]["samples"]
    final_checkpoint = next(
        sample for sample in reversed(terminal) if sample["phase"] == "checkpoint"
    )
    post_gc = next(sample for sample in terminal if sample["phase"] == "post_gc")
    arrays_released = post_gc["live_array_count"] < final_checkpoint["live_array_count"] * 0.5
    rss_retained = post_gc["rss_bytes"] > final_checkpoint["rss_bytes"] * 0.9
    nonmetric_arrays_retained = discarded["live_array_count"] > 1.0

    findings = []
    if async_removed:
        findings.append("per-step synchronization removes RSS growth: async backlog")
    else:
        findings.append("per-step synchronization does not remove RSS growth")
    if retention_removed:
        findings.append("discarding StepMetrics removes RSS growth: retained JAX buffers")
    elif (
        retained["live_array_count"] > 0
        and discarded["live_array_count"] < retained["live_array_count"] * 0.8
    ):
        findings.append(
            "StepMetrics retain JAX arrays but do not explain the RSS growth"
        )
    if arrays_released and rss_retained:
        findings.append("live arrays are released while RSS remains high: allocator caching")
    if nonmetric_arrays_retained:
        findings.append(
            "non-metric JAX arrays remain live after GC and cache clearing: "
            "reference retention is the primary cause, not allocator-only caching"
        )
    heap_flat = abs(retained["python_heap_bytes"]) <= max(1024.0, abs(rss) * 0.2)
    if meaningful and heap_flat:
        findings.append("RSS growth is predominantly outside the Python heap")
    return {"slopes": slopes, "findings": findings}


def _selected(raw: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    invalid = set(values) - set(allowed)
    if invalid or not values:
        raise ValueError(f"invalid {label}: {sorted(invalid)}")
    return values


def _worker_command(
    args: argparse.Namespace, workload: str, variant: str, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--workload",
        workload,
        "--variant",
        variant,
        "--worker-json",
        str(output),
        "--steps",
        str(args.steps),
        "--frames",
        str(args.frames),
        "--residues",
        str(args.residues),
        "--seed",
        str(args.seed),
    ]


def _run_monitored(command: list[str], limit_bytes: int) -> None:
    process = subprocess.Popen(command)
    observed = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            if observed.memory_info().rss > limit_bytes:
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError(
                    f"worker exceeded RSS safety limit ({limit_bytes / 2**30:.1f} GiB)"
                )
        except psutil.NoSuchProcess:
            break
        time.sleep(0.25)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _write_csv(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    fields = ["workload", "variant", *MemorySample.__dataclass_fields__.keys()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, report in reports.items():
            workload, variant = key.split("/", 1)
            for sample in report["samples"]:
                writer.writerow(
                    {"workload": workload, "variant": variant, **sample}
                )


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    workloads = _selected(args.workloads, WORKLOADS, "workloads")
    variants = _selected(args.variants, VARIANTS, "variants")
    if set(variants) != set(VARIANTS):
        raise ValueError("classification requires all three variants")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="jaxent-native-memory-") as temporary:
        for workload in workloads:
            for variant in variants:
                output = Path(temporary) / f"{workload}-{variant}.json"
                _run_monitored(
                    _worker_command(args, workload, variant, output),
                    int(args.max_worker_rss_gb * 2**30),
                )
                reports[f"{workload}/{variant}"] = json.loads(output.read_text())

    parity = {}
    classifications = {}
    for workload in workloads:
        selected = {
            variant: reports[f"{workload}/{variant}"] for variant in variants
        }
        reference = selected["retained_checkpoint_sync"]
        parity[workload] = all(
            report["parameter_fingerprint"] == reference["parameter_fingerprint"]
            and np.isclose(
                report["final_loss"], reference["final_loss"], rtol=1e-4, atol=1e-5
            )
            for report in selected.values()
        )
        classifications[workload] = classify(selected)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "steps": args.steps,
        "fresh_process_per_variant": True,
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "backend": jax.default_backend(),
        },
        "numerical_parity": parity,
        "classification": classifications,
        "reports": reports,
    }
    json_path = args.output_dir / "eager_native_memory.json"
    csv_path = args.output_dir / "eager_native_memory_samples.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(csv_path, reports)
    print(json.dumps(
        {
            "steps": args.steps,
            "numerical_parity": parity,
            "classification": classifications,
        },
        indent=2,
    ))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.reclassify_json is not None:
        payload = json.loads(args.reclassify_json.read_text())
        workloads = {
            key.split("/", 1)[0] for key in payload["reports"]
        }
        payload["classification"] = {
            workload: classify(
                {
                    variant: payload["reports"][f"{workload}/{variant}"]
                    for variant in VARIANTS
                }
            )
            for workload in sorted(workloads)
        }
        args.reclassify_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload["classification"], indent=2))
        return 0
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.variant is not None:
        _run_worker(args)
        return 0
    report = _run_parent(args)
    return 0 if all(report["numerical_parity"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
