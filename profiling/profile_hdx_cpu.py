#!/usr/bin/env python3
"""Benchmark or profile one synthetic HDX uptake optimisation path on CPU."""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import hashlib
import json
import platform
import pstats
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.opt import optimiser as optimiser_module
from jaxent.src.opt.base import OptimizationHistory, OptimizationState
from jaxent.src.opt.chunk import run_sequential
from jaxent.src.opt.run import _build_chunk_state, _optimise


SCHEMA_VERSION = 2
PATHS = ("eager", "jit", "pure")
OPTIMIZERS = ("adam", "sgd", "adagrad", "adamw", "rmsprop", "lbfgs")
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5
TIMEPOINT_MINUTES = (0.167, 120.0)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "profile"), default="timing")
    parser.add_argument("--path", choices=PATHS, required=True)
    parser.add_argument("--steps", type=positive_int, default=None)
    parser.add_argument("--frames", type=positive_int, default=500)
    parser.add_argument("--residues", type=positive_int, default=144)
    parser.add_argument("--timepoints", type=positive_int, default=5)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--target-frame", type=nonnegative_int, default=0)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warm-repeats", type=positive_int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--profile-limit", type=positive_int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("profiling-output"))
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--terminal-npz", type=Path, default=None)
    return parser


@dataclass(frozen=True)
class Fixture:
    simulation: Simulation
    models: tuple[Any, ...]
    data: tuple[Any, ...]
    indexes: tuple[int, ...]
    losses: tuple[Callable, ...]
    timepoint_values: tuple[float, ...]


def _timepoint_values(count: int) -> jax.Array:
    return jnp.asarray(
        np.geomspace(TIMEPOINT_MINUTES[0], TIMEPOINT_MINUTES[1], count, dtype=np.float32)
    )


def _make_fixture(
    frames: int,
    residues: int,
    timepoints: int,
    seed: int,
    target_frame: int,
) -> Fixture:
    if target_frame >= frames:
        raise ValueError(f"target-frame must be less than frames ({frames})")

    rng = np.random.default_rng(seed)
    residue_axis = np.linspace(0.6, 1.6, residues, dtype=np.float32)[:, None]
    frame_axis = np.linspace(0.75, 1.25, frames, dtype=np.float32)[None, :]
    noise = rng.normal(0.0, 0.01, size=(residues, frames)).astype(np.float32)
    heavy = jnp.asarray(residue_axis * frame_axis + 0.2 + noise)
    acceptor = jnp.asarray(
        residue_axis * np.flip(frame_axis, axis=1) + 0.1 + noise[::-1]
    )
    k_ints = jnp.asarray(np.linspace(0.3, 1.1, residues, dtype=np.float32))
    feature = BV_input_features(
        heavy_contacts=heavy,
        acceptor_contacts=acceptor,
        k_ints=k_ints,
    )

    timepoint_array = _timepoint_values(timepoints)
    model_config = BV_model_Config(
        num_timepoints=timepoints,
        timepoints=timepoint_array,
    )
    model = BV_model(model_config)
    model_parameters = model_config.forward_parameters
    params = Simulation_Parameters(
        frame_weight_logits=jnp.zeros(frames, dtype=jnp.float32),
        model_parameters=[model_parameters],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.float32),
    )
    simulation = Simulation(
        input_features=[feature],
        forward_models=[model],
        params=params,
        raise_jit_failure=True,
    )
    simulation.initialise()
    simulation = Simulation.forward(simulation, simulation.params)

    target_log_pf = (
        jnp.asarray(model_parameters.bv_bc) * heavy[:, target_frame]
        + jnp.asarray(model_parameters.bv_bh) * acceptor[:, target_frame]
    )
    target = 1.0 - jnp.exp(
        -timepoint_array[:, None] * k_ints[None, :] / jnp.exp(target_log_pf)[None, :]
    )
    return Fixture(
        simulation=simulation,
        models=(model,),
        data=(target,),
        indexes=(0,),
        losses=(_loss,),
        timepoint_values=tuple(float(value) for value in np.asarray(timepoint_array)),
    )


def _loss(model: Simulation, target: Any, index: int) -> tuple[jax.Array, jax.Array]:
    prediction = jnp.asarray(model.outputs[index].y_pred())
    loss = jnp.mean(jnp.square(prediction - target))
    return loss, loss


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


@contextlib.contextmanager
def count_host_materialisation() -> Iterator[Counter[str]]:
    counts: Counter[str] = Counter()
    array_types = {jax.Array, type(jnp.asarray(0.0))}
    originals: dict[tuple[type, str], Any] = {}
    for array_type in array_types:
        for name in ("__float__", "__int__", "__bool__", "item"):
            if hasattr(array_type, name):
                originals[(array_type, name)] = getattr(array_type, name)

    for (array_type, name), original in originals.items():

        def counted(self, *args, _original=original, _name=name, **kwargs):
            counts[f"{type(self).__name__}.{_name}"] += 1
            return _original(self, *args, **kwargs)

        setattr(array_type, name, counted)
    try:
        yield counts
    finally:
        for (array_type, name), original in originals.items():
            setattr(array_type, name, original)


@contextlib.contextmanager
def count_compiles() -> Iterator[dict[str, float | int]]:
    import jax._src.compiler as compiler

    stats: dict[str, float | int] = {"count": 0, "seconds": 0.0}
    original = compiler.backend_compile

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stats["count"] = int(stats["count"]) + 1
            stats["seconds"] = float(stats["seconds"]) + (time.perf_counter() - start)

    compiler.backend_compile = wrapped
    try:
        yield stats
    finally:
        compiler.backend_compile = original


def _make_settings(args: argparse.Namespace) -> OptimiserSettings:
    early_stop = args.mode == "profile" and args.allow_early_stop
    return OptimiserSettings(
        name="profile_hdx_cpu",
        n_steps=args.steps,
        tolerance=1e-12 if early_stop else 0.0,
        learning_rate=args.learning_rate,
        optimiser_type=args.optimizer,
        convergence=200.0 if early_stop else 0.0,
        min_steps_per_threshold=2 if early_stop else args.steps + 1,
    )


class PreparedPath:
    """A reusable path executor whose compiled callables survive warm repetitions."""

    def __init__(
        self,
        path: str,
        fixture: Fixture,
        settings: OptimiserSettings,
    ) -> None:
        self.path = path
        self.fixture = fixture
        self.settings = settings
        self.optimizer = optimiser_module.OptaxOptimizer(
            learning_rate=settings.learning_rate,
            optimizer=settings.optimiser_type,
        )
        self.initial_state = self.optimizer.initialise(fixture.simulation, None)

        self._pure_runner: Callable[[OptimizationState], Any] | None = None
        if path in ("pure", "jit"):

            def pure_runner(state: OptimizationState):
                carry, inputs, loss_functions, indexes = _build_chunk_state(
                    fixture.simulation,
                    fixture.data,
                    settings.tolerance,
                    settings.convergence,
                    fixture.indexes,
                    fixture.losses,
                    state,
                    self.optimizer,
                )
                inputs = inputs._replace(ema_alpha=jnp.asarray(settings.ema_alpha))
                result = run_sequential(
                    carry,
                    inputs,
                    settings.n_steps,
                    settings.step_chunk_size,
                    self.optimizer,
                    loss_functions,
                    indexes,
                    settings.min_steps_per_threshold,
                )
                return result.carry.opt_state

            self._pure_runner = jax.jit(pure_runner)

    def reset(self) -> None:
        self.optimizer.history = OptimizationHistory()
        self.optimizer._current_lr = float(self.optimizer.learning_rate)
        self.optimizer._current_model_lr = float(
            self.optimizer.learning_rate * self.optimizer.model_parameters_lr_scale
        )

    def run_once(self) -> OptimizationState:
        if self.path in ("pure", "jit"):
            if self._pure_runner is None:
                raise RuntimeError("pure runner was not prepared")
            return self._pure_runner(self.initial_state)

        terminal_states: list[OptimizationState] = []
        _optimise(
            self.fixture.simulation,
            self.fixture.data,
            self.settings.n_steps,
            self.settings.tolerance,
            self.settings.convergence,
            self.fixture.indexes,
            self.fixture.losses,
            self.initial_state,
            self.optimizer,
            ema_alpha=self.settings.ema_alpha,
            min_steps_per_threshold=self.settings.min_steps_per_threshold,
            silent=True,
            terminal_state_callback=terminal_states.append,
        )
        if len(terminal_states) != 1:
            raise RuntimeError(
                f"{self.path} captured {len(terminal_states)} terminal states; expected one"
            )
        return terminal_states[0]


def _completed_steps(state: OptimizationState) -> int:
    return int(np.asarray(state.step))


def _final_loss(state: OptimizationState) -> float:
    if state.losses is None:
        raise RuntimeError("terminal optimization state has no losses")
    return float(np.asarray(state.losses.total_train_loss))


def _parameter_snapshot(state: OptimizationState) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(leaf).copy()
        for leaf in jax.tree_util.tree_leaves(state.params)
        if hasattr(leaf, "shape")
    )


def _parameter_fingerprint(snapshot: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for leaf in snapshot:
        contiguous = np.ascontiguousarray(leaf)
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _snapshots_close(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> bool:
    return len(left) == len(right) and all(
        np.allclose(a, b, rtol=PARITY_RTOL, atol=PARITY_ATOL, equal_nan=False)
        for a, b in zip(left, right)
    )


@dataclass
class SampleReport:
    elapsed_s: float
    steps_completed: int
    steps_per_s: float
    compiles: int
    compile_s: float
    host_materialisations: int
    host_materialisations_per_step: float
    host_breakdown: dict[str, int]
    final_loss: float
    parameter_fingerprint: str


def _measure_sample(
    prepared: PreparedPath,
    requested_steps: int,
) -> tuple[SampleReport, tuple[np.ndarray, ...]]:
    prepared.reset()
    with count_compiles() as compile_stats, count_host_materialisation() as hosts:
        start = time.perf_counter()
        state = prepared.run_once()
        _block_tree(state)
        elapsed = time.perf_counter() - start

    completed = _completed_steps(state)
    if completed != requested_steps:
        raise RuntimeError(
            f"{prepared.path} completed {completed} steps; expected {requested_steps}"
        )
    final_loss = _final_loss(state)
    if not np.isfinite(final_loss):
        raise RuntimeError(f"{prepared.path} produced non-finite terminal loss {final_loss}")
    snapshot = _parameter_snapshot(state)
    host_total = sum(hosts.values())
    sample = SampleReport(
        elapsed_s=elapsed,
        steps_completed=completed,
        steps_per_s=requested_steps / elapsed,
        compiles=int(compile_stats["count"]),
        compile_s=float(compile_stats["seconds"]),
        host_materialisations=host_total,
        host_materialisations_per_step=host_total / requested_steps,
        host_breakdown=dict(sorted(hosts.items())),
        final_loss=final_loss,
        parameter_fingerprint=_parameter_fingerprint(snapshot),
    )
    return sample, snapshot


@dataclass
class TimingReport:
    path: str
    steps_requested: int
    fixture_setup_s: float
    path_setup_s: float
    cold: SampleReport
    warm: list[SampleReport]
    warm_median_s: float
    warm_mad_s: float
    warm_median_steps_per_s: float
    warm_compiles: int
    warm_cache_reused: bool
    repetitions_deterministic: bool
    validation_errors: list[str]


@dataclass
class ProfileReport:
    path: str
    steps_requested: int
    steps_completed: int
    fixture_setup_s: float
    path_setup_s: float
    profile_execution_s: float
    compiles: int
    compile_s: float
    host_materialisations: int
    host_breakdown: dict[str, int]
    final_loss: float
    parameter_fingerprint: str
    profile_file: str
    trace_dir: str | None
    hotspots: list[dict[str, Any]]


def _prepare(
    args: argparse.Namespace,
    settings: OptimiserSettings,
) -> tuple[Fixture, PreparedPath, float, float]:
    fixture_start = time.perf_counter()
    fixture = _make_fixture(
        args.frames,
        args.residues,
        args.timepoints,
        args.seed,
        args.target_frame,
    )
    fixture_setup_s = time.perf_counter() - fixture_start

    path_start = time.perf_counter()
    prepared = PreparedPath(args.path, fixture, settings)
    path_setup_s = time.perf_counter() - path_start
    return fixture, prepared, fixture_setup_s, path_setup_s


def _time_one(
    args: argparse.Namespace,
    settings: OptimiserSettings,
) -> tuple[TimingReport, tuple[np.ndarray, ...]]:
    _, prepared, fixture_setup_s, path_setup_s = _prepare(args, settings)
    cold, cold_snapshot = _measure_sample(prepared, args.steps)

    warm_reports: list[SampleReport] = []
    warm_snapshots: list[tuple[np.ndarray, ...]] = []
    for _ in range(args.warm_repeats):
        report, snapshot = _measure_sample(prepared, args.steps)
        warm_reports.append(report)
        warm_snapshots.append(snapshot)

    warm_times = np.asarray([sample.elapsed_s for sample in warm_reports])
    median = float(np.median(warm_times))
    mad = float(np.median(np.abs(warm_times - median)))
    warm_compiles = sum(sample.compiles for sample in warm_reports)
    deterministic = all(
        _snapshots_close(cold_snapshot, snapshot)
        and np.isclose(
            cold.final_loss,
            sample.final_loss,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
        for sample, snapshot in zip(warm_reports, warm_snapshots)
    )
    validation_errors = []
    if warm_compiles:
        validation_errors.append(
            f"warm repetitions performed {warm_compiles} backend compilations"
        )
    if not deterministic:
        validation_errors.append("cold and warm terminal states are not deterministic")

    return (
        TimingReport(
            path=args.path,
            steps_requested=args.steps,
            fixture_setup_s=fixture_setup_s,
            path_setup_s=path_setup_s,
            cold=cold,
            warm=warm_reports,
            warm_median_s=median,
            warm_mad_s=mad,
            warm_median_steps_per_s=args.steps / median,
            warm_compiles=warm_compiles,
            warm_cache_reused=warm_compiles == 0,
            repetitions_deterministic=deterministic,
            validation_errors=validation_errors,
        ),
        cold_snapshot,
    )


def _profile_one(
    args: argparse.Namespace,
    settings: OptimiserSettings,
) -> ProfileReport:
    _, prepared, fixture_setup_s, path_setup_s = _prepare(args, settings)
    prepared.reset()
    profile_path = args.output_dir / f"{args.path}.prof"
    trace_dir = args.trace_path or (args.output_dir / f"trace_{args.path}")
    if args.trace:
        trace_dir.mkdir(parents=True, exist_ok=True)

    profile = cProfile.Profile()
    with count_host_materialisation() as hosts, count_compiles() as compile_stats:
        start = time.perf_counter()
        trace_context = (
            jax.profiler.trace(str(trace_dir), create_perfetto_link=False)
            if args.trace
            else contextlib.nullcontext()
        )
        with trace_context:
            state = profile.runcall(prepared.run_once)
            _block_tree(state)
        execution_s = time.perf_counter() - start

    profile.dump_stats(str(profile_path))
    stats = pstats.Stats(profile).sort_stats("cumtime")
    hotspots: list[dict[str, Any]] = []
    for key in (stats.fcn_list or [])[: args.profile_limit]:
        filename, lineno, function = key
        primitive_calls, calls, total_s, cumulative_s, _ = stats.stats[key]
        hotspots.append(
            {
                "function": f"{filename}:{lineno}({function})",
                "primitive_calls": primitive_calls,
                "calls": calls,
                "tottime_s": total_s,
                "cumtime_s": cumulative_s,
            }
        )
    snapshot = _parameter_snapshot(state)
    return ProfileReport(
        path=args.path,
        steps_requested=args.steps,
        steps_completed=_completed_steps(state),
        fixture_setup_s=fixture_setup_s,
        path_setup_s=path_setup_s,
        profile_execution_s=execution_s,
        compiles=int(compile_stats["count"]),
        compile_s=float(compile_stats["seconds"]),
        host_materialisations=sum(hosts.values()),
        host_breakdown=dict(sorted(hosts.items())),
        final_loss=_final_loss(state),
        parameter_fingerprint=_parameter_fingerprint(snapshot),
        profile_file=str(profile_path),
        trace_dir=str(trace_dir) if args.trace else None,
        hotspots=hotspots,
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.target_frame >= args.frames:
        parser.error("--target-frame must be less than --frames")
    if args.steps is None:
        args.steps = 1000 if args.mode == "timing" else 100
    if args.mode == "timing":
        if args.trace or args.trace_path is not None:
            parser.error("--trace and --trace-path are profile-mode options")
        if args.allow_early_stop:
            parser.error("timing mode requires fixed-step execution")


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_parameter_snapshot(
    path: Path,
    snapshot: Sequence[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{f"leaf_{index:03d}": leaf for index, leaf in enumerate(snapshot)},
        )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = _make_settings(args)

    if args.mode == "timing":
        timing_report, terminal_snapshot = _time_one(args, settings)
        timing = [asdict(timing_report)]
        profile_reports: list[dict[str, Any]] = []
        validation_errors = timing_report.validation_errors
    else:
        profile_report = _profile_one(args, settings)
        terminal_snapshot = ()
        timing = []
        profile_reports = [asdict(profile_report)]
        validation_errors = []

    run_config = {
        "mode": args.mode,
        "path": args.path,
        "steps": args.steps,
        "frames": args.frames,
        "residues": args.residues,
        "timepoints": args.timepoints,
        "timepoint_values": [
            float(value) for value in np.asarray(_timepoint_values(args.timepoints))
        ],
        "seed": args.seed,
        "target_frame": args.target_frame,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "warm_repeats": args.warm_repeats,
        "trace": args.trace,
        "trace_path": str(args.trace_path) if args.trace_path else None,
        "profile_limit": args.profile_limit,
        "output_dir": str(args.output_dir),
        "allow_early_stop": args.allow_early_stop,
        "json": str(args.json) if args.json else None,
        "terminal_npz": str(args.terminal_npz) if args.terminal_npz else None,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_config": run_config,
        "environment": _environment(),
        "timing": timing,
        "profile": profile_reports,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
    }
    report_path = args.json or args.output_dir / (
        "timing_report.json" if args.mode == "timing" else "profile_report.json"
    )
    if args.terminal_npz is not None and args.mode == "timing":
        _write_parameter_snapshot(args.terminal_npz, terminal_snapshot)
    _atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
