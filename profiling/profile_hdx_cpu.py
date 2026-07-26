#!/usr/bin/env python3
"""Benchmark or profile the HDX-only BV optimisation paths on CPU."""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import json
import pstats
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

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
from jaxent.src.opt.run import _optimise_pure, run_optimise


PATHS = ("eager", "jit", "pure")
OPTIMIZERS = ("adam", "sgd", "adagrad", "adamw", "rmsprop", "lbfgs")


def parse_paths(value: str) -> list[str]:
    paths = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not paths or len(set(paths)) != len(paths) or any(path not in PATHS for path in paths):
        raise argparse.ArgumentTypeError("paths must be a comma-separated subset of eager,jit,pure")
    return paths


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
    parser.add_argument("--steps", type=positive_int, default=None)
    parser.add_argument("--frames", type=positive_int, default=500)
    parser.add_argument("--residues", type=positive_int, default=140)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--target-frame", type=nonnegative_int, default=0)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--paths", type=parse_paths, default=None)
    parser.add_argument("--path", choices=PATHS, default=None)
    parser.add_argument("--warm-repeats", type=positive_int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--profile-limit", type=positive_int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("profiling-output"))
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def _make_fixture(frames: int, residues: int, seed: int, target_frame: int) -> tuple[Any, list, tuple, list[int], list[Callable]]:
    if target_frame >= frames:
        raise ValueError(f"target-frame must be less than frames ({frames})")
    rng = np.random.default_rng(seed)
    residue_axis = np.linspace(0.6, 1.6, residues, dtype=np.float32)[:, None]
    frame_axis = np.linspace(0.75, 1.25, frames, dtype=np.float32)[None, :]
    noise = rng.normal(0.0, 0.01, size=(residues, frames)).astype(np.float32)
    heavy = jnp.asarray(residue_axis * frame_axis + 0.2 + noise)
    acceptor = jnp.asarray(residue_axis * np.flip(frame_axis, axis=1) + 0.1 + noise[::-1])
    k_ints = jnp.asarray(np.linspace(0.3, 1.1, residues, dtype=np.float32))
    feature = BV_input_features(heavy_contacts=heavy, acceptor_contacts=acceptor, k_ints=k_ints)
    model_config = BV_model_Config()
    model = BV_model(model_config)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(frames, dtype=jnp.float32) / frames,
        frame_mask=jnp.ones(frames, dtype=jnp.float32),
        model_parameters=[model_config.forward_parameters],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.float32),
    )
    simulation = Simulation(input_features=[feature], forward_models=[model], params=params)
    simulation.initialise()
    simulation = Simulation.forward(simulation, simulation.params)
    target = (jnp.asarray(model_config.forward_parameters.bv_bc) * heavy[:, target_frame]
              + jnp.asarray(model_config.forward_parameters.bv_bh) * acceptor[:, target_frame])
    return simulation, [model], (target,), [0], [_loss]


def _loss(model: Simulation, target: Any, index: int):
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
    originals = {}
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
def count_compiles() -> Iterator[Counter[str]]:
    import jax._src.compiler as compiler
    counts: Counter[str] = Counter()
    original = compiler.backend_compile
    def wrapped(*args, **kwargs):
        counts["backend_compile"] += 1
        return original(*args, **kwargs)
    compiler.backend_compile = wrapped
    try:
        yield counts
    finally:
        compiler.backend_compile = original


def _run(path: str, fixture: tuple, settings: OptimiserSettings) -> Any:
    simulation, models, data, indexes, losses = fixture
    if path == "pure":
        optimizer = optimiser_module.OptaxOptimizer(
            learning_rate=settings.learning_rate, optimizer=settings.optimiser_type
        )
        state = optimizer.initialise(simulation, None)
        return _optimise_pure(
            simulation, data, settings.n_steps, settings.tolerance, settings.convergence,
            indexes, losses, state, optimizer, ema_alpha=settings.ema_alpha,
            min_steps_per_threshold=settings.min_steps_per_threshold,
        )
    return run_optimise(
        simulation=simulation, data_to_fit=data, config=settings, forward_models=models,
        indexes=indexes, loss_functions=losses, jit_update_step=path == "jit", silent=True,
    )


def _make_settings(args: argparse.Namespace) -> OptimiserSettings:
    early_stop = args.mode == "profile" and args.allow_early_stop
    return OptimiserSettings(
        name="profile_hdx_cpu", n_steps=args.steps, tolerance=1e-12,
        learning_rate=args.learning_rate, optimiser_type=args.optimizer,
        convergence=200.0 if early_stop else 0.0,
        min_steps_per_threshold=2 if early_stop else args.steps + 1,
    )


def _completed_steps(path: str, result: Any) -> int:
    if path == "pure":
        return int(result.opt_state.step)
    return len(result[1].states)


def _result_params(path: str, result: Any) -> Any:
    if path == "pure":
        return result.opt_state.params
    history = result[1]
    return history.states[-1].params if history.states else result[0].params


def _final_loss(path: str, result: Any) -> float:
    losses = result.opt_state.losses if path == "pure" else result[1].states[-1].losses
    return float(np.asarray(losses.total_train_loss))


def _parameter_digest(path: str, result: Any) -> float:
    return float(sum(np.asarray(leaf, dtype=np.float64).sum() for leaf in jax.tree_util.tree_leaves(_result_params(path, result))))


def _validate_steps(path: str, result: Any, requested: int) -> int:
    completed = _completed_steps(path, result)
    if completed != requested:
        raise RuntimeError(f"{path} completed {completed} steps; expected {requested}")
    return completed


def _counter_total(counter: Counter[str]) -> int:
    return sum(counter.values())


@dataclass
class TimingReport:
    path: str
    steps_requested: int
    steps_completed: int
    setup_s: float
    cold_s: float
    warm_samples_s: list[float]
    warm_median_s: float
    warm_mad_s: float
    cold_compiles: int
    warm_compiles: int
    host_transfers_total: int
    host_transfers_per_step: float
    host_breakdown: dict[str, int]
    final_loss: float
    parameter_digest: float


@dataclass
class ProfileReport:
    path: str
    steps_requested: int
    steps_completed: int
    profile_execution_s: float
    compiles: int
    host_transfers_total: int
    host_breakdown: dict[str, int]
    profile_file: str
    trace_dir: str | None
    hotspots: list[dict[str, Any]]


def _fixture(args: argparse.Namespace) -> tuple[Any, list, tuple, list[int], list[Callable]]:
    return _make_fixture(args.frames, args.residues, args.seed, args.target_frame)


def _time_one(path: str, args: argparse.Namespace, settings: OptimiserSettings, out_dir: Path) -> TimingReport:
    setup_start = time.perf_counter()
    fixture = _fixture(args)
    setup_s = time.perf_counter() - setup_start
    with count_compiles() as cold_compiles, count_host_materialisation() as cold_hosts:
        start = time.perf_counter()
        result = _run(path, fixture, settings)
        _block_tree(result)
        cold_s = time.perf_counter() - start
    completed = _validate_steps(path, result, args.steps)
    warm_samples: list[float] = []
    warm_compile_count: Counter[str] = Counter()
    warm_host_count: Counter[str] = Counter()
    for _ in range(args.warm_repeats):
        warm_fixture = _fixture(args)
        with count_compiles() as compile_counts, count_host_materialisation() as host_counts:
            start = time.perf_counter()
            warm_result = _run(path, warm_fixture, settings)
            _block_tree(warm_result)
            warm_samples.append(time.perf_counter() - start)
        _validate_steps(path, warm_result, args.steps)
        warm_compile_count.update(compile_counts)
        warm_host_count.update(host_counts)
    median = float(np.median(warm_samples))
    mad = float(np.median(np.abs(np.asarray(warm_samples) - median)))
    all_hosts = cold_hosts + warm_host_count
    return TimingReport(
        path, args.steps, completed, setup_s, cold_s, warm_samples, median, mad,
        _counter_total(cold_compiles), _counter_total(warm_compile_count),
        _counter_total(all_hosts), _counter_total(all_hosts) / args.steps,
        dict(sorted(all_hosts.items())), _final_loss(path, result), _parameter_digest(path, result),
    )


def _profile_one(path: str, args: argparse.Namespace, settings: OptimiserSettings, out_dir: Path) -> ProfileReport:
    fixture = _fixture(args)
    profile_path = out_dir / f"{path}.prof"
    trace_dir = args.trace_path or (out_dir / f"trace_{path}")
    if args.trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
    profile = cProfile.Profile()
    with count_host_materialisation() as host_counts, count_compiles() as compile_counts:
        start = time.perf_counter()
        with (jax.profiler.trace(str(trace_dir), create_perfetto_link=False) if args.trace else contextlib.nullcontext()):
            result = profile.runcall(_run, path, fixture, settings)
            _block_tree(result)
        execution_s = time.perf_counter() - start
    completed = _completed_steps(path, result)
    profile.dump_stats(str(profile_path))
    stats = pstats.Stats(profile).sort_stats("cumtime")
    hotspots = []
    for key in (stats.fcn_list or [])[:args.profile_limit]:
        filename, lineno, function = key
        primitive_calls, calls, total_s, cumulative_s, _ = stats.stats[key]
        hotspots.append({"function": f"{filename}:{lineno}({function})", "primitive_calls": primitive_calls,
                         "calls": calls, "tottime_s": total_s, "cumtime_s": cumulative_s})
    return ProfileReport(path, args.steps, completed, execution_s, _counter_total(compile_counts),
                         _counter_total(host_counts), dict(sorted(host_counts.items())),
                         str(profile_path), str(trace_dir) if args.trace else None, hotspots)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.target_frame >= args.frames:
        parser.error("--target-frame must be less than --frames")
    if args.steps is None:
        args.steps = 1000 if args.mode == "timing" else 100
    if args.mode == "timing":
        if args.paths is None:
            parser.error("timing mode requires --paths")
        if args.path is not None or args.trace or args.trace_path is not None:
            parser.error("--path, --trace, and --trace-path are profile-mode options")
        if args.allow_early_stop:
            parser.error("timing mode requires fixed-step execution; remove --allow-early-stop")
    else:
        if args.path is None:
            parser.error("profile mode requires exactly one --path")
        if args.paths is not None:
            parser.error("profile mode accepts --path, not --paths")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = _make_settings(args)
    if args.mode == "timing":
        reports = [_time_one(path, args, settings, args.output_dir) for path in args.paths]
        timing = [asdict(report) for report in reports]
        profile = []
    else:
        report = _profile_one(args.path, args, settings, args.output_dir)
        timing = []
        profile = [asdict(report)]
    run_config = {"mode": args.mode, "steps": args.steps, "frames": args.frames, "residues": args.residues,
                  "seed": args.seed, "target_frame": args.target_frame, "optimizer": args.optimizer,
                  "learning_rate": args.learning_rate, "paths": args.paths, "path": args.path,
                  "warm_repeats": args.warm_repeats, "trace": args.trace,
                  "trace_path": str(args.trace_path) if args.trace_path else None,
                  "profile_limit": args.profile_limit, "output_dir": str(args.output_dir),
                  "allow_early_stop": args.allow_early_stop, "json": str(args.json) if args.json else None,
                  "platform": jax.default_backend()}
    report = {"run_config": run_config, "timing": timing, "profile": profile}
    report_path = args.json or args.output_dir / ("timing_report.json" if args.mode == "timing" else "profile_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
