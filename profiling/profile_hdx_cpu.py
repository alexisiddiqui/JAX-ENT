#!/usr/bin/env python3
"""Profile the HDX-only BV optimisation paths on the CPU backend.

The fixture is deliberately synthetic and small enough to make repeated CPU
measurements practical.  Each path gets a new fixture and optimiser state.
The JSON report is intended to be both machine-readable and sufficient to
reproduce a run from its ``run_config`` object.
"""

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
TRACE_MODES = ("none", "selected", "all")
OPTIMIZERS = ("adam", "sgd", "adagrad", "adamw", "rmsprop", "lbfgs")


def parse_paths(value: str) -> list[str]:
    paths = [part.strip().lower() for part in value.split(",") if part.strip()]
    if (
        not paths
        or len(set(paths)) != len(paths)
        or any(path not in PATHS for path in paths)
    ):
        raise argparse.ArgumentTypeError(
            "paths must be a comma-separated subset of eager,jit,pure"
        )
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
    parser.add_argument("--steps", type=positive_int, default=300)
    parser.add_argument("--frames", type=positive_int, default=500)
    parser.add_argument("--residues", type=positive_int, default=140)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--target-frame", type=nonnegative_int, default=0)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--paths", type=parse_paths, default=list(PATHS))
    parser.add_argument("--trace", choices=TRACE_MODES, default="all")
    parser.add_argument("--trace-path", choices=PATHS, default=None)
    parser.add_argument("--profile-limit", type=positive_int, default=30)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("profiling-output"))
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def _make_fixture(
    frames: int, residues: int, seed: int, target_frame: int
) -> tuple[Any, list, tuple, list[int], list[Callable]]:
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
        heavy_contacts=heavy, acceptor_contacts=acceptor, k_ints=k_ints
    )

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
    simulation = Simulation(
        input_features=[feature], forward_models=[model], params=params
    )
    simulation.initialise()
    simulation = Simulation.forward(simulation, simulation.params)
    target = (
        jnp.asarray(model_config.forward_parameters.bv_bc) * heavy[:, target_frame]
        + jnp.asarray(model_config.forward_parameters.bv_bh) * acceptor[:, target_frame]
    )
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
        result = _optimise_pure(
            simulation,
            data,
            settings.n_steps,
            settings.tolerance,
            settings.convergence,
            indexes,
            losses,
            state,
            optimizer,
            ema_alpha=settings.ema_alpha,
            min_steps_per_threshold=settings.min_steps_per_threshold,
        )
        _block_tree(result)
        return result
    result = run_optimise(
        simulation=simulation,
        data_to_fit=data,
        config=settings,
        forward_models=models,
        indexes=indexes,
        loss_functions=losses,
        jit_update_step=path == "jit",
        silent=True,
    )
    _block_tree(result)
    return result


@dataclass
class PathReport:
    path: str
    repeats: list[float]
    timing_min_s: float
    timing_mean_s: float
    wall_s: float
    compiles: int
    host_transfers_total: int
    host_transfers_per_step: float
    host_breakdown: dict[str, int]
    hotspots: list[dict[str, Any]]
    profile_file: str
    trace_dir: str | None


def _profile_one(
    path: str, args: argparse.Namespace, settings: OptimiserSettings, out_dir: Path
) -> PathReport:
    profile_path = out_dir / f"{path}.prof"
    trace_dir = out_dir / f"trace_{path}"
    trace_enabled = args.trace == "all" or (
        args.trace == "selected" and args.trace_path == path
    )
    if trace_enabled:
        trace_dir.mkdir(parents=True, exist_ok=True)

    def invoke():
        fixture = _make_fixture(
            args.frames, args.residues, args.seed, args.target_frame
        )
        return _run(path, fixture, settings)

    profile = cProfile.Profile()
    with (
        count_host_materialisation() as host_counts,
        count_compiles() as compile_counts,
    ):
        start = time.perf_counter()
        with (
            jax.profiler.trace(str(trace_dir), create_perfetto_link=False)
            if trace_enabled
            else contextlib.nullcontext()
        ):
            result = profile.runcall(invoke)
            _block_tree(result)
        wall = time.perf_counter() - start
    profile.dump_stats(str(profile_path))
    stats = pstats.Stats(profile).sort_stats("cumtime")
    hotspots = []
    for key in (stats.fcn_list or [])[: args.profile_limit]:
        filename, lineno, function = key
        primitive_calls, calls, total_s, cumulative_s, _callers = stats.stats[key]
        hotspots.append(
            {
                "function": f"{filename}:{lineno}({function})",
                "primitive_calls": primitive_calls,
                "calls": calls,
                "tottime_s": total_s,
                "cumtime_s": cumulative_s,
            }
        )

    repeats = [wall]
    for _ in range(args.repeats - 1):
        repeat_start = time.perf_counter()
        _run(
            path,
            _make_fixture(args.frames, args.residues, args.seed, args.target_frame),
            settings,
        )
        repeats.append(time.perf_counter() - repeat_start)
    total_host = sum(host_counts.values())
    return PathReport(
        path,
        repeats,
        min(repeats),
        sum(repeats) / len(repeats),
        wall,
        compile_counts["backend_compile"],
        total_host,
        total_host / args.steps,
        dict(sorted(host_counts.items())),
        hotspots,
        str(profile_path),
        str(trace_dir) if trace_enabled else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.target_frame >= args.frames:
        parser.error("--target-frame must be less than --frames")
    if args.trace == "selected" and (
        args.trace_path is None or args.trace_path not in args.paths
    ):
        parser.error("--trace selected requires --trace-path to name a selected path")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = OptimiserSettings(
        name="profile_hdx_cpu",
        n_steps=args.steps,
        # Keep the default run genuinely fixed-step.  The library default
        # (1e-2) is useful for normal optimisation but can stop this synthetic
        # fixture long before n_steps.
        tolerance=1e-12,
        learning_rate=args.learning_rate,
        optimiser_type=args.optimizer,
        convergence=200.0 if args.allow_early_stop else 0.0,
        min_steps_per_threshold=2 if args.allow_early_stop else args.steps + 1,
    )
    reports = [
        _profile_one(path, args, settings, args.output_dir) for path in args.paths
    ]
    run_config = {
        "steps": args.steps,
        "frames": args.frames,
        "residues": args.residues,
        "seed": args.seed,
        "target_frame": args.target_frame,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "paths": args.paths,
        "trace": args.trace,
        "trace_path": args.trace_path,
        "profile_limit": args.profile_limit,
        "repeats": args.repeats,
        "output_dir": str(args.output_dir),
        "allow_early_stop": args.allow_early_stop,
        "json": str(args.json) if args.json else None,
        "platform": jax.default_backend(),
    }
    report = {"run_config": run_config, "paths": [asdict(item) for item in reports]}
    report_path = args.json or args.output_dir / "profile_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
