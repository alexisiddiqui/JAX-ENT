#!/usr/bin/env python3
"""CPU profiling harness for jaxENT optimise paths.

This script is intentionally separate from package code. It builds synthetic
optimisation fixtures from existing tests and reports wall time, cProfile hot
spots, and simple host-materialisation counters.
"""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import importlib
import io
import pstats
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]


def _block_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


@contextlib.contextmanager
def count_host_materialisation():
    """Count common host materialisation paths used in Python loops."""

    counts: Counter[str] = Counter()
    array_types = {jax.Array, type(jnp.asarray(0.0))}
    originals = {}
    for array_type in array_types:
        for name in ("__float__", "__int__", "__bool__", "item"):
            if hasattr(array_type, name):
                originals[(array_type, name)] = getattr(array_type, name)

    def counted(array_type: type, name: str, original: Callable):
        def wrapper(self, *args, **kwargs):
            counts[f"{array_type.__name__}.{name}"] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for (array_type, name), original in originals.items():
        setattr(array_type, name, counted(array_type, name, original))
    try:
        yield counts
    finally:
        for (array_type, name), original in originals.items():
            setattr(array_type, name, original)


def make_synthetic_fixture(n_steps: int, n_frames: int, n_res: int, n_q: int, *, full_steps: bool):
    helpers = importlib.import_module("jaxent.tests.modules.optimise._mixed_model_optimise_helpers")
    run_mod = importlib.import_module("jaxent.src.opt.run")

    simulation, models, targets = helpers.build_mixed_simulation(
        include_saxs=True,
        include_xlms=True,
        n_frames=n_frames,
        n_res=n_res,
        n_q=n_q,
    )
    mappings, mapped_targets = helpers.build_mapping_targets_for_three_models(*targets)
    data_to_fit = tuple(zip(mappings, mapped_targets))
    loss_functions = (helpers.mapped_output_l2_loss,) * 3
    indexes = (0, 1, 2)
    config = helpers.make_test_optimiser_settings("perf_probe", n_steps=n_steps)
    if full_steps:
        config = replace(config, convergence=0.0, min_steps_per_threshold=n_steps + 1)
    return run_mod, simulation, models, data_to_fit, config, indexes, loss_functions


def run_python_loop(fixture, *, jit_update_step: bool):
    run_mod, simulation, models, data_to_fit, config, indexes, loss_functions = fixture
    return run_mod.run_optimise(
        simulation=simulation,
        data_to_fit=data_to_fit,
        config=config,
        forward_models=models,
        indexes=indexes,
        loss_functions=list(loss_functions),
        jit_update_step=jit_update_step,
        silent=True,
    )


def run_pure_loop(fixture):
    run_mod, simulation, _models, data_to_fit, config, indexes, loss_functions = fixture
    optimiser_mod = importlib.import_module("jaxent.src.opt.optimiser")

    optimizer = optimiser_mod.OptaxOptimizer(
        learning_rate=config.learning_rate,
        optimizer=config.optimiser_type,
    )
    opt_state = optimizer.initialise(simulation, None)
    carry = run_mod._optimise_pure(
        simulation,
        data_to_fit,
        config.n_steps,
        config.tolerance,
        config.convergence,
        indexes,
        loss_functions,
        opt_state,
        optimizer,
        ema_alpha=config.ema_alpha,
        min_steps_per_threshold=config.min_steps_per_threshold,
    )
    _block_tree(carry)
    return carry


def time_call(name: str, func: Callable, *, repeats: int = 1):
    timings = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func()
        _block_tree(last_result)
        timings.append(time.perf_counter() - start)
    print(f"{name}: min={min(timings):.6f}s mean={sum(timings) / len(timings):.6f}s runs={timings}")
    return last_result


def profile_call(name: str, func: Callable, limit: int):
    profile = cProfile.Profile()
    with count_host_materialisation() as counts:
        start = time.perf_counter()
        result = profile.runcall(func)
        _block_tree(result)
        elapsed = time.perf_counter() - start
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats("cumtime")
    stats.print_stats(limit)
    print(f"\n=== {name} cProfile elapsed={elapsed:.6f}s ===")
    print(stream.getvalue())
    print(f"=== {name} host materialisation counters ===")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    if not counts:
        print("(none)")
    return result


def profile_micro_paths(fixture, limit: int):
    run_mod, simulation, _models, data_to_fit, config, indexes, loss_functions = fixture
    optimizer_mod = importlib.import_module("jaxent.src.opt.optimiser")

    optimizer = optimizer_mod.OptaxOptimizer(
        learning_rate=config.learning_rate,
        optimizer=config.optimiser_type,
    )
    opt_state = optimizer.initialise(simulation, None)

    def compute_loss_once():
        return optimizer_mod.compute_loss(
            simulation,
            opt_state.params,
            tuple(data_to_fit),
            tuple(indexes),
            tuple(loss_functions),
        )

    profile_call("compute_loss_once", compute_loss_once, limit)

    def step_once():
        return optimizer.step(
            optimizer=optimizer,
            state=opt_state,
            simulation=simulation,
            data_targets=tuple(data_to_fit),
            loss_functions=tuple(loss_functions),
            indexes=tuple(indexes),
        )

    profile_call("optimizer_step_once", step_once, limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--residues", type=int, default=128)
    parser.add_argument("--q", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile-limit", type=int, default=30)
    parser.add_argument("--mode", choices=["summary", "profile", "micro"], default="profile")
    parser.add_argument("--full-steps", action="store_true")
    args = parser.parse_args()

    print("JAX platform:", jax.default_backend())
    print(
        "fixture:",
        f"steps={args.steps}",
        f"frames={args.frames}",
        f"residues={args.residues}",
        f"q={args.q}",
    )
    fixture = make_synthetic_fixture(
        n_steps=args.steps,
        n_frames=args.frames,
        n_res=args.residues,
        n_q=args.q,
        full_steps=args.full_steps,
    )

    if args.mode == "summary":
        time_call("python_loop_eager_step", lambda: run_python_loop(fixture, jit_update_step=False), repeats=args.repeats)
        time_call("python_loop_jit_step", lambda: run_python_loop(fixture, jit_update_step=True), repeats=args.repeats)
        time_call("pure_lax_loop", lambda: run_pure_loop(fixture), repeats=args.repeats)
    elif args.mode == "micro":
        profile_micro_paths(fixture, args.profile_limit)
    else:
        profile_call("python_loop_eager_step", lambda: run_python_loop(fixture, jit_update_step=False), args.profile_limit)
        profile_call("python_loop_jit_step", lambda: run_python_loop(fixture, jit_update_step=True), args.profile_limit)
        profile_call("pure_lax_loop", lambda: run_pure_loop(fixture), args.profile_limit)


if __name__ == "__main__":
    main()
