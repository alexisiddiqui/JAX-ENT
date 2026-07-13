#!/usr/bin/env python3
"""Regression-oriented metrics probe for the jaxENT optimise paths.

Complements ``optimise_perf_probe.py`` (which gives cProfile hot spots). This
script focuses on the two numbers that predict GPU behaviour but are cheap to
measure on the CPU backend:

  * host materialisations  -> device->host syncs per optimisation step. Each one
    is a pipeline stall on GPU. The CPU cost is tiny, so wall time hides them;
    the COUNT is the proxy metric.
  * XLA compilations        -> number of ``backend_compile`` calls. Recompilation
    inside the loop (constants folded into programs, eager per-primitive
    dispatch) shows up here.

It reuses the fixtures and host-materialisation counter from
``optimise_perf_probe.py`` so the two scripts stay in sync.

Usage:
    python profiling/optimise_perf_metrics.py --steps 150 --frames 2048 --residues 140
    python profiling/optimise_perf_metrics.py --json metrics.json   # machine-readable

Exit code is non-zero if --max-host-transfers-per-step is exceeded by any path
(useful as a CI guard once a target budget is agreed).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import jax

THIS_DIR = Path(__file__).resolve().parent

# Import the sibling probe as a module without requiring a package install.
_spec = importlib.util.spec_from_file_location(
    "optimise_perf_probe", THIS_DIR / "optimise_perf_probe.py"
)
assert _spec and _spec.loader
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@contextlib.contextmanager
def count_compiles():
    """Count XLA ``backend_compile`` invocations during the block."""
    import jax._src.compiler as compiler_mod

    counts = Counter()
    original = compiler_mod.backend_compile

    def wrapper(*args, **kwargs):
        counts["backend_compile"] += 1
        return original(*args, **kwargs)

    compiler_mod.backend_compile = wrapper
    try:
        yield counts
    finally:
        compiler_mod.backend_compile = original


@dataclass
class PathMetrics:
    name: str
    steps: int
    wall_s: float
    compiles: int
    host_transfers_total: int
    host_transfers_per_step: float
    host_breakdown: dict[str, int]


def measure(name: str, func: Callable, steps: int) -> PathMetrics:
    with probe.count_host_materialisation() as host_counts, count_compiles() as compile_counts:
        start = time.perf_counter()
        result = func()
        probe._block_tree(result)
        wall = time.perf_counter() - start

    total_host = sum(host_counts.values())
    return PathMetrics(
        name=name,
        steps=steps,
        wall_s=wall,
        compiles=compile_counts["backend_compile"],
        host_transfers_total=total_host,
        host_transfers_per_step=total_host / steps if steps else float("nan"),
        host_breakdown=dict(sorted(host_counts.items())),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument("--residues", type=int, default=140)
    parser.add_argument("--q", type=int, default=256)
    parser.add_argument(
        "--full-steps",
        action="store_true",
        default=True,
        help="Force every path to run all steps (apples-to-apples). On by default.",
    )
    parser.add_argument("--allow-early-stop", dest="full_steps", action="store_false")
    parser.add_argument("--json", type=str, default=None, help="Write metrics JSON to this path.")
    parser.add_argument(
        "--max-host-transfers-per-step",
        type=float,
        default=None,
        help="If set, exit non-zero when any path exceeds this budget.",
    )
    args = parser.parse_args()

    print("JAX platform:", jax.default_backend())
    print(f"fixture: steps={args.steps} frames={args.frames} residues={args.residues} q={args.q} "
          f"full_steps={args.full_steps}")

    # NOTE: build one fixture per path. The fixtures mutate optimiser/simulation
    # state during a run, so reusing across paths would contaminate measurements.
    def fresh_fixture():
        return probe.make_synthetic_fixture(
            n_steps=args.steps,
            n_frames=args.frames,
            n_res=args.residues,
            n_q=args.q,
            full_steps=args.full_steps,
        )

    paths: list[PathMetrics] = []
    paths.append(measure(
        "python_loop_eager",
        lambda: probe.run_python_loop(fresh_fixture(), jit_update_step=False),
        args.steps,
    ))
    paths.append(measure(
        "python_loop_jit",
        lambda: probe.run_python_loop(fresh_fixture(), jit_update_step=True),
        args.steps,
    ))
    paths.append(measure(
        "pure_lax_loop",
        lambda: probe.run_pure_loop(fresh_fixture()),
        args.steps,
    ))

    # Report
    header = f"{'path':<20}{'wall_s':>10}{'compiles':>10}{'host/step':>12}{'host_total':>12}"
    print("\n" + header)
    print("-" * len(header))
    for m in paths:
        print(f"{m.name:<20}{m.wall_s:>10.3f}{m.compiles:>10d}{m.host_transfers_per_step:>12.2f}"
              f"{m.host_transfers_total:>12d}")

    print("\nhost materialisation breakdown (device->host syncs):")
    for m in paths:
        print(f"  {m.name}: {m.host_breakdown or '(none)'}")

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(m) for m in paths], indent=2))
        print(f"\nwrote {args.json}")

    if args.max_host_transfers_per_step is not None:
        offenders = [m for m in paths if m.host_transfers_per_step > args.max_host_transfers_per_step]
        if offenders:
            print(f"\nFAIL: budget {args.max_host_transfers_per_step} host transfers/step exceeded by:")
            for m in offenders:
                print(f"  {m.name}: {m.host_transfers_per_step:.2f}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
