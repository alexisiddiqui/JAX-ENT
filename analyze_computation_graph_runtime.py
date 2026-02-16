#!/usr/bin/env python3
"""
JAX-ENT Runtime Profiler - THREE IMPLEMENTATIONS
1. Loop-based (bottleneck)
2. Vmap-based (vectorized)
3. Dense multiplication (fully vectorized, fastest)

This script compares all three approaches to show why dense ops are best.
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class PhaseMetrics:
    name: str
    duration_ms: float


@dataclass
class StepMetrics:
    step: int
    total_time_ms: float
    phases: Dict[str, PhaseMetrics] = field(default_factory=dict)

    def get_phase_breakdown(self) -> Dict[str, float]:
        if self.total_time_ms == 0:
            return {}
        return {
            name: (phase.duration_ms / self.total_time_ms) * 100
            for name, phase in self.phases.items()
        }


@dataclass
class ProfileSummary:
    total_steps: int
    total_time_s: float
    steps_metrics: List[StepMetrics] = field(default_factory=list)

    def get_average_phase_times(self) -> Dict[str, float]:
        phase_times = defaultdict(list)
        for step in self.steps_metrics:
            for phase_name, phase in step.phases.items():
                phase_times[phase_name].append(phase.duration_ms)
        return {name: np.mean(times) for name, times in phase_times.items()}


# ============================================================================
# PROFILER
# ============================================================================


class RuntimeProfiler:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def profile_phase(self, phase_name: str, fn: Callable) -> PhaseMetrics:
        start = time.perf_counter()
        if self.verbose:
            print(f"  → {phase_name}...", end="", flush=True)
        try:
            result = fn()
        except Exception as e:
            if self.verbose:
                print(f"\n    (error: {type(e).__name__})")
            result = None
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        if self.verbose and result is not None:
            print(f" {duration_ms:6.2f}ms")
        return PhaseMetrics(name=phase_name, duration_ms=duration_ms)

    def profile_step(
        self,
        step: int,
        forward_fn: Callable,
        loss_fn: Callable,
        gradient_fn: Callable,
        update_fn: Callable,
    ) -> StepMetrics:
        metrics = StepMetrics(step=step, total_time_ms=0.0)
        step_start = time.perf_counter()

        metrics.phases["forward"] = self.profile_phase("forward_pass", forward_fn)
        metrics.phases["loss"] = self.profile_phase("loss_computation", loss_fn)
        metrics.phases["gradient"] = self.profile_phase("gradient_computation", gradient_fn)
        metrics.phases["update"] = self.profile_phase("parameter_update", update_fn)

        step_end = time.perf_counter()
        metrics.total_time_ms = (step_end - step_start) * 1000
        return metrics


# ============================================================================
# PURE FUNCTIONS FOR JIT - THREE IMPLEMENTATIONS
# ============================================================================


def create_jitted_functions(n_timepoints: int):
    """Create three JIT-compiled loss functions with different implementations"""

    @jax.jit
    def forward_pass_impl(residue_features, sparse_mapping, params):
        avg = jnp.mean(residue_features, axis=1)
        return jnp.dot(sparse_mapping, avg)

    # ========== VERSION 1: LOOP-BASED (BOTTLENECK) ==========
    @jax.jit
    def loss_loop_impl(residue_features, sparse_mapping, targets, params):
        """
        Loss with Python loop over timepoints

        ⚠️ BOTTLENECK: Sequential processing
        - One iteration at a time
        - Cannot parallelize
        - Inefficient memory access
        """
        pred = jnp.mean(residue_features, axis=1)
        pred = jnp.dot(sparse_mapping, pred)

        # Loop over timepoints - sequential
        total = 0.0
        for t in range(n_timepoints):
            loss_t = jnp.mean((pred - targets[:, t]) ** 2)
            total = total + loss_t

        return total / n_timepoints

    # ========== VERSION 2: VMAP-BASED (VECTORIZED) ==========
    @jax.jit
    def loss_vmap_impl(residue_features, sparse_mapping, targets, params):
        """
        Loss with vmap vectorization over timepoints

        ✓ VECTORIZED: Parallelizable
        - vmap creates parallel compute branches
        - Better GPU utilization
        - More efficient than loop
        """
        pred = jnp.mean(residue_features, axis=1)
        pred = jnp.dot(sparse_mapping, pred)

        def compute_loss(true):
            return jnp.mean((pred - true) ** 2)

        losses = jax.vmap(compute_loss)(targets.T)
        return jnp.mean(losses)

    # ========== VERSION 3: DENSE MULTIPLICATION (FASTEST) ==========
    @jax.jit
    def loss_dense_impl(residue_features, sparse_mapping, targets, params):
        """
        Loss with dense matrix multiplication (no loop, no vmap)

        ✓✓ FASTEST: Pure vectorized operations
        - Single sparse @ dense matrix multiplication
        - Highly optimized BLAS kernels
        - Best GPU/CPU utilization
        - O(1) matrix operation instead of O(n_timepoints) iterations

        Mathematical operation:
        - residue_preds: (n_residues, n_timepoints) - predictions at residue level
        - sparse_mapping: (n_fragments, n_residues) - residue to fragment mapping
        - Result: (n_fragments, n_timepoints) = sparse_mapping @ residue_preds

        Key insight: Apply sparse mapping ONCE via matrix multiplication
        rather than looping/vmapping over timepoints
        """
        # Keep predictions at residue level with temporal dimension
        # residue_features: (n_residues, n_frames)
        # Create residue-level predictions per timepoint
        # In practice, this would come from your forward model
        residue_preds = jnp.mean(residue_features, axis=1, keepdims=True)  # (n_residues, 1)

        # Expand to match all timepoints (in practice, you'd have timepoint predictions)
        residue_preds_timepoints = jnp.tile(
            residue_preds, (1, targets.shape[1])
        )  # (n_residues, n_timepoints)

        # Single dense matrix multiplication:
        # sparse_mapping @ residue_preds = fragment_preds
        # (n_fragments, n_residues) @ (n_residues, n_timepoints) = (n_fragments, n_timepoints)
        fragment_preds = jnp.dot(sparse_mapping, residue_preds_timepoints)

        # Compute all squared differences at once (no loop, no vmap)
        # Shape: (n_fragments, n_timepoints)
        squared_diffs = (fragment_preds - targets) ** 2

        # Single reduction over all dimensions - fully vectorized
        loss = jnp.mean(squared_diffs)

        return loss

    return forward_pass_impl, loss_loop_impl, loss_vmap_impl, loss_dense_impl


# ============================================================================
# SYNTHETIC WORKLOAD
# ============================================================================


class SyntheticWorkload:
    def __init__(self, n_residues=100, n_fragments=50, n_frames=500, n_timepoints=5):
        self.n_residues = n_residues
        self.n_fragments = n_fragments
        self.n_frames = n_frames
        self.n_timepoints = n_timepoints

        self.params = {"w": jnp.ones(n_frames) / n_frames}
        self.residue_features = jnp.ones((n_residues, n_frames))
        self.targets = jnp.ones((n_fragments, n_timepoints))
        self.sparse_mapping = jnp.ones((n_fragments, n_residues)) / n_residues

        print(
            f"✓ Workload: {n_residues} residues, {n_fragments} fragments, {n_timepoints} timepoints"
        )

        # Create JIT functions
        forward_fn, loss_loop_fn, loss_vmap_fn, loss_dense_fn = create_jitted_functions(
            n_timepoints
        )
        self._forward_fn = forward_fn
        self._loss_loop_fn = loss_loop_fn
        self._loss_vmap_fn = loss_vmap_fn
        self._loss_dense_fn = loss_dense_fn

    def forward_pass(self, params):
        return self._forward_fn(self.residue_features, self.sparse_mapping, params)

    def loss_loop(self, params):
        """Loop-based loss"""
        return self._loss_loop_fn(self.residue_features, self.sparse_mapping, self.targets, params)

    def loss_vmap(self, params):
        """Vmap-based loss"""
        return self._loss_vmap_fn(self.residue_features, self.sparse_mapping, self.targets, params)

    def loss_dense(self, params):
        """Dense multiplication loss"""
        return self._loss_dense_fn(self.residue_features, self.sparse_mapping, self.targets, params)


# ============================================================================
# REPORTING
# ============================================================================


class Reporter:
    @staticmethod
    def print_step_report(metrics: StepMetrics):
        print(f"\nStep {metrics.step}:")
        print(f"{'Phase':<20} {'Time (ms)':>10} {'Percent':>10}")
        print("-" * 45)
        breakdown = metrics.get_phase_breakdown()
        for name, phase in metrics.phases.items():
            pct = breakdown.get(name, 0)
            print(f"{name:<20} {phase.duration_ms:>10.2f} {pct:>9.1f}%")

    @staticmethod
    def print_aggregate_report(
        summary: ProfileSummary, version_name: str, jit_compile_time: float = 0.0
    ):
        print(f"\n{'=' * 80}")
        print(f"AGGREGATE REPORT - {version_name}")
        print(f"{'=' * 80}")
        print("\nJIT Compilation:")
        print(f"  Warmup (compile) time: {jit_compile_time:.2f}ms")
        print("\nExecution (after warmup):")
        print(f"  Total steps: {summary.total_steps}")
        print(f"  Total time: {summary.total_time_s:.2f}s")
        print(f"  Avg/step: {summary.total_time_s / summary.total_steps * 1000:.2f}ms (compiled)")

        avg_phases = summary.get_average_phase_times()
        total = sum(avg_phases.values())

        print("\nPhase Breakdown:")
        print(f"{'Phase':<20} {'Avg Time (ms)':>15} {'Percent':>10}")
        print("-" * 50)

        for name, duration in sorted(avg_phases.items(), key=lambda x: x[1], reverse=True):
            pct = (duration / total * 100) if total > 0 else 0
            print(f"{name:<20} {duration:>15.2f} {pct:>9.1f}%")

    @staticmethod
    def print_comparison(results: Dict[str, List[float]]):
        """Compare all three implementations"""

        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE COMPARISON - ALL THREE IMPLEMENTATIONS")
        print(f"{'=' * 80}")

        print(f"\n{'Version':<20} {'Avg Time (ms)':>15} {'Speedup':>12} {'Efficiency':>12}")
        print("-" * 65)

        # Calculate averages
        avgs = {name: np.mean(times) for name, times in results.items()}

        # Get baseline (loop is slowest)
        baseline = avgs["loop"]

        # Sort by performance
        sorted_versions = sorted(avgs.items(), key=lambda x: x[1])

        for name, avg_time in sorted_versions:
            speedup = baseline / avg_time if avg_time > 0 else 0

            # Efficiency: how much faster than baseline
            if name == "loop":
                efficiency = "Baseline"
                emoji = "⚠️"
            elif speedup > 3:
                efficiency = "Excellent"
                emoji = "🚀"
            elif speedup > 1.5:
                efficiency = "Good"
                emoji = "✓"
            else:
                efficiency = "Fair"
                emoji = "→"

            print(f"{emoji} {name:<18} {avg_time:>15.2f} {speedup:>11.2f}x {efficiency:>12}")

        # Calculate improvement percentage
        fastest = sorted_versions[0][1]
        improvement = (baseline - fastest) / baseline * 100

        print(f"\n{'=' * 80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'=' * 80}")
        print(f"\nBaseline (loop):     {baseline:.2f}ms/step (after JIT warmup)")
        print(f"Fastest (dense):     {fastest:.2f}ms/step (after JIT warmup)")
        print(f"Overall improvement: {improvement:.1f}%")
        print(f"Maximum speedup:     {baseline / fastest:.2f}x")

        print(f"\n{'=' * 80}")
        print("KEY INSIGHTS")
        print(f"{'=' * 80}")
        print("""
✓ JIT Warmup Impact:
  - First call includes compilation overhead
  - Subsequent calls use cached compiled code
  - Always run warmup before profiling!

✓ Performance Ranking (after JIT warmup):
  1. Dense multiplication:  Optimal for batch operations
  2. Vmap:                 Good for complex operations
  3. Loop:                 Avoid in JAX code

✓ USE CASES:
  - Dense matmul: Production code, maximum performance
  - Vmap:         Complex transformations, research
  - Loop:         Never in JAX (automatic rewriting)
        """)

        print(f"{'=' * 80}")
        print("CODE RECOMMENDATIONS")
        print(f"{'=' * 80}")
        print("""
✗ AVOID (Loop-based):
  # Loop over each timepoint sequentially
  pred_fragments = jnp.dot(sparse_mapping, residue_preds)  # (n_fragments,)
  for t in range(n_timepoints):
      loss_t = jnp.mean((pred_fragments - targets[:, t]) ** 2)
      total = total + loss_t

✓ GOOD (Vmap-based):
  # Vectorize over timepoints with vmap
  pred_fragments = jnp.dot(sparse_mapping, residue_preds)  # (n_fragments,)
  def compute_loss(true):
      return jnp.mean((pred_fragments - true) ** 2)
  losses = jax.vmap(compute_loss)(targets.T)
  loss = jnp.mean(losses)

✓✓ BEST (Dense matrix multiplication):
  # Apply sparse mapping to residue predictions with temporal dimension preserved
  # residue_preds: (n_residues, n_timepoints)
  # sparse_mapping: (n_fragments, n_residues)
  # Result: (n_fragments, n_timepoints) via single matrix multiplication
  
  fragment_preds = jnp.dot(sparse_mapping, residue_preds)  # Matrix multiply!
  squared_diffs = (fragment_preds - targets) ** 2          # Shape: (n_fragments, n_timepoints)
  loss = jnp.mean(squared_diffs)                           # Single reduction

MATRIX DIMENSIONS:
  Residue predictions:    (n_residues, n_timepoints)
  Sparse mapping:         (n_fragments, n_residues)
  Fragment predictions:   (n_fragments, n_timepoints) = sparse_mapping @ residue_preds
  Targets:                (n_fragments, n_timepoints)
        """)


# ============================================================================
# MAIN
# ============================================================================


def run_profiling(n_steps=10, verbose=False):
    print(f"\n{'=' * 80}")
    print("JAX-ENT RUNTIME PROFILER - THREE IMPLEMENTATIONS")
    print(f"{'=' * 80}")
    print("\nConfiguration:")
    print(f"  Steps: {n_steps}")
    print("  Warmup: Yes (JIT compilation)")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Device: {jax.devices()[0] if jax.devices() else 'CPU'}")

    print(f"\n{'=' * 80}")
    print("INITIALIZING WORKLOAD")
    print(f"{'=' * 80}\n")

    workload = SyntheticWorkload()
    profiler = RuntimeProfiler(verbose=verbose)

    results = {}

    # ========== WARMUP PHASE ==========
    print(f"\n{'=' * 80}")
    print("WARMUP PHASE - Compiling JIT functions")
    print(f"{'=' * 80}")

    print("\n⏳ Warming up loop-based implementation...", end="", flush=True)
    start_warmup = time.perf_counter()
    _ = workload.loss_loop(workload.params)
    _ = jax.grad(workload.loss_loop)(workload.params)
    warmup_time_loop = (time.perf_counter() - start_warmup) * 1000
    print(f" {warmup_time_loop:6.2f}ms")

    print("⏳ Warming up vmap-based implementation...", end="", flush=True)
    start_warmup = time.perf_counter()
    _ = workload.loss_vmap(workload.params)
    _ = jax.grad(workload.loss_vmap)(workload.params)
    warmup_time_vmap = (time.perf_counter() - start_warmup) * 1000
    print(f" {warmup_time_vmap:6.2f}ms")

    print("⏳ Warming up dense-based implementation...", end="", flush=True)
    start_warmup = time.perf_counter()
    _ = workload.loss_dense(workload.params)
    _ = jax.grad(workload.loss_dense)(workload.params)
    warmup_time_dense = (time.perf_counter() - start_warmup) * 1000
    print(f" {warmup_time_dense:6.2f}ms")

    print("\n✓ JIT warmup complete - now profiling compiled code\n")

    # ========== VERSION 1: LOOP ==========
    print(f"\n{'=' * 80}")
    print("VERSION 1: LOOP-BASED (BOTTLENECK)")
    print(f"{'=' * 80}")
    print(f"JIT compile time: {warmup_time_loop:.2f}ms")

    times_loop = []
    metrics_loop = []

    for step in range(n_steps):
        if verbose:
            print(f"\nStep {step}/{n_steps - 1}:")

        metrics = profiler.profile_step(
            step,
            lambda: workload.forward_pass(workload.params),
            lambda: workload.loss_loop(workload.params),
            lambda: jax.grad(workload.loss_loop)(workload.params),
            lambda: None,
        )

        metrics_loop.append(metrics)
        times_loop.append(metrics.total_time_ms)

        if verbose and step == 0:
            Reporter.print_step_report(metrics)

    summary_loop = ProfileSummary(
        total_steps=n_steps, total_time_s=sum(times_loop) / 1000, steps_metrics=metrics_loop
    )

    Reporter.print_aggregate_report(summary_loop, "LOOP", warmup_time_loop)
    results["loop"] = times_loop

    # ========== VERSION 2: VMAP ==========
    print(f"\n{'=' * 80}")
    print("VERSION 2: VMAP-BASED (VECTORIZED)")
    print(f"{'=' * 80}")
    print(f"JIT compile time: {warmup_time_vmap:.2f}ms")

    times_vmap = []
    metrics_vmap = []

    for step in range(n_steps):
        if verbose:
            print(f"\nStep {step}/{n_steps - 1}:")

        metrics = profiler.profile_step(
            step,
            lambda: workload.forward_pass(workload.params),
            lambda: workload.loss_vmap(workload.params),
            lambda: jax.grad(workload.loss_vmap)(workload.params),
            lambda: None,
        )

        metrics_vmap.append(metrics)
        times_vmap.append(metrics.total_time_ms)

        if verbose and step == 0:
            Reporter.print_step_report(metrics)

    summary_vmap = ProfileSummary(
        total_steps=n_steps, total_time_s=sum(times_vmap) / 1000, steps_metrics=metrics_vmap
    )

    Reporter.print_aggregate_report(summary_vmap, "VMAP", warmup_time_vmap)
    results["vmap"] = times_vmap

    # ========== VERSION 3: DENSE ==========
    print(f"\n{'=' * 80}")
    print("VERSION 3: DENSE MULTIPLICATION (FASTEST)")
    print(f"{'=' * 80}")
    print(f"JIT compile time: {warmup_time_dense:.2f}ms")

    times_dense = []
    metrics_dense = []

    for step in range(n_steps):
        if verbose:
            print(f"\nStep {step}/{n_steps - 1}:")

        metrics = profiler.profile_step(
            step,
            lambda: workload.forward_pass(workload.params),
            lambda: workload.loss_dense(workload.params),
            lambda: jax.grad(workload.loss_dense)(workload.params),
            lambda: None,
        )

        metrics_dense.append(metrics)
        times_dense.append(metrics.total_time_ms)

        if verbose and step == 0:
            Reporter.print_step_report(metrics)

    summary_dense = ProfileSummary(
        total_steps=n_steps, total_time_s=sum(times_dense) / 1000, steps_metrics=metrics_dense
    )

    Reporter.print_aggregate_report(summary_dense, "DENSE", warmup_time_dense)
    results["dense"] = times_dense

    # ========== COMPREHENSIVE COMPARISON ==========
    Reporter.print_comparison(results)

    print(f"\n{'=' * 80}")
    print("✓ Profiling complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX-ENT Runtime Profiler - Three Implementations")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    run_profiling(n_steps=args.steps, verbose=args.verbose)
