# `lax.cond` Step Branch — High-Frame Performance Regression

Status: **diagnosed, fix validated, not applied.** The working tree is unchanged; every
experiment below was reverted. This document is the handoff for deciding whether to adopt
the fix.

Date: 2026-07-27. Machine: Apple M1 Max, CPU backend. All numbers are `warm_median_s` from
`profiling/profile_hdx_cpu.py --mode timing`, 1000 steps.

## Summary

The `cpu_optimisation_20260727` sweep flagged two baseline regressions, `frames_5000/jit`
(+15.9%) and `corner_high/jit` (+85.4%). Both are real. The cause is the per-step
`jax.lax.cond` in `optimisation_step` (`jaxent/src/opt/chunk.py:242`), which prevents XLA
from optimising the step body. Replacing it with an unconditional step plus a `select` on
the results recovers **2.20× on pure and 1.99× on jit** at the largest configuration, with
identical loss and parameter fingerprints and all 110 optimise tests passing.

The regression is confined to high frame counts. Below 5000 frames the chunked runner
remains a clear win over every pre-refactor path.

## Where the regression is

Comparing the current runner against the best pre-refactor path per configuration
(`initial_baseline_20260726`; "old jit" is the step-only JIT loop, "old pure" the
`while_loop`):

| config | frames | old jit | old pure | current | vs best old |
|---|---:|---:|---:|---:|---:|
| corner_low | 173 | 0.641 | 0.139 | 0.066 | 2.11× |
| anchor | 500 | 0.715 | 0.433 | 0.208 | 2.08× |
| residues_096 | 500 | 0.683 | 0.344 | 0.126 | 2.72× |
| residues_600 | 500 | 0.885 | 0.937 | 0.774 | 1.14× |
| frames_1125 | 1125 | 0.831 | 0.804 | 0.454 | 1.77× |
| frames_5000 | 5000 | 1.646 | 3.043 | 1.926 | **0.85×** |
| corner_high | 5000 | 4.149 | 9.150 | 7.980 | **0.52×** |

The driver is frames, not residues — `residues_600` at 500 frames is fine. The old
step-only JIT loop carried roughly 0.6 ms/step of Python overhead (visible as its ~0.63 s
floor at every small config), which masked better per-step device efficiency. The chunked
runner removed that floor, which is why it wins everywhere overhead dominated and loses
where compute dominates.

## Experiments

All at `corner_high` (600 residues × 5000 frames × 10 timepoints), `--path pure`.

| experiment | warm | vs current | verdict |
|---|---:|---:|---|
| current (`lax.cond` over full carry) | 7.510 s | — | baseline |
| exclude `sim` from the `tree_map` select | 7.377 s | 1.02× | refuted |
| exclude `sim` from the cond operand/result | 7.350 s | 1.02× | refuted |
| remove the cond entirely | 3.411 s | 2.20× | isolates the cause |
| **unconditional step + select on results** | **3.413 s** | **2.20×** | **candidate fix** |

Two plausible-sounding memory hypotheses were tested and **refuted**. Both assumed the
24 MB of loop-invariant input features (`heavy_contacts` / `acceptor_contacts`, shaped
`n_residues × n_frames`) were being copied per step because `sim` rides in `ChunkCarry`
(`chunk.py:39`). XLA already folds `select(p, x, x) → x` and handles the cond operand
fine, so neither cost anything. Recording this so the same ground is not re-covered: the
memory-traffic explanation is wrong, despite the regression correlating cleanly with array
size.

The cause is the `lax.cond` construct itself. Removing it and replacing it with a select
give the same number to within noise (3.411 vs 3.413 s), which is what pins the
attribution.

## The candidate fix

In `optimisation_step` (`jaxent/src/opt/chunk.py:242`), replace:

```python
return jax.lax.cond(carry.active, step, _no_op_step, carry)
```

with an unconditional step plus a select on the results — mirroring the finite/frozen
pattern already used inside `step` at `chunk.py:229-233`:

```python
stepped_carry, stepped_metrics = step(carry)
frozen_carry, frozen_metrics = _no_op_step(carry)
next_carry = jax.tree_util.tree_map(
    lambda good, frozen: jax.lax.select(carry.active, good, frozen),
    stepped_carry, frozen_carry,
)
next_metrics = jax.tree_util.tree_map(
    lambda good, frozen: jax.lax.select(carry.active, good, frozen),
    stepped_metrics, frozen_metrics,
)
return next_carry, next_metrics
```

### Measured effect — both paths

Matched runs, same machine, same session, unmodified vs patched:

| config | path | current | with fix | speedup |
|---|---|---:|---:|---:|
| corner_high | pure | 7.510 s | **3.413 s** | 2.20× |
| corner_high | jit | 7.401 s | **3.721 s** | 1.99× |
| anchor | pure | 0.2061 s | 0.2069 s | 1.00× |
| anchor | jit | 0.2043 s | 0.2077 s | 0.98× |

`anchor` is unchanged within noise on both paths — the fix neither helps nor hurts where
the step body is small. At `corner_high` the fix turns the worst regression into
**1.22× faster than the old best path** (4.149 s) on pure, and 1.12× on jit.

Note pure and jit are the same code post-refactor; the small residual spread between them
is run-to-run variation, and both were measured for this handoff because the sweep flagged
the regression on the jit rows specifically.

### Correctness

- All **110** optimise tests pass (`jaxent/tests/unit/opt/`,
  `jaxent/tests/modules/optimise/`, `jaxent/tests/integration/optimise/test_batch_optimise.py`),
  including the freeze-semantics tests: `test_stopped_batch_lane_remains_frozen`,
  `test_tolerance_termination_stops_before_n_steps`,
  `test_nonfinite_loss_freezes_last_finite_state`.
- Identical final loss and parameter fingerprint at `corner_high` on both paths
  (`42d5efe122878608…`).
- Zero host materialisations, zero warm compiles, unchanged.

## The tradeoff — read before adopting

The fix trades wasted compute on inactive steps for a step body XLA can optimise.
Previously a converged step took a cheap no-op branch; now it runs the full step and
discards the result via select.

**Runs that converge early in sequential mode will get slower, and this is unmeasured.**
None of the benchmarks above stop early — `corner_high` runs all 1000 steps — so the
regression case is not covered by any number in this document. Quantifying it is the first
thing to do before adopting.

This also reopens a decision settled in the item-7 plan: `run_chunks` deliberately avoids
early exit at chunk boundaries to keep host syncs at zero. That was clearly right when
inactive steps were cheap. If inactive steps now cost full compute, one host sync per chunk
boundary (10 for a 1000-step run at the default `step_chunk_size=100`) is likely the better
trade. The two decisions should be made together, not separately.

Consistency note: the item-7 plan already documented that `lax.cond` becomes a `select`
under `vmap`, so batched lanes never got per-lane compute savings. This change makes the
sequential path behave the same way as the batch path already does.

## Recommended next steps

1. Measure an early-converging sequential run, with and without the fix, at a realistic
   configuration. This is the only missing evidence.
2. Decide the fix and chunk-boundary early exit together.
3. If adopted: add a regression test asserting `corner_high`-class configurations stay
   under the old-best 4.149 s, and one covering early-convergence step counts so the
   tradeoff is pinned by a test rather than by this document.
4. Re-run the full sweep and update `initial_baseline_20260726` — several of its rows
   describe execution paths that no longer exist.

## Reproduction

```sh
# regression, both paths
uv run python profiling/profile_hdx_cpu.py --mode timing --path pure \
  --steps 1000 --residues 600 --frames 5000 --timepoints 10 --warm-repeats 2
uv run python profiling/profile_hdx_cpu.py --mode timing --path jit \
  --steps 1000 --residues 600 --frames 5000 --timepoints 10 --warm-repeats 2

# control: unaffected configuration
uv run python profiling/profile_hdx_cpu.py --mode timing --path pure --steps 1000
uv run python profiling/profile_hdx_cpu.py --mode timing --path jit  --steps 1000
```

Sweep artifacts: `profiling/_output/hdx_cpu_scaling/cpu_optimisation_20260727/`
(26 valid cells, pure + jit, zero parity failures). Pre-refactor baseline:
`profiling/_output/hdx_cpu_scaling/initial_baseline_20260726/`.

## Unrelated open issues found during this investigation

Neither is caused by the chunked runner, and both currently fail in the test suite.

1. **Eager path recompiles once per step.** Measured `warm.compiles=2` at `steps=2`, and
   47 compiles cold; the sweep's eager `corner_low` cell took 202.4 s with 3,000 warm
   compilations. Production paths are unaffected (`jit` and `pure` both report 0 warm
   compiles). Introduced when `_optimise` was unified onto the shared primitives. Fails
   `tests/unit/profiling/test_hdx_cpu_scaling.py::test_each_prepared_path_reuses_warm_cache_and_reports_terminal_step`.
   Confirmed not caused by the pytree-caching commit by reverting that commit's three
   source files and re-running.

2. **Lost float64 precision in HDX analysis.** Commit `0343fa4` removed
   `jax.config.update("jax_enable_x64", True)` from
   `jaxent/src/analysis/hdx_target_variance.py`. `positive_two_moment_uptake` no longer
   matches its analytic limit `1 - exp(-t·mean)`. Fails
   `tests/unit/analysis/test_hdx_target_variance.py::test_zero_variance_recovers_fixed_mean_limit`.
   Removing a module-level global `jax.config.update` is defensible — it is an import side
   effect that can perturb benchmarks — but it silently changed analysis numerics. The
   scoped fix is to enable x64 only around the functions that require it.
