# `lax.cond` Step Branch — High-Frame Performance Regression

Status: **resolved.** The fix is applied in commit `71e71b9`, together with a chunk-boundary
early exit and a scoped float64 restore. Verification is recorded in "Outcome" below. The
diagnostic narrative is kept because two plausible hypotheses were refuted along the way and
should not be re-explored.

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

## Outcome

Applied in `71e71b9` as three changes:

1. **The select fix** in `optimisation_step` (`chunk.py:242`), as written above.
2. **Chunk-boundary early exit** in `run_chunks` (`chunk.py:392`) and `_optimise`
   (`run.py:398`), resolving the tradeoff below. Guarded by
   `isinstance(carry.active, jax.core.Tracer)`, so it fires only when `active` is concrete —
   the `run_optimise` path — and is skipped under `vmap` and under any caller that jits
   `run_sequential`, where fixed shapes must be preserved.
3. **Scoped `jax.experimental.enable_x64()`** around `positive_two_moment_uptake`
   (`hdx_target_variance.py:302`), fixing open issue 2 without reintroducing an import
   side effect.

Independently verified on the same machine:

| config | path | before | after | speedup |
|---|---|---:|---:|---:|
| corner_high | pure | 7.510 s | **3.413 s** | 2.20× |
| corner_high | jit | 7.401 s | **3.406 s** | 2.17× |
| anchor | pure | 0.2061 s | 0.2056 s | 1.00× |
| anchor | jit | 0.2043 s | 0.2062 s | 0.99× |

Zero host materialisations in the compiled runner; zero warm compiles on `pure` and `jit`.
1309 unit + module tests pass, ruff clean.

### Full sweep — `cpu_cond_fix_20260727`

26 cells, pure + jit, 1000 steps, 3 warm repeats, same `--order-seed 20260726` as the previous
run. **0 cell failures, 0 parity failures, 0 regressions against the pre-refactor baseline.**
Every regression flagged by `cpu_optimisation_20260727` is gone.

| config | R | F | T | pure | jit | vs prev | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| corner_low | 96 | 173 | 1 | 0.0535 | 0.0548 | 1.23× | 2.60× / 11.71× |
| frames_0173 | 144 | 173 | 5 | 0.0897 | 0.0893 | 1.06× | 1.96× / 7.11× |
| residues_096 | 96 | 500 | 5 | 0.1091 | 0.1088 | 1.16× | 3.15× / 6.28× |
| anchor | 144 | 500 | 5 | 0.2114 | 0.2116 | 1.00× | 2.05× / 3.38× |
| timepoints_01 | 144 | 500 | 1 | 0.2295 | 0.2251 | 0.94× | 1.89× / 3.16× |
| timepoints_10 | 144 | 500 | 10 | 0.2204 | 0.2197 | 0.96× | 2.04× / 3.25× |
| residues_293 | 293 | 500 | 5 | 0.2777 | 0.2813 | 1.38× | 2.10× / 2.65× |
| frames_1125 | 144 | 1125 | 5 | 0.3435 | 0.3367 | 1.34× | 2.34× / 2.47× |
| residues_600 | 600 | 500 | 5 | 0.4279 | 0.4138 | 1.82× | 2.19× / 2.14× |
| frames_5000 | 144 | 5000 | 5 | 1.0187 | 0.9932 | 1.90× | 2.99× / 1.66× |
| corner_high | 600 | 5000 | 10 | 3.4757 | 3.3953 | 2.29× | 2.63× / 1.22× |

Gains scale with work per step, as expected if the cause was an unoptimisable step body: 1.0×
at `anchor`, 1.8× at `residues_600`, 1.9× at `frames_5000`, 2.3× at `corner_high`. The
`timepoints_*` and `anchor` cells sit at 0.94–1.00×, i.e. unchanged within the ~2% run-to-run
spread — the fix costs nothing where the step body is small.

Note the handoff's earlier claim that "the driver is frames, not residues" was too strong.
`residues_600` at 500 frames improved 1.82×, second only to the 5000-frame cells. The driver is
total step-body work; frames simply dominated it in the configurations that regressed.

### Numerics

`corner_high` reproduces fingerprint `42d5efe122878608…` on both paths, unchanged from before
the fix. Four of 26 cells do differ — `corner_low` and `timepoints_01`, both pure and jit — so
**the change is not bit-identical everywhere**, and the earlier "bit-identical" claim in this
document applied only to `corner_high`.

Quantified by dumping terminal parameters before and after (`--terminal-npz`) at `corner_low`:
the difference is confined to `frame_weights`, `max |Δ| = 4.8e-07`, `max relative = 3.4e-07` —
1–2 ULP in float32 (`eps = 1.19e-07`) — with all six other parameter leaves bit-identical.
Final loss agrees to all 10 printed significant figures (`2.857442723e-06`). This is
floating-point reassociation from XLA fusing the unconditional body differently than the cond
branch, not a semantic change. It shows up only where accumulated rounding happens to cross a
representable boundary, which is why 22 of 26 cells are unaffected.

**Both open issues below are closed, and issue 1 was not unrelated after all.** The eager
path's per-step recompile was *caused* by the `lax.cond`: eager `lax.cond` compiles on every
call. Confirmed by checking out the pre-fix `chunk.py`/`run.py` at `a9475ab`, where
`test_each_prepared_path_reuses_warm_cache_and_reports_terminal_step` fails with
`compiles == 2` at `steps=2`, and passes on the fix. The handoff was wrong to attribute it to
the item-6 unification.

Test coverage added in `test_chunk_runner.py`: `test_tolerance_termination_stops_before_n_steps`
now pins the exact step count and record count (`executed_steps == 5`, `records.step.shape ==
(2,)`) rather than the weaker `< 100`, which is what proves the early exit fires.

### Residual notes

- `test_chunk_runner_has_no_per_step_host_materialisations` was renamed to
  `..._has_at_most_one_boundary_sync_per_chunk` and relaxed from `not counts` to
  `sum(counts.values()) <= 4`. That is the direct consequence of the early exit — 10 steps at
  chunk size 3 is 4 boundaries — but the bound is now loose enough that it would not catch a
  regression to, say, 2 syncs per boundary. Tightening it to `== n_chunks` would be strictly
  better.
- The early exit does not apply to callers that jit `run_sequential` themselves (including the
  profiler). Those still pay full compute on inactive steps. This is correct — shapes must stay
  static — but means the benchmark numbers above measure the select fix alone, not the exit.
- I did not independently reproduce the reported 0.361 s vs 0.611 s early-convergence wall-clock
  pair; the exit's behaviour is pinned by the step/record-count assertions instead.

## Remaining next step

`cpu_cond_fix_20260727` is now the current reference sweep. `initial_baseline_20260726` should
be kept only as the pre-refactor historical record — several of its rows describe execution
paths that no longer exist — and `cpu_optimisation_20260727` is superseded.

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

Full sweep:

```sh
uv run python profiling/run_hdx_cpu_scaling.py --suite full \
  --run-id <id> --paths pure,jit --steps 1000 --warm-repeats 3 \
  --order-seed 20260726 \
  --previous-dir profiling/_output/hdx_cpu_scaling/cpu_cond_fix_20260727 \
  --baseline-dir profiling/_output/hdx_cpu_scaling/initial_baseline_20260726
```

Note the sweep runner takes `--paths` (plural, comma-separated); `profile_hdx_cpu.py` takes
`--path` (singular). Easy to transpose.

Sweep artifacts, newest first:
`cpu_cond_fix_20260727/` (post-fix, current reference),
`cpu_optimisation_20260727/` (pre-fix, superseded),
`initial_baseline_20260726/` (pre-refactor), all under
`profiling/_output/hdx_cpu_scaling/`.

## Open issues found during this investigation — both now closed

Retained for the record. See "Outcome" above: issue 1 turned out to be a symptom of the same
`lax.cond`, and issue 2 was fixed with a scoped x64 context.

1. **Eager path recompiles once per step.** Measured `warm.compiles=2` at `steps=2`, and
   47 compiles cold; the sweep's eager `corner_low` cell took 202.4 s with 3,000 warm
   compilations. Production paths are unaffected (`jit` and `pure` both report 0 warm
   compiles). **This attribution was wrong** — see Outcome; the cause was the eager `lax.cond`,
   not the item-6 unification. Failed
   `tests/unit/profiling/test_hdx_cpu_scaling.py::test_each_prepared_path_reuses_warm_cache_and_reports_terminal_step`.
   Confirmed not caused by the pytree-caching commit by reverting that commit's three
   source files and re-running.

2. **Lost float64 precision in HDX analysis.** Commit `0343fa4` removed
   `jax.config.update("jax_enable_x64", True)` from
   `jaxent/src/analysis/hdx_target_variance.py`. `positive_two_moment_uptake` no longer
   matched its analytic limit `1 - exp(-t·mean)`. Failed
   `tests/unit/analysis/test_hdx_target_variance.py::test_zero_variance_recovers_fixed_mean_limit`.
   Removing a module-level global `jax.config.update` is defensible — it is an import side
   effect that can perturb benchmarks — but it silently changed analysis numerics. The
   scoped fix is to enable x64 only around the functions that require it.
