# HDX CPU Optimisation Profiling Handoff

## Harness redesign completed (2026-07-26)

`profiling/profile_hdx_cpu.py` now has explicit `timing` and `profile` modes.
Timing runs are fixed-step, exclude fixture and optimizer setup, synchronize
inside the timed interval, and report independent cold/warm compilation and
host-materialisation counts. One prepared fixture and callable are reused for
warm samples. Profile runs handle one path, keep cProfile/JAX trace artifacts
out of benchmark metrics, and report diagnostic elapsed time separately. The
harness validates the terminal optimizer step directly and records final loss
and SHA-256 parameter fingerprints.

Typical commands:

```text
uv run python profiling/profile_hdx_cpu.py --mode timing \
  --path pure --steps 1000 --timepoints 5
uv run python profiling/profile_hdx_cpu.py --mode profile \
  --path pure --steps 100 --timepoints 5 --trace
```

The harness uses a private, optional terminal-state callback in `_optimise`; the
public optimizer return contract and optimizer math are unchanged. Focused tests
cover uptake shape, exact terminal steps, warm cache reuse, deterministic
repetitions, suite definitions, parity helpers, and non-negative scaling fits.

## Clean Uptake Scaling Baseline Completed (2026-07-26)

The complete clean baseline is stored at
`profiling/_output/hdx_cpu_scaling/initial_baseline_20260726/`. The `_output`
directory follows the repository ignore convention. The campaign contains the
manifest, 39 per-cell JSON/NPZ artifacts, aggregate JSON/CSV, report, and
annotated runtime and path-speedup heatmaps.

Environment and protocol:

- Apple M1 Max, 10 cores, 32 GB RAM.
- Python 3.13.2, JAX 0.4.35, jaxlib 0.4.34, CPU backend.
- 13 residue/frame/timepoint configurations and all three paths.
- 1,000 fixed steps, one cold sample, and three warm samples per cell.
- 39/39 cells valid, zero warm compilations, and zero cross-path parity failures.

Representative warm medians:

| Configuration `(residues, frames, timepoints)` | Eager | JIT step | Pure loop |
|---|---:|---:|---:|
| Low corner `(96, 173, 1)` | 11.34 s | 0.641 s | 0.139 s |
| Anchor `(144, 500, 5)` | 11.47 s | 0.715 s | 0.433 s |
| 600 residues `(600, 500, 5)` | 11.62 s | 0.885 s | 0.937 s |
| 5,000 frames `(144, 5000, 5)` | 11.82 s | 1.646 s | 3.043 s |
| High corner `(600, 5000, 10)` | 13.10 s | 4.149 s | 9.150 s |

Key findings:

- At the anchor, pure is 26.5x faster than eager and 1.65x faster than JIT step.
- At the low corner, pure is 81.6x faster than eager and 4.61x faster than JIT.
- The current pure path scales much more steeply with residues and frames. This
  is consistent with its full `n_steps` history buffers remaining in the
  compiled loop carry; the lean-history stage will test that explanation
  directly. JIT step becomes slightly faster by 600 residues and substantially
  faster at 5,000 frames and the high corner.
- Increasing timepoints from 1 to 10 has little measurable effect at 144
  residues and 500 frames. The residue-by-frame work dominates this
  average-first uptake fixture.
- Warm host materialisations are 7.001/step for eager, 4.001/step for JIT step,
  and zero for pure.

This baseline makes lean/final-only history a prerequisite for selecting the
compiled loop as the production default at large frame counts. The compiled
architecture remains the correct direction, but the existing full-history pure
implementation is not uniformly the fastest path.

Use the five-configuration sentinel suite after every optimization stage:

```text
uv run python profiling/run_hdx_cpu_scaling.py \
  --suite stage --run-id <stage-id> --steps 1000 --warm-repeats 3 \
  --baseline-dir profiling/_output/hdx_cpu_scaling/initial_baseline_20260726 \
  --previous-dir profiling/_output/hdx_cpu_scaling/<previous-stage-id>
```

Run the full suite again after the final stage by replacing `--suite stage` with
`--suite full`.

## Clean Single-Configuration Baseline (2026-07-26)

The Stage 0 baseline was measured on the current tree with the timing harness,
using the anchor fixture `(144 residues, 500 frames, 5 timepoints)`, 1,000 fixed
steps, one cold sample, and three warm samples. Each path was run in a separate
process with no profiler or trace enabled.

Commands:

```text
uv run python profiling/profile_hdx_cpu.py --mode timing --path pure --steps 1000
uv run python profiling/profile_hdx_cpu.py --mode timing --path jit --steps 1000
uv run python profiling/profile_hdx_cpu.py --mode timing --path eager --steps 1000
```

Warm median results:

| Path | Wall time | Compiles | Host materialisations |
|---|---:|---:|---:|
| Eager | 11.448 s | 0 | 7,001 |
| JIT step | 0.744 s | 0 | 4,001 |
| Pure compiled loop | 0.437 s | 0 | 0 |

The clean pure path is 1.69x faster than the JIT-step path and has zero host
materialisations; eager has 7.001 host materialisations per step and JIT has 4.001.

## Main Conclusion

Make the pure compiled optimisation loop the normal production path after
removing its full-history carry. Do not spend significant effort
micro-optimising the eager implementation.

The current JIT mode only compiles `optimizer.step`; the surrounding optimisation
loop, convergence tracking, stopping decisions, history handling, and diagnostic
checks remain in Python. Consequently, it still synchronizes with the device on
almost every iteration.

Recommended execution modes:

- `compiled`: production default using `jax.lax.while_loop`, `scan`, or
  `fori_loop`.
- `chunked`: execute compiled blocks of steps and return to Python only at
  logging/checkpoint boundaries.
- `python`: retain for debugging and detailed diagnostics.

Fixed-step runs should use a dedicated `scan` or `fori_loop` path that omits
convergence and early-stopping logic.

## Resolved Design Decisions

The following decisions were made after the initial profile review.

### Shared Chunked Optimisation Architecture

Sequential and batch optimisation will use the same pure interfaces rather than
maintaining separate loop implementations.

The common structure should be:

1. A pure, jittable one-step transition.
2. A pure, jittable fixed-size chunk runner.
3. A convergence evaluator applied at chunk boundaries.
4. Sequential and vmapped batch adapters around the same chunk runner.
5. A common result-to-history adapter.

`batch_optimise` does not perform per-step early stopping. A fixed
`step_chunk_size` will group optimisation steps, after which convergence is
evaluated across all batch members in parallel. No Python/device boundary is
required within a chunk.

The scalar error EMA should remain in the compiled carry and may be updated each
step. The convergence decision is made only at the end of a chunk. Parameter EMA
will be removed.

Recommended primitive split:

```text
optimisation_step(carry, inputs) -> carry
run_step_chunk(carry, chunk_size, inputs) -> carry + chunk metrics
evaluate_convergence(carry, chunk metrics) -> convergence state
run_sequential(...) / run_batch(...) -> shared result
```

For batched execution, convergence predicates should be vmapped. If different
batch members converge at different chunk boundaries, retain an active mask so
converged members become no-ops in subsequent chunks while unconverged members
continue.

### JIT Defaults

JIT will be enabled by default for:

- The simulation forward function.
- The optimisation step/chunk function.

The Python/eager path remains available as an explicit debugging mode, not the
normal execution path.

The implementation should avoid silently reverting to eager execution in a
performance-oriented mode. A JIT failure should either raise or require an
explicitly configured fallback; otherwise users cannot know which execution
contract they received.

### History Contract

Full per-step state history is no longer the target default. The default history
should retain only:

- Convergence records at chunk boundaries.
- The best state.
- The final status and step count.

History configuration must control which parameter partitions are retained in
the best/convergence snapshots. It should also control optional state fields.

Recommended defaults:

```text
save_best = true
save_convergence = true
parameter_partitions = active optimisation partitions
save_gradients = false
save_optimizer_state = false
save_parameter_ema = false
```

The exact configuration representation should use the existing
`Optimisable_Parameters` partition identifiers rather than introducing a second
parameter naming system.

### Logging Contract

- Pure uninterrupted execution logs after completion.
- Chunked execution may log at each chunk boundary.
- Users who explicitly require per-step logging can select a chunk size of one
  or use the Python diagnostic path.
- Per-step Python callbacks are not part of the fast-path contract.

### EMA Policy

Keep the error/loss EMA used for convergence detection. Remove parameter EMA and
the corresponding EMA parameter/history storage.

This removes the parameter-wide multiply/add operations identified in the JIT
profile without changing loss-smoothing precision.

### Loss Scaling

Loss and gradient scaling behavior is out of scope for this optimisation work.
It exists primarily for numerical precision and must not be altered as part of
the performance refactor.

Any future change to how learning-rate or loss scaling interacts with Optax must
be handled as a separate numerical/algorithmic change with dedicated tests.

### Gradient Mask Transitions and Parameter Specialisation

Remove `initial_steps` and the scheduled initial/final gradient-mask transition
machinery completely for now. Use a single static gradient mask and apply it
after gradient calculation.

Do not add active-subtree optimizer specialisation in this refactor. If more
selective optimisation is needed later, zero unwanted gradients post hoc using
the static mask. This preserves one common optimizer structure and avoids
additional phase-specific compilation paths.

This is an intentional API simplification. Remove:

- `OptaxOptimizer.initial_steps` and its constructor argument.
- Initial-versus-final mask construction and selection.
- `gradient_mask_idx` from optimizer state and pure carries.
- Initial-step gates in convergence checks.
- Initial-step learning-rate selection branches.
- Tests and serialization compatibility paths specific to `initial_steps`.

Callers that previously relied on delayed parameter activation must instead
choose the desired static optimization mask before starting a run. A staged
optimization can be expressed as two explicit optimization calls if needed.

## Ranked Optimisation Work

### P0: Move the Complete Loop On-Device

Relevant code:

- `jaxent/src/opt/run.py`, `_optimise`, around lines 273-348
- `jaxent/src/opt/run.py`, `_optimise_pure`, around lines 141-230

The JIT path still executes a Python `for` loop. Replace this with a shared pure
carry and chunk-runner design used by both `run_optimise` and `batch_optimise`.
Add adapters that preserve their public return contracts.

For callers requiring periodic logging, run compiled chunks of perhaps 50-500
steps and materialize only summary metrics between chunks.

### P0: Remove Per-Step Host Synchronization

The 3,000 JIT `.item()` calls map directly to `jaxent/src/opt/run.py` around
lines 309-315:

```python
current_loss.item()
jnp.isnan(current_loss).item()
jnp.isinf(current_loss).item()
```

Replace these with a single device predicate:

```python
continue_running = jnp.isfinite(current_loss) & (current_loss >= tolerance)
```

Keep the predicate inside compiled control flow.

The gradient oscillation branch around `run.py:305` contributes approximately
one boolean host synchronization per step. `_step_with_rates` already computes
the gradient dot product in `jaxent/src/opt/optimiser.py` around lines 398-401,
but `_step` discards it and the Python loop recomputes it through
`check_gradient_oscillation`.

Carry this value through the compiled loop rather than recomputing or
materializing it.

### P0: Fix JIT Dynamic-State Persistence

Potential correctness issue in the current implementation:

- `_step` calculates `new_lr`, `new_model_lr`, and `new_mask_idx`.
- The Python assignments in `jaxent/src/opt/optimiser.py:486-491` cannot execute
  when those values are JAX tracers.
- `_step` then drops these values from its return tuple.

Oscillation-driven learning-rate reductions may therefore fail to persist across
separately jitted step calls. The mask-index issue disappears when
`initial_steps` and mask transitions are removed. The pure carry explicitly
preserves learning-rate state and should be treated as the reference
implementation.

Add a parity test that uses:

- Deliberately oscillating gradients
- Multiple learning-rate reductions
- Assertions over per-chunk LR, loss, and final parameters

The existing simple eager/JIT parity test may not exercise this behavior.

This is one instance of a broader `aux_data` mechanism documented in the next
section; see that section for the underlying cause and two additional, more
severe instances of the same mechanism.

### P0: Unsafe Pytree `aux_data` — Array-Equality Crashes

Root cause behind the reported "successive `optimise()` calls can break"
behavior that motivated the current try/except JIT guards in
`Simulation.initialise()` and `OptaxOptimizer.initialise()`.

JAX pytree registration splits an object into dynamic leaves and static
`aux_data`. `aux_data` must be a well-behaved, hashable/equality-comparable
Python value because JAX compares it (via `==`) between calls to decide
whether a previously compiled function can be reused. When `aux_data`
contains a `numpy.ndarray` or `jax.Array`, that comparison returns an array
instead of a bool, and converting it to a bool raises:

```text
ValueError: The truth value of an array with more than one element is ambiguous.
```

This was confirmed with a minimal reproduction: registering a pytree with an
array in `aux_data` and calling a jitted function on it works the first time,
and works on repeated calls with the *exact same array object*, but raises
the above error on any subsequent call with a different array object of the
same shape/value. This matches an intermittent, call-order-dependent failure
rather than a deterministic one, which fits the reported symptom.

Two concrete instances, both outside the `OptaxOptimizer` case above:

1. **`ExpD_Dataloader.tree_flatten`** (`jaxent/src/data/loader.py:262-277`)
   puts `self.y_true` (`numpy.ndarray`) and `self.covariance_matrix`
   (`Array | None`) into `aux_data`. Neither is read inside any jitted loss
   function — losses read `dataset.train.y_true` / `dataset.val.y_true` off
   the nested `Dataset`, which is already correctly registered as dynamic via
   `jax.tree_util.register_dataclass(data_fields=["y_true", "data_mapping",
   "covariance_matrix"], meta_fields=["data"])`. `ExpD_Dataloader` instances
   are constructed fresh per fold/config in cross-validation and sweep
   workflows, so a new array object with the same value is a routine
   occurrence, not an edge case.

   Fix: move `y_true` and `covariance_matrix` from `aux_data` into the
   `leaves` tuple alongside `train`/`val`/`test`. Zero behavioral change —
   these fields are not part of `state.params`, so making them dynamic leaves
   does not expose them to `value_and_grad`.

2. **`BV_Model_Parameters.static_params` / `linear_BV_Model_Parameters.static_params`**
   (`jaxent/src/models/HDX/BV/parameters.py`) include `"timepoints"`, a real
   `jnp.array(...)` built via `field(default_factory=...)`. Every call to
   `BV_model_Config.forward_parameters` constructs a fresh `BV_Model_Parameters`
   instance — i.e. on every `run_optimise()` call across a sweep — so a new
   `timepoints` array object recurs on essentially every successive
   optimisation run.

   Unlike case 1, `timepoints` lives inside `Simulation_Parameters.model_parameters`,
   which *is* the argument `value_and_grad` differentiates, and
   `create_gradient_masks` (`jaxent/src/opt/gradients.py`) masks gradients per
   whole `Model_Parameters` object rather than per field. Naively making
   `timepoints` an ordinary dynamic leaf would let it silently receive
   optimizer updates whenever BV coefficients are being trained — a physics
   correctness regression, not just a JIT fix.

   Fix: give `BV_Model_Parameters`/`linear_BV_Model_Parameters` a custom
   `tree_flatten`/`tree_unflatten` that keeps `timepoints` in `aux_data` (no
   gradient — correct) but stores it as a plain hashable `tuple(float, ...)`
   instead of an array, converting back to `jnp.array(...)` on unflatten.
   `bv_bc`/`bv_bh` are unaffected and remain ordinary dynamic leaves.

General rule for the rest of this refactor: `aux_data` must never contain a
`numpy.ndarray` or `jax.Array` directly. Anything that must stay
compile-time-static belongs in `aux_data` as a hashable Python-native type
(tuple/scalar/enum), converted at the `tree_unflatten` boundary if the live
attribute needs to be an array.

### P1: Remove Parameter EMA Work

Profile evidence from JIT:

- `ConvergenceTracker.update`: 0.628 s cumulative
- `Simulation_Parameters._apply_op`: 0.514 s
- Parameter multiplication: 0.352 s
- Parameter addition: 0.176 s

The source is the parameter-wide EMA in
`jaxent/src/opt/track.py:150-168`.

Actions:

1. Retain only scalar error/loss EMA in the compiled carry.
2. Remove `ema_params`, parameter-wide EMA arithmetic, and EMA parameter history.
3. Evaluate convergence at chunk boundaries.
4. Vmap the convergence evaluation for batch optimisation.

### P1: Add Lean History Policies

The pure path currently allocates full `n_steps` history buffers in
`jaxent/src/opt/run.py:125-138` and dynamically writes every parameter and loss
leaf in `jaxent/src/opt/optimiser.py:555-567`.

Replace full trajectory buffers with a configurable lean history. The default
retains convergence records and the best state only. Parameter partitions saved
in those records must be configurable.

Carry the best validation loss and selected best-state parameters on-device.
Avoid building and synchronizing a complete trajectory. A legacy full-history
mode may be retained temporarily only where required for compatibility tests or
downstream callers.

`Simulation_Parameters.normalize_weights` is also applied to `save_state` every
step around `optimiser.py:429-435`. In fixed-step final-only mode, normalize once
at exit unless normalized parameters are required for convergence semantics.

### Out of Scope: Active-Subtree Optimizer Specialisation

Do not introduce frame-only or other active-subtree optimizer variants in this
refactor. Keep one optimizer tree and zero unwanted gradients after gradient
calculation using a static mask.

Also remove `initial_steps`, the scheduled initial/final mask switch, and its
dynamic mask-index state. A more elaborate staged optimization API can be
introduced later if a concrete use case justifies it.

### P2: Cache PyTree Schemas

Profile evidence:

- JIT: 31,456 `_get_grouped_slots` calls, approximately 0.50 s cumulative.
- Eager: 62,028 calls, approximately 1.12 s cumulative.
- JIT `tree_flatten` and `tree_unflatten` each cost roughly 0.4 s cumulative.

`jaxent/src/interfaces/model.py:22-49` repeatedly scans class MROs and partitions
slots.

Actions:

- Cache `_get_ordered_slots` and `_get_grouped_slots` with `functools.cache`, or
  calculate the tuples once in `__init_subclass__`.
- Make flatten/unflatten consume precomputed tuples.
- Consider specialized flatten/unflatten implementations for common HDX/BV
  parameter classes.
- Preserve runtime type validation at public boundaries, but exclude internal
  PyTree protocol methods from package-wide beartype checking where practical.

### P2: Remove Redundant Staged Work

Low-risk cleanup:

- `jaxent/src/opt/run.py:213-230`: the `while_loop` condition already rejects
  converged carries, making the inner `lax.cond(converged, ...)` redundant.
- `jaxent/src/models/core.py:154-156`: `Simulation.forward` constructs the same
  `new_sim` twice.
- `jaxent/src/models/core.py:256-258`: `Simulation.tree_unflatten` unpacks
  `dynamic_values` twice.
- `jaxent/src/interfaces/simulation.py`: `normalize_weights` computes projected
  model parameters and then discards them.
- `_pure_step` receives `updated_sim` from `_step_with_rates` but does not use it.
  Refactor the loss auxiliary output to return only values required by the loop.

Some dead numerical operations may already be eliminated by XLA, but removing
them still reduces eager work, tracing, and code complexity.

### P2: Disambiguate Frame-Weight Logits vs. Normalized Weights

`Simulation_Parameters.frame_weights` (and, less critically, `frame_mask`)
represents two different quantities depending on where in the pipeline it is
read, distinguished only by convention rather than by type:

- Raw, unnormalized logits — the actual optimizer state that gradient descent
  updates and that propagates step-to-step (`new_state.params` in
  `_step_with_rates`, `jaxent/src/opt/optimiser.py`).
- Softmax-normalized simplex weights — produced by
  `Simulation_Parameters.normalize_weights` (`jaxent/src/interfaces/simulation.py`)
  and used for the forward pass and for reported/history state
  (`save_state.params`).

`frame_mask` has the same raw-logit-versus-`sigmoid`-plus-`clip` duality
inside `normalize_weights`, though it is currently inert in practice since
`Optimisable_Parameters.frame_mask` optimization raises `NotImplementedError`
in `create_gradient_masks`.

This ambiguity is not hypothetical: confirming that the lean-history change
(P1, above) does not alter reported numbers required explicitly tracing which
of `new_state.params` (logits, feeds the next step) versus `save_state.params`
(normalized, feeds history/reporting) held which representation, since
nothing at the call site signals it.

Recommended fix (deliberately the cheap version, not a full type split):
rename the field so its logit-space nature is explicit at every call site
(e.g. `frame_weights` -> `frame_weight_logits`), or introduce a lightweight
`NewType`/alias distinguishing logits from normalized weights so static
analysis can flag a value used in the wrong space. This does not require a
new pytree-registered wrapper type and should not change runtime behavior.

Explicitly out of scope here: a full separate wrapper type for
logits-vs-weights. That would touch every consumer of `Simulation_Parameters`
(optax's `multi_transform` param labels, gradient masking, the forward pass,
HDF persistence) and would compete for scope with the refactors already
queued above. Revisit only if the ambiguity keeps causing bugs.

## Benchmark Redesign

Profiling and benchmarking must be separate runs.

### Timed benchmark

- Disable cProfile.
- Disable JAX tracing.
- Place `block_until_ready()` inside the timed region.
- Exclude fixture construction when measuring optimizer execution; report setup
  separately.
- Report cold compile-plus-execution time.
- Reuse the same compiled callable and shapes for multiple warm repetitions.
- Report median and dispersion across warm repetitions.
- Verify every path completed exactly the requested number of steps.

### Profiling run

- Profile one path per invocation.
- Collect cProfile and JAX traces separately where possible.
- Trace a representative window rather than the complete 1,000-step eager run.
- Do not use trace-export-inclusive wall time as the benchmark result.

Suggested metrics:

- Cold wall time
- Warm wall time
- Compilation count and compilation time
- Steps per second
- Host materialisations per step
- Peak memory
- Number of retained history states
- Final loss and parameter parity

## Implementation Sequence

1. Redesign the harness to separate clean timing from profiling.
2. Define common pure step, chunk, convergence, carry, and result interfaces.
3. Add correctness tests for chunk sizes, fixed-step counts, and sequential/batch
   parity.
4. Make simulation forward and optimization chunks JIT by default.
5. Refactor `run_optimise` and `batch_optimise` onto the shared chunk runner.
6. Remove parameter EMA, `initial_steps`, and scheduled gradient-mask transition
   state.
7. Add configurable convergence/best-only history with parameter partition
   selection.
8. Cache PyTree schemas and reduce beartype involvement in PyTree internals.
9. Remove redundant control flow and object construction.
10. Benchmark each change independently against the clean baseline.

## Stage 1 Completion (2026-07-26)

Stage 1 items 2–5 are implemented on the current tree. The optimisation loop now uses
the shared pure-JAX chunk runner in `jaxent/src/opt/chunk.py`:

- `ChunkInputs`, `ChunkCarry`, `StepMetrics`, `ChunkRecord`, and `ChunkResult` define
  the shared dynamic state and boundary interfaces.
- `optimisation_step` is the common step transition, including EMA convergence state,
  learning-rate state, finite-loss rollback, executed-step counting, and the running
  best snapshot.
- `run_step_chunk` compiles one scan plus boundary convergence evaluation. Sequential
  execution runs fixed chunks with a remainder chunk; batch execution is a bounded
  `vmap` of that same runner.
- `run_optimise` defaults to `execution_mode="compiled"`; `"python"` remains available
  for diagnostics. `jit_update_step` remains as a deprecated compatibility alias.
- `batch_optimise` reconstructs public histories from stacked chunk results without
  per-run write-index synchronisation.
- `history.states` is boundary-granular and `history.convergence_states` is the active
  threshold-event subset. Post-stop boundaries are dropped, and the running best is
  selected over executed steps. Boundary history entries use the final carry's
  optimizer state and gradients, as specified for this slice.
- The eager diagnostic path delegates to the same step and convergence primitives;
  its former oscillation-triggered host-side threshold reset is intentionally removed.
  Note this is the convergence-counter reset only. Oscillation-triggered **learning-rate
  decay** now ratchets cumulatively from the carried rate: `_step_with_rates` computes
  `reduce_lr = (grad_dot_product < 0) & (step > 1)` and divides both rates by
  `plateau_denominator` (`jaxent/src/opt/optimiser.py:391-404`). It is now more correct
  than before — `grad_dot_product` is carried through `StepMetrics` rather than being
  dropped by `_step` and recomputed host-side, which was the P0 "LR reductions may fail
  to persist across separately jitted step calls" defect. Covered by
  `test_learning_rate_state_persists_across_chunks`.

  Behavioural effect of the dropped counter reset: on an oscillating run, convergence
  thresholds can now be met sooner than the old eager path allowed, since
  `steps_since_threshold_start` is no longer pushed back on oscillation.
  This supersedes the earlier Stage 1 note that the learning-rate trajectory was
  unaffected; cumulative decay is an intentional Item 6 numerical change.
- The profiler's `pure` and `jit` paths now exercise the shared compiled runner. The
  old step-only JIT loop is no longer a production execution path.

Correctness verification:

```
uv run pytest jaxent/tests/unit/opt/ jaxent/tests/modules/optimise/ -x
# 100 passed
uv run pytest jaxent/tests/integration/optimise/test_batch_optimise.py -x
# 1 passed
```

The new chunk-runner tests cover chunk remainders, tolerance and non-finite stopping,
terminal threshold events, non-monotonic best-state selection, fixed-step compiled/
Python parity, sequential/batch frozen lanes, learning-rate persistence, history
granularity, and host-materialisation instrumentation. The full optimisation suites
pass with the existing beartype warnings and two intentional deprecation warnings.

Clean 1,000-step shared-runner timing on the profiling fixture:

| path | cold wall | warm median | completed | host materialisations/step |
|---|---:|---:|---:|---:|
| pure | 3.396 s | 0.213 s | 1000 | 0 |
| jit | 3.377 s | 0.214 s | 1000 | 0 |

The `pure` and `jit` runs produced identical final loss and parameter fingerprints.
The earlier Stage 0 clean baseline remains recorded above for comparison; the new
compiled path is below the clean pure-path baseline after warm-up. Items 6 and 8 were
not changed. The item-7 history changes are limited to the explicit contract decisions
listed in the design section.

## Acceptance Criteria

Functional:

- Eager, JIT-step, and compiled-loop paths agree on final losses and parameters
  within defined tolerances.
- Every complete chunk executes exactly `step_chunk_size` steps, with the final
  chunk correctly handling a remainder.
- Sequential and batched execution use the same chunk transition and agree on
  equivalent inputs.
- Convergence is evaluated at chunk boundaries and is vectorized for batch
  members.
- Learning-rate state persists identically across execution modes.
- Parameter EMA, `initial_steps`, and scheduled mask-index state no longer exist.
- History contains the best state and configured convergence snapshots without
  gradients or optimizer state by default.
- History parameter snapshots contain only the configured parameter partitions.
- Loss scaling behavior remains unchanged. The following numerical changes are
  intentional and tested: two warm-up steps are removed, the full static gradient
  mask is active from step 0, and learning-rate decay compounds from carried state.

Performance:

- Compiled fixed-step mode performs zero per-step host materialisations.
- Warm timing is measured without cProfile or trace export.
- The public compiled path remains within a small tolerance of the internal pure
  path.
- Final-only history is measurably faster and uses less memory than full history.
- PyTree schema caching materially reduces flatten/unflatten CPU time.

## Recommended First Patch

The highest-value first patch should combine:

1. Shared pure `optimisation_step`, `run_step_chunk`, and
   `evaluate_convergence` interfaces.
2. Sequential and batch adapters using the shared fixed-size chunk runner.
3. JIT-default simulation forward and optimizer chunk execution.
4. Error EMA retained, with parameter EMA removed.
5. Best/convergence-only history with configurable parameter partitions.
6. Static post-gradient masking with `initial_steps` and scheduled mask
   transitions removed.
7. A clean unprofiled benchmark mode.
8. Parity tests covering loss, parameters, chunk/remainder step counts,
   convergence, and learning-rate state.

This establishes a trustworthy performance baseline and removes the dominant
Python/device boundary before smaller optimizations are attempted.
