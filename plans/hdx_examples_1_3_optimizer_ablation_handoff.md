# HDX Examples 1–3 Optimizer Ablation Handoff

Status: **ready to implement and run; full campaigns not started.**

Date: 2026-07-30. Baseline code: commit `549f33a` (`fixed LR semantics`).
The tracked working tree was clean when this handoff was written.

## Objective

Find the smallest production fitting loop that recovers the expected
Examples 1–3 model rankings. The decision is based on recovery after model
selection by **validation MSE**, not validation loss, optimizer best state, the
terminal state, or recovery itself.

The required directional gates are:

1. Example 1: `Sigma_MSE > MSE` and `ISO_BI > ISO_TRI`.
2. Example 2: `Sigma_MSE > MSE` and `AF2_filtered > AF2_MSAss`.
3. Example 3: `MSE > Sigma_MSE`.

`mcMSE` is out of scope. Report the `AF2_filtered - AF2_MSAss` result for
Example 3, but do not make it a hard gate unless the scientific requirement is
expanded.

The campaign must report individual data-split recoveries, their mean, sample
variance (`ddof=1`), and standard deviation. A positive mean paired recovery
difference defines the directional gate. Also show the direction separately
for `sequence_cluster` and `spatial`; a pooled pass with a split-type reversal
must be marked **fragile**, not a clean pass.

> **MaxEnt scaling defect (discovered in ISO Stage 0):** before Phase 1c,
> `examples/common/optimization.py` applied `maxent_scaling` to loss slot 0
> (the primary data loss), while the MaxEnt loss in slot 1 remained fixed at
> one. Historical nominal MaxEnt sweeps through `run_optimization` therefore
> swept data-vs-MaxEnt strength in the inverse direction. Do not interpret or
> combine those sweep labels without accounting for this defect; Phase 1c
> aligns weights to `[data=1, MaxEnt=maxent_scaling, ...]`.

> **Stage 0 convergence/alignment finding (Phase 1d):** the ISO target
> generator originally prepended its synthetic 294th terminal column even
> though feature-topology residues match `segments[:-1]`; split materialization
> then selected target rows by `fragment_index`, shifting every live fitting
> target by one residue. A standalone `[1:]` comparison hid this mismatch.
> Corrected targets append the unmatched terminal column. On the corrected BI
> open=0.90 cell, true-weight initialization gives production train MSE
> `3.45e-12`, and a uniform-start MaxEnt=0.001 fit reaches full-layout MSE
> `1.39e-4` in its best retained state. Separately, `best_state` currently
> stores pre-update losses beside post-update parameters, and the last
> convergence state was worse than `best_state` in 405/480 Phase 1c histories
> under external forward evaluation. Treat historical convergence-selected
> results as unscored until their target alignment and state selection are
> audited; do not revisit the earlier ablation verdicts yet.

> **Checkpoint correction (Phase 2):** the optimizer now evaluates loss again
> after applying an update, so every retained state, convergence snapshot, and
> `best_state` pairs post-update parameters with their own post-update loss.
> This has repo-wide blast radius. It is verified independently before the
> frame-averaging change; prior ablation results remain intentionally
> unrevisited.

## Current numerical verdict

The earlier “new logits geometry” hypothesis was wrong. The April code already
optimized frame-weight logits.

The evidence now supports two independent numerical changes:

- Commit `bf528da` changed oscillation LR behavior. Historical production runs
  used `initial_steps=0`, which restored the configured LR every step and
  applied the `1 / 1.005` damping only on an oscillating step. The refactor
  carried the damped LR and compounded it. Commit `549f33a` restores the
  historical noncompounding behavior.
- Commit `0afc52a` adopted `tensordot` for frame averaging. It is much faster on
  CPU but reassociates float32 reduction. Against the April explicit
  multiply/sum it changes the initial averaged features by up to
  `3.814697e-6`; chaotic Adam trajectories amplify that difference.

The following have been tested and should not enter the main grid:

- `einsum` and `tensordot` lower to the same `dot_general` and remain paired.
- `lax.scan` and a JIT loop without scan are bit-for-bit equal in the isolated
  optimizer test. Whole-step JIT versus eager execution can differ, so keep
  the production execution mode fixed at `compiled`.
- A float64 contraction narrows the direct reduction error but does not restore
  the April trajectory. Full 64-bit operation is deferred.
- Legacy simplex values are converted to logits on load and softmaxed exactly
  once for analysis. There is no demonstrated analysis-time double softmax.
- The matched April/current Example 3 input arrays were identical, so no
  refeaturization is required for this ablation.
- The constant logit gauge differs between April and current code and produces
  a very small additional float32 perturbation. Keep it fixed in the main grid;
  use it only as a final diagnostic if every main cell fails.

These facts explain trajectory divergence but do not determine which
implementation recovers the scientific rankings. Only the full scored
Examples 1–3 ablation can do that.

## Do not conflate the two “reset” mechanisms

There are two unrelated controls:

1. **Oscillation LR adjustment**: when consecutive gradients have a negative
   dot product, use `base_lr / plateau_denominator` for that optimizer step.
   This is the binary `lr_adjustment` factor below.
2. **Initial-LR reset step**: use `initial_learning_rate` for the first `k`
   optimizer steps, then switch to `learning_rate`. This is the later integer
   `initial_steps` factor.

The launchers currently parse and pass `--initial-learning-rate` and
`--initial-steps`, but the Python fitting scripts do not use either value.
They are dead CLI options at commit `549f33a`. Do not label an unchanged run as
an `initial_steps` ablation. Implement and test the two-phase schedule before
Stage 2, or remove the dead options if Stage 2 is unnecessary.

`reset_threshold_cooldown_on_oscillation` is a third, convergence-diagnostic
control. It must remain fixed. It resets only the checkpoint cooldown and must
not change optimizer parameters.

## Factors held fixed

Every campaign cell must use the same:

- commit, dependencies, backend, device count, and `compiled` execution mode;
- pre-existing `_featurise` and `_datasplits` inputs;
- three split replicates;
- split types `sequence_cluster,spatial`;
- 5000 optimizer steps;
- convergence ladder and chunk size;
- Adam, loss normalization, MaxEnt grid, BV grid, and BV parameter LR scale;
- initial frame weights/logit gauge;
- only `MSE,Sigma_MSE`.

Do not mix HDF files across commits or append cells to an existing output
directory. Each cell gets a unique directory containing its factor values and
commit hash. Store the effective settings in every run's config JSON, including
the reduction implementation and LR-adjustment state.

Use the current production launcher grids:

| Example | ensembles | MaxEnt values | BV values | fits per cell |
|---|---|---:|---:|---:|
| 1 | `ISO_TRI,ISO_BI` | current 20-value launcher grid | — | 480 |
| 2 | `AF2_filtered,AF2_MSAss` | current 35-value launcher grid | — | 840 |
| 3 | `AF2_filtered,AF2_MSAss` | current 20-value launcher grid | `0,0.25,0.5,0.75,1` | 2400 |

This is 3720 optimizer fits per ablation cell after removing `mcMSE`.

## Required temporary controls

Before starting the long runs, expose and persist these controls:

- `learning_rate`: already exposed; values `1.0` and `0.1`.
- `lr_adjustment`: new explicit boolean. `on` means the restored,
  noncompounding historical adjustment; `off` means no gradient-dot-product LR
  change. Do not emulate `off` by deleting convergence cooldown handling.
- `frame_average_impl`: temporary enum `tensordot` or `legacy_sum`, where
  `legacy_sum` is:

  ```python
  weights = frame_weights.reshape(1, -1)
  jnp.sum(x * weights, axis=-1)
  ```

- `step_chunk_size`: expose it for the preflight only, then hold it fixed at
  `100`.

Use the same names all the way through the three shell launchers, Python
fitting scripts, `OptimizationConfig`, and saved JSON:

```text
--lr-adjustment on|off
--frame-average-impl tensordot|legacy_sum
--step-chunk-size INT
```

Add focused tests before the preflight:

- adjustment `off` always uses the base LR, including on a forced oscillation;
- adjustment `on` applies `base_lr / 1.005` only on the oscillating step and
  returns to the base LR on the following non-oscillating step;
- both reduction implementations have the documented direct float32
  tolerance, and `legacy_sum` matches the April expression;
- saved config JSON records the effective values, not just parser defaults.

The temporary alternative paths are ablation scaffolding. Once a winner is
chosen, remove the losing implementation and flags rather than maintaining a
permanent matrix of production modes.

## Preflight: prove the chunking is observational

Do this before any full cell:

1. Use one fixed Example 3 run with both BV parameters and frame weights
   optimized.
2. Run enough steps to cross more than one convergence threshold.
3. Compare `step_chunk_size=1` with `step_chunk_size=100`.
4. Require equal final parameters/losses under the existing float32 test
   tolerances.
5. Require the same ordered convergence labels and the same retained state for
   every crossed threshold, including multiple thresholds crossed inside one
   100-step chunk.
6. Process and run `score_models`; require the same validation-MSE-selected
   checkpoint and recovery.

Convergence checkpoints remain diagnostic/model-selection candidates only.
They must not stop, reset, or select the optimizer trajectory. A chunk may
contain multiple threshold events, and all such events must be retained rather
than keeping only the last event at the chunk boundary.

## Stage 1: five-cell deletion-first screen

Run all three full examples for these cells:

| cell | LR | LR adjustment | frame averaging | purpose |
|---|---:|---|---|---|
| `A0` | 1.0 | on | `tensordot` | current committed baseline |
| `A1` | 1.0 | off | `tensordot` | can adjustment be deleted? |
| `A2` | 0.1 | off | `tensordot` | lower-LR stability test |
| `A3` | 1.0 | off | `legacy_sum` | reduction-order test |
| `A4` | 0.1 | off | `legacy_sum` | LR × reduction interaction |

This is intentionally not an arbitrary one-factor-at-a-time screen. With the
feature being considered for deletion held off, it fully crosses the two
remaining binary factors. `A0` supplies the direct on/off comparison at the
current LR and reduction.

Decision:

- If one or more of `A1:A4` cleanly passes every hard gate, delete LR
  adjustment. Select among passing cells by gate margin and lower split
  variance. If numerically tied, prefer `tensordot` for its measured CPU speed.
- If none of `A1:A4` passes, do not conclude that LR adjustment is necessary
  yet. Complete the missing interaction cells:

  | cell | LR | LR adjustment | frame averaging |
  |---|---:|---|---|
  | `B1` | 0.1 | on | `tensordot` |
  | `B2` | 1.0 | on | `legacy_sum` |
  | `B3` | 0.1 | on | `legacy_sum` |

  Together with `A0`, these complete the adjustment-on half of the `2^3`
  factorial. Compare matched on/off pairs before retaining the feature.

Do not reuse the July 29 outputs as `A0`: they predate commit `549f33a`, so
their LR semantics do not match the committed baseline.

After the new flags exist, each Stage 1 cell is launched with the following
templates, substituting its values and a unique cell/commit label:

```bash
bash jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/run_maxent_parallel_SIGMA.sh \
  --ensembles ISO_TRI,ISO_BI \
  --losses MSE,Sigma_MSE \
  --split-types sequence_cluster,spatial \
  --learning-rate "${LR}" \
  --lr-adjustment "${LR_ADJUSTMENT}" \
  --frame-average-impl "${FRAME_AVERAGE_IMPL}" \
  --step-chunk-size 100 \
  --dir-name "_optimise_ABLATION_${CELL}_${COMMIT}"

bash jaxent/examples/2_CrossValidation/fitting/jaxENT/run_maxent_parallel.sh \
  --ensembles AF2_filtered,AF2_MSAss \
  --losses MSE,Sigma_MSE \
  --split-types sequence_cluster,spatial \
  --learning-rate "${LR}" \
  --lr-adjustment "${LR_ADJUSTMENT}" \
  --frame-average-impl "${FRAME_AVERAGE_IMPL}" \
  --step-chunk-size 100 \
  --dir-name "_optimise_ABLATION_${CELL}_${COMMIT}"

bash jaxent/examples/3_CrossValidationBV/fitting/jaxENT/run_maxent_parallel_BV_objective.sh \
  --ensembles AF2_filtered,AF2_MSAss \
  --losses MSE,Sigma_MSE \
  --split-types sequence_cluster,spatial \
  --learning-rate "${LR}" \
  --lr-adjustment "${LR_ADJUSTMENT}" \
  --frame-average-impl "${FRAME_AVERAGE_IMPL}" \
  --step-chunk-size 100 \
  --dir-name "_optimise_ABLATION_${CELL}_${COMMIT}"
```

Leave the production MaxEnt and BV grids at their launcher defaults. Record the
fully expanded arrays in a cell manifest before launch so a later default edit
cannot make cells incomparable.

## Stage 2: initial-LR reset integer

Run this stage only if the low base LR (`0.1`) is selected or strongly improves
Example 3 stability. Otherwise remove `--initial-learning-rate` and
`--initial-steps` as dead options.

Implement the historical schedule explicitly:

```text
steps [0, k): learning_rate = 1.0
steps [k, ...): learning_rate = 0.1
```

The transition must be based on the global optimizer step, not chunk number,
and must work identically for chunk sizes 1 and 100.

Add tests at steps `k-1`, `k`, and `k+1`, including `k=0`, before running the
integer screen. Save both configured rates and the transition step in JSON.

Screen `k = 0, 2, 10, 100` on full Example 3 using the best Stage 1 reduction
and LR-adjustment settings. `k=0` is the pure low-LR control and `k=2` is the
historical default sentinel. If no `k > 0` improves the Example 3 MSE-over-Sigma
gate and split variance, keep `k=0` and delete the schedule.

If a positive `k` wins, run only `k=0` and the winning `k` on full Examples 1
and 2. Retain the schedule only if the winner still passes all three examples'
hard gates. This gives full Examples 1–3 confirmation without multiplying the
entire integer grid across all examples.

## Scoring and selection protocol

Run each example's `process_optimisation_results.py` and then its
`score_models_ISO_TRI_BI.py`. The legacy recovery-analysis outputs are useful
plots but are not the gate input.

For each cell's `model_scores.csv`:

1. Apply
   `jaxent.examples.common.analysis.filter_best_convergence_by_validation_mse`.
   This retains the minimum finite `val_mse` checkpoint for each fitted
   hyperparameter model.
2. For each
   `(ensemble, loss_function, split_type, split_idx)` group, select the
   remaining MaxEnt value with minimum `val_mse`.
3. For Example 3, include `bv_reg_function` in the model family and minimize
   across both MaxEnt and `bv_reg_value`.
4. Never select using `val_loss`, `recovery_percent`, terminal state, or an
   optimizer-internal best state.
5. Preserve the selected `maxent_value`, `bv_reg_value`, and
   `convergence_value` in the report for auditability.

The filtered `model_selection_performance_by_split.csv` produced by the common
analysis pipeline can be used by taking rows with `score_metric == "val_mse"`.
A dedicated ablation summarizer is preferable because it can enforce complete
groups and emit all cells in one schema.

## Required report

Write both a machine-readable CSV and a Markdown summary.

Per selected model:

```text
cell, commit, example, ensemble, loss_function, split_type, split_idx,
learning_rate, lr_adjustment, frame_average_impl, initial_steps,
maxent_value, bv_reg_value, convergence_value, val_mse, recovery_percent
```

Per group:

```text
cell, example, ensemble, loss_function, split_type, n,
recovery_mean, recovery_sample_variance, recovery_sd,
val_mse_mean, val_mse_sample_variance
```

Per gate, compute paired recovery differences using the same split type and
split index:

- Example 1 loss: `Sigma_MSE - MSE`.
- Example 1 ensemble: `ISO_BI - ISO_TRI`.
- Example 2 loss: `Sigma_MSE - MSE`.
- Example 2 ensemble: `AF2_filtered - AF2_MSAss`.
- Example 3 loss: `MSE - Sigma_MSE`.

Report the paired values, mean, sample variance, SD, and count. A hard gate
passes when its pooled paired mean is positive. Mark it fragile if either split
type has a non-positive paired mean. Do not call a three-split directional
result statistically significant without a separately specified inferential
test.

Also report:

- missing/failed fits and incomplete hyperparameter grids;
- number of convergence candidates per fit;
- selected threshold/MaxEnt/BV distributions;
- wall time and backend;
- whether any ranking depends on one split;
- historical comparison as context, not as mixed-format input.

## Historical anchors

Use these only as external references:

- Last matched pre-performance code used for numerical isolation:
  `cd3b8cd` (2026-04-16).
- Correct April Example 3 production output:
  `_optimise_quick_FIGURE_SIGMA_5000_lr1.0_BV_objectve_scale1.0__20260416_024125`.
- Its previously reproduced matched Example 3 recovery values were
  `70.238280`, `60.961404`, and `68.623772`; mean
  `66.607818697`, sample variance `24.563154305`, SD
  `4.956122911`.
- July 29 runs are post-performance diagnostic references, not a valid current
  `A0`, because commit `549f33a` was made afterward.

Legacy HDF histories require the format-1 migration path. Never silently load
an old HDF with current positional-state assumptions. New ablation cells should
all use the current format and their own processing manifests.

## Final production decision

Choose the minimum feature set that cleanly passes all hard gates:

1. Remove LR adjustment if an adjustment-off cell passes.
2. Keep `tensordot` if it passes; restore `legacy_sum` only if it is necessary
   for the gates.
3. Use LR `1.0` or `0.1` according to the full scored result, not trajectory
   similarity.
4. Remove the initial-LR schedule unless a positive reset step survives the
   Stage 2 confirmation.
5. Keep convergence checkpoint machinery solely for diagnostic curves and
   validation-MSE model selection.
6. Defer float64, `mcMSE`, eager/compiled comparisons, and further logit-gauge
   work until the staged grid fails.

After selecting the winner, remove temporary ablation switches, run the
optimizer unit/integration tests, and run one final full Examples 1–3 production
campaign from a clean commit.
