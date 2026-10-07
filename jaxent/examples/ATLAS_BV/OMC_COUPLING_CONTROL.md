# Total graph coupling versus relative edge weights

This controlled experiment uses the twenty frozen stratified candidates from `1tzw_A` and `1dd3_B`, at 25% and 12.5% retention and five sampling seeds. Sources, targets, candidate counts, actual numerical bandwidths and OMC strength 0.1 are unchanged.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coupling-control --workers 10
# Separate phases; completed valid fits resume without optimisation:
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coupling-control --phase prepare
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coupling-control --phase fit --workers 10
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coupling-control --phase report
```

Outputs are separate: `outputs/analysis/pairwise_geometry/omc_coupling_control/`.

## Controls

Let `Kq` be the existing kernel and `Cq` its mean off-diagonal weight over retained frames. Fix the reference at the existing `q=0.08` bandwidth.

| Variant | Kernel | What varies? |
|---|---|---|
| existing | `Kq` | Total coupling and relative edge weights |
| fixed_coupling | `Kq * Cref/Cq` | Relative edge weights |
| fixed_pattern | `Kref * Cq/Cref` | Total coupling |

The Gaussian graphs have nonzero edges everywhere. “Pattern” refers to relative weights, not the existence of discrete connections. The fixed-pattern arm uses the reference sigma throughout; its displayed bandwidth labels identify the original coupling levels being reproduced.

Kernel scaling is fixed before fitting and uses neither uptake targets nor fitted weights. This is **not** the existing `wᵀKw`-normalised objective. Candidate padding and diagonal edges are excluded from coupling calculations. The full kernel is scaled consistently; diagonal edges contribute zero to OMC energy.

All variants are exactly identical at the reference. Each candidate therefore needs only ten new fits: five non-reference points for each new variant. There are **200 new fits**, **120 reused original OMC fits**, and **140 reused MaxEnt/unregularised fits**. Forty reference aliases make 500 displayed rows but only 460 distinct fitted trajectories. Reference aliases must never be counted as independent fits.

Use the original optimiser, initial weights, all-residue/time MSE and convergence checks. Reporting is a separate module and excluded from fitting cache identity. Changing fitting code or frozen input hashes invalidates reuse; changing report code does not.

## Outcomes and interpretation

Report population TV, MSE, ESS, conditional ESS, correct-population ESS ceilings, residual cancellation and saved-fit MaxEnt matching at tolerances 0.02/0.01/0.005. Raw effects compare each variant with the existing fit at the same candidate and bandwidth.

For each outcome, relative to the reference fit:

```
coupling effect = Y_fixed_pattern - Y_reference
edge-pattern effect = Y_fixed_coupling - Y_reference
interaction = Y_existing - Y_fixed_pattern - Y_fixed_coupling + Y_reference
```

Their sum is exactly `Y_existing - Y_reference`. This is a factorial contrast in these fixed ensembles, not an independent biological effect or an additive attribution with interaction omitted. Separately calculated medians need not sum.

A factorial contrast requires convergence of all its fits. Repetition-level paired medians and endpoint/span summaries require complete curves. Each repetition contributes once, with median/minimum/maximum and sign counts across five repetitions. Cells with fewer than three complete repetitions are labelled insufficient for a stable descriptive conclusion. No bandwidth is chosen using population truth.

## Artifacts

- `index.html`, `findings.md`: measured conclusions, three-curve overlays and interaction plots.
- `results.csv`: all variants and reused comparators, including `fit_id`, origin and convergence.
- `kernel_audit.csv`: actual coupling, scale, original sigma label and kernel sigma, within/between coupling.
- `paired_differences.csv`, `repetition_effects.csv`, `effect_summary.csv`: paired effects and coverage.
- `factorial_contrasts.csv`, `factorial_summary.csv`: individual contrasts and descriptive summaries.
- `curve_summary.csv`, `endpoint_summary.csv`, `ess_transitions.csv`: ESS endpoint changes, spans and exact inverse-ESS changes.
- `basin_metrics.csv`, `residual_terms.csv`, `ess_matches.csv`: conditional population/ESS, observation-wise decomposition and matching sensitivity.
- `manifest.json`, candidate completion receipts and `audit.json`: frozen provenance, new-fit deduplication and source immutability.

Tables also have Parquet versions. Residue `feature_row` values use the parent coverage experiment's per-system `residue_mapping.csv`.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-coupling-mpl uv run --no-sync pytest \
  tests/test_omc_coupling_control.py tests/test_omc_bandwidth_diagnostics.py -q
```

Tests check constant coupling, invariant relative edge weights, exact reference equality, padding exclusion, energy linearity and diagonal invariance, factorial identities and cache invalidation. Reporting independently reconstructs objectives and verifies reused metrics, weight validity, unique-fit counts, reference aliases and local links. The full run completed all 200 new fits with convergence.

No new candidates, systems, bandwidths, independently selected strengths, alternative structural graph metrics or ISO runs are introduced.
