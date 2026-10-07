# Population imbalance versus within-basin coverage

This experiment compares **random** and **structurally stratified** candidate filtering in the frozen `1tzw_A` and `1dd3_B` source ensembles. Both methods have identical basin counts and numerical bandwidths. The question is whether reducing within-basin coverage distortion improves OMC population recovery relative to unregularised fitting.

## Run and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coverage-control --workers 10
# Individual phases; fitting resumes completed candidates after verification:
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coverage-control --phase prepare
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coverage-control --phase fit --workers 10
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-coverage-control --phase report
```

Output is separate from previous experiments:
`outputs/analysis/pairwise_geometry/omc_coverage_control/`.

- `index.html`, `findings.md`: descriptive conclusions, coverage plots, paired recovery effects and per-system plots.
- `manifest.json`: complete candidate indices, shared seeds, frozen sigmas, source hashes and implementation identity.
- `coverage_audit.csv`, `coverage_gate.csv`: pre-fit structural and observable coverage checks.
- `results.csv`, `populations.csv`: all 780 fits and conditional basin metrics, including convergence flags.
- `paired_effects.csv`, `repetition_summary.csv`, `summary.csv`: primary outcome and descriptive repetition summaries.
- `residual_decomposition.csv`, `residue_terms.csv`: exact population/coverage/reweighting residual decomposition, including squared-error cross terms.
- `ess_transitions.csv`, `ess_matches.csv`: inverse-ESS decomposition and MaxEnt matching sensitivity.
- `analytic_controls.csv`: uniform full-source zero-population-error control, without fitting.
- `audit.json`: final fit count, convergence, source immutability and manifest identity.

CSV tables also have Parquet equivalents. Candidate folders retain weights and verifiable completion receipts. Feature-row numbers in the residue tables join to each system's `residue_mapping.csv`.

## Frozen design

Use the original 512 source frames, population basins and synthetic uptake targets. Thin the largest basin to 50%, 25% or 12.5%, using the original floor-rounded counts, while retaining the other basin completely. Use five paired seeds, `20260909` through `20260913`.

Structural strata are built from Euclidean distances between complete Cα pair-distance vectors, divided by the square root of the vector length. Recursively split along the farthest-pair direction until the number of strata equals the required retained count. Allocate frames so strata differ in size by at most one; source-frame index breaks ties. Draw one frame per stratum using its minimum random priority. The random comparator retains the globally lowest-priority frames. Both methods use the same priorities in each repetition.

No uptake values, fitting results or recovered populations enter selection. The random subsets are nested across retention levels; stratified subsets are built independently at each retention. Comparisons are paired within retention. Stratum sizes can differ by one, so coverage preservation is approximate rather than an exact distribution identity.

Before fitting, require the median stratified/random **structural energy-distance ratio** to be below one at each system/retention. Freeze all five repetitions; do not search alternative seeds if the manipulation fails. Report conditional uptake-mean MSE, balanced-truth MSE, Cα Rg Wasserstein distance and contact-map RMS differences as additional diagnostics, without using them to select candidates.

The prepared experiment passed all six checks:

| System | Retain 50% | Retain 25% | Retain 12.5% |
|---|---:|---:|---:|
| 1tzw_A | 0.821 | 0.815 | 0.751 |
| 1dd3_B | 0.258 | 0.286 | 0.450 |

Values are median structural-error ratios, stratified/random.

## Fitting and analysis

OMC strength remains 0.1. The six **actual numerical sigmas** from the previous random candidate are frozen separately for each system/retention and shared by both methods and all repetitions. Quantile labels remain 0.02, 0.04, 0.08, 0.16, 0.32 and 0.64; these are legacy labels, not quantiles recalculated from the new candidates.

Reuse the existing six reverse-KL MaxEnt strengths and unregularised comparator. All-residue/time MSE, uniform candidate priors, Adam learning rate 0.05 and uninterrupted checkpoints 1000/3000/10000 remain unchanged. Relative objective change over the final 250 steps must be at most 0.01. Ten spawned workers use at most two CPU cores each and one BLAS thread.

Total: 2 systems × 3 retentions × 5 repetitions × 2 methods × 13 arms = **780 fits**.

The primary per-bandwidth effect is:

```
(TV_OMC - TV_unregularised)_stratified
    - (TV_OMC - TV_unregularised)_random
```

Negative favours stratification. Each paired point requires both OMC fits and both corresponding unregularised fits to converge. A repetition-level median requires all six bandwidth pairs. Report median/minimum/maximum across complete repetitions, alongside missing coverage; bandwidths are not independent replicates. No bandwidth is chosen by population truth.

Inspect absolute population errors as well: an improved relative effect does not necessarily imply OMC beats unregularised fitting. Compare MaxEnt at saved-fit ESS-fraction tolerances 0.02, 0.01 and 0.005, retaining gaps and exclusions. Check the exact residual decomposition, conditional ESS and the ESS ceiling compatible with correct basin populations.

Interpret outcomes separately: coverage and relative recovery improve; coverage improves without relative recovery improvement; or structural coverage improves without uptake coverage improvement. These are descriptive interventions within two fixed ensembles, not independent biological replication or proof of a universal mechanism. Fewer than three complete repetitions are explicitly labelled insufficient for a stable descriptive verdict.

## Verification and boundaries

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-coverage-mpl uv run --no-sync pytest \
  tests/test_omc_coverage_control.py tests/test_omc_bandwidth_diagnostics.py -q
```

Tests cover deterministic balanced partitions, degenerate geometry, matching basin counts, no duplicate selections, shared frame priorities and exact supplied bandwidth use. Existing diagnostic tests cover residual/ESS identities and graph energy. Every completed candidate passes the original saved-weight/metric verification. Cache identity includes the complete frozen manifest and implementation hashes; changing candidate indices, seeds or bandwidths changes that identity. Corrupt or mismatched completion receipts are not reused.

No new systems, bandwidths, OMC strengths, alternative graph kernels or ISO testing are introduced. Synthetic uptake uses all eligible feature residues at three times, not experimental exchange measurements. Previous experiment artifacts remain unchanged.
