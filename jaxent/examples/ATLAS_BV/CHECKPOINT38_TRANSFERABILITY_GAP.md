# Checkpoint 38: global-alpha transferability gap

This checkpoint pairs the held-out replica-C results from checkpoints 36 and 37
without refitting either model. It quantifies the absolute global-alpha recovery
gap for every system, predictor, and eligible structural-W1 band, then tests
whether that gap is explained by cohort heterogeneity or by the mismatch between
the per-system and pooled alpha values.

Run the analysis with:

```bash
jaxent/examples/ATLAS_BV/commands.sh geometry-transfer-gap --workers 4
```

The structural descriptors use all 2,700 post-equilibration frames per system.
C-alpha radius of gyration and MDAnalysis DSSP helix/sheet/coil fractions are
combined with chain length into a cohort-atypicality score. The primary analysis
contains the 11 direct Work, PF, and coordinate predictors. Energy, local-variance,
and local-dispersion predictors are sensitivity analyses.

Across the primary predictors, alpha mismatch strongly predicts the absolute
recovery gap after adjustment for predictor and band availability (standardized
coefficient 0.851, 95% clustered-bootstrap CI 0.692--1.000, block-permutation
`p=1.0e-4`). The pre-specified heterogeneity mechanism is not supported:
cohort atypicality does not predict alpha mismatch (`p=0.129`) or the unadjusted
recovery gap (`p=0.512`). After alpha mismatch is included, its coefficient is
negative rather than attenuated toward zero. This establishes that alpha
non-transferability matters, but the tested chain-length/Rg/DSSP composite does
not explain why the system-specific alphas differ.

Outputs are written under
`outputs/analysis/pairwise_geometry/checkpoint38_transferability_gap/`. The main
artifacts are `transferability_gap_by_predictor.png`,
`alpha_mismatch_mechanism.png`, `predictor_heterogeneity_correlations.png`,
`system_gap_and_atypicality.png`, the paired parquet tables, and
`checkpoint38_report.yaml`.

## Plain-language interpretation

The strongest result is narrower than “PF and Work predict every protein with one
universal model.” Checkpoint 38 changed only the scalar alpha. For direct PF L1/L2
and the strongest direct Work predictors, replacing each protein's fitted alpha
with one pooled alpha changed held-out distribution recovery relatively little:
the mean absolute gaps are about 0.031--0.044. Direct Work Scale is a less portable
exception at 0.088. The corresponding local-variance gaps are much larger,
approximately 0.152--0.197.

Thus the **scale of the direct PF/Work relationship is comparatively portable
across these 24 systems**, whereas the scale attached to local variance is much
more system-dependent. “Comparatively” matters: individual systems still have
nonzero errors, and this experiment does not establish a universal physical
constant. The local-variance conclusion is also descriptive. Checkpoint 37 kept
each system's selected neighbourhood size and shrinkage, and the tested
chain-length/Rg/secondary-structure descriptors did not explain the remaining
alpha differences.

This result also does not answer how many known populations are needed inside one
protein. The checkpoint-36/37 predictors estimate the **magnitude** of a pairwise
log-density difference. They do not directly say which member of the pair is more
populated. Checkpoint 39 therefore supplies a small number of signed population
anchors and tests whether the absolute PF/Work distances can reconstruct unseen
local occupancies and forced structural-cluster populations.
