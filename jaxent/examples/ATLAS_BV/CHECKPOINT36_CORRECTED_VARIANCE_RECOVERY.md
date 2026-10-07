# Checkpoint 36: corrected BV variance recovery

This checkpoint repeats the checkpoint-28 direct and local variance-magnitude
comparison after the BV contact and intrinsic-rate unit corrections. It is intentionally
separate from the OMC/Laplacian work and writes to a fresh checkpoint-36 cache, leaving all
historical feature and result directories untouched.

The locked BV protocol is the Bradshaw/Radou rational 6--12 switch with a 0.1 Angstrom
scale, 6.5/2.4 Angstrom heavy/acceptor midpoints, `bc=0.35`, `bh=2.0`, all explicitly
modelled atoms, the chain-local residue exclusion `[-2, 2]`, and intrinsic rates stored in
`min^-1`. Every replica is independently checked at frames 101, 550, and 1000 against the
contact equation and against rates recalculated from sequence. Where the checkpoint-26
graph audit cache exists, the new cache is also required to match it.

Replica A fits alpha, replica B chooses `k` from 5, 10, 20, and 50 and shrinkage from
0.001, 0.01, and 0.1, and replica C is held out for evaluation. The target remains the
magnitude of the rank-10 structural-W1 KDE log-density change with at most 10,000
deterministically sampled pairs. Recovery is reported in six global W1 bands, requiring at
least 30 test pairs per band.

Metrics include all provided Work outputs (Scale, Shape, Density, Fitting, Magnitude, and
Opt), PF L1/L2, and coordinate RMSD/W1. Local variance magnitude is evaluated for the
primitive Work and PF representations. RMSD and W1 receive the analogous local
RMS-radius dispersion control. The report records raw alpha, the dimensionless normalized
scale `alpha * ||x|| / ||y||`, feature and target RMS, selected local parameters, and an
explicit warning for alpha at or above 1000.

Run the mandatory single-system review:

```bash
jaxent/examples/ATLAS_BV/commands.sh geometry-corrected-variance-recovery
```

The outputs are under
`outputs/analysis/pairwise_geometry/checkpoint36_corrected_variance_recovery/single`,
including `recovery_vs_w1.png`, `parameter_distributions.png`, parquet tables, an alpha
comparison to checkpoint 28, and `checkpoint36_report.yaml`.

## Single-system review: `2w86_A`

The corrected run completed on all three replicas. Every independently recalculated
contact and rate check had zero numerical discrepancy, including comparison with the
corrected checkpoint-26 graph cache. The log-PF medians were 0.758, 0.824, and 0.806 for
replicas A, B, and C, respectively.

No 1000-scale alpha pathology remains. Direct alphas span 0.00969--9.60,
variance-magnitude alphas span 7.15--31.52, and coordinate-dispersion alphas span
2.82--3.63. Their scale-independent normalized alphas collectively span 0.548--0.924.
The small raw Work Density/Fitting/Opt alphas reflect those features' much larger
numerical RMS (about 94 versus target RMS 1.30), rather than a degenerate fit. This is why
the normalized-alpha panel accompanies the raw parameter plot.

These are single-system diagnostics, not evidence for a cohort-level ranking. In held-out
replica C, q5 recovery is 77.4% for direct Work Scale, 44.9% for direct Work Magnitude,
35.5% for direct PF L2, 42.4% for direct RMSD, and 30.3% for direct W1. The corresponding
local controls reach 84.7% for Work Scale variance magnitude, 66.1% for PF variance
magnitude, 70.3% for RMSD dispersion, and 86.4% for W1 dispersion. The q0 band is omitted
because this system has fewer than the locked minimum of 30 held-out pairs there.

The 24-system frozen checkpoint-26 pilot is deliberately gated. Only after reviewing a
clean single-system report can it be started explicitly:

```bash
jaxent/examples/ATLAS_BV/commands.sh geometry-corrected-variance-recovery \
  --scope pilot --approve-pilot --workers 2
```

## Frozen 24-system pilot

The approved pilot completed for all 24 systems and all 72 replicas. Recalculated
contacts, intrinsic rates, and corrected checkpoint-26 reference features agreed exactly.
No parameter pathology was detected: direct alphas span 0.00409--19.05, local coordinate
dispersion spans 1.00--5.00, and variance-magnitude alphas span 0.904--83.10. Across all
models, normalized alpha remains between 0.464 and 0.985.

Mean q0/q5 recovery is 60.7%/54.6% for direct Work Scale, 68.2%/53.8% for its local
variance magnitude, 67.8%/55.6% for Work Density variance magnitude, and 68.4%/50.9%
for PF variance magnitude. Direct RMSD reaches 25.9%/41.4%, while local RMSD dispersion
reaches 63.0%/57.6%. Local W1 dispersion gives 79.9%/79.2%, but is a deliberately circular
positive control because both its neighbourhood and the KDE target are defined from
structural W1. Direct W1 gives 64.4%/39.0%.

Only systems with at least 30 held-out pairs in a band contribute to that band: 12 systems
contribute to q0 and q5, 18 to q1, 22 to q2 and q3, and 19 to q4. Consequently these
bandwise means should not be interpreted as a single balanced 24-system trajectory.

## Total-energy column (17 September 2026)

The 24-system figure now includes PyRosetta `ref2015` total scores (REU) and OpenMM
CHARMM36 protein-vacuum total energies (kJ/mol), reusing the checkpoint-24 and
checkpoint-23 per-frame caches. All 144 energy files were checked for finite values and
exact replica/frame alignment with the 2,700 analysed frames per system. Source hashes
are recorded in `corrected_variance_runtime.parquet`. Direct predictors are absolute
total-energy differences with alpha fitted on A. The lower row applies the same
W1-neighbour variance-magnitude statistical test, selecting k and shrinkage on B and
evaluating on C. Native energy units are retained, so raw energy alphas have reciprocal
REU or mol/kJ units; normalized alpha permits comparisons across predictor scales.

| Total-energy predictor | q0 recovery | q5 recovery |
|---|---:|---:|
| PyRosetta direct | 69.5% | 55.7% |
| OpenMM direct | 63.7% | 58.0% |
| PyRosetta variance test | 73.3% | 60.3% |
| OpenMM variance test | 67.7% | 52.4% |

Each endpoint band includes 12 eligible systems. The original Work/PF/coordinate
results agree with the previous run within 1.2e-15. All 528 fits are finite with no
alpha-threshold warning. The recovery figure is exported as PNG and PDF; the parameter
plot includes both energies. The preceding pilot artifacts are preserved in
`checkpoint36_corrected_variance_recovery/pilot_before_energies_20260917`.

## Radius-of-gyration control

The coordinate column additionally includes equal-weight C-alpha radius of gyration,
`Rg = sqrt(mean_i |r_i - mean_j r_j|^2)`, in Angstroms, calculated from the same
coordinates as the structural controls. Its direct predictor is `|Rg(frame a)-Rg(frame b)|`.
The lower panel uses the same local RMS-radius dispersion construction as RMSD/W1,
with pairwise Rg differences and W1-selected neighbours. Alpha fitting, B-replica
parameter selection, sampled pairs, and C-replica distribution evaluation are unchanged.
Artifacts preceding this addition are retained in `pilot_before_rg_20260917`.
