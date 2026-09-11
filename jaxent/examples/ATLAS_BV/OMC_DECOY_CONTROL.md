# Clustered-candidate OMC decoy experiment

The initial implementation uses 1tzw_A, a fixed set of 100 structural representatives,
independently clustered empirical targets, and internal/random/donor decoy challenges.
The existing BV featurisation pipeline is used with an experiment-specific smooth
contact cutoff. Its rational tail has a 0.5 Angstrom scale and contacts inside the
radius retain weight 1. Physical uptake is checked before performance fitting.

## Run and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-decoy-control --phase all --workers 10
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-decoy-control --phase prepare
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-decoy-control --phase run --limit 1 --workers 1
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-decoy-control --phase report
```

The default output is
`outputs/analysis/pairwise_geometry/omc_decoy_control/index.html`, with CSV/Parquet
tables and PNG/SVG figures. Preparation produces a report even when physical
preflight prevents fitting. `--limit` limits fitting cases only; it cannot bypass
the physical gate. `run_status.json` distinguishes a blocked run from a completed
or partial fitting run. Fits resume per case using hashes and file locks. Use a
fresh `--output` directory after scientific code, configuration or input changes.
Report-only changes do not invalidate the fit identity.

## Frozen design

- Reuse the previous 512 post-equilibration R1 source-frame indices. Cluster full
  C-alpha pair-distance vectors with KMeans, `k=100`, 20 initialisations and seed
  20260909. Keep the actual frame nearest each centre. These representatives are
  frozen across all challenges.
- Independently select target k from 3–8 using maximum silhouette, requiring at
  least 20 frames in each state. Break ties toward smaller k. Require a candidate
  representative in every selected target state.
- Baseline: full empirical reference target. Internal: remove each target state
  in turn and condition the empirical source weights on the retained states.
  The 100 candidate representatives stay unchanged.
- External: append 25 profiles to the native 100, with the unchanged full target.
  Random profiles independently sample each residue's reference log-PF marginal.
  Donor profiles sample 2w86_A R1 frames, randomly crop contiguous ordered profiles,
  and support tiling for shorter profiles. Recipient intrinsic rates are always
  retained. Five seeds, 20260909–20260913, are used separately for each external
  decoy type. Frame/crop/tile provenance is stored in each case directory.
- Donor profiles are synthetic feature decoys, with no claimed residue homology
  or recipient structural coordinates. Structural coverage and conditional ESS
  are evaluated on the native representatives; total ESS and decoy mass are
  reported separately.

The contact override is frozen in `omc_decoy_config.yaml`: `smooth_cutoff`, 0.5
Angstrom scales for both heavy and acceptor contacts. The equation is
`1 / (1 + (max(0, distance - radius) / scale)**6)`. Radii remain 6.5 and 2.4
Angstroms, coefficients remain 0.35 and 2.0, and the atom environment and sequence
exclusions are unchanged. This is a new explicit mode; `bradshaw_switch`,
`legacy_switch` and `hard` retain their existing meanings and the shared ATLAS
configuration is unchanged.

Recipient and donor features are generated into this experiment's `features/`
directory with the normal BV CLI, using 1,001 R1 frames for each system. Previous
ATLAS caches are not overwritten. The new features are checked against independent
contact pair sums on three frozen frames, including periodic distances. Cached
intrinsic rates are checked against full-sequence rates at 300 K and pD 7 in
seconds^-1. Provenance includes the exact command, effective protocol and hashes.

For both target generation and fitting:

```
mean_logPF = logPF @ population_weights
uptake(t, r) = -expm1(-t * k_int[r] * exp(-mean_logPF[r]))
```

This is the existing BV mean-log-PF convention with stable small-value evaluation.
Times are 10, 60, 600, 3600 and 14400 seconds. There is no noise, residue centring,
peptide aggregation or automatic coefficient/exposure adjustment. The gate
requires at least one full-reference observable inside (1e-8, 1-1e-8), and at least
one state-removal contrast greater than 1e-8 in absolute uptake. This is a numerical
informativeness guard, not an experimental detection limit. Individual unresolved
state-removal challenges are excluded from fitting even if other contrasts pass.

Each informative case has 13 arms: OMC strength 0.1 at the six existing quantiles
0.02, 0.04, 0.08, 0.16, 0.32 and 0.64; six existing reverse-KL MaxEnt strengths
1e-5 through 1; and unregularised fitting. Bandwidths are recalculated from the core
candidate's positive pairwise scalar mean-log-PF distances, then frozen when decoys
are appended. The original OMC weight regulariser is retained. The all-residue,
all-timepoint MSE is divided by the full-reference target variance plus 1e-8,
fixed across challenges. No truth-based setting selection is permitted.

Adam uses learning rate 0.05 and checkpoints 1000/3000/10000. Two initialisations
(uniform and small seeded logit perturbations) are run. The lower-objective result
is saved, alongside both starts. Both must satisfy a 1% final-window objective
change and agree in objective within 1% for the arm to enter selected comparisons.
Weight disagreement and gradient norms remain visible. Plateau checks alone do
not establish global optimality.

## Interpretation and diagnostics

The report measures population TV including external mass, conditional retained
population error, decoy mass, total/native ESS, and native coverage using weights
at least 1e-4. It plots complete OMC curves and MSE-selected populations. MaxEnt
comparisons use the nearest native ESS fraction and explicitly flag differences
greater than two percentage points; there is no interpolation or extrapolation.

Representation diagnostics include candidate/target cluster mixing, source-cluster
occupancy-weight uptake, and nearest-supported-representative quadrature. The
quadrature's population and uptake errors are shown with its OMC penalty; it is
not an exact ground-truth set of representative weights. Graph diagnostics report
cross-decoy connections and mean off-diagonal coupling. Absolute local population
Jacobian singular values expose weak observable information. Native graph proximity
uses the existing scalar mean-log-PF metric, not a guarantee of structural similarity.
Natural empirical targets need not imply graph-smooth representative weights.

## Physical checks after the 0.5 Angstrom change: 2026-09-09

Preparation selected the same **100 representatives and three target states** as
before (silhouette 0.1754; smallest state 127 source frames). All source-frame,
candidate and target-label arrays match the previous preparation exactly. There
are **14 cases**: one baseline, three internal removals, five random sets and five
donor sets. Contact-feature and intrinsic-rate checks passed for both systems.

The physical gate now passes: **613 of 690 reference observations** lie between
1e-8 and 1-1e-8. Reference-averaged log-PFs reach down to **1.151**; source-frame
log-PF quantiles (minimum, median, maximum) are approximately **0.000053, 12.061,
22.533**. Removing each target state changes at least one uptake observation by
**0.797, 0.459 and 0.323**, respectively. These are observable differences under
the frozen synthetic BV model, not experimental measurements.

The initial 10 Angstrom Bradshaw-switch diagnostic is preserved in
`outputs/analysis/pairwise_geometry/omc_decoy_control_bradshaw10/`. That preparation
had maximum uptake 1.3565e-39 and ran no fits. Its failure concerns the old physical
mapping; it is not a population-recovery result for the current contact protocol.

## Completed fitting results

All **182 arms across 14 cases** completed, including two initialisations per arm.
**174 arms passed** the frozen convergence/initialisation checks; eight remain
flagged and are excluded from selected comparisons. All saved weight vectors are
finite and normalised. No extra systems, strengths or bandwidths were added.

Using the all-residue-MSE-selected converged OMC arm:

| Challenge | Population TV | Decoy mass | Conditional native ESS fraction |
|---|---:|---:|---:|
| Baseline | 0.0250 | — | 0.7436 |
| Remove state 0 | 0.0280 | 0.0005 | 0.4463 |
| Remove state 1 | 0.0977 | 0.0977 | 0.5476 |
| Remove state 2 | 0.1078 | 0.1078 | 0.6163 |
| Donor profiles, median of five sets | 0.0226 | 0.0033 | 0.6771 |
| Random profiles, median of five sets | 0.1672 | 0.1672 | 0.7231 |

For the baseline, unregularised population TV is 0.0218 and native ESS fraction
0.4632. Removing state 0 favours OMC in population recovery (0.0280 versus
unregularised 0.0819); removing states 1 and 2 does not (OMC 0.0977/0.1078 versus
unregularised 0.0881/0.0703). Donor profiles receive little weight. Random profiles
remain a harder challenge. These are descriptive observations on one system;
no universal recovery verdict follows from them.

Unresolved arms are OMC q=0.02 and q=0.08 for internal state 2; OMC q=0.04 for
random seed 20260911; unregularised fits for random seeds 20260910, 20260911 and
20260913; one MaxEnt arm for donor seed 20260910; and the unregularised arm for
donor seed 20260911. All starts, objective disagreements, weight disagreements,
residuals and complete bandwidth curves are retained for diagnosis.

## Verification

```bash
uv run --no-sync pytest jaxent/tests/modules/HDX/test_contacts.py \
  jaxent/tests/modules/models/test_bv_config_contacts.py tests/test_omc_decoy_control.py -q
```

The 32 tests cover forward parity, seconds/minutes equivalence, small-uptake
numerics, nonlinear gradients and recovery, clustering independence, crop/tile
reconstruction, empirical target masses, frozen graph edges, ESS denominators,
the mandatory gate, prepared-input integrity, the fitted HTML/report path and
contact plateau/tail behaviour, hydrogen and sequence-neighbour exclusion, and
backward compatibility of the existing contact modes.
