# Fixed-strength OMC bandwidth control

This experiment tests bandwidth-driven ESS control and population recovery on 12 new
ATLAS systems. `1yoz_B` is a separately reported calibration reference. The experiment
uses synthetic fixed targets from each source ensemble before candidate filtering.

## Frozen design

Select four eligible systems from each RMSF tercile using seed `20260826`. Within
each tercile, round-robin the available CATH classes; deterministic seeded ordering
breaks ties between classes and systems. Exclude the calibration system. Replace
ineligible inputs or partitions using the same ordering, without fitting. Freeze
all 12 systems and the reference in `manifest.json` before starting any fits.

Use replica 1, at most 512 evenly spaced source frames, the existing structural W1
distance profiles, and the existing work-distance RBF kernels. Compute pseudo-uptake
on the complete source once; its uniform average is the fixed target. Neither the
target nor observable centering changes when frames are filtered.

Evaluate KMeans k=2–8 with `n_init=20`. Select the greatest positive precomputed-W1
silhouette score among partitions with at least 20 frames in every cluster; ties
choose smaller k. Freeze labels. Retain an unfiltered control and candidates retaining
50%, 25% or 12.5% of the largest cluster. All other clusters remain intact. Filtering
is nested and seeded per system, retains at least two frames per cluster, and ties
for the largest cluster choose its smaller label. Each candidate starts uniformly.

Every candidate has exactly 13 fits:

| Method | Fixed settings |
|---|---|
| Original OMC | Strength 0.1; bandwidth quantiles 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 |
| Existing MaxEnt | Strengths 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 |
| Unregularised | One zero-strength reference |

The original OMC and existing MaxEnt definitions are unchanged from calibration;
the latter uses KL(uniform candidate prior || fitted weights). MaxEnt's six strengths
span the onset of ESS changes through near-uniform fits in the calibration. There
is no OMC strength sweep or extra fitting for ESS matching.

Use all-residue/time MSE, scaled by `var(target) + 1e-8` during optimisation, Adam
learning rate 0.05, and the same JAX numerical precision as the calibration. Check
at 1,000 steps; continue unresolved trajectories to 3,000, then 10,000, retaining
Adam state. Convergence is at most 1% absolute objective change relative to the final
objective (denominator floor 1e-12) over the last 250 steps. Retain absolute changes,
gradient norms and unresolved flags. Near-zero-error controls can fail the relative
criterion despite small absolute errors.

## Running and resuming

```bash
bash jaxent/examples/ATLAS_BV/commands.sh geometry-omc-bandwidth-control --phase screen
bash jaxent/examples/ATLAS_BV/commands.sh geometry-omc-bandwidth-control --phase run --workers 10
bash jaxent/examples/ATLAS_BV/commands.sh geometry-omc-bandwidth-control --phase report
```

`run` also screens if no manifest exists. `report` never fits. `--output PATH` selects
a separate output directory. Defaults are under
`outputs/analysis/pairwise_geometry/omc_bandwidth_control/`. The completed earlier
calibration is preserved. Input content, numerical protocol, frozen source arrays
and clustering audits are fingerprinted. A candidate resumes only when its weights,
fit metrics and populations match a completion receipt and pass reconstruction
checks. Invalid or interrupted candidate shards are recomputed.

The runner defaults to ten spawned worker processes. Each candidate has an exclusive
file lock and is rechecked inside the lock, preventing duplicate fitting when workers
or runners overlap. CPU affinity partitions the available cores between workers
(two cores each on the current 20-core machine); native BLAS threads remain limited
to one. `--workers 1` provides serial execution. Parallelism changes scheduling only;
the frozen numerical protocol and existing completed results are reused.

The full budget is 624 new-system fits plus 52 reference fits: **676 total**. This
pass uses one replica, one partition and one filtering seed per system. It does not
run ISO tests, introduce additional regularisers or expand bandwidths.

## Interpretation and artifacts

`index.html` links the full tables and shows cohort summaries, per-system bandwidth
curves, ESS–population-error and ESS–MSE tradeoffs, and target/candidate/recovered
cluster populations. Every setting stays visible; red crosses identify unresolved
fits. Tables include within-cluster ESS, structural dispersion and unfiltered controls.

Match each converged OMC fit to the converged MaxEnt setting nearest in ESS fraction,
breaking ties toward smaller strength. A match is valid only within 0.02 absolute
ESS fraction. Save the actual mismatch, missing-match reason, population-error and
MSE differences. Do not interpolate; a MaxEnt setting can match several OMC settings.

Headline ESS ranges and rank correlations require all six OMC settings to converge;
changes from the baseline also require baseline convergence. Matched comparisons
require both fits to converge. Aggregate settings within cases, retention levels
within systems, then systems. Report contributing-system counts and incomplete
curves; show the calibration reference separately. Unfiltered controls are excluded
from recovery headline summaries.

ESS measures weight spread. Population total-variation error measures recovery of
the frozen source populations. Neither low MSE nor greater ESS alone establishes
population recovery; inspect all three and the matched MaxEnt comparison together.

## Frozen cohort

Screening selected all systems below without exclusions. Although k=2–8 were
evaluated, the highest eligible silhouette score selected **k=2 for all 13 systems**.
This run therefore evaluates cross-protein and filtering-severity variation, but
does not establish generality to more complex partitions.

| RMSF tercile | New systems |
|---|---|
| Low | 5noh_A, 1cuo_A, 1c1k_A, 1pch_A |
| Middle | 4g6d_B, 3kvd_D, 1tzw_A, 5x1u_B |
| High | 1dd3_B, 3fpu_A, 2ad6_D, 1ef1_D |

The separate reference is `1yoz_B`. Source frames, labels, retained indices, input
hashes and all attempted silhouette scores are archived with the cohort.

## Completed results

All **676 fits** and **1,352 cluster-population rows** are complete. The run was
switched from serial execution to ten workers after 429 fits; completed candidates
were retained, and the interrupted candidate was recomputed.
The reference source arrays and target exactly match the earlier calibration.

| Cohort result (12 new systems) | Value |
|---|---:|
| Converged filtered-candidate fits | 460 / 468 (98.3%) |
| Fully converged six-bandwidth curves | 31 / 36 |
| Complete curves with higher ESS at q=0.64 than q=0.02 | 31 / 31 |
| Median system ESS-fraction span | 0.1202 (12.0 percentage points) |
| Median system bandwidth–ESS Spearman correlation | 1.0 |
| ESS-matched OMC points | 115 / 216 |
| Median population-TV change versus unregularised | +0.00608 |
| Median population-TV difference versus matched MaxEnt | −0.000963 |
| Median MSE difference versus matched MaxEnt | +9.69e-6 |

Medians first aggregate within each system as specified above; the calibration
reference is excluded. Lower TV and MSE are better. At matched ESS, system-level
population error is lower for OMC in seven systems and higher in five; MSE is higher
in all twelve. These summaries establish consistent bandwidth-driven ESS control
in this cohort, with population recovery depending on the system and operating
point. The small pooled recovery difference does not establish a general advantage.

There are 176 unresolved fits overall: 168 unfiltered controls and eight filtered
cohort fits. The near-uniform controls are particularly sensitive to the relative
convergence criterion at small objective values. They remain flagged in the output
and are excluded from recovery headline summaries.

Validation covered 26 tests, including spawned-process locking and safe resumption.
All saved weights were independently checked for frame identity, normalisation,
removed-frame zeros, all-residue MSE, cluster populations, total and within-cluster
ESS, structural dispersion and matched-comparison differences. All 27 plots and
34 report links were checked. See `artifact_audit.json` in the output directory.

Open `outputs/analysis/pairwise_geometry/omc_bandwidth_control/index.html` for the
complete report; `cohort_summary.png` is the standalone cohort comparison, and
`system_summary.csv` contains the system-level table with contribution counts.
