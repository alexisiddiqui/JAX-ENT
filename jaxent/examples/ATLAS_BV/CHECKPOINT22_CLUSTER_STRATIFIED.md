# Checkpoint 22: structural-cluster-stratified population recovery

## Question

Does the loss of fixed-BV population recovery at large structural W1 arise mainly from comparing
different structural basins? This pilot replaces the earlier trajectory-medoid interpretation with
explicit structural clusters and reports every previously retained standalone metric separately for
pairs within and between clusters.

## Leakage controls and definitions

- Clusters use structural coordinates only; PF/BV features never define a cluster.
- PCA and clustering are fit on replica A only. Replica B and C frames are projected into that fixed
  partition.
- The selected partition is K-means after an A-only 95%-variance PCA. Aligned C-alpha coordinates
  and C-alpha internal-distance W1 signatures are compared; the higher A-only silhouette wins.
- HDBSCAN is audited independently in five exact distance spaces: C-alpha RMS, structural W1,
  periodic backbone dRMSD, circular-z dRMSD, and circular-z quadratic distance.
- The MD target remains the rank-10 structural-W1 kernel density. Model parameters are fit on A,
  ridge hyperparameters use B where applicable, and recovery is measured on C.
- The x axis retains the pre-existing global structural-W1 bands: q0 0.017–0.053 A, q1
  0.053–0.108 A, q2 0.108–0.205 A, q3 0.205–0.445 A, q4 0.445–1.859 A, and q5
  1.859–23.269 A.
- Each cell requires at least 30 pairs and 20 unique frames. Recovery is
  `100 * (1 - sqrt(JSD))` between the predicted and MD-target population-change distributions.

The complete metric set contains 26 standalone predictors: Work Scale/Shape and all density
definitions; PF L1, L2, cosine, correlation, raw and centred PF-W1, variance-scaled information
distances and per-residue ridge; contact-channel metrics and ridges; and structural W1, aligned
C-alpha RMS, raw periodic dRMSD, circular-z dRMSD, and z-quadratic controls. BV coefficients are
fixed.

## Pilot result

The fixed 12-system pilot completed. Eleven systems have evaluable within and between cells
somewhere on the W1 axis. A-only W1-PCA was selected for all 12 systems (11 two-cluster and one
three-cluster solution). At least one valid exact-space HDBSCAN solution was found for 11 systems;
raw periodic dRMSD was the most consistently viable density space (11/12), followed by C-alpha RMS
(8/12) and structural W1 (5/12).

The tail is not sufficiently populated for a full-cohort expansion. Only two systems contribute a
q5 within-cluster estimate and five contribute q5 between-cluster estimates; only two contribute
both. The predeclared expansion gate now explicitly requires at least 8/12 systems in both q5
strata and therefore fails.

Provisional q5 medians are nevertheless suggestive. Work Scale recovery is 69.8% within clusters
(two systems) and 50.4% between clusters (five systems). PF correlation is the leading q5 metric in
both evaluated partitions, at 75.0% within and 57.3% between, but these ranks are not stable enough
to select a model given the system counts. For Work Scale, common-support between-cluster recovery
is 59.7% (four systems), while novel-support recovery is 40.4% (five systems).

Thus clustering supports the hypothesis that cross-basin and novel-frame comparisons are harder,
but it does not yet quantify the q5 contrast robustly. Expanding the same design to all systems
would add computation without fixing the sparse conditional cell. A better next design is to use
cluster-aware sampling (minimum within- and between-cluster tail pairs per system) or report a
continuous same-cluster interaction rather than six hard W1 bands.

## Outputs

- `outputs/analysis/pairwise_geometry/checkpoint22_cluster_stratified/within_between_recovery.png`
- `cluster_stratified_results.parquet`: system-level recovery, pair and unique-frame counts
- `cluster_stratified_summary.parquet`: median and SD across systems
- `cluster_audit.parquet`: K-means and HDBSCAN feasibility/sensitivity
- `cluster_metric_fits.parquet`: fitted metric parameters and selected cluster metadata
- `checkpoint22_report.yaml`: feasibility and expansion decision

Run with:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-cluster-stratified
```

## Full 111-system single assignment

The authorized full A-fit/B-tune/C-test run completed for 111/111 systems. Thirteen systems have a
q5 within-cluster estimate, 21 have a q5 between-cluster estimate, and 11 have both. The unpaired
Work Scale medians are 68.2% within and 53.7% between; these must not be interpreted as a paired
cluster effect because they use different system subsets. Across the same 11 systems, the median
paired within-minus-between change is -1.46 recovery points. The larger run therefore does not
support a systematic within-cluster q5 advantage for Work Scale, despite confirming that novel
support is difficult (46.6% q5 median across 22 systems).

Full outputs are under `checkpoint22_cluster_stratified/full_single_assignment/`, including the
paired contrast table and plot. Run or regenerate plots with:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-cluster-stratified --full
./jaxent/examples/ATLAS_BV/commands.sh geometry-cluster-stratified --full --plot-only
```
