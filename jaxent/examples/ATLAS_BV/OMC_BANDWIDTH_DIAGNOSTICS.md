# OMC bandwidth mechanism diagnostics

This is a read-only analysis of the frozen 12-system cohort and `1yoz_B` calibration reference. It diagnoses the four questions in the bandwidth report using the saved weights, source features and exact source trajectory frames. It performs **no optimisation, target redesign, hyperparameter search or ISO experiment**.

Run from the repository root:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-bandwidth-diagnostics --workers 10
# Rebuild narrative/plots and recheck artifacts from completed diagnostic tables:
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-bandwidth-diagnostics --report-only
# Independent mapping/analysis smoke check, with no cohort-wide conclusions:
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-bandwidth-diagnostics \
  --systems 1tzw_A --workers 1 --output /tmp/omc-diagnostic-smoke
```

Inputs remain in `outputs/analysis/pairwise_geometry/omc_bandwidth_control`.
Outputs are separate, under `outputs/analysis/pairwise_geometry/omc_bandwidth_diagnostics`:

- `index.html`: conclusions, overlap matrix, system and severity tables, annotated structural context, residue-error plots and weight-distribution plots.
- `findings.md`: mechanistic interpretation, case studies and proposed tests for discussion.
- `mechanism_verdicts.csv`: directly demonstrated / supported association / contradicted / unresolved.
- `system_overlap.csv`, `case_overlap.csv`: continuous magnitudes alongside descriptive flags.
- `ess_transitions.csv`: every adjacent bandwidth transition and exact inverse-ESS contributions.
- `fit_diagnostics.csv`: population ceiling, MSE decomposition, cancellation and fitted mean shape.
- `graph_diagnostics.csv`: unnormalised within/between coupling and OMC energies, including deterministic balanced weights.
- `observability.csv`: population projection onto weak singular directions, including numerical nullspace.
- `matches.csv`, `matching_sensitivity.csv`: actual saved-fit nearest matches at 0.02, 0.01 and 0.005 ESS-fraction tolerances; evaluate both weight vectors under the same OMC objective.
- Per-system `residual_terms`, `matched_residues`, `residue_structure`, `residue_coverage`, `residue_mapping`, `frames`, `coverage`, `basin_weights`, and contact-map arrays, with CSV/Parquet downloads.
- `audit.json`: original-summary and matching parity, input hashes, identities and all local report links.

## Exact definitions

Saved float32 weights are promoted to float64 and normalised to the simplex; each system receipt reports the largest original sum correction. The recalculated MSE, population TV and ESS must agree with saved metrics. All paired comparisons requiring fitted solutions exclude the original unconverged fits. Controls are retained for auditing but excluded from mechanistic cohort summaries.

**All-residue MSE** uses every eligible feature residue at synthetic times 0.1, 1 and 10. It does not imply that excluded terminal/proline residues have measurements. The synthetic rates use residue-centred log-PF on the 512 selected source frames. These are not experimental exchange rates; acceptor counts are a contact-based proxy rather than observed hydrogen-bond occupancy.

With basin mass `p_c` and conditional ESS `E_c`,

```
1 / ESS = sum_c p_c² / E_c
Δpopulation = sum_c (p2_c² - p1_c²) (1/E1_c + 1/E2_c) / 2
Δwithin     = sum_c (1/E2_c - 1/E1_c) (p1_c² + p2_c²) / 2
```

The last two sum to **Δ inverse ESS**, not Δ ESS. Reversals require convergence at both adjacent endpoints; even transitions within otherwise incomplete curves are retained if both endpoints converge. System ESS spans and comparisons against unregularised fitting use complete curves, preserving the original summary convention. Low response means the bottom quartile of complete cohort curves within each retention, not a new failure gate.

At exact target masses, the maximum ESS is

```
ESS_max = 1 / sum_c (p_true_c² / n_retained_c)
```

It is attained by deterministic weights `p_true_c / n_retained_c` within each basin. Higher ESS necessarily changes at least one target mass. These weights are **not** the MSE-optimal solution subject to true populations, and their MSE is not a lower bound on that solution's error.

Let source, retained-uniform and fitted-conditional uptake means be `μs`, `μk` and `μw`. The observation-wise residual is exactly

```
A = sum_c (pfit_c - ptrue_c) μs_c
B = sum_c pfit_c (μk_c - μs_c)
C = sum_c pfit_c (μw_c - μk_c)
prediction - target = A + B + C
```

The tables preserve `A²`, `B²`, `C²`, `2AB`, `2AC`, `2BC`; these sum to actual squared error. Cancellation fraction is `1 - MSE / mean(A²+B²+C²)`. It quantifies residual cancellation, not lost biological information.

The original graph energy is

```
R = 0.5 N² sum_ij wi wj Kij (wi - wj)²
```

Ordered within/between edges partition the energy exactly. Kernel coupling excludes self edges when reported as graph density. The unnormalised kernel changes total coupling as bandwidth changes, even at fixed strength 0.1. Evaluating saved MaxEnt weights under this same energy is analysis, not refitting. The original MaxEnt arm uses reverse KL from uniform to fitted weights, not forward KL.

Observable SVD uses the candidate matrix centred across frames, which removes the uniform/simplex-normal direction. Basin indicators are likewise centred. Relative singular-value thresholds are `1e-6`, `1e-4`, `1e-2`; the full right-singular basis includes nullspace. Numerical nullity uses `s_max * max(matrix.shape) * eps(float64)`. Projection assesses local linear sensitivity only; positivity and nonlinearity of the regulariser constrain permissible alternatives.

## Structural interpretation

The trajectory frame indices are resolved through post-equilibration feature columns, and the reconstructed synthetic uptake, scalar work distances and Cα pair-distance-signature W1 matrix must match the frozen source. Cα coordinates are Kabsch-aligned. Rg uses equal-weight Cα positions in Å; shape anisotropy is based on the gyration tensor. Contacts use 8 Å and sequence separation greater than three. Contact maps and retained/source distributions measure structural coverage, not experimental stability.

Feature topology rows map to simulated residue numbers. The correspondence table is sequence-checked against the actual simulation and supplies original deposited residue numbers; corrected simulation residues follow the local `UnP_seq` column. Do not mistake fragment-local `UnP_num` values for authoritative full-length UniProt residue positions. ATLAS crystallographic partner-contact counts are context, not proof of an obligate assembly or an interface mechanism. Visual representatives are structural-W1 medoids of the saved source basins, not newly predicted structures.

Primary structure annotations are curated in `analysis/omc_diagnostic_annotations.txt`, with PDB/PDBe/NCBI links in the report. Functional context does not establish a cause of recovery error. The diagnostic does not infer biological activity from frame weights. Spearman associations across twelve systems are descriptive; CATH labels are metadata, not causal explanations.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-mpl uv run --no-sync pytest \
  tests/test_omc_bandwidth_diagnostics.py tests/test_omc_bandwidth_control.py -q
```

Tests cover exact ESS and residual identities, the target-population ceiling, graph-energy decomposition and diagonal invariance, observable nullspace handling, and rigid-body alignment. The full run checks all source mappings, numerical metrics, saved fit hashes, original summary and nearest-match parity, and report links. Ten spawned workers use one BLAS thread each. The runner does not import or invoke the optimisation module.

## Result

The full diagnostic run reproduces the original summary counts. Recovery failures against both comparators overlap in `1c1k_A`, `1pch_A`, `1tzw_A`, `2ad6_D`, and `5x1u_B`. ESS reversals/limited response form a partly different group. Exact residual cancellation and the changing graph penalty explain why MSE, population recovery and ESS need not improve together. Tighter saved-fit matches weaken the apparent population advantage over MaxEnt. See `findings.md` for measured evidence, biological context, and the limits of each causal claim.
