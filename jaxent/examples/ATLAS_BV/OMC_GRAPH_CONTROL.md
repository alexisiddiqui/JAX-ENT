# Coupling-matched OMC graph comparison

This experiment reuses the completed 1tzw_A decoy experiment with 100 fixed
representatives, three independently selected target states and 14 challenges.
The 0.5 Angstrom contact features, physical mean-log-PF HDX forward model, MSE
normalisation, OMC strength 0.1 and reference target populations are unchanged.

## Run and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-graph-control --phase prepare
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-graph-control --phase run --smoke --workers 1
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-graph-control --phase run --workers 10
./jaxent/examples/ATLAS_BV/commands.sh geometry-omc-graph-control --phase report
```

The default `--phase all` prepares, verifies, runs and reports. Output is
`outputs/analysis/pairwise_geometry/omc_graph_control/index.html`. The report has
PNG/SVG plots and CSV/Parquet tables. Case directories retain distances,
bandwidths, raw and scaled kernels, fit diagnostics and both initialisations.

The manifest hashes all copied inputs, original fit provenance and energy files.
Scientific code or input changes require a fresh `--output` directory. Report
changes do not invalidate fits. Each graph/case job is locked and resumable.
Numerical fitting failures produce `failure.json`; failed baseline smoke jobs
stop the remaining run. Nonconverged finite solutions are retained and flagged.

## Frozen graph comparison

| Graph | Distance | Cases | New arms |
|---|---|---:|---:|
| Scalar log-PF | Absolute difference in mean residue log-PF | All 14; reused | 0 |
| Full-profile log-PF | RMS difference over corresponding raw residue log-PFs | All 14 | 84 |
| Structural | RMS difference over full C-alpha pair-distance vectors | Baseline + 3 internal | 24 |
| PyRosetta energy | Absolute ref2015 total-score difference | Baseline + 3 internal | 24 |

There are 132 new scientific arms (264 initialisation trajectories), plus a
six-arm scalar replay for adapter validation. All 182 original arms are reused,
including MaxEnt/unregularised controls and their original convergence flags.
The combined result table therefore contains 314 arms.

Each representation uses the existing Gaussian equation and bandwidth quantiles
0.02, 0.04, 0.08, 0.16, 0.32 and 0.64. Numerical bandwidths are calculated from
positive distances among the 100 native representatives and remain frozen when
external profiles are appended. Off-diagonal kernel entries are multiplied by
`scalar_mean_coupling / new_raw_mean_coupling` separately for each case and
bandwidth. Diagonals stay one. Thus graph comparisons at a given quantile have
matched mean coupling; coupling still varies across quantiles. Scaling after
appending profiles can change native edge amplitudes, but not native distances
or bandwidths. Scaled kernel entries are weights and can exceed one.

Energy scores come from the checkpoint-24 1tzw_A R1 archive and are joined by
original trajectory frame identifier. The archive hash is checked against its
scoring manifest. All 100 frames have unique finite matches. These are cached
unrelaxed ref2015 scores; no PyRosetta runtime or new scoring is needed.
Energy similarity does not favour low-energy frames, imply structural similarity,
or convert Rosetta scores into Boltzmann population weights.

Structural and energy graphs are not applicable to the ten external-profile
cases. No coordinates or energies are assigned to synthetic profiles. The
structural graph shares its metric with target clustering and is interpreted
as a geometry benchmark, not independent validation of a physical population prior.

## Optimisation and verification

The new six-arm adapter uses the existing nonlinear objective and exact OMC
regulariser, Adam learning rate 0.05, checkpoints 1000/3000/10000 and the same
uniform/seeded initialisations. Convergence requires both starts to pass the
1% final-window objective criterion and agree in objective within 1%.

The original thirteen-arm baseline replay reproduced its saved results exactly.
The six-arm XLA batch changed rounding near uniform wide-bandwidth solutions:
maximum weight difference 3.37e-6, maximum weight-distribution TV 6.38e-6 and
maximum relative objective difference 4.69e-8. Step counts and convergence labels
were identical. Adapter acceptance bounds weight-distribution TV and maximum
weight error at 1e-5 and objective agreement at rtol 1e-7, atol 1e-11. These
observed errors and explicit tolerances are saved in `adapter_verification.json`.

```bash
uv run --no-sync pytest tests/test_omc_graph_control.py tests/test_omc_decoy_control.py -q
```

The 23 tests cover profile cancellation, rigid-transform invariance, exact energy
joins, energy offset/rescaling invariance, frozen bandwidths, coupling equality,
external-case restrictions, optimiser parity, convergence filtering, empty
comparisons, manifest integrity, resume behaviour, numerical-failure records and
the existing decoy model/report behaviour.

## Completed results: 2026-09-09

All 22 new graph/case jobs and 132 new arms completed. **130 new arms converged**:
84/84 full-profile, 23/24 structural and 23/24 energy. The eight original
unresolved arms remain flagged. The largest mean-coupling mismatch was 7.44e-15.

All-residue-MSE-selected, converged OMC population TV:

| Challenge | Scalar | Full profile | Structure | Energy |
|---|---:|---:|---:|---:|
| Baseline | 0.0250 | 0.0398 | 0.0398 | 0.0312 |
| Remove state 0 | 0.0280 | 0.0479 | 0.0560 | 0.0476 |
| Remove state 1 | 0.0977 | 0.0708 | 0.0596 | 0.0738 |
| Remove state 2 | 0.1078 | 0.0923 | 0.0922 | 0.0904 |
| Donor profiles, median of 5 sets | 0.0226 | 0.0304 | N/A | N/A |
| Random profiles, median of 5 sets | 0.1672 | 0.1726 | N/A | N/A |

The new geometries help state-1 and state-2 removal at their MSE-selected settings,
but degrade baseline and state-0 recovery. Full-profile distances do not resolve
random-profile rejection; median random decoy mass is 0.1726 compared with 0.1672
for the scalar graph. Conditional native ESS is higher (0.8595 versus 0.7231).

Selected-setting comparisons and fixed-bandwidth comparisons answer different
questions. Across valid paired internal arms at the same bandwidth and coupling,
median population-TV differences (new minus scalar) are +0.0013 for full-profile,
+0.0026 for structural and +0.0064 for energy graphs. Thus the selected-case gains
are not evidence of improvement across the whole bandwidth range. The report
retains all valid-pair counts, complete curves, missing comparisons and
nearest-native-ESS diagnostics; it does not pool bandwidths or seeds as independent
biological replicates or declare a universal winning graph.
