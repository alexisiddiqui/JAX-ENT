# Checkpoint 19: thermodynamic-combination pilot

This cheap screen asks whether Work Density or Work Shape contains information complementary to Work
Scale before launching another 111-system, six-rotation experiment.

- Twelve systems are selected reproducibly using three systems from each protein-size quartile. No
  recovery result is used in selection.
- Replica 1 fits, replica 2 selects the nonnegative ridge penalty, and replica 3 tests.
- KDE neighbour rank is fixed at 10 and each replica is capped at 10,000 deterministic frame pairs.
- Features are divided by their replica-1 RMS without mean centring; the regression is nonnegative and
  has no intercept, preserving the zero-work/zero-change origin.
- Candidates are Work Scale alone, Scale+legacy density, Scale+normalized density,
  Scale+legacy+Shape, and all five Work metrics.

The gate for a full run is a median held-out recovery improvement of at least three percentage points
overall or in q4--q5, with at least 8/12 systems improving by at least three points. MAE improvement,
coefficient stability, residual complementarity, and feature collinearity are reported as safeguards.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-thermodynamic-combination-pilot
```

## Result

The full-run gate did not pass. Work Scale alone achieved 84.5% median overall recovery in this
size-stratified pilot. Scale+legacy density achieved 78.6%, a median change of -2.7 percentage points;
its median q4--q5 improvement was exactly 0 points, and only 1/12 systems cleared the +3-point gate.
Every larger combination reduced median recovery by 15.6--17.9 points overall and 7.9--11.9 points
in q4--q5.

The median replica-2 residual correlation after fitting Work Scale was 0.016 for legacy density,
0.008 for unnormalized density, 0.002 for normalized density, and 0.028 for Work Shape. Thus the
additional metrics do not show consistent complementary signal. Some combinations slightly improved
MAE while substantially worsening distribution recovery, so that MAE change does not justify a full
distribution-fit run. The recommended decision is to retain Work Scale alone and not launch the full
six-rotation combination experiment.
