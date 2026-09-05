# Checkpoint 25: system alpha versus system variance

## Definition

For every system, the rank-10 MD structural-W1 KDE magnitude target and each unscaled predictor are
constructed on the same deterministic replica-A sample of at most 10,000 frame pairs. The primary
system variance is the variance of that unscaled pairwise predictor. This definition is valid for
both scalar frame scores and the vector-valued legacy Work Density metric.

The fitted coefficient is the same nonnegative least-squares scalar used by the population analyses.
Spearman correlation is measured across systems with a 10,000-resample system bootstrap interval.
Correlation with the variance of the common MD target is reported separately.

OpenMM total energies were rescored for all 111 systems (three replicas and 1,001 frames per
replica) with the validated OpenCL backend. The available-system and matched-OpenMM analyses are
therefore the same complete 111-system cohort.

## Results

| Predictor | Systems | Spearman rho | Bootstrap 95% interval |
|---|---:|---:|---:|
| Work Scale | 111 | -0.86 | [-0.90, -0.79] |
| Work Density, legacy Zq | 111 | -0.85 | [-0.91, -0.76] |
| OpenMM total | 111 | -0.56 | [-0.69, -0.41] |
| PyRosetta ref2015 total | 111 | -0.98 | [-0.99, -0.97] |
| PyRosetta ref2015_cart total | 111 | -0.93 | [-0.95, -0.88] |

With full sampling, OpenMM's inverse alpha-versus-predictor-variance relationship is moderate and
unambiguous rather than the uncertain pilot estimate. Its alpha also correlates positively with MD
target variance (rho=0.56, 95% bootstrap interval [0.40, 0.69]). By contrast, target-variance rho is
0.13 for Work Scale, 0.24 for legacy Work Density, -0.26 for `ref2015`, and -0.04 for
`ref2015_cart`.

Thus descriptor-scale compensation remains the dominant pattern for the thermodynamic and Rosetta
metrics. OpenMM is different: its fitted scale reflects both the inverse spread of its raw energy
differences and the spread of the MD target. This is an association across systems, not evidence
that the fitted vacuum-energy scale is a physical temperature or transferable free energy.

## Command

```bash
OPENMM_WORKERS=4 OPENMM_PLATFORM=OpenCL \
  ./jaxent/examples/ATLAS_BV/commands.sh openmm-score-parallel
./jaxent/examples/ATLAS_BV/commands.sh geometry-alpha-variance --workers 6
```
