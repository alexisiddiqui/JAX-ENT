# Checkpoint 24: PyRosetta energy-score pilot

## Protocol

The existing `neuralplexer_dev` Python 3.10 environment reads the ATLAS PDB/XTC files and loads the
local PyRosetta 2026.15 build. No environment is installed or modified. The scorer evaluates raw,
periodically unwrapped trajectory coordinates without repacking, relaxation, or minimization using
both `ref2015` and `ref2015_cart`.

PDB atoms are mapped one-to-one to real Rosetta Pose atoms using explicit GROMACS/Rosetta naming
aliases and a residue- and element-constrained coordinate fallback. Rosetta virtual atoms are not
assigned trajectory coordinates. Every nonzero weighted score term and the total score are cached
per frame in Rosetta Energy Units (REU).

REU are not kJ/mol. The primary population comparison therefore fits a nonnegative scale to the
absolute score difference on replica A. Raw signed and absolute score differences are retained only
as nonphysical unit-scale controls; they are not interpreted as `-Delta E/(RT)`.

## Population comparison

The target, sampling, structural clustering, global structural-W1 bands, and A-fit/C-test split are
identical to checkpoints 22 and 23. Results include total-score and score-family recovery, matched
Work Scale and OpenMM controls, and paired q4/q5 bootstrap comparisons.

## Pilot result

All 111 systems passed the topology and score-function audit. Five systems contain doubly
protonated histidines; standard Rosetta neutral HIS lacks one of those imidazole protons, which is
explicitly recorded and ignored while all heavy atoms remain mapped. The 12-system pilot contains
36 trajectories and 36,036 scored frames. Weighted terms reconstruct both totals exactly.

For the five systems represented in the global q5 tail, median recovery was 62.4% for fitted
`ref2015`, 69.0% for fitted `ref2015_cart`, 59.2% for fitted OpenMM total energy, and 52.8% for Work
Scale. The `ref2015_cart` Cartesian-bonded component reached 70.0%. Between structural clusters,
q5 recovery was 63.7%, 69.0%, 58.6%, and 50.4%, respectively; Cartesian-bonded reached 72.4%.

The paired q5 improvement of `ref2015_cart` over Work Scale was 12.9 percentage points globally
and 16.2 points between clusters. Only five systems contribute to q5, so this is promising pilot
evidence rather than a cohort-level ranking. Raw unit-scale REU differences recovered only about
18–29%, confirming that a fitted scale is necessary and must not be interpreted as a physical
Boltzmann factor.

## Full-cohort result

All 111 systems and 333 trajectories were scored, giving 333,333 validated frames. Eligibility
requirements leave 23 systems in the global q5 band and 21 in the between-cluster q5 comparison.

Across global q5 pairs, median recovery is 59.2% for Work Scale, 61.9% for `ref2015`, 61.2% for
`ref2015_cart`, and 61.6% for the Cartesian torsional/rotamer component. Between clusters the
corresponding recoveries are 53.7%, 61.9%, 63.3%, and 56.8%. Standard `ref2015` total and its
torsional/rotamer component remain nearly indistinguishable.

The apparent q5 median advantage is not robust in paired system differences. Against Work Scale,
the global q5 paired median is +0.7 percentage points for `ref2015` and +0.4 points for
`ref2015_cart`; both bootstrap confidence intervals cross zero. Between clusters, the paired
medians are +5.6 and +2.2 points, again with intervals crossing zero. Work Scale remains clearly
stronger from q0 through q2 and is comparable or better through q4.

Thus, the 12-system Cartesian-bonded improvement does not generalize as a decisive full-cohort
gain. Rosetta provides useful complementary tail signal, particularly between clusters, but it
does not replace Work Scale as the most stable overall predictor.

## Commands

```bash
./jaxent/examples/ATLAS_BV/commands.sh pyrosetta-audit
PYROSETTA_WORKERS=6 ./jaxent/examples/ATLAS_BV/commands.sh pyrosetta-score-parallel
./jaxent/examples/ATLAS_BV/commands.sh geometry-pyrosetta-energy
```

The full single-assignment cohort uses the same resumable scorer and analysis:

```bash
PYROSETTA_WORKERS=12 ./jaxent/examples/ATLAS_BV/commands.sh pyrosetta-score-parallel --full
./jaxent/examples/ATLAS_BV/commands.sh geometry-pyrosetta-energy --full
```
