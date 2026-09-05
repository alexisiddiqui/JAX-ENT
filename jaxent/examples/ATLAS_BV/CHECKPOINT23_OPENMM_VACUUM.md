# Checkpoint 23: OpenMM protein-vacuum energy control

## Protocol

The protein-only ATLAS PDB/XTC coordinates are rescored without minimization using OpenMM 8.5.2,
`charmm36.xml`, `NoCutoff`, dielectric 1, no solvent, no implicit solvent, and no constraints. The
available TPR files contain the full solvated system, so this is an independent CHARMM36-family
control rather than an exact replay of the embedded GROMACS CHARMM36m system.

The disposable environment is defined by `openmm_vacuum_environment.yml` and created under
`/tmp/jaxent-openmm-8.5.2`. OpenMM CUDA could not load because the solver selected CUDA 13.3 PTX
while the installed driver supports CUDA 13.0. The validated OpenCL backend was therefore used.
The machine has only the NVIDIA OpenCL ICD, so OpenCL execution is on the RTX 3090.

All 111 systems passed atom-order, TPR-protein-order, frame-count, and OpenMM template-assignment
checks. The 12-system pilot scored all 36 trajectories (36,036 frames) at a mean 110 frames/s.
Bond-graph periodic unwrapping retained maximum bond lengths below 2.18 A. Force-group energies sum
to total energy within `5.9e-11 kJ/mol`.

## Population comparison

The target remains the rank-10 MD structural-W1 KDE log-density difference. Replica A fits the
non-negative scale for absolute energy differences and replica C is evaluated. Direct physical
predictions use `-Delta E/(RT)` at 300 K without fitting or exponentiation. The same global W1 bands
and checkpoint-22 structural clusters are retained.

Across all eligible pilot pairs, q5 median recovery is:

| Predictor | Recovery | Systems |
|---|---:|---:|
| Work Scale | 52.8% | 5 |
| OpenMM total, fitted absolute difference | 59.2% | 5 |
| OpenMM nonbonded, fitted absolute difference | 58.1% | 5 |
| OpenMM torsional, fitted absolute difference | 61.7% | 5 |
| OpenMM total, direct signed Boltzmann | 24.2% | 5 |
| OpenMM total, direct Boltzmann magnitude | 19.1% | 5 |

Vacuum energy therefore contains useful empirical tail information once its scale is fitted, but
its physical magnitude is incompatible with the MD target. The fitted total-energy scale
corresponds to a median effective temperature of about 108,000 K rather than 300 K. Vacuum
potential energy must be treated as a rescaled structural descriptor, not a calibrated free energy.

The q5 pilot contains only five systems globally and two within clusters. These results justify
considering a full-cohort score but do not establish the relative ranking of total, torsional, and
nonbonded energy.

## Commands

```bash
./jaxent/examples/ATLAS_BV/commands.sh openmm-env
./jaxent/examples/ATLAS_BV/commands.sh openmm-audit
./jaxent/examples/ATLAS_BV/commands.sh openmm-score --pilot --platform OpenCL
./jaxent/examples/ATLAS_BV/commands.sh geometry-openmm-energy
```
