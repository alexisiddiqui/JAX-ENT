# Checkpoint 10 — opening-probability and naive-distance screen

Completed on 2026-08-28 for 111 systems and all six ordered A-fit/B-calibrate/C-test assignments.
The screen converts fixed-BV log protection factors with `p_open = expit(-logPF)` and compares raw,
frame-centred, A-only residue-z-scored, and combined profiles. Scalar candidates include mean L1,
RMS L2, cosine, correlation, mean residue-wise Bernoulli sqrt(JSD), and symmetric Bernoulli KLD
(Jeffreys). Vector candidates use ridge on absolute per-residue opening-probability changes.

Candidate selection uses endpoint-frame-disjoint validation within A. B supplies conformal
calibration and C supplies every reported result. The adaptive arm selects independently for each
system, target and ordered assignment. The test set therefore does not choose the representation.

## Primary result

No opening-probability arm improves either log-PF baseline in common-support W1 q5:

| Model | Recovery | Mondrian coverage | Interval score |
|---|---:|---:|---:|
| Raw log-PF vector ridge | **51.57%** | **68.53%** | **0.905** |
| Raw log-PF Absolute-L1 | 44.02% | 53.97% | 1.331 |
| Best opening scalar: centred/z-scored L1 | 40.63% | 27.62% | 3.759 |
| Opening Bernoulli sqrt(JSD) | 40.56% | 24.51% | 2.232 |
| Opening Bernoulli symmetric KLD | 40.54% | 24.29% | 2.254 |
| A-selected opening pipeline | 40.54% | 25.20% | 2.461 |

The adaptive pipeline loses `11.21` recovery percentage points against raw log-PF vector ridge
(95% system-bootstrap CI `-12.90` to `-8.37`). Every opening scalar is also worse than raw log-PF
Absolute-L1 after familywise correction; the least-negative median effect is `-2.83` points.

The most frequently A-selected W1 arm is centred/z-scored correlation (290/666 W1 assignments), but
it transfers poorly to C. This is direct evidence that within-replica feature selection is not a
substitute for cross-replica information. All 1,332 target-specific adaptive assignments used a
valid endpoint-frame-disjoint split; the prespecified no-split fallback was not required in the
final run.

## Interpretation and decision

The nonlinear conversion to opening probability strongly compresses the fixed-BV profiles: most
residues have very small opening probabilities, so geometrically distinct log-PF changes become
nearly indistinguishable in probability space. Mean centring, A-only z-scoring, cosine/correlation,
and Bernoulli divergences do not restore the lost magnitude information.

The Stage 2 gate fails. Do not promote opening probability as a replacement for log-PF geometry.
Proceed to Stage 3, but retain raw log-PF per-residue ridge as the primary conditional-likelihood
baseline. The prespecified raw `|Delta p_open|` arm remains as the physically interpretable opening
comparison; it is not expected to win based on this screen.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-screen --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-screen-report
```

Outputs are under `outputs/analysis/pairwise_geometry/checkpoint10_opening_screen/`.
