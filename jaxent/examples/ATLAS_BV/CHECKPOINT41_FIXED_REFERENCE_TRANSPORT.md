# Checkpoint 41: fixed-reference and self-distance transport

## Question

Checkpoint 40 showed that direct PF L1 and Work Opt assignment changed from
negative to positive when source and query were represented by one common
descriptor. That result did not establish *why*. In particular, the common
descriptor used the same feature geometry for both sides, so it could not
distinguish a genuine reference-coordinate correction from a source-copy
control.

This checkpoint asks three separate questions:

1. Are PF L1 or Work Opt implicitly referenced to a replica-specific origin?
2. Can A/B population labels be transported using C's own internal feature
   distances, without using any C population values?
3. Does the nonlinear Work Opt calculation need to be applied after local
   neighbourhood averaging rather than to every frame before averaging?

The third question is essential because Work Opt is nonlinear. In general,

\[
  \operatorname{WorkOpt}\!\left(E[z]\right)
  \ne E\!\left[\operatorname{WorkOpt}(z)\right].
\]

## External reference and invariance control

Each system uses its shared simulation starting structure at frame 0. Its BV
profile is

\[
  z_0 = 0.35\,N_C + 2.0\,N_H.
\]

The frame-0 profile is exactly identical in replicas A, B, and C for all 24
systems. Analysed trajectories start no earlier than frame 101, so the reference
is outside every fitting and evaluation subset.

PF L1 has no hidden ensemble-derived origin: it is simply the L1 distance
between residue profiles. Subtracting a common starting profile must therefore
do nothing:

\[
  \|(x-z_0)-(y-z_0)\|_1 = \|x-y\|_1.
\]

The fixed-origin PF calculation was retained as a negative control. Its distance
matrices were audited before the mathematically identical raw matrices were used
downstream to avoid floating-point tie-breaking differences. The final maximum
difference in alpha, assignment rho, and recovery is exactly zero.

For Work Opt, the existing calculation first centres each frame on that frame's
mean PF and then applies the entropy transformation. The external-reference
arms replace this with either the fixed scalar
\(\mu_0=\operatorname{mean}(z_0)\) or the full residue profile \(z_0\).

Each Work definition is evaluated in both orders:

- **frame then pool:** calculate Work Opt for each frame, then kernel-average;
- **pool then transform:** kernel-average raw PF, then calculate Work Opt for
  the neighbourhood.

## Transfer tests

The checkpoint repeats C1→C2, C2→C1, and pooled A/B→C transfer with 64
neighbourhood landmarks, 100 deterministic label selections, and 2, 4, 8, 16,
or 32 known neighbourhood populations.

Four distance constructions are distinguished:

- `cross`: target descriptors are compared directly with source descriptors;
- `target_self`: alpha is fitted on source self-distances, while target
  predictions use target-to-target landmark distances;
- `source_common`: source geometry is used for both fit and query;
- `joint_common`: a source+target pooled geometry is used for both.

`target_self` and `joint_common` use unlabeled target features, but no target
population value enters fitting. Alpha remains the nonnegative, no-intercept
least-squares fit on known-known population differences. Conclusions use
held-out Spearman rho and MD distribution recovery; MAE is not a decision
metric.

## 24-system result

### Headline assignment result

At 32 known populations:

| transfer | original PF L1 | PF target-self | original Work Opt | Work target-self | dynamic Work, pool then transform | fixed-mean Work, pool then transform |
|---|---:|---:|---:|---:|---:|---:|
| A/B→C | -0.395 | 0.284 | -0.293 | 0.136 | **0.642** | **0.726** |
| C1→C2 | -0.104 | 0.232 | 0.144 | 0.123 | **0.661** | **0.581** |
| C2→C1 | -0.220 | 0.409 | -0.203 | 0.276 | **0.638** | **0.686** |

The fixed-mean calculation clears rho 0.5 in every transfer and reaches 0.726
for A/B→C. However, that fact alone does **not** confirm the fixed-reference
hypothesis, because changing the nonlinear aggregation order already produces
rho 0.638–0.661 without any external reference.

### What caused the rescue?

The paired, system-level 95% bootstrap intervals separate reference choice from
operator order:

| contrast at 32 labels | A/B→C | C1→C2 | C2→C1 |
|---|---:|---:|---:|
| fixed mean minus dynamic, both pool-then-transform | [-0.029, 0.013] | [-0.011, 0.046] | [-0.00002, 0.075] |
| dynamic pool-then-transform minus original frame-then-pool | **[0.408, 1.374]** | **[0.256, 0.711]** | **[0.253, 0.971]** |
| pool minus frame order under fixed mean | **[0.563, 1.317]** | **[0.353, 0.868]** | **[0.368, 1.075]** |

The external reference adds no reliable improvement once operator order is held
fixed: every interval includes zero. In contrast, pool-then-transform has an
entirely positive improvement interval in every transfer, both with dynamic
centering and with the fixed reference.

The full starting-profile reference gives the same qualitative answer. Applying
it frame-first does not rescue assignment; applying it after pooling does. It is
weaker than scalar or dynamic centering, reaching rho 0.312, 0.188, and 0.369.

Therefore the confirmed mechanism is **nonlinear aggregation order**, not a
replica-specific coordinate origin. The transferable object is the Work Opt of
the locally averaged PF profile, not the average of framewise Work Opt values.

### What does target-self tell us?

Using target-internal distances improves raw PF L1 and Work Opt substantially
for A/B→C and C2→C1. The lower confidence bounds for the rho improvements are:

| predictor | A/B→C | C1→C2 | C2→C1 |
|---|---:|---:|---:|
| PF L1 | 0.080 | -0.010 | 0.292 |
| Work Opt | 0.075 | -0.114 | 0.171 |

Thus cross-estimator distance mismatch is real in two transfers, but it is not a
universal explanation: C1→C2 remains ambiguous. More importantly, target-self
alone stays below rho 0.5. It helps, but the Work aggregation-order correction
is much stronger.

### Assignment versus distribution recovery

The improved ordering does not improve every aspect of prediction. For A/B→C,
original Work Opt recovery is 61.7%, while dynamic and fixed-mean
pool-then-transform recover 58.0% and 57.8%. The temporal transfers show similar
four-to-six-point reductions. The new representation is therefore much better
at placing individual neighbourhoods in the correct rank order, but modestly
worse at reproducing the complete distribution of population differences.

This distinction matters practically: use pool-then-transform Work Opt when the
goal is local population assignment or prioritisation, but do not treat its high
rho as proof that absolute population spreads are calibrated.

## Variance-magnitude interpretation

Checkpoint 40 already performed the relevant order for variance magnitude: it
pooled frames, found neighbours in the pooled structural subset, and then
recomputed local variance. It did not average independently computed replica
variances. Its failure to reach strong transfer therefore cannot be attributed
to a compute-then-pool implementation error.

## Conclusion

The proposed fixed-reference mechanism is rejected. It is mathematically
inapplicable to PF L1 and adds no significant Work Opt improvement after
matching aggregation order.

The experiment instead identifies a concrete implementation correction:
**average PF within the structural neighbourhood first, then calculate Work
Opt.** This raises median assignment rho above 0.5 in all three transfers and to
0.642 without any external reference in A/B→C. The remaining recovery loss and
temporal asymmetry mean that this is a useful transferable assignment metric,
not yet a complete population model.

## Run and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-fixed-reference-transport \
  --phase all --workers 4
```

Primary outputs are under
`outputs/analysis/pairwise_geometry/checkpoint41_fixed_reference_transport/`:

- `fixed_reference_learning_curves.png`;
- `fixed_reference_32label_summary.png`;
- `cohort_summary.csv` and `paired_contrasts.csv`;
- `mechanism_decisions.csv` and `reference_audit.parquet`;
- `checkpoint41_report.yaml`.
