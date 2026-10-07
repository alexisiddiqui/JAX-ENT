# Checkpoint 30: prior-relative BV graph Laplacian

## Decision

The original complete-graph Boltzmann consistency loss is retired. An equilibrium MD
trajectory already represents its population through frame multiplicity, so assigning
each frame a new weight proportional to `exp(G)` would replace rather than preserve that
information. The implemented candidate instead regularises the correction to an explicit
input prior:

```text
r_i = log(w_i / p0_i) + constant
L_graph = sum_(i,j) a_ij (r_i - r_j)^2 / sum_(i,j) a_ij
```

The graph is built once from fixed-BV frame coordinates. `pf_l2` uses the complete
per-residue log-PF profile; `work_scale` uses its residue mean. A symmetric-union kNN
graph and self-tuning Gaussian edge weights make the locality assumption explicit.
The MD/Boltzmann ensemble is therefore a useful prior, not a per-frame target.

The loss is zero at `w=p0`, invariant to a common logit shift, and has no gradient into
BV parameters. Disconnected graph components deliberately retain independent population
offsets; the HDX likelihood must identify transfers between them.

## Qualification before ISO validation

The ATLAS qualification has two preregistered stages:

1. **Graph audit.** Replica A fits and replica B selects metric and `k`; replica C tests
   whether BV neighbours are structurally closer and share structural-density labels
   more often than node-relabelled controls. A structural-W1 graph is an oracle, not a
   deployable arm.
2. **Controlled reweighting.** Uniform frame weights are truth. Smooth structural and
   basin-population tilts create biased priors at several ESS levels. HDX observables are
   split by residue/time, then no regularisation, MaxEnt KL, BV Laplacian, structural
   oracle, and rewired-graph controls are compared. Advancement requires improved weight
   recovery for both physical biases, no more than 1% median held-out HDX-MSE regression,
   and a smaller benefit after graph rewiring.

The shuffled bias remains a specificity diagnostic, but is excluded from the rewiring
gate because it has no physical locality for rewiring to destroy. Both requested and
achieved prior ESS are persisted; a binary basin tilt may not be able to attain every
requested ESS exactly.

## Commands and status

Smoke tests (one system, deliberately too few permutations/steps to pass gates):

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-prior \
  --phase audit --limit 1 --permutations 5
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-prior \
  --phase reweight --limit 1 --steps 20 --frame-cap 64 \
  --ess-fractions 0.4 --strengths 0,0.1
```

Run the pilot, and only then the full confirmatory analysis:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-prior --phase all
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-prior --phase all --full
```

## Pilot result (24 systems)

The pilot completed on 2026-09-06 and selected `pf_l2, k=5`. The graph audit passed:
mean held-out structural-density energy gain was 0.646 (95% bootstrap CI 0.589–0.700),
median structural-edge gain was 0.604, and all 24 systems beat their rewired null at
`p <= 0.05`.

The reweighting gate did not pass. Smooth-bias weight-TV recovery improved over MaxEnt
by 0.0429 (95% CI 0.0371–0.0488), with median held-out HDX MSE 46.8% lower. Basin-bias
recovery improved by only 0.00620 and its CI crossed zero (-0.00075–0.0131), although
held-out HDX MSE remained non-inferior. Rewiring reduced the median physical-bias gain
from 0.0224 to 0.00393, so the negative-control gate passed. The structural oracle's
substantially lower errors show that locality can help, but the fixed-BV graph does not
yet encode basin membership reliably enough for promotion.

Therefore the preregistered stop rule applies: **do not run the full analysis or ISO
validation from this candidate.** The runner writes machine-readable Parquet tables and YAML gate reports beneath
`outputs/analysis/pairwise_geometry/laplacian_prior_validation/{pilot,full}/`.
