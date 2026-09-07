# Checkpoint 33: residue-scaled Work Scale kNN

## Stage 1: scaling grid and graph smoke test

This checkpoint tests whether the Work Scale neighbourhood size should increase with
protein length. It is separate from the locked checkpoint-32 analysis. The scaling rule is

```text
k(N) = clip(round_half_up(k_ref * f(N) / f(109)), 3, min(160, n_frames - 1))
```

with constant, square-root, linear and `N log N` families and
`k_ref = 10, 20, 40, 80, 120`. Both self-tuned weighted and uniform-edge kNN graphs are
included so that neighbourhood scaling can be separated from Gaussian edge weighting.

Stage 1 only validates the candidate definitions and graph construction. It deliberately
does not select a rule or run the 24-system reweighting study. The plot and candidate table
always show the planned 256-frame experiment; `--frame-cap` only reduces the smoke graph audit.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-residue-scaling \
  --phase prepare --smoke --limit 1 --workers 2 --frame-cap 32
```

## Stage 2: development selection

All 40 candidates are screened on replica A at 192 frames and 100 steps. The eight selected
family/topology finalists are fitted on replica B at 256 frames and 500 steps. Replica B
selects a family with a one-standard-error preference for the slower scaling rule; replica C
evaluates only the locked candidate, the best constant candidate, and checkpoint 32's Work
Scale all-pairs RBF 16% graph. This staged evaluation avoids padding all 40 full-resolution
graphs to the largest candidate's edge count.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-residue-scaling \
  --phase develop --workers 6 --frame-cap 256 --steps 500
```

## Stage 2 result

Replica-B connectivity auditing moved every family to `k_ref=20`. All eight finalists were
connected. The one-standard-error rule selected weighted constant `k=20`; square-root,
linear and `N log N` scaling were not supported over a constant neighbourhood.

On replica C, constant `k=20` retained positive smooth recovery in all size quartiles. Basin
recovery was unresolved in Q3 and Q4; Q4 mean gain was 0.00217 with a 95% interval from
-0.000014 to 0.00653, and only 2/6 Q4 systems improved. It was slightly worse than the
all-pairs RBF graph in Q4 (paired mean difference -0.00112, 95% interval -0.00321 to
-0.000004). The development gate therefore failed. The 87-system exploratory diagnostic has
not been run.

The subsequent no-refit analysis is recorded in
[`CHECKPOINT34_BASIN_DIFFICULTY.md`](CHECKPOINT34_BASIN_DIFFICULTY.md). It retains the locked
all-pairs RBF graph and shows that MaxEnt difficulty explains a substantial part, but not
all, of the apparent protein-length trend.
