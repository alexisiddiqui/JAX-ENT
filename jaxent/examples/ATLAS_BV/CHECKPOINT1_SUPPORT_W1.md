# Checkpoint 1 — support and C-alpha W1 audit

Completed on 2026-08-27 for all 111 systems and all 333 replica holdouts. The command is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-support-audit --workers 4
```

Exact frame-pair W1 is the mean absolute difference between sorted distributions of unique
intraframe C-alpha distances. Diagonal zeros and symmetric duplicates are excluded. The
nearest-frame support search uses 256 inverse-CDF quantiles; its median exact-W1 error is
`0.00396 Å` and its worst observed error is `0.0875 Å`.

For frame-centred PF L2, median interval coverage is `80.7%`, `84.9%`, and `76.8%` in the
hyperlocal (`<1.25 Å`), local (`1.25–2.5 Å`), and global (`>2.5 Å`) RMSD regimes. Global coverage
is associated with structural support: across folds, coverage correlates `+0.51` with the fraction
inside the training target range, `-0.46` with novel-endpoint fraction, and `+0.38` with effective
endpoint frames. Thus tail support matters, but median global effective support is still about 688
frames and scarcity alone does not explain the failure.

W1 is complementary rather than redundant with RMSD: median held-out Spearman correlation is only
`0.324` (IQR `0.214–0.492`). W1 coverage is high in the middle quantiles but fails in both tails;
frame-centred L2 gives `34.7%` in q0 and `20.4%` in q5. The q5 band has a median 1,640 pairs but
only 162 effective frames and 14.1% novel-endpoint pairs. Conversely, q0 retains about 535 effective
frames with negligible novelty, so its failure cannot be attributed to rare-frame support.

Checkpoint decision: pause for review before common-support/extrapolation work.
