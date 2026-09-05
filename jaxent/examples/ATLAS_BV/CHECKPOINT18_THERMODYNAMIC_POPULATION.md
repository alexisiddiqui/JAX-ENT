# Checkpoint 18: thermodynamic BV population metrics

This is a third, magnitude-only arm of the checkpoint-17 W1-KDE experiment. Each MD frame supplies
an aligned fixed-BV log-PF residue profile. Frame pairs are assigned the notebook Work Shape
(`delta H_opt / RT`), Work Scale (`delta H_abs / RT`), or Work Density (`-T delta S_opt / RT`)
metric, and these are compared with the target magnitude `abs(log rho_i - log rho_j)`.

Work Density is deliberately evaluated three ways:

- notebook legacy: `Pi = Z*q`;
- no normalization: `Pi = q`;
- conventional normalization: `p = q/Z`.

Here `q = exp(-abs(logPF - mean(logPF)))` and `Z = sum(q)`. Only `q/Z` is a normalized probability.
All predictors are stored as dimensionless `W/(RT)` values so alpha=1 is a Boltzmann-scale diagnostic.
Energy summaries use the notebook values `R=8.31 J mol-1 K-1` and `T=300 K`, giving
`RT=2.493 kJ/mol`.

Each scalar is evaluated at alpha=1, with a nonnegative replica-A per-system alpha, and with an
equal-system-weighted leave-one-system-out alpha. Replica B calibrates 90% conformal intervals and
replica C is untouched until evaluation; all six ordered assignments are rotated.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-thermodynamic-population --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-thermodynamic-population-report
```

The headline output is `outputs/analysis/pairwise_geometry/checkpoint18_thermodynamic_population/
thermodynamic_recovery_global_w1.png`. Recovery is `100*(1-sqrt(JSD))`; angstroms occur only in the
global structural W1/RMSD band definitions.

## Completed result (111 systems)

Work Scale is the strongest magnitude predictor. With a per-system alpha it recovers 87.7%, 85.9%,
78.2%, 69.1%, 53.6%, and 42.6% across global W1 bands q0--q5 (69.5% mean). Its leave-one-system-out
version is nearly as strong at 68.2% mean recovery, indicating that its scale largely transfers.
Algebraically, this metric is the absolute value of the signed mean-BV coordinate from checkpoint 17;
the new result tests population-change magnitude rather than signed ordering.

The notebook legacy Work Density (`Pi=Zq`) is second strongest at 60.2% mean local-alpha recovery,
but loses more under leave-one-system-out scaling (56.3%). The unnormalized `q` and conventional
normalized `q/Z` definitions recover only 41.3% and 38.6%, respectively. The legacy result must not be
called a Shannon/Gibbs entropy: its apparent advantage comes from the non-normalized transformation,
which introduces profile- and residue-count-dependent amplitude.

Work Shape averages 35.9%, close to absolute L1/L2 and below cosine/correlation. Bandwidth dependence
is negligible for Work Scale (69.4--69.5% mean across neighbour ranks 5--50) and modest for the other
metrics. Strict nominal-90% coverage averages 88.1% for Work Scale and 85.8--86.8% for the remaining
local-alpha thermodynamic metrics.
