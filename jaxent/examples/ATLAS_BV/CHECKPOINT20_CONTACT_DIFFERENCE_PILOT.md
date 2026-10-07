# Checkpoint 20: contact-coordinate difference pilot

This pilot tests whether retaining the stored continuous heavy-contact (`Nc`) and acceptor-contact
(`Nh`) coordinates separately improves the W1-KDE population target over the fixed BV projection
`0.35*Nc + 2.0*Nh`. These are Bradshaw-switch contact coordinates, not raw atom-pair distances.

The experiment reuses checkpoint 19's 12 size-stratified systems and deterministic frame pairs.
Replica 1 fits, replica 2 tunes ridge regularization, and replica 3 tests at KDE neighbour rank 10.
It compares contact L1, L2, cosine, correlation, two-channel ridge, and concatenated per-residue ridge
against Work Scale, legacy Work Density, and fixed-PF cosine/correlation. A separate signed two-channel
diagnostic tests population ordering.

The full-run gate requires at least +3 percentage points median recovery in q4 or q5, at least two
thirds of the systems contributing to that band improving by that amount, and no worsening of median
MAE. Contributor-relative counting is required because only six pilot systems contain global-q5 pairs.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-contact-difference-pilot
```

## Result

No contact-coordinate model passed the full-run gate. At q4, Work Scale retained the best median
recovery (50.5%); the best contact model was per-residue ridge at 44.5%, with a paired median change
of -8.3 points. At q5, contact correlation reached 64.3% versus 56.6% for Work Scale, but only six
systems contributed q5 observations: only 3/6 improved by at least three points, the paired median
gain was 3.8 points, and median MAE worsened by 0.061. This is an unstable distribution-shape gain,
not a transferable tail repair.

Freeing the signed heavy/acceptor coefficients also failed to improve population ordering: fixed
signed BV recovered 53.1% at q5 versus 43.5% for free signed contacts, and was stronger in four of six
bands. The stored contact coordinates therefore do not justify a full six-rotation run. Work Scale
remains the headline model; fixed-PF/contact correlation can remain a tail diagnostic.
