# BV uptake approximations

JAX-ENT provides three distinct BV uptake paths. They share the physical
protection score

```text
z[r,f] = beta_c C[r,f] + beta_h H[r,f]
k[r,f] = k_int[r] exp(-z[r,f])
```

`beta_c` and `beta_h` are the only condition-level physical contact
coefficients. They are stored as unconstrained raw optimizer coordinates and
mapped through `softplus`, so their physical values remain positive.

## Additive interval-hazard model (`linear_bv`)

This is the inexpensive phenomenological model. Contacts are averaged once,
linearly in the frame weights. For ordered observation times, each interval
adds the positive hazard

```text
h[r,j] = k_int[r] delta_t[j]
         exp(alpha[j] - beta_c mean(C)[r] - beta_h mean(H)[r])
U[r,j] = 1 - exp(-sum(i <= j, h[r,i]))
```

The `alpha[j]` values are condition-level interval offsets shared by all
residues; they are not per-residue or per-frame coefficients. This construction
is monotone over the configured time grid, continuous under its piecewise
constant interval hazard, smooth in every fitted parameter, and bounded in
`[0, 1]`. With all offsets zero it is exactly the single-exponential EX2 curve
computed from the averaged contacts. It is intentionally defined only on its
configured time grid.

## Soft rate mixture (`bv_rate_distribution --backend soft_mixture`)

Learned, ordered log-protection supports form a numerical rate basis. Every
frame is assigned smoothly to those supports from its BV score. Frame weights
remain coupled to uptake through the component masses

```text
a[r,f,q] = softmax_q(-0.5 ((z[r,f] - zeta[q]) / tau)^2)
pi[r,q] = sum_f weight[f] a[r,f,q]
U[r,t] = sum_q pi[r,q] (1 - exp(-t k_int[r] exp(-zeta[q])))
```

Thus changing a frame population changes the uptake curve, while time cost is
`O(residues * timepoints * Q)` rather than `O(residues * timepoints * frames)`.
Supported benchmark sizes are `Q = 2, 4, 8`. Initial supports are empirical BV
score quantiles when features are available. The model config exposes a weak
(`1e-3` by default) anchor penalty for optimization objectives to include.

## Gamma moment closure (`bv_rate_distribution --backend gamma_moments`)

This backend derives the weighted mean and variance directly from current
frame rates. It does not fit Gamma moments or support points:

```text
mu[r] = sum_f weight[f] k[r,f]
v[r]  = sum_f weight[f] (k[r,f] - mu[r])^2
U[r,t] = 1 - (1 + t v[r] / mu[r])^(-mu[r]^2 / v[r])
```

The implementation has a stable zero-variance limit equal to
`1 - exp(-mu t)`.

## Validation

Unit validation covers bounds, monotonicity, unit conversion, the EX2 and
zero-variance limits, frame-population coupling, support ordering for
`Q = 2, 4, 8`, specialized simulation dispatch, HDF round trips, JIT, and
finite gradients.

The scientific benchmark is implemented in
`jaxent/examples/ATLAS_BV/analysis/bv_uptake_model_validation.py`. It regenerates
exact frame-wise TeaA targets at six known open/closed populations and evaluates
a frozen 12-system ATLAS subset spanning length and RMSF terciles. ATLAS uses a
known 60/30/10 mixture of its three replicas. Fits use alternating peptide-like
residue windows for training and held-out testing, both without noise and with
uptake noise of 0.01 over five seeds. Each model is calibrated once per system
against an exact frame-wise target on training windows, then frozen for population
recovery. TeaA uses a 50/50 calibration mixture. An exact oracle distinguishes
approximation bias from population non-identifiability.

On the current data, `linear_bv` passes both the median uptake-error and
population-recovery gates on TeaA and ATLAS. Median held-out uptake MAE is 0.0094
for TeaA and 0.0193 for ATLAS; median maximum population error is 0.030 and 0.017,
respectively. Eleven of twelve ATLAS systems individually meet the uptake MAE
threshold, so the aggregate result should not be read as universal accuracy.

The same calibration protocol was repeated for the Gamma-moment and default
Q=4 soft-mixture models, with fitted and unfitted baselines retained. Fitting Q=4
improves median noiseless population recovery from 54.1% to 94.8% on TeaA and
from 32.5% to 74.3% on ATLAS, but it does not meet the uptake or recovery gates.
Fitting Gamma gives 88.5% recovery on TeaA and 32.3% on ATLAS; the combined
ATLAS calibration/recovery pipeline converges in only 6 of 12 cases. These
backends therefore remain experimental rather than validated alternatives to
the additive linear model.
The full generated report and machine-readable outputs are under
`jaxent/examples/ATLAS_BV/outputs/analysis/bv_uptake_validation/`.

MoPrP remains useful as an exploratory real-data check, but is not sufficient
as the sole validation dataset because its ground-truth populations are unknown.
