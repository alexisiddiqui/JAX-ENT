# TeaA intermediate-structure BV validation

The independent target is exact frame-wise uptake from the reference open/closed
ensembles at 40% open, 60% closed, and 0% intermediate. The ISO_TRI candidate
contains 44 open, 830 closed, and
1351 intermediate frames classified by the established
1-Angstrom RMSD rule. Parameters are calibrated on alternating training peptide
windows and frozen before population recovery. Gamma and Q4 calibration uses
32 balanced frames per state; final predictions and metrics use all frames.

| model | noise_sigma | heldout_uptake_mae | population_max_abs | population_recovery_percent | recovered_open | recovered_closed | recovered_intermediate | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact | 0 | 0.0023091 | 0.0459433 | 84.6736 | 0.36903 | 0.585026 | 0.0459433 | 1 |
| exact | 0.01 | 0.00307498 | 0.103905 | 76.6988 | 0.356334 | 0.555967 | 0.103905 | 1 |
| gamma_moments | 0 | 0.0220102 | 0.0442258 | 96.1257 | 0.355774 | 0.644226 | 0 | 1 |
| gamma_moments | 0.01 | 0.0221165 | 0.0487827 | 95.7211 | 0.351217 | 0.648783 | 3.55356e-16 | 1 |
| gamma_moments_default | 0 | 0.0410253 | 0.085932 | 92.3767 | 0.314068 | 0.685932 | 5.24575e-15 | 1 |
| gamma_moments_default | 0.01 | 0.0410806 | 0.0877992 | 92.2061 | 0.312201 | 0.687799 | 1.38406e-14 | 1 |
| linear_bv | 0 | 0.0111529 | 0.188348 | 68.1592 | 0.336142 | 0.47551 | 0.188348 | 1 |
| linear_bv | 0.01 | 0.0113567 | 0.232525 | 64.311 | 0.313234 | 0.450081 | 0.232525 | 1 |
| linear_bv_default | 0 | 0.0177702 | 0.0405981 | 96.5067 | 0.440598 | 0.559402 | 4.0357e-17 | 1 |
| linear_bv_default | 0.01 | 0.0177709 | 0.0372078 | 92.978 | 0.437208 | 0.562792 | 5.90691e-18 | 1 |
| mixture_q4 | 0 | 0.0226074 | 0.118432 | 77.9873 | 0.43177 | 0.481568 | 0.0866626 | 1 |
| mixture_q4 | 0.01 | 0.0227544 | 0.133485 | 73.9105 | 0.431252 | 0.466515 | 0.123969 | 1 |
| mixture_q4_default | 0 | 0.0611126 | 0.4 | 45.5318 | 4.23273e-16 | 0.834398 | 0.165602 | 1 |
| mixture_q4_default | 0.01 | 0.0611111 | 0.4 | 45.9376 | 0 | 0.845724 | 0.154276 | 1 |
