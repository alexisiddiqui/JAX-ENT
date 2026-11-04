"""[s1] K1: self-tuning kernel in L2-lnPF space per ensemble; walk covariance + lambda_geo / N_eff; L_st.
Variants: k in {10, 20} x sigma_i = mean | RMS kNN distance x support = symmetric-union kNN | dense; lazy (primary) and
non-lazy walk. MoPrP: lnPF = scaled b' (clamped) per frame (500 x 97); C_walk of the mean predicted uptake (210 cells) by
  'delta' = J C_walk(rate) J^T, J = d uptake / d kbar at uniform weights (rate mode, linearised)  [primary]
  'upt'   = per-frame uptake curves (uptake-mode approximation).
ISO BI/TRI: lnPF = 0.35 Nc + 2.0 Nh (294 res), F = per-frame residue uptake (1470 cells; linear model, exact).
Walk factor g capped at 2N-1 (finite chain; s0.walk_G). ISO also saves the dense primary-k kernel (connected) as sensitivity.
Check: gci2 h2 construction in predicted-uptake space (default BV, dense, mean, k=10/20, c=1, lazy) -> gci2 0.54 / 0.25 (MSAss).
-> s1_kernel.csv, s1_primary.npz (L_st, d, C_walk / C_ens for the primary kernel), s1.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
import s0_common as S0
Q, D = S0.Q, S0.D

rows, SAVE = [], {}
# --- check: gci2 uptake-space construction (default BV 0.35/2.0, uptake mode, dense, mean, c=1, lazy)
for ens in D.ENSS:
    res, h, a, k = D.load(ens)
    Fg = D.C.frame_pep_uptake(D.BC0 * h + D.BH0 * a, k, D.pmap(res), D.TIMES).reshape(500, -1)
    for kk in (10, 20):
        kern = S0.build_kernel(Fg, kk, 'mean', 'dense'); _, _, inf = S0.walk_cov(kern, Fg, True)
        print(f'[check gci2 h2] {ens} k={kk}: lam_geo {inf["lam_geo"]:.4f} N_eff {inf["N_eff"]:.2f}', flush=True)
        rows.append(dict(system='MoPrP', ensemble=ens, space='gci2-uptake(default BV)', k=kk, bw='mean', support='dense', lazy=True,
                         F='upt', **inf, mu2=kern['mu'][1], ncomp=kern['ncomp']))

for ens in D.ENSS:
    m = Q.MoPrP(ens, floor=True); X = S0.moprp_lnpf(ens); Dm = cdist(X, X)
    sig2 = (m.sig ** 2).ravel()
    for kk in (10, 20):
        for bw in ('mean', 'rms'):
            for sup in ('union', 'dense'):
                kern = S0.build_kernel(X, kk, bw, sup, Dm=Dm)
                for lazy in (True, False):
                    for mode in ('delta', 'upt'):
                        Cw, Ce, inf = S0.moprp_cwalk(m, kern, mode, lazy)
                        dr = np.diag(Cw) / sig2
                        r = dict(system='MoPrP', ensemble=ens, space='lnPF(b-prime)', k=kk, bw=bw, support=sup, lazy=lazy, F=mode, **inf,
                                 mu2=kern['mu'][1], mu11=kern['mu'][10], n_mu_gt_0p9=int(np.sum(kern['mu'][1:] > 0.9)),
                                 ncomp=kern['ncomp'], n_edges=kern['n_edges'], sig_med=float(np.median(kern['sig'])),
                                 cw_over_sig2_med=float(np.median(dr)), cw_over_sig2_max=float(dr.max()),
                                 cells_cw_gt_sig2=int(np.sum(dr > 1)), cw_sd_max=float(np.sqrt(np.diag(Cw)).max()))
                        rows.append(r)
                        if (kk, bw, sup) == (S0.PRIMARY['k'], S0.PRIMARY['bw'], 'dense') and lazy and mode == 'upt':
                            SAVE[f'{ens}|Cw|uptD'] = Cw; SAVE[f'{ens}|Ce|uptD'] = Ce; SAVE[f'{ens}|lam_geo|uptD'] = inf['lam_geo']
                        if (kk, bw, sup) == (S0.PRIMARY['k'], S0.PRIMARY['bw'], S0.PRIMARY['support']) and lazy:
                            SAVE[f'{ens}|Cw|{mode}'] = Cw; SAVE[f'{ens}|Ce|{mode}'] = Ce; SAVE[f'{ens}|lam_geo|{mode}'] = inf['lam_geo']
                if (kk, bw, sup) == (S0.PRIMARY['k'], S0.PRIMARY['bw'], S0.PRIMARY['support']):
                    SAVE[f'{ens}|Lst'] = kern['Lst']; SAVE[f'{ens}|d'] = kern['d']; SAVE[f'{ens}|sig'] = kern['sig']
                print(f'{ens} k={kk} {bw} {sup}: ' + ' | '.join(f'{r["F"]}{"" if r["lazy"] else "(nl)"} lam {r["lam_geo"]:.4f} Neff {r["N_eff"]:.1f}'
                                                              for r in rows[-4:]) + f' | ncomp {kern["ncomp"]} mu2 {kern["mu"][1]:.4f}', flush=True)
    # attractor of the pure-OMC term: w* ∝ sqrt(d) (null vector of L_st) for the primary kernel
    w = np.sqrt(SAVE[f'{ens}|d']); w /= w.sum()
    print(f'{ens} L_st attractor w∝sqrt(d): ESS {Q.ess(w):.3f} rec {m.recovery(w):.1f} pops {np.round(100 * m.pops(w), 2)}', flush=True)
    SAVE[f'{ens}|attractor'] = w

for which in ('bi', 'tri'):
    t0 = time.time()
    m = Q.ISO(which, 0.01, 0); X = S0.iso_lnpf(which); Dm = cdist(X, X); N = len(X)
    for kk in (10, 20):
        for bw in ('mean', 'rms'):
            for sup in ('union', 'dense'):
                kern = S0.build_kernel(X, kk, bw, sup, Dm=Dm)
                for lazy in (True, False):
                    Cw, Ce, inf = S0.walk_cov(kern, m.Ff, lazy)
                    rows.append(dict(system=f'ISO_{which.upper()}', ensemble=f'ISO_{which.upper()}', space='lnPF', k=kk, bw=bw, support=sup,
                                     lazy=lazy, F='upt(exact)', **inf, mu2=kern['mu'][1], mu11=kern['mu'][10],
                                     n_mu_gt_0p9=int(np.sum(kern['mu'][1:] > 0.9)), ncomp=kern['ncomp'], n_edges=kern['n_edges'],
                                     n_capped=int(np.sum(S0.walk_G(kern, lazy) >= 2 * len(kern['mu']) - 1)),
                                     sig_med=float(np.median(kern['sig'])), lam_over_5N=inf['lam_geo'] / (5 / N),
                                     cw_sd_max=float(np.sqrt(np.diag(Cw)).max())))
                    if (kk, bw, sup) == (S0.PRIMARY['k'], S0.PRIMARY['bw'], 'dense') and lazy:
                        SAVE[f'ISO_{which}|Cw|dense'] = Cw; SAVE[f'ISO_{which}|lam_geo|dense'] = inf['lam_geo']
                    if (kk, bw, sup) == (S0.PRIMARY['k'], S0.PRIMARY['bw'], S0.PRIMARY['support']) and lazy:
                        SAVE[f'ISO_{which}|Cw'] = Cw; SAVE[f'ISO_{which}|Ce'] = Ce; SAVE[f'ISO_{which}|lam_geo'] = inf['lam_geo']
                print(f'ISO_{which} k={kk} {bw} {sup}: ' + ' | '.join(f'{"lazy" if r["lazy"] else "nonlazy"} lam {r["lam_geo"]:.5f} Neff {r["N_eff"]:.1f}'
                                                                   for r in rows[-2:]) + f' | ncomp {kern["ncomp"]} mu2 {kern["mu"][1]:.4f} 5/N {5 / N:.5f} t={time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(rows); df.to_csv(S0.OUT + 's1_kernel.csv', index=False)
np.savez(S0.OUT + 's1_primary.npz', **SAVE)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
print(df[df.lazy].drop(columns=['space']).round(5).to_string(index=False))
