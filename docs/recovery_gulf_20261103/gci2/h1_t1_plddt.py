"""T1 (secondary): bc/bh refit restricted to high-pLDDT cells (CA B-factor of MoPrP_max_plddt_4334.pdb).
Cell = (peptide, t) is hi-conf if all featurised trimmed residues (s+2..e) have pLDDT>=thr; weighted variant uses
fraction-in-mask as cell weight. Fit at uniform frame weights, objectives mean(W r^2) and MSE.
GCI here uses the previous-pass nominal-style Sigma = D_pep_rms + C_ens/500 (kappa=1) only for S_n/T_n flags.
-> h1_t1.csv, h1_params.json"""
import sys, json
import numpy as np, pandas as pd, MDAnalysis as mda
from scipy.optimize import minimize
sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import g0_common as G
OUT = '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2/'
u = mda.Universe('jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb'); ca = u.select_atoms('name CA')
PL = dict(zip(ca.resids.astype(int), ca.tempfactors))
SV = G.sigma_variants()
rows, params = [], {}
for ens in G.ENSS:
    res, h, a, k = G.load(ens); M = G.pmap(res)
    for thr in (85, 90, 95):
        frac = np.array([np.mean([PL[int(r)] >= thr for r in res[M[i] > 0]]) for i in range(G.NP)])
        full = frac == 1
        variant = 'strict' if full.sum() >= 4 else 'weighted'
        cw = (full.astype(float) if variant == 'strict' else frac)[:, None] * np.ones((1, G.NT))
        for obj, W in (('wr2', G.WTS), ('mse', np.ones_like(G.WTS))):
            def f(p):
                F = G.frame_uptake(p[0] * h + p[1] * a, k, M)
                return np.sum(cw * W * (G.DEXP - F.mean(0)) ** 2) / np.sum(cw)
            best = None
            for x0 in [(0.35, 2.0), (0.2, 1.0), (0.1, 0.5), (0.3, 0.0), (0.0, 1.5), (0.5, 3.0)]:
                r = minimize(f, x0, method='L-BFGS-B', bounds=[(0, 2), (0, 10)])
                if best is None or r.fun < best.fun: best = r
            bc, bh = best.x
            F = G.frame_uptake(bc * h + bh * a, k, M); Cm = G.cov_ens(F)
            r = (G.DEXP - F.mean(0)).ravel()
            Sig = np.diag(SV['pep_rms'] ** 2) + Cm / 500
            hi = np.repeat(full, G.NT)
            row = dict(ensemble=ens, thr=thr, variant=variant, n_hi_pep=int(full.sum()),
                       hi_peps=';'.join(np.array(G.LAB)[full]), objective=obj, bc=bc, bh=bh, bh_at_zero=bh < 1e-6,
                       obj_fit=best.fun, obj_default=f((G.BC0, G.BH0)),
                       mean_dlnPF=float(np.mean((bc - G.BC0) * h.mean(1) + (bh - G.BH0) * a.mean(1))),
                       mean_resid_all=r.mean())
            g = G.gci(r, Sig); row.update(S_all=g['S'], T_all=g['T'], nflag_all=g['nflag'], flags_all=G.flagged(g['z'], g['t_n']))
            if hi.sum():
                g2 = G.gci(r[hi], Sig[np.ix_(hi, hi)])
                row.update(mean_resid_hi=r[hi].mean(), S_hi=g2['S'], T_hi=g2['T'], nflag_hi=g2['nflag'], thrS_hi=g2['thrS'])
            for i, lab in enumerate(G.LAB): row[f'r_{lab}'] = r[i * G.NT:(i + 1) * G.NT].mean()
            rows.append(row); params[f'{ens}|{thr}|{obj}'] = [bc, bh]
            print(ens, thr, variant, obj, round(bc, 4), round(bh, 4), flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT + 'h1_t1.csv', index=False)
json.dump(params, open(OUT + 'h1_params.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_colwidth', 80)
print(df.drop(columns=[c for c in df if c.startswith('r_') or c == 'flags_all']).round(4).to_string())
print(df[['ensemble', 'thr', 'objective'] + [c for c in df if c.startswith('r_')]].round(3).to_string())
print(df[['ensemble', 'thr', 'objective', 'flags_all']].to_string())
