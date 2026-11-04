"""T3: HDX consistency of the dG-restrained ensembles under bias-removed BV.
Forward models: 'scale' = median-scaled BV (bc,bh = s*(0.35,2.0), identical to pop/p1 'scale|matched');
                'shift' = BV(0.35,2.0) + delta, delta = median(pfact - mean lnPF) over pfact>0 (-2.46 / -2.79).
Modes: frame_uptake, rate. All 14 peptides x 15 times.  Scores: MSE, mean(w r^2) (moprp.weights), pooled-sigma chi2 (mean).
-> d4_t3.csv (global), d4_t3_pep.csv (per peptide), d4.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import d0_common as D

W = dict(np.load(D.OUT + 'd3_weights.npz'))
SETS = {'uniform': None, 'A_S11_exact': 'A|S11|exact|inf', 'A_S11_soft100': 'A|S11|soft|100', 'A_SN_soft100': 'A|SN|soft|100',
        'B70_S11_soft100': 'B70|S11|soft|100'}
rows, prow = [], []
for ens in D.ENSS:
    res, h, a, k = D.load(ens); M = D.pmap(res); asg = D.assign(ens)
    for model in ['scale', 'shift']:
        L = D.lnpf(ens, model)
        for mode, f in D.PRED.items():
            base = None
            for wn, key in SETS.items():
                w = np.ones(500) / 500 if key is None else W[f'{ens}|{key}']
                P = f(L, k, M, w); sc = D.scores(P); r = D.DEXP - P
                P6 = D.state_pops(asg, w)
                rows.append(dict(ensemble=ens, model=model, mode=mode, weights=wn, ESS=D.ess(w), rec_ORIG=D.recovery(asg, w),
                                 Folded=P6[0], PUF1=P6[1], PUF2=P6[2], unfolded=P6[4], **sc))
                pc = np.mean((r / D.SIG_POOL) ** 2, 1); pm = r.mean(1)
                if key is None: base = (pc, pm)
                for i in range(D.NP):
                    prow.append(dict(ensemble=ens, model=model, mode=mode, weights=wn, peptide=D.LAB[i], chi2_pool=pc[i], mean_res=pm[i],
                                     dchi2_vs_uniform=pc[i] - base[0][i], dmeanres_vs_uniform=pm[i] - base[1][i]))
df = pd.DataFrame(rows); pp = pd.DataFrame(prow)
df.to_csv(D.OUT + 'd4_t3.csv', index=False); pp.to_csv(D.OUT + 'd4_t3_pep.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
print('\nper-peptide pooled chi2 (scale model), uniform vs A_S11_soft100 / B70_S11_soft100 (delta):')
for ens in D.ENSS:
    for mode in D.PRED:
        x = pp[(pp.ensemble == ens) & (pp.model == 'scale') & (pp['mode'] == mode)]
        t = x.pivot(index='peptide', columns='weights', values='chi2_pool').loc[D.LAB]
        mr = x.pivot(index='peptide', columns='weights', values='mean_res').loc[D.LAB]
        out = pd.DataFrame({'uni_chi2': t.uniform, 'A_chi2': t.A_S11_soft100, 'dA': t.A_S11_soft100 - t.uniform,
                            'B70_chi2': t.B70_S11_soft100, 'dB70': t.B70_S11_soft100 - t.uniform,
                            'uni_meanres': mr.uniform, 'A_meanres': mr.A_S11_soft100})
        print(f'\n{ens} {mode}'); print(out.to_string(float_format=lambda v: f'{v:.3f}'))
