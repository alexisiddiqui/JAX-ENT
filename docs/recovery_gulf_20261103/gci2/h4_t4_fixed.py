"""T4: externally fixed Sigma = D_exp + lam*C_ens (+ representation term), kappa = 1, NO residual-based tuning.
D_exp = per-peptide pooled RMS sigma floored at 0.0586 (all cells 0.0586 except 75-79: 0.0703).
lam_geo (h2: lazy, c=1, m=all, default F; geometric mean of k=10,20) x {0.5,1,2}; lam_struct (h2 sandwich, rule labels);
rep variants (h3): +5/N C_ens ('rep_mult', ISO worst-case size multiplier) or + sigma_rep^2 = 0.0067^2 on diag ('rep_floor').
Corrections: none, global, smooth5 (s* = argmin |S_n(prior)| under the lam_geo x1 Sigma, 0.05 grid on [0,1.5], also s=1),
bcbh_all (prev. refit 0.26/0), bcbh_hc90, bcbh_hc85 (h1, wr2). C_ens = uniform-weight covariance of that corrected model.
Models: prior, Folded, NNLS_in (wt 1/sigma_D), NNLS_cv (2-fold contiguous blocks, cross-fitted), path w(t)=(1-t)u+t w_NNLS.
-> h4_gci.csv, h4_path.csv, h4_lams.json"""
import sys, json; sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import numpy as np, pandas as pd
import g0_common as G

h2 = pd.read_csv(G.OUT + 'h2_t2.csv'); st = pd.read_csv(G.OUT + 'h2_struct.csv'); h3 = pd.read_csv(G.OUT + 'h3_t3.csv')
T1P = json.load(open(G.OUT + 'h1_params.json'))
tab = G.corr_table()
for ens in G.ENSS:
    for thr in ('90', '85'):
        tab[f'{ens}|bcbh_hc{thr}'] = np.array(T1P[f'{ens}|{thr}|wr2'])
LAMG = {}
for ens in G.ENSS:
    x = h2[(h2.ensemble == ens) & (h2.params == 'default') & h2.lazy & (h2.c == 1.0) & (h2.m == 'all')]
    LAMG[ens] = dict(lam_geo=float(np.exp(np.log(x.lam_geo).mean())), lam_geo_k10=float(x[x.k == 10].lam_geo.iloc[0]),
                     lam_geo_k20=float(x[x.k == 20].lam_geo.iloc[0]),
                     lam_struct=float(st[(st.ensemble == ens) & (st.labels == 'rule-based')].lam.iloc[0]))
SREP = float(h3[h3.residual == 'true_40_60'].sigma_rep.iloc[0]); NF = 500
json.dump(dict(LAMG, sigma_rep=SREP), open(G.OUT + 'h4_lams.json', 'w'), indent=1)
print(LAMG, SREP)
sD = G.sigma_variants()['pep_rms_floor']; Dd = sD ** 2
CORRS = ['none', 'global', 'smooth5', 'bcbh_all', 'bcbh_hc90', 'bcbh_hc85']
TS = np.round(np.arange(0, 1.0001, 0.05), 2)


def settings(ens):
    lg = LAMG[ens]['lam_geo']
    return {'geo x0.5': (0.5 * lg, 0), 'geo x1': (lg, 0), 'geo x2': (2 * lg, 0), 'struct': (LAMG[ens]['lam_struct'], 0),
            'geo+rep_mult': (lg + 5 / NF, 0), 'geo+rep_floor': (lg, SREP ** 2)}


rows, acc = [], []
for ens in G.ENSS:
    a = G.assign(ens); u = np.ones(NF) / NF
    wF = (a == 0).astype(float); wF /= wF.sum()
    for corr in CORRS:
        if corr in ('global', 'smooth5'):
            lg = LAMG[ens]['lam_geo']; best = None
            for s in np.round(np.arange(0, 1.5001, 0.05), 2):
                F = G.get_F(ens, corr, s); Cm = G.cov_ens(F)
                S = G.gci(G.DEXP.ravel() - F.reshape(NF, -1).mean(0), np.diag(Dd) + lg * Cm)['S']
                if best is None or abs(S) < best[1]: best = (s, abs(S))
            svals = sorted({best[0], 1.0}); sbest = best[0]
        else:
            svals = [1.0]; sbest = 1.0
        for s in svals:
            F = G.get_F(ens, corr, s); Cm = G.cov_ens(F); Ff = F.reshape(NF, -1)
            wt = 1 / sD.reshape(G.NP, G.NT)
            wN = G.C.nnls_simplex(F, G.DEXP, wt)
            wA = G.C.nnls_simplex(F[:, G.FOLD_A], G.DEXP[G.FOLD_A], wt[G.FOLD_A])
            wB = G.C.nnls_simplex(F[:, G.FOLD_B], G.DEXP[G.FOLD_B], wt[G.FOLD_B])
            pcv = np.empty((G.NP, G.NT)); pcv[G.FOLD_B] = np.tensordot(wA, F[:, G.FOLD_B], 1); pcv[G.FOLD_A] = np.tensordot(wB, F[:, G.FOLD_A], 1)
            models = {'prior': (Ff.T @ u, G.recovery(a, u)), 'Folded': (Ff.T @ wF, G.recovery(a, wF)),
                      'NNLS_in': (Ff.T @ wN, G.recovery(a, wN)), 'NNLS_cv': (pcv.ravel(), np.nan)}
            for t in TS:
                w = (1 - t) * u + t * wN; models[f'path_{t:.2f}'] = (Ff.T @ w, G.recovery(a, w))
            for lab, (lam, fl) in settings(ens).items():
                stt = G.sig_stats(np.diag(Dd + fl) + lam * Cm)
                base = dict(ensemble=ens, corr=corr, s=s, s_star=bool(np.isclose(s, sbest)), lam_set=lab, lam=lam, floor=fl,
                            a_n=stt['a_n'], b_n=stt['b_n'], spec=stt['spec'], thrS=stt['thrS'], thrQ=stt['thrQ'], t_n=stt['t_n'])
                block = []
                for m, (pred, rc) in models.items():
                    r = G.DEXP.ravel() - pred; d = G.triple(r, stt); d.pop('z')
                    row = dict(base, model=m, rec=rc, rec_cvA=G.recovery(a, wA) if m == 'NNLS_cv' else np.nan,
                               rec_cvB=G.recovery(a, wB) if m == 'NNLS_cv' else np.nan, mean_resid=r.mean(), code=G.code(d), **d)
                    rows.append(row); block.append(row)
                ok = [x for x in block if x['model'].startswith('path_') and x['passAll']]
                acc.append(dict(ensemble=ens, corr=corr, s=s, lam_set=lab, lam=lam, t_pass=','.join(x['model'][5:] for x in ok) or 'none',
                                rec_min=min((x['rec'] for x in ok), default=np.nan), rec_max=max((x['rec'] for x in ok), default=np.nan),
                                rec_prior=models['prior'][1], rec_nnls=models['NNLS_in'][1], rec_folded=models['Folded'][1],
                                nnls_comp=str(G.C.composition(a, wN)), cvA=G.recovery(a, wA), cvB=G.recovery(a, wB)))
            print(ens, corr, s, flush=True)
df = pd.DataFrame(rows); df.to_csv(G.OUT + 'h4_gci.csv', index=False)
ac = pd.DataFrame(acc); ac.to_csv(G.OUT + 'h4_path.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120); pd.set_option('display.max_rows', 500)
m4 = df[df.model.isin(['prior', 'Folded', 'NNLS_in', 'NNLS_cv'])]
mat = m4.pivot_table(index=['ensemble', 'corr', 's', 'lam_set'], columns='model', values='code', aggfunc='first')[['prior', 'Folded', 'NNLS_in', 'NNLS_cv']]
print(mat.to_string())
print(ac.round(2).to_string())
