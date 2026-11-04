"""T4 diagnostics: (1) flagged cells per peptide for prior / Folded / NNLS_in at s* and lam_geo x1;
(2) smallest diagonal inflation kappa (Sigma = kappa*D_exp + lam_geo*C_ens) at which each model passes all three
(the 'missing model-error variance'), expressed also as an extra homoscedastic sigma_model = sqrt((kappa-1)*0.0586^2).
-> h4b_flags.csv, h4b_kappa.csv"""
import sys, json, re; sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import numpy as np, pandas as pd
import g0_common as G
df = pd.read_csv(G.OUT + 'h4_gci.csv'); L = json.load(open(G.OUT + 'h4_lams.json'))
T1P = json.load(open(G.OUT + 'h1_params.json')); tab = G.corr_table()
for ens in G.ENSS:
    for thr in ('90', '85'): tab[f'{ens}|bcbh_hc{thr}'] = np.array(T1P[f'{ens}|{thr}|wr2'])
sstar = df[df.s_star].groupby(['ensemble', 'corr']).s.first()
fr = []
for (ens, corr), s in sstar.items():
    for m in ('prior', 'Folded', 'NNLS_in', 'NNLS_cv'):
        r = df[(df.ensemble == ens) & (df['corr'] == corr) & (df.s == s) & (df.lam_set == 'geo x1') & (df.model == m)].iloc[0]
        cnt = {}
        for f in (r.flagged if isinstance(r.flagged, str) else '').split(';'):
            if f: p = f.split('@')[0]; cnt[p] = cnt.get(p, 0) + 1
        fr.append(dict(ensemble=ens, corr=corr, s=s, model=m, T=r['T'], S=r.S, Q=r.Q, thrS=r.thrS, thrQ=r.thrQ, nflag=r.nflag,
                       mean_resid=r.mean_resid, by_peptide=' '.join(f'{k}:{v}' for k, v in sorted(cnt.items(), key=lambda x: -x[1]))))
fr = pd.DataFrame(fr); fr.to_csv(G.OUT + 'h4b_flags.csv', index=False)
sD = G.sigma_variants()['pep_rms_floor']; Dd = sD ** 2
KG = np.exp(np.linspace(0, np.log(400), 121))
kr = []
for (ens, corr), s in sstar.items():
    F = G.get_F(ens, corr, s); Cm = G.cov_ens(F); Ff = F.reshape(500, -1); a = G.assign(ens)
    u = np.ones(500) / 500; wF = (a == 0) / np.sum(a == 0); wN = G.C.nnls_simplex(F, G.DEXP, 1 / sD.reshape(14, 15))
    for m, w in (('prior', u), ('Folded', wF), ('NNLS_in', wN)):
        r = G.DEXP.ravel() - Ff.T @ w
        k_all = next((k for k in KG if G.gci(r, k * np.diag(Dd) + L[ens]['lam_geo'] * Cm)['passAll']), np.nan)
        k_T = next((k for k in KG if G.gci(r, k * np.diag(Dd) + L[ens]['lam_geo'] * Cm)['passT']), np.nan)
        kr.append(dict(ensemble=ens, corr=corr, s=s, model=m, rec=G.recovery(a, w), kappa_pass_all=k_all, kappa_pass_T=k_T,
                       sigma_model_all=np.sqrt(max(k_all - 1, 0)) * 0.0586, rms_resid=np.sqrt(np.mean(r ** 2))))
kr = pd.DataFrame(kr); kr.to_csv(G.OUT + 'h4b_kappa.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
print(fr.round(2).to_string()); print(kr.round(3).to_string())
