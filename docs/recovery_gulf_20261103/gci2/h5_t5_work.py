"""T5: work-scale selector vs GCI acceptance.
work_scale (jaxent/examples/common/analysis/scoring.py:100) = R*T*|mean(pred_lnpf) - mean(prior_lnpf)|, R=8.314, T=298 (function
default is 300; ranking unaffected), pred_lnpf = per-residue frame-weighted mean lnPF over the 97 featurised residues
(as in score_models_aSyn_conditions.py: pred_ln_pf per residue), BV 0.35/2.0 (the fitted model; additive corrections cancel).
=> WS = R T |sum_f (w_f - 1/N) lbar_f|/1000 kJ/mol with lbar_f = mean_res lnPF_f.
Pools: (a) path w(t) = (1-t)u + t w_NNLS for corr none and global s*; (b) ex2_fresh MaxEnt candidates under P2 (h0).
GCI Sigma: 'fixed' = D_exp(floored) + lam_geo C_ens (h4, kappa=1); 'prior-floor' = kappa_p D_exp + lam_geo C_ens with kappa_p the
smallest kappa at which the uniform prior passes (h4b; residual-tuned, for contrast only).
-> h5_candidates.csv, h5_path.csv, h5_select.csv, F4_work_vs_recovery.png"""
import sys, json; sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import g0_common as G
RT = 8.314 * 298 / 1000
L4 = json.load(open(G.OUT + 'h4_lams.json')); kap = pd.read_csv(G.OUT + 'h4b_kappa.csv')
h4 = pd.read_csv(G.OUT + 'h4_gci.csv'); SST = h4[h4.s_star].groupby(['ensemble', 'corr']).s.first()
pool = pd.read_csv(G.OUT + 'h0_pool.csv'); Wz = np.load(G.OUT + 'h0_weights.npz')
sD = G.sigma_variants()['pep_rms_floor']; Dd = sD ** 2
crow, prow = [], []
for ens in G.ENSS:
    res, h, a_, k = G.load(ens); lbar = (G.BC0 * h + G.BH0 * a_).mean(0)  # (500,)
    a = G.assign(ens); u = np.ones(500) / 500
    P = pool[pool.ensemble == ens]
    Wm = np.stack([Wz[f'{r}|{i}'] for r, i in zip(P.run_id, P.candidate_index)])
    ws = RT * np.abs((Wm - u) @ lbar)
    for corr in ('none', 'global'):
        s = SST[(ens, corr)]
        F = G.get_F(ens, corr, s); Cm = G.cov_ens(F); Ff = F.reshape(500, -1)
        kp = kap[(kap.ensemble == ens) & (kap['corr'] == corr) & (kap.model == 'prior')].kappa_pass_all.iloc[0]
        if not np.isfinite(kp): kp = kap[(kap.ensemble == ens) & (kap['corr'] == 'global') & (kap.model == 'prior')].kappa_pass_all.iloc[0]
        STs = {'fixed': G.sig_stats(np.diag(Dd) + L4[ens]['lam_geo'] * Cm), 'prior-floor': G.sig_stats(kp * np.diag(Dd) + L4[ens]['lam_geo'] * Cm)}
        R = G.DEXP.ravel()[None] - Wm @ Ff
        for sig, st in STs.items():
            out = [G.triple(r, st) for r in R]
            for j, (idx, row) in enumerate(P.iterrows()):
                d = out[j]
                crow.append(dict(ensemble=ens, split_type=row.split_type, split_idx=row.split_idx, strength=row.strength, step=row.step,
                                 run_id=row.run_id, candidate_index=row.candidate_index, corr=corr, s=s, sigma=sig, kappa=1.0 if sig == 'fixed' else kp,
                                 work_scale=ws[j], rec=row.recovery_percent, ess=row.ess_percent, val_wmse=row.val_weighted_mse,
                                 T=d['T'], S=d['S'], Q=d['Q'], passAll=d['passAll'], nflag=d['nflag']))
            # path
            wN = G.C.nnls_simplex(F, G.DEXP, 1 / sD.reshape(14, 15))
            for t in np.round(np.arange(0, 1.0001, 0.05), 2):
                w = (1 - t) * u + t * wN; d = G.triple(G.DEXP.ravel() - Ff.T @ w, st)
                prow.append(dict(ensemble=ens, corr=corr, s=s, sigma=sig, kappa=1.0 if sig == 'fixed' else kp, t=t,
                                 work_scale=RT * abs((w - u) @ lbar), rec=G.recovery(a, w), ess=100 / np.sum(w ** 2) / 500,
                                 T=d['T'], S=d['S'], Q=d['Q'], passAll=d['passAll'], nflag=d['nflag']))
        print(ens, corr, flush=True)
C_ = pd.DataFrame(crow); C_.to_csv(G.OUT + 'h5_candidates.csv', index=False)
Pp = pd.DataFrame(prow); Pp.to_csv(G.OUT + 'h5_path.csv', index=False)
# selection per replicate
sel = []
for key, g in C_.groupby(['ensemble', 'corr', 'sigma', 'split_type', 'split_idx']):
    b = g.sort_values(['work_scale', 'step']).iloc[0]
    acc = g[g.passAll]
    ba = acc.sort_values(['work_scale', 'step']).iloc[0] if len(acc) else None
    bv = g.sort_values(['val_wmse', 'step']).iloc[0]
    sel.append(dict(zip(['ensemble', 'corr', 'sigma', 'split_type', 'split_idx'], key), n=len(g), n_pass=len(acc),
                    ws_min=b.work_scale, rec_wsmin=b.rec, ess_wsmin=b.ess, strength_wsmin=b.strength, step_wsmin=b.step, wsmin_passes=b.passAll,
                    rec_wsmin_in_accept=ba.rec if ba is not None else np.nan, rec_accept_min=acc.rec.min() if len(acc) else np.nan,
                    rec_accept_max=acc.rec.max() if len(acc) else np.nan, rec_valsel=bv.rec, ws_valsel=bv.work_scale, valsel_passes=bv.passAll,
                    rho_ws_rec=spearmanr(g.work_scale, g.rec).statistic))
S_ = pd.DataFrame(sel); S_.to_csv(G.OUT + 'h5_select.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(S_.round(2).to_string())
agg = S_.groupby(['ensemble', 'corr', 'sigma', 'split_type']).agg(rec_wsmin_mean=('rec_wsmin', 'mean'), rec_wsmin_sd=('rec_wsmin', 'std'),
      ess_wsmin=('ess_wsmin', 'mean'), wsmin_pass=('wsmin_passes', 'sum'), n_pass=('n_pass', 'sum'), n=('n', 'sum'),
      rec_valsel_mean=('rec_valsel', 'mean'), rec_valsel_sd=('rec_valsel', 'std'), rho=('rho_ws_rec', 'mean'))
print(agg.round(2).to_string())
print(Pp.round(3).to_string())
# figure
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
for i, ens in enumerate(G.ENSS):
    for j, sig in enumerate(('fixed', 'prior-floor')):
        ax = axs[j, i]; g = C_[(C_.ensemble == ens) & (C_['corr'] == 'global') & (C_.sigma == sig)]
        for st, mk in (('sequence_cluster', 'o'), ('spatial', 's')):
            x = g[g.split_type == st]
            ax.scatter(x.work_scale, x.rec, s=6, marker=mk, c='lightgray', label=f'{st} (fail)')
            y = x[x.passAll]; ax.scatter(y.work_scale, y.rec, s=10, marker=mk, c='tab:green', label=f'{st} (GCI pass)')
        p = Pp[(Pp.ensemble == ens) & (Pp['corr'] == 'global') & (Pp.sigma == sig)]
        ax.plot(p.work_scale, p.rec, 'k-', lw=1, label='path u->NNLS (global s*)')
        ax.scatter(p[p.passAll].work_scale, p[p.passAll].rec, c='tab:red', s=25, zorder=5, label='path GCI pass')
        ss = S_[(S_.ensemble == ens) & (S_['corr'] == 'global') & (S_.sigma == sig)]
        ax.scatter(ss.ws_min, ss.rec_wsmin, marker='*', s=120, c='tab:blue', edgecolor='k', zorder=6, label='min work-scale per replicate')
        ax.set_xscale('symlog', linthresh=0.01); ax.set_title(f'{ens} | Sigma {sig} (corr global s*)'); ax.set_xlabel('work scale (kJ/mol)')
        ax.set_ylabel('recovery (%)'); ax.axhline(p.rec.iloc[0], ls=':', c='k')
axs[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(G.OUT + 'F4_work_vs_recovery.png', dpi=130)
