"""T4: joint objective  J(w) = chi2_pool(w) + beta * sum_s (ln p_s(w) - ln p_exp,s)^2 + (1/S) KL(w||u)
chi2_pool = mean over 210 cells of ((dexp - pred)/sigma_pool)^2 under the median-scaled BV model; w = softmax(theta), L-BFGS.
Restraints: family A (rule features) and B70 (Q_seg<0.7), segment set S11 (one row per segment).
beta in {0, 0.1, 1, 10} (+100 for illustration), S in {10, 100, 1000}, modes frame_uptake / rate.
-> d5_t4.csv, d5_weights.npz, F_t4_joint.png, d5.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import d0_common as D, d0b_solvers as V

T = D.seg_table().set_index('seg'); IND = dict(np.load(D.OUT + 'd2_indicators.npz')); TG = D.alt_targets()
BETAS = [0, 0.1, 1, 10, 100]; SS = [0.1, 1, 10, 100, 1000]   # brief: S in {10,100,1000}; 0.1 and 1 added (entropy-relevant regime)
rows, W = [], {}
for ens in D.ENSS:
    asg = D.assign(ens)
    for mode in ['frame_uptake', 'rate']:
        data = V.Data(ens, mode, 'scale')
        for fam in ['A', 'B70']:
            O = np.array([IND[f'{ens}|{s}|{fam}'] for s in D.S11 if not np.isnan(IND[f'{ens}|{s}|{fam}']).all()])
            segs = [s for s in D.S11 if not np.isnan(IND[f'{ens}|{s}|{fam}']).all()]
            t = np.array([T.loc[s, 'p_open'] for s in segs])
            for S in SS:
                x0 = None
                for beta in BETAS:
                    if fam == 'B70' and beta == 0: continue          # identical to A beta=0
                    w, r = V.soft_fit(O, t, S, beta=beta, data=data)   # cold start from uniform (warm starts trap in one-hot softmax saturation)
                    c2, _ = data.chi2_grad(w); P = data.pred(w); sc = D.scores(P); P6 = D.state_pops(asg, w)
                    R = float(np.sum((np.log(O @ w + V.EPS) - np.log(t)) ** 2))
                    feas = O.max(1) > 0
                    Rf = float(np.sqrt(np.mean((np.log(O[feas] @ w + V.EPS) - np.log(t[feas])) ** 2)))
                    row = dict(ensemble=ens, mode=mode, family=fam, S=S, beta=beta, conv=bool(r.success), J=float(r.fun), chi2_pool=c2,
                               mse=sc['mse'], wr2=sc['wr2'], restraint_R=R, rms_logratio_feasible=Rf, ESS=D.ess(w),
                               **{f'pop_{s}': P6[i] for i, s in enumerate(D.STATES)},
                               **{f'rec_{k}': D.recovery_vs(P6, v) for k, v in TG.items()})
                    for s in D.S11:
                        for f2 in ['A', 'B70']:
                            o = IND[f'{ens}|{s}|{f2}']; row[f'{f2}:{s}'] = float(o @ w) if not np.isnan(o).all() else np.nan
                    rows.append(row); W[f'{ens}|{mode}|{fam}|{S}|{beta}'] = w
df = pd.DataFrame(rows); df.to_csv(D.OUT + 'd5_t4.csv', index=False); np.savez(D.OUT + 'd5_weights.npz', **W)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
cols = ['ensemble', 'mode', 'family', 'S', 'beta', 'conv', 'chi2_pool', 'wr2', 'restraint_R', 'rms_logratio_feasible', 'ESS', 'pop_Folded', 'pop_PUF1', 'pop_PUF2',
        'pop_PUF3', 'pop_unfolded', 'pop_PUF2-like', 'rec_ORIG', 'rec_ALT1', 'rec_ALT2']
print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.4g}'))

# figure: recovery (ORIG solid, ALT2 dotted) and chi2 vs beta, per ensemble x mode, lines per S; family A (B70 thin grey)
fig, ax = plt.subplots(3, 4, figsize=(17, 10.5), sharex=True)
cmap = {0.1: '#e7298a', 1: '#66a61e', 10: '#1b9e77', 100: '#d95f02', 1000: '#7570b3'}
xb = lambda b: np.where(np.asarray(b) == 0, 0.01, b)
for j, (ens, mode) in enumerate([(e, m) for e in D.ENSS for m in ['frame_uptake', 'rate']]):
    for S in SS:
        for fam, lw, alpha in [('A', 2.2, 1.0), ('B70', 1.0, 0.6)]:
            x = df[(df.ensemble == ens) & (df['mode'] == mode) & (df.S == S) & ((df.family == fam) | (df.beta == 0))].sort_values('beta')
            ls = '-' if fam == 'A' else '--'
            ax[0, j].plot(xb(x.beta), x.rec_ORIG, ls, color=cmap[S], lw=lw, alpha=alpha, marker='o', ms=3, label=f'S={S} {fam}')
            ax[1, j].plot(xb(x.beta), x.chi2_pool, ls, color=cmap[S], lw=lw, alpha=alpha, marker='o', ms=3)
            ax[2, j].plot(xb(x.beta), 100 * x.ESS, ls, color=cmap[S], lw=lw, alpha=alpha, marker='o', ms=3)
    u = D.recovery(D.assign(ens), np.ones(500) / 500)
    uc = V.Data(ens, mode, 'scale').chi2_grad(np.ones(500) / 500)[0]
    ax[0, j].axhline(u, color='k', lw=0.8, ls=':', label='uniform'); ax[1, j].axhline(uc, color='k', lw=0.8, ls=':')
    ax[0, j].set_title(f'{ens} · {mode}', fontsize=10); ax[2, j].set_xscale('log'); ax[2, j].set_yscale('log'); ax[2, j].set_xlabel('β (restraint weight; β=0 plotted at 0.01)')
    ax[0, j].set_ylim(0, 100)
ax[0, 0].set_ylabel('recovery vs 97.1/2.4/0.5 target (%)'); ax[1, 0].set_ylabel('pooled-σ χ² (mean per cell)'); ax[2, 0].set_ylabel('ESS (% of 500)')
ax[0, 0].legend(fontsize=7, ncol=2, loc='lower right')
fig.suptitle('T4: χ²(scaled BV) + β·Σ(ln p_seg − ln K_op)² + KL/S   (solid = rule-feature restraints A, dashed = Q-contact B70)', fontsize=11)
fig.tight_layout(); fig.savefig(D.OUT + 'F_t4_joint.png', dpi=130)
