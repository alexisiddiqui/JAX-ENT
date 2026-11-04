"""[s3] T1 selection: rules A / B / D (+ D|ESS>=50, refs p0 / oracle) on the s2 sweep, per ensemble x F-mode x lam x version
(diag: flags + F_diag; full: flags (marginal diag z, Sidak) + F_full = 1/2 r^T Sigma^-1 r + KL). Candidates ESS >= 10% U {S=0}.
Checks lam = 0 reproduces selgci q3 (floored) A / B / D picks. Flag bookkeeping: flags at p0 and at every pick under lam vs
lam = 0 (same w); disappeared cells vs C_walk,ii / sigma_ii^2.
-> s3_selected.csv, s3_summary.csv, s3_flags_p0.csv (per fold x peptide), s3.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s0_common as S0
Q, D = S0.Q, S0.D

df = pd.read_csv(S0.OUT + 's2_sweep.csv'); FL = dict(np.load(S0.OUT + 's2_flags.npz'))
PR = dict(np.load(S0.OUT + 's1_primary.npz'))
RULES = ['A_flags', 'B_F', 'D_flags_then_F', 'D_ess50', 'ref_p0', 'ref_oracle']
LKEYS = ['0', '1/N', 'lg/3', 'lg', '3lg']

sel = []
for (ens, md, lk, split, held), g in df.groupby(['ensemble', 'mode', 'lam_key', 'split', 'heldout'], sort=False):
    for ver, fcol in (('diag', 'F_diag'), ('full', 'F_full')):
        if lk == '0' and ver == 'full':
            pass
        for r in S0.select(g, fcol):
            r.update(version=ver, F=r[fcol])
            f0 = FL[f'{ens}|{split}|{held}|{r["S"]:.6g}|{md}|0']
            r['n_flags_lam0_same_w'] = len(f0)
            sel.append(r)
S = pd.DataFrame(sel)

# --- reproduction check vs selgci q3 (floored)
q3 = pd.read_csv(os.path.join(S0.RG, 'selgci', 'q3_selected.csv'))
q3 = q3[(q3.system == 'MoPrP') & (q3.sigma == 'floor') & q3.rule.isin(['A_flags', 'B_F', 'D_flags_then_F', 'ref_oracle'])]
mine = S[(S['mode'] == 'upt') & (S.lam_key == '0') & (S.version == 'diag')]
mg = q3.merge(mine, on=['ensemble', 'split', 'heldout', 'rule'], suffixes=('_q3', ''))
print(f'[check] lam=0 vs selgci q3: {len(mg)} picks, S identical {np.isclose(mg.S_q3, mg.S).sum()}/{len(mg)}, '
      f'max |drec| {np.abs(mg.rec_q3 - mg.rec).max():.2e}, flags identical {(mg.ho_n_flags == mg.n_flags).sum()}/{len(mg)}', flush=True)

popc = [c for c in S.columns if c.startswith('pop_')]
keys = ['ensemble', 'mode', 'lam_key', 'version', 'rule']
grp = S.groupby(keys, sort=False)
summ = grp.agg(lam=('lam', 'first'), S_med=('S', 'median'), S_min=('S', 'min'), S_max=('S', 'max'), ESS_mean=('ESS', 'mean'), ESS_sd=('ESS', 'std'),
               rec_mean=('rec', 'mean'), rec_sd=('rec', 'std'), rec_min=('rec', 'min'), flags_mean=('n_flags', 'mean'),
               flags_lam0_mean=('n_flags_lam0_same_w', 'mean'), **{c: (c, 'mean') for c in popc}).reset_index()
summ['n_ESS_gt50'] = grp.ESS.apply(lambda x: int((x > 0.5).sum())).values
summ['n_rec_ge95'] = grp.rec.apply(lambda x: int((x >= 95).sum())).values
summ['n_p0'] = grp.S.apply(lambda x: int((x == 0).sum())).values
# folds whose pick differs from the lam=0 pick (same ensemble / mode / rule)
base = S[S.lam_key == '0'].set_index(['ensemble', 'mode', 'version', 'rule', 'split', 'heldout']).S
S['S_lam0'] = [base.get((r.ensemble, r['mode'], r.version, r.rule, r.split, r.heldout), np.nan) for _, r in S.iterrows()]
summ['n_changed_vs_lam0'] = S.groupby(keys, sort=False).apply(lambda x: int((~np.isclose(x.S, x.S_lam0)).sum())).values
S.to_csv(S0.OUT + 's3_selected.csv', index=False); summ.to_csv(S0.OUT + 's3_summary.csv', index=False)

# --- flags at p0: which cells disappear when C_walk enters, and are they the uncertain cells?
frows = []
for ens in D.ENSS:
    m = Q.MoPrP(ens, floor=True); sig2 = (m.sig ** 2).ravel()
    for md in ('upt', 'delta', 'uptD'):
        lg = float(PR[f'{ens}|lam_geo|{md}']); cw = np.diag(PR[f'{ens}|Cw|{md}']); ratio = cw / sig2   # lam_geo C_shape,ii / sigma^2
        for split, held, tr, va in Q.moprp_splits():
            cells = np.load(S0.OUT + f'_cells_{ens}_{split}_{held.replace(" ", "")}.npy')
            f0 = set(cells[FL[f'{ens}|{split}|{held}|0|{md}|0']])
            for lk in LKEYS[1:]:
                lam = df[(df.ensemble == ens) & (df['mode'] == md) & (df.lam_key == lk)].lam.iloc[0]
                f1 = set(cells[FL[f'{ens}|{split}|{held}|0|{md}|{lk}']])
                gone = sorted(f0 - f1); kept = sorted(f0 & f1)
                rr = (lam / lg) * ratio
                for p in sorted(set(c // m.NT for c in f0)):
                    frows.append(dict(ensemble=ens, mode=md, lam_key=lk, split=split, heldout=held, peptide=D.LAB[p],
                                      flags_lam0=sum(1 for c in f0 if c // m.NT == p), flags_lam=sum(1 for c in f1 if c // m.NT == p),
                                      gone=sum(1 for c in gone if c // m.NT == p),
                                      ratio_gone_med=float(np.median([rr[c] for c in gone if c // m.NT == p])) if any(c // m.NT == p for c in gone) else np.nan,
                                      ratio_kept_med=float(np.median([rr[c] for c in kept if c // m.NT == p])) if any(c // m.NT == p for c in kept) else np.nan))
FP = pd.DataFrame(frows); FP.to_csv(S0.OUT + 's3_flags_p0.csv', index=False)

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
show = summ[summ.rule.isin(['A_flags', 'B_F', 'D_flags_then_F', 'D_ess50', 'ref_oracle'])]
print(show.drop(columns=['S_min', 'S_max', 'pop_PUF3', 'pop_unfolded', 'pop_PUF2-like']).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
print('\n[flags at p0, primary upt] totals per ensemble x lam (sum over 6 folds):')
t = FP[FP['mode'] == 'upt'].groupby(['ensemble', 'lam_key']).agg(flags_lam0=('flags_lam0', 'sum'), flags_lam=('flags_lam', 'sum'), gone=('gone', 'sum'))
print(t.to_string())
print('\n[flags at p0, upt, lam_geo] per peptide (sum over folds holding it out): flags lam0 -> lam_geo; median lam*C_ii/sig^2 of gone / kept cells')
t = FP[(FP['mode'] == 'upt') & (FP.lam_key == 'lg')].groupby(['ensemble', 'peptide']).agg(flags_lam0=('flags_lam0', 'sum'), flags_lam=('flags_lam', 'sum'),
                                                                                          gone=('gone', 'sum'), ratio_gone=('ratio_gone_med', 'median'),
                                                                                          ratio_kept=('ratio_kept_med', 'median'))
print(t.round(3).to_string())
for ens in D.ENSS:
    m = Q.MoPrP(ens, floor=True); r = np.diag(PR[f'{ens}|Cw|upt']) / (m.sig ** 2).ravel()
    pr = r.reshape(14, 15)
    print(f'{ens} upt lam_geo C_walk,ii/sigma^2 per peptide (median over t): ' + ', '.join(f'{D.LAB[p]}:{np.median(pr[p]):.2f}' for p in range(14)))
