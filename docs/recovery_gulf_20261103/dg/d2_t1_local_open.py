"""T1: per-frame local 'open' indicators per HDX segment, uniform-weight open fractions vs experimental K_op.
Indicator families:
  A  rule-spec features (MoPrP_rules_spec.json transforms, MoPrP_unfolding_spec.json features, median-centred):
       b1_open   = NOT b1_strand_intact            (dih_b1 phi in [-17,15] AND psi in [-21,30])
       a1_open   = NOT a1_helix_intact (|dih_a1 phi|,|psi| <= 10)  OR  displacement_a1_vs_a3 > 0.75   (a1_displacement)
       b2_open   = NOT b2_strand_intact (|dih_b2 phi|,|psi| <= 21) OR  displacement_b2_vs_core > 0.25 (b2_displacement)
       core_open = compactness_core > 0.5 A  OR extreme_displacement   (rules spec has NO threshold on compactness_core;
                   0.5 A chosen: above Filtered max 0.23 and MSAss 95th pct 0.15)
  B  per-segment CA native-contact fraction Q_seg (d2a): open = Q_seg < 0.7 (B70) / < 0.5 (B50); continuous 1-Q
  C  scaled-BV implied opening: K_BV,f = mean over HDX-NMR slow residues of exp(-lnPF_rf) (EX2: K_op = 1/PF),
     lnPF from the median-scaled BV model (bc,bh = s*(0.35,2.0)); uniform K_BV = mean_f K_BV,f
-> d2_t1.csv, d2_indicators.npz, d2.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import d0_common as D

T = D.seg_table()
Q = dict(np.load(D.OUT + 'd2a_Q.npz'))
REGION_A = {'127-132': 'b1', '133-143': None, '133-148': 'a1', '149-153': 'a1', '154-167': 'b2', '155-162': 'b2',
            '182-196': 'core', '197-204': 'core', '197-206': 'core', '204-224': 'core', '205-212': 'core', '217-223': 'core'}
FEAT_A = {'b1': 'dih_b1 phi/psi', 'a1': 'dih_a1 phi/psi + displacement_a1_vs_a3', 'b2': 'dih_b2 phi/psi + displacement_b2_vs_core',
          'core': 'compactness_core (+extreme_displacement)'}


def inr(x, lo, hi):
    return (x >= lo) & (x <= hi)


def region_open(ff):
    b1 = ~(inr(ff.dih_b1_integrity_phi, -17, 15) & inr(ff.dih_b1_integrity_psi, -21, 30))
    a1 = ~(inr(ff.dih_a1_integrity_phi, -10, 10) & inr(ff.dih_a1_integrity_psi, -10, 10)) | (ff.displacement_a1_vs_a3 > 0.75)
    b2 = ~(inr(ff.dih_b2_integrity_phi, -21, 21) & inr(ff.dih_b2_integrity_psi, -21, 21)) | (ff.displacement_b2_vs_core > 0.25)
    extreme = inr(ff.displacement_b2_vs_core, 3.5, 20) | inr(ff.displacement_a1_vs_core, 5, 20) | inr(ff.displacement_a1_vs_a3, 10, 20)
    core = (ff.compactness_core > 0.5) | extreme
    return {k: np.asarray(v, float) for k, v in dict(b1=b1, a1=a1, b2=b2, core=core).items()}


rows, ind = [], {}
for ens in D.ENSS:
    ff = D.frame_features(ens); a = D.assign(ens); nF = len(ff)
    RO = region_open(ff)
    res, h, ac, k = D.load(ens)
    L = D.lnpf(ens, 'scale')
    ridx = {r: i for i, r in enumerate(res)}
    print(f'{ens}: region open fractions (uniform) ' + ', '.join(f'{r} {v.mean():.3f}' for r, v in RO.items()))
    print('   by state (frames per state, region open frac): ')
    for s in range(6):
        m = a == s
        if m.sum(): print(f'     {D.STATES[s]:9s} n={m.sum():3d} ' + ' '.join(f'{r}={v[m].mean():.2f}' for r, v in RO.items()))
    for _, s in T.iterrows():
        key = s.mouse
        regA = REGION_A[key]
        oA = RO[regA] if regA else np.full(nF, np.nan)
        q = Q[f'{ens}|{key}']
        sl = [ridx[r] for r in D.SLOW_RES[key] if r in ridx]
        kbv_f = np.exp(-L[sl]).mean(0)                         # per-frame EX2 K implied by scaled BV
        kint_sl = k[sl]
        # rate-weighted (what a single-exponential slow phase sees): K_eff = sum_r kint_r <1/PF_r> / sum_r kint_r
        kbv_rate = (kint_sl[:, None] * np.exp(-L[sl])).sum(0) / kint_sl.sum()
        pB70, pB50 = (q < 0.7).astype(float), (q < 0.5).astype(float)
        ind[f'{ens}|{s.seg}|A'] = oA; ind[f'{ens}|{s.seg}|B70'] = pB70; ind[f'{ens}|{s.seg}|B50'] = pB50
        ind[f'{ens}|{s.seg}|Bc'] = 1 - q; ind[f'{ens}|{s.seg}|C'] = kbv_rate
        Kbv = kbv_rate.mean()
        row = dict(ensemble=ens, seg=s.seg, seq=f'{s.seq_lo}-{s.seq_hi}', label=s.label, region_A=regA or '-',
                   feature_A=FEAT_A.get(regA, 'not featured'), slow_seq=','.join(map(str, D.SLOW_RES[key])),
                   dG_exp=s.dG, K_exp=s.K, p_exp=s.p_open,
                   open_A=np.nanmean(oA) if regA else np.nan, open_B70=pB70.mean(), open_B50=pB50.mean(), mean_1mQ=(1 - q).mean(),
                   K_BV_scaled=Kbv, dG_BV_scaled=-D.RT * np.log(Kbv),
                   nframes_open_A=int(np.nansum(oA)) if regA else -1, nframes_open_B70=int(pB70.sum()))
        for nm in ['open_A', 'open_B70', 'K_BV_scaled']:
            ref = row['p_exp'] if nm != 'K_BV_scaled' else row['K_exp']
            row[f'ratio_{nm}'] = row[nm] / ref if row[nm] == row[nm] else np.nan
        # which states carry the open frames (B70)
        row['B70_open_states'] = ';'.join(f'{D.STATES[c]}:{int(((a == c) & (pB70 > 0)).sum())}' for c in range(6) if ((a == c) & (pB70 > 0)).sum())
        row['A_open_states'] = ';'.join(f'{D.STATES[c]}:{int(((a == c) & (oA > 0)).sum())}' for c in range(6) if regA and ((a == c) & (oA > 0)).sum())
        rows.append(row)
    for r, v in RO.items(): ind[f'{ens}|region|{r}'] = v
df = pd.DataFrame(rows)
df.to_csv(D.OUT + 'd2_t1.csv', index=False)
np.savez(D.OUT + 'd2_indicators.npz', **ind)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 60)
for ens in D.ENSS:
    print(f'\n== {ens} ==')
    x = df[df.ensemble == ens][['seg', 'seq', 'region_A', 'dG_exp', 'p_exp', 'open_A', 'ratio_open_A', 'open_B70', 'ratio_open_B70', 'open_B50',
                                 'K_BV_scaled', 'dG_BV_scaled', 'A_open_states', 'B70_open_states']]
    print(x.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
