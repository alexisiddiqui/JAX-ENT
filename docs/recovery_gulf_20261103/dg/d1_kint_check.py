"""T0: k_int cross-check of Moulick et al. Table 1 against our intrinsic rates.  -> d1_kint.csv, d1.log
Paper k_int(seg) = mean k_int of the HDX-NMR slow residues of the segment (Table 2), pD 4, 25 C, s^-1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import d0_common as D

kh = D.read_kint(D.C.DATA + 'hdxrate_poly_pD4p0_298K_min.dat')
ke = D.read_kint(D.C.DATA + 'expfact_kint_pH4p4_298K_min.dat')
rows = []
for lo, hi, lab, nex, kp, ph in D.TABLE1:
    key = f'{lo}-{hi}'
    sr = D.SLOW_RES.get(key)
    if not sr: continue
    a = np.array([kh[r] for r in sr]) / 60.0; b = np.array([ke[r] for r in sr]) / 60.0
    rows.append(dict(seg=key, slow_seq=','.join(map(str, sr)), slow_aa=''.join(D.SEQ[r - 1] for r in sr), kint_paper=kp,
                     hdxrate_pD4_mean=a.mean(), hdxrate_pD4_geo=np.exp(np.log(a).mean()), ratio_hdxrate=a.mean() / kp,
                     expfact_pH44_mean=b.mean(), ratio_expfact=b.mean() / kp,
                     log10_ratio_hdxrate=np.log10(a.mean() / kp)))
df = pd.DataFrame(rows)
df.to_csv(D.OUT + 'd1_kint.csv', index=False)
pd.set_option('display.width', 250)
print(df.to_string(float_format=lambda x: f'{x:.3g}'))
oth = df[df.seg != '133-143']
print(f'\nmedian ratio (ours/paper) over the other {len(oth)} segments: HDXrate pD4.0 {oth.ratio_hdxrate.median():.2f} '
      f'(range {oth.ratio_hdxrate.min():.2f}-{oth.ratio_hdxrate.max():.2f}); exPfact pH4.4 {oth.ratio_expfact.median():.2f}')
r = df[df.seg == '133-143'].iloc[0]
for cand in [9e-4, 9e-3]:
    print(f'133-143 candidate k_int {cand:g}: ours(HDXrate)/cand = {r.hdxrate_pD4_mean / cand:.2f}, ours(exPfact)/cand = {r.expfact_pH44_mean / cand:.2f}')
# implied k_int for 133-143 from the others' median ratio
print(f'133-143 implied paper-scale k_int = ours / median ratio = {r.hdxrate_pD4_mean / oth.ratio_hdxrate.median():.2e} s^-1')
# 133-148 contains Ile138 + R147 + Y148: consistency of 133-148's paper kint with Ile138's kint
