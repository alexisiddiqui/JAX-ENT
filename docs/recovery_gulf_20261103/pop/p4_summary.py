"""T4: summary table + sharpness of the state-level optimum.
Sharpness: near-optimal set on the grid = points whose score is within a tolerance of the best (mse/wr2/chi2_pool: treated as a
chi2 with sigma rescaled so the minimum has chi2/nu = 1, tolerance Delta chi2 <= 2  <=>  210*(s - s_min)/s_min <= 2;
maxz: within 1% relative; spearman: within 0.002). Report its recovery range and Folded/PUF1 range.
-> p4_summary.csv (state-level T2 + frame-level T3 rows), p4_sharp.csv, p4.log"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import p0_common as P

pd.set_option('display.width', 320); pd.set_option('display.max_rows', 3000); pd.set_option('display.max_columns', 40)
NCELL = P.DEXP.size
sharp = []
for f in sorted(glob.glob(P.OUT + 'p2_grid_*.npz')):
    z = np.load(f, allow_pickle=True); G = z['P']; names = list(z['names'])
    tag = os.path.basename(f)[8:-4]; ens = 'AF2_MSAss' if tag.startswith('AF2_MSAss') else 'AF2_filtered'
    rest = tag[len(ens) + 1:]; mode = 'frame_uptake' if rest.endswith('frame_uptake') else 'rate'; model = rest[:-len(mode) - 1]
    for c in [None, 0.0, 0.01, 0.05]:
        m = np.ones(len(G), bool) if c is None else G[:, 3:].sum(1) <= c + 1e-9
        for sc in ['mse', 'wr2', 'chi2_pool', 'maxz', 'spearman']:
            v = z[sc][m]
            if sc == 'spearman': near = v >= v.max() - 0.002; best = v.max()
            elif sc == 'maxz': near = v <= v.min() * 1.01; best = v.min()
            else: near = NCELL * (v - v.min()) / v.min() <= 2; best = v.min()
            rec = z['recovery'][m][near]; Gn = G[m][near]
            sharp.append(dict(ensemble=ens, model=model, mode=mode, constraint='none' if c is None else f'decoy<={c}', score=sc,
                              n_near=int(near.sum()), frac_near=float(near.mean()), rec_min=rec.min(), rec_max=rec.max(),
                              F_min=Gn[:, 0].min(), F_max=Gn[:, 0].max(), PUF1_min=Gn[:, 1].min(), PUF1_max=Gn[:, 1].max(),
                              PUF2_max=Gn[:, 2].max(), decoy_max=Gn[:, 3:].sum(1).max() if Gn.shape[1] > 3 else 0.0))
S = pd.DataFrame(sharp); S.to_csv(P.OUT + 'p4_sharp.csv', index=False)

B = pd.read_csv(P.OUT + 'p2_best.csv'); B = B[B.how == 'refined'].copy()
B['decoys'] = B[['PUF3', 'unfolded', 'PUF2-like']].fillna(0).sum(1)
B['level'] = 'state(T2)'; B['selection'] = B['score']; B['ESS'] = np.nan
T3 = pd.read_csv(P.OUT + 'p3_t3.csv'); T3['decoys'] = T3[['PUF3', 'unfolded', 'PUF2-like']].sum(1)
T3['level'] = 'frame(T3)'; T3['selection'] = T3['objective']
T3['constraint'] = T3['constraint'].map({'none': 'none', 'decoy=0.0': 'decoy<=0.0', 'decoy=0.01': 'decoy=0.01'})
cols = ['level', 'ensemble', 'model', 'mode', 'selection', 'constraint', 'Folded', 'PUF1', 'PUF2', 'decoys', 'recovery', 'ESS', 'mse', 'wr2', 'chi2_pool', 'maxz', 'spearman']
A = pd.concat([B[cols], T3[cols]], ignore_index=True)
A['script'] = np.where(A.level == 'state(T2)', 'p2_landscape', 'p3_frames')
A['F_major_P1P2_pos'] = (A.Folded > 0.5) & (A.PUF1 > 1e-3) & (A.PUF2 > 1e-3)
A['F_major_P1_pos'] = (A.Folded > 0.5) & (A.PUF1 > 1e-3)
A.to_csv(P.OUT + 'p4_summary.csv', index=False)

print('=== configurations with recovery >= 80 (state level, constraint none or decoy<=0) ===')
print(A[(A.level == 'state(T2)') & (A.recovery >= 80) & A.constraint.isin(['none', 'decoy<=0.0'])].round(3).to_string())
print('\n=== count of state-level selections with recovery>=80, by model x mode (of 5 scores x 5 constraints) ===')
st = A[A.level == 'state(T2)']
print(st.assign(hit=st.recovery >= 80).groupby(['ensemble', 'model', 'mode']).hit.agg(['sum', 'count']).to_string())
print('\n=== sharpness (decoy<=0.0 and none) ===')
print(S[S.constraint.isin(['none', 'decoy<=0.0'])].round(3).to_string())
print('\n=== frame-level (T3) recovery>=80 by model x mode ===')
fr = A[A.level == 'frame(T3)']
print(fr.assign(hit=fr.recovery >= 80).groupby(['ensemble', 'model', 'mode', 'constraint']).hit.agg(['sum', 'count']).to_string())
