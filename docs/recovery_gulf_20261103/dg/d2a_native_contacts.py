"""Per-frame, per-segment CA native-contact fraction Q_seg from the archived MoPrP101 trajectories (the ones that were
both featurised (_featurise_a101_k109 manifest sha256) and rule-clustered (101 CA atoms)).
Also re-computes compactness_core / compactness_a1 to verify frame order against the clustering CSV.
-> d2a_Q.npz, d2a.log"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hashlib, json
import numpy as np, pandas as pd
import MDAnalysis as mda
from scipy.spatial.distance import pdist, cdist
import d0_common as D

TOP = D.C.EX + 'data/MoPrP_max_plddt_4334.pdb'
XTC = {'AF2_MSAss': D.C.EX + 'data/_archived_MoPrP101/_cluster_MoPrP/clusters/all_clusters.xtc',
       'AF2_filtered': D.C.EX + 'data/_archived_MoPrP101/_cluster_MoPrP_filtered/clusters/all_clusters.xtc'}
man = json.load(open(D.FEAT + 'manifest.json'))
ref = mda.Universe(TOP); ca_ref = ref.select_atoms('name CA'); X0 = ca_ref.positions.copy(); rid = ca_ref.resids
assert len(rid) == 101 and rid[0] == 1
d0 = cdist(X0, X0); sep = np.abs(rid[:, None] - rid[None])
NATIVE = (d0 < 8.0) & (sep >= 3)
segs = D.seg_table().drop_duplicates('mouse')
out = {}
for ens, xtc in XTC.items():
    sha = hashlib.sha256(open(xtc, 'rb').read()).hexdigest()
    print(ens, 'xtc sha matches featurisation manifest:', sha == man['trajectories'][ens]['sha256'])
    u = mda.Universe(TOP, xtc); ca = u.select_atoms('name CA')
    X = np.array([ca.positions.copy() for _ in u.trajectory])          # F,101,3
    nF = len(X)
    core = np.isin(rid, list(range(49, 72)) + list(range(77, 102))); a1 = (rid >= 22) & (rid <= 31)
    comp_core = np.array([pdist(x[core]).mean() for x in X]); comp_a1 = np.array([pdist(x[a1]).mean() for x in X])
    ff = D.frame_features(ens)
    for nm, v in [('compactness_core', comp_core), ('compactness_a1', comp_a1)]:
        print(f'  {nm}: max|recomputed - csv| = {np.abs((v - np.median(v)) - ff[nm].values).max():.2e}')
    Dm = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)                 # F,101,101
    formed = Dm < (d0[None] + 1.5)
    for _, s in segs.iterrows():
        m = (rid >= s.seq_lo) & (rid <= s.seq_hi)
        pair = NATIVE & (m[:, None] | m[None])
        pair = np.triu(pair, 1)
        q = formed[:, pair].mean(1)
        out[f'{ens}|{s.mouse}'] = q
        print(f'  {s.mouse:8s} seq {s.seq_lo}-{s.seq_hi}: {pair.sum():4d} native CA contacts; Q median {np.median(q):.3f}, '
              f'frac Q<0.7 {np.mean(q < 0.7):.3f}, frac Q<0.5 {np.mean(q < 0.5):.3f}')
np.savez(D.OUT + 'd2a_Q.npz', **out)
