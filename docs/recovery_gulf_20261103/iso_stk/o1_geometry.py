"""[o1] geometry + kernels for the scalar-distance OMC (sidecar construction) and the T1 self-tuning / sidecar-kNN graphs.
ISO : all-pairs Work Scale, Calpha RMSD, PyRosetta |d ref2015| from artifacts/iso_policy_sidecars/geometry (TRI, 2225 frames).
      BI frames are located inside TRI by exact (Nc, Nh) row match -> BI distances = TRI submatrix, re-median-scaled on BI.
MoPrP: AF2_MSAss Work Scale + periodic phi/psi RMS from the fresh sidecar geometry (stored); AF2_filtered recomputed with the
      sidecar recipe (moprp_sidecar_support.prepare_geometry: same 96 feature residues, same periodic_dihedral_pair_distance),
      the recipe validated by recomputing MSAss from the same archived trajectory.
Kernel diagnostics per metric x h (sidecar K = exp(-min(d~^2/2h^2, 80)), capped -> 0): off-diagonal fill, mean degree,
connected components, relative distance of L=D-K from the complete-graph Laplacian (h >= 100 collapse check).
T1 graphs: self-tuning L_st (stk s0.build_kernel, k=20 mean union alpha=1/2) for BI/TRI (+ lambda_geo check vs stk s1);
sidecar logPF-kNN20 union graph (unweighted) L_sym for TRI and its induced BI subgraph.
-> o1_bi_in_tri.npy, o1_moprp_<ens>_<metric>.npz, o1_graphs.npz, o1_kernels.csv, o1.log"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
import i0_common as C
Q, D, S0 = C.Q, C.D, C.S0
REPO = C.REPO

# ---------------- ISO: BI inside TRI
B = os.path.join(REPO, Q.ISO_B, 'fit_features/')
b = np.load(B + 'features_iso_bi.npz'); t = np.load(B + 'features_iso_tri.npz')
Xb = np.hstack([b['heavy_contacts'].T, b['acceptor_contacts'].T]); Xt = np.hstack([t['heavy_contacts'].T, t['acceptor_contacts'].T])
look = {r.tobytes(): i for i, r in enumerate(Xt)}
assert len(look) == len(Xt)
idx = np.array([look[r.tobytes()] for r in Xb]); assert np.all(np.diff(idx) > 0) and np.array_equal(Xt[idx], Xb)
np.save(C.OUT + 'o1_bi_in_tri.npy', idx)
mb, mt = Q.ISO('bi'), Q.ISO('tri')
print(f'[ISO] BI (874) = exact ordered subset of TRI (2225); labels agree: {np.array_equal(mb.labels, mt.labels[idx])}; '
      f'BI open/closed {np.sum(mb.labels == 0)}/{np.sum(mb.labels == 1)}; TRI open/closed/unassigned '
      f'{np.sum(mt.labels == 0)}/{np.sum(mt.labels == 1)}/{np.sum(mt.labels == -1)}', flush=True)
man = json.load(open(C.ISO_GEOM + 'manifest.json'))
for met in ('work_scale', 'rmsd', 'pyrosetta'):
    z = np.load(C.ISO_GEOM + f'{met}.npz'); R = z['raw_distances'].astype(float); Dn, s = C.median_scale(R)
    print(f'[ISO TRI {met}] median {s:.6g} (manifest {man["distances"][met]["median_positive_distance"]:.6g}); '
          f'max |normalised - stored| {np.abs(Dn - z["normalized_distances"]).max():.2e}', flush=True)

# ---------------- MoPrP geometry
sys.path.insert(0, os.path.join(REPO, 'jaxent/examples/2_CrossValidation/fitting/jaxENT'))
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import periodic_dihedral_pair_distance  # noqa
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
EX = os.path.join(REPO, 'jaxent/examples/2_CrossValidation/')
TOP = EX + 'data/MoPrP_max_plddt_4334.pdb'
TRJ = {'AF2_MSAss': EX + 'data/_archived_MoPrP101/_cluster_MoPrP/clusters/all_clusters.xtc',
       'AF2_filtered': EX + 'data/_archived_MoPrP101/_cluster_MoPrP_filtered/clusters/all_clusters.xtc'}
FD = EX + 'fitting/jaxENT/_featurise/'


def feat_res(ens):
    return {int(r) for tp in json.load(open(FD + f'topology_{ens}.json'))['topologies'] for r in tp['residues']}


def work_scale(ens, src='sidecar'):
    if src == 'sidecar':
        z = np.load(FD + f'features_{ens}.npz'); L = D.BC0 * np.asarray(z['heavy_contacts'], float) + D.BH0 * np.asarray(z['acceptor_contacts'], float)
    else:
        res, h, a, k = D.load(ens); L = D.BC0 * h + D.BH0 * a
    v = L.mean(0); return np.abs(v[:, None] - v[None, :])


def dihedral_rmsd(ens, resset):
    u = mda.Universe(TOP, TRJ[ens])
    residues = [r for r in u.select_atoms('protein').residues if int(r.resid) in resset]
    ags = []
    for r in residues:
        phi, psi = r.phi_selection(), r.psi_selection()
        if phi is not None and psi is not None: ags.extend((phi, psi))
    ang = np.deg2rad(np.asarray(Dihedral(ags).run().results.angles, float)).T
    n = ang.shape[1]; l, r_ = np.triu_indices(n, 1); T = np.zeros((n, n)); T[l, r_] = periodic_dihedral_pair_distance(ang, l, r_); T += T.T
    return T, len(ags)


res96 = feat_res('AF2_MSAss')
for ens in D.ENSS:
    ws = work_scale(ens); ws_a = work_scale(ens, 'a101')
    T, nang = dihedral_rmsd(ens, res96)
    print(f'[MoPrP {ens}] frames {len(ws)}; work_scale(sidecar feats) vs a101 feats: corr {np.corrcoef(ws.ravel(), ws_a.ravel())[0, 1]:.4f}; '
          f'dihedral angles {nang}', flush=True)
    if ens == 'AF2_MSAss':
        zw = np.load(C.MO_GEOM + 'work_scale.npz'); zd = np.load(C.MO_GEOM + 'dihedral_rmsd.npz')
        print(f'   recipe check vs stored sidecar geometry: work_scale max|diff| {np.abs(ws - zw["raw_distances"]).max():.2e}, '
              f'dihedral max|diff| {np.abs(T - zd["raw_distances"]).max():.2e} (median stored {float(zd["median_positive_distance"]):.4f})', flush=True)
        ws, T = zw['raw_distances'].astype(float), zd['raw_distances'].astype(float)      # use the stored files themselves
    for met, R in (('work_scale', ws), ('dihedral_rmsd', T)):
        Dn, s = C.median_scale(R)
        np.savez_compressed(C.OUT + f'o1_moprp_{ens}_{met}.npz', raw=R, normalized=Dn, median=s)
        print(f'   {met}: median {s:.5g}, max d~ {Dn.max():.2f}', flush=True)

# ---------------- kernel diagnostics
rows = []
def diag(system, met, Dn):
    N = len(Dn); off = ~np.eye(N, dtype=bool)
    Lc = N * np.eye(N) - np.ones((N, N))
    for h in C.HS:
        K = C.sidecar_kernel(Dn, h); d = K.sum(1); L = np.diag(d) - K
        nz = (K[off] > 0).mean(); ncomp = connected_components(K > 0, directed=False)[0]
        # relative distance to the complete graph after optimal scalar rescale (the gamma rescaling removes scale)
        a = np.sum(L * Lc) / np.sum(Lc * Lc); rel = np.linalg.norm(L - a * Lc) / np.linalg.norm(a * Lc) if a > 0 else np.nan
        rows.append(dict(system=system, metric=met, h=h, offdiag_fill=nz, mean_degree_w=d.mean(), min_degree_w=d.min(),
                         isolated=int(np.sum(d == 0)), ncomp=int(ncomp), rel_dev_complete=rel, d_med_scaled_max=float(Dn.max())))
        r = rows[-1]
        print(f'{system:12s} {met:13s} h={h:<6g} fill {nz:.4f} meanW {r["mean_degree_w"]:9.3f} isolated {r["isolated"]:4d} '
              f'ncomp {ncomp:4d} rel.dev(complete) {rel:.2e}', flush=True)


for which in ('bi', 'tri'):
    for met, (Dn, s) in C.iso_distances(which).items():
        diag(f'ISO_{which.upper()}', met, Dn)
for ens in D.ENSS:
    for met, (Dn, s) in C.moprp_distances(ens).items():
        diag(ens, met, Dn)
pd.DataFrame(rows).to_csv(C.OUT + 'o1_kernels.csv', index=False)

# ---------------- T1 graphs
SAVE = {}
PR = np.load(os.path.join(C.RG, 'stk', 's1_primary.npz'))
for which in ('bi', 'tri'):
    X = S0.iso_lnpf(which); kern = S0.build_kernel(X, 20, 'mean', 'union')
    m = Q.ISO(which); Cw, Ce, inf = S0.walk_cov(kern, m.Ff, True)
    print(f'[T1 L_st ISO_{which.upper()}] ncomp {kern["ncomp"]} edges {kern["n_edges"]} lam_geo {inf["lam_geo"]:.5g} (stk s1 {float(PR[f"ISO_{which}|lam_geo"]):.5g})', flush=True)
    SAVE[f'{which}|Lst'] = kern['Lst']; SAVE[f'{which}|d'] = kern['d']
    att = np.sqrt(kern['d']); att /= att.sum()
    print(f'   L_st null vector w∝sqrt(d): ESS {Q.ess(att):.3f} rec {m.recovery(att):.1f} pops(open/closed/unassigned) {np.round(100 * m.pops(att), 2)}', flush=True)
    if kern['ncomp'] > 1:
        from scipy.sparse.csgraph import connected_components as cc
        _, lab = cc(kern['Kt'] > 0, directed=False)
        for c_ in np.unique(lab):
            mm = lab == c_; print(f'   component {c_}: {mm.sum()} frames, labels {dict(zip(*np.unique(m.labels[mm], return_counts=True)))}', flush=True)
z = np.load(C.ISO_KNN); A = np.zeros((2225, 2225)); A[z['edge_sources'], z['edge_targets']] = 1; A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
Xt = S0.iso_lnpf('tri'); Dm = cdist(Xt, Xt)
print(f'[T1 sidecar kNN20] edges {int(A.sum() // 2)}; stored distances vs lnPF L2: max|diff| {np.abs(z["distances"] - Dm).max():.2e}', flush=True)
for which, AA in (('tri', A), ('bi', A[np.ix_(idx, idx)])):
    d = AA.sum(1); dm = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-300)), 0)
    SAVE[f'{which}|knn_Lsym'] = np.diag((d > 0).astype(float)) - dm[:, None] * AA * dm[None]
    nc, lab = connected_components(AA > 0, directed=False)
    m = Q.ISO(which); att = np.sqrt(d); att /= att.sum()
    print(f'   ISO_{which.upper()} kNN20 graph: ncomp {nc}, isolated {int(np.sum(d == 0))}, null w∝sqrt(deg): rec {m.recovery(att):.1f} '
          f'pops {np.round(100 * m.pops(att), 2)}', flush=True)
np.savez(C.OUT + 'o1_graphs.npz', **SAVE)
