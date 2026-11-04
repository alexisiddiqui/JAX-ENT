"""Recreated verbatim (functional parts) from calib/c0_common.py of the previous pass (scratch was wiped).
NumPy BV frame_uptake forward model + row-normalised map rule."""
import json
import numpy as np, pandas as pd

EX = 'jaxent/examples/2_CrossValidation/'
DATA = EX + 'data/_MoPrP/'
CLUS = EX + 'analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters/'
BC, BH = 0.35, 2.0
ENS = {'AF2_MSAss': 'AF2_MSAss', 'AF2_filtered': 'AF2_Filtered'}
SM = {0: "Folded", 1: "PUF1", 2: "PUF2", 3: "PUF3", 4: "unfolded", 5: "PUF2-like"}
TARGET = {"Folded": 0.97118501, "PUF1": 0.0236435785, "PUF2": 0.00517140695, "PUF3": 0., "unfolded": 0., "PUF2-like": 0.}
TIMES = np.loadtxt(DATA + 'moprp.times')[1:] * 60.0
DEXP = np.loadtxt(DATA + 'moprp.dexp')[:, 1:].T
WTS = np.loadtxt(DATA + 'moprp.weights')[:, 1:].T
SEGS = np.loadtxt(DATA + '_output/MoPrP_segments.txt', dtype=int)
PEP_LAB = [f"{s}-{e}" for s, e in SEGS]
PFACT = dict(np.loadtxt(DATA + 'median.pfact'))
SIGMA_HOMO = 0.0586


def assignments(ens):
    e = ENS[ens]
    return pd.read_csv(CLUS + f'{e}/{e}_frame_to_cluster.csv').cluster_label.values


def pep_map(res, segs=SEGS, trim=2):
    M = np.zeros((len(segs), len(res)))
    for i, (s, e) in enumerate(segs):
        m = (res >= s + trim) & (res <= e)
        M[i, m] = 1.0 / (e - (s + trim) + 1)
    return M


def pep_map_norm(res, segs=SEGS, trim=2):
    M = pep_map(res, segs, trim)
    return M / M.sum(1, keepdims=True)


def frame_pep_uptake(lnpf, kint, M, t=TIMES):
    rates = kint[:, None] * np.exp(-lnpf)
    D = 1 - np.exp(-rates[:, :, None] * t[None, None])
    return np.einsum('pr,rft->fpt', M, D)


def recovery(assign, w):
    from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
    w = np.asarray(w, float); w = w / w.sum()
    return float(calculate_recovery_percentage(assign, w, TARGET, SM))


def composition(assign, w):
    w = np.asarray(w, float); w = w / w.sum()
    return {SM[c]: round(float(w[assign == c].sum()), 3) for c in np.unique(assign) if w[assign == c].sum() > 1e-3}


def nnls_simplex(F, obs, wt, lam=1e3):
    from scipy.optimize import nnls
    n = F.shape[0]
    A = (F * wt[None]).reshape(n, -1).T; b = (obs * wt).ravel()
    A2 = np.vstack([A, lam * np.ones(n)]); b2 = np.append(b, lam)
    x, _ = nnls(A2, b2, maxiter=50 * n)
    return x / x.sum()
