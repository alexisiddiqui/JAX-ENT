"""[s0] Shared helpers for the selection-function study (read-only; run from repo root with .venv/bin/python).
New-feature (a101_k109, row-normalised map, 14 peptides) NumPy forward pass reuses gci2/c0_common.py."""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path('/home/alexi/Documents/JAX-ENT')
SEL = REPO / 'artifacts/recovery_gulf_20261103/sel'
sys.path.insert(0, str(REPO / 'artifacts/recovery_gulf_20261103/gci2'))
import c0_common as C  # noqa: E402

EX2 = REPO / 'jaxent/examples/2_CrossValidation'
ARMS = {'full_uptake': EX2 / 'analysis/_moprp_policy_sidecars_fresh_20261023',
        'rate': EX2 / 'analysis/_moprp_policy_sidecars_rate_fresh_20261023'}
CAMPAIGN = {'full_uptake': EX2 / 'fitting/jaxENT/_moprp_policy_sidecars_fresh_20261023',
            'rate': EX2 / 'fitting/jaxENT/_moprp_policy_sidecars_rate_fresh_20261023'}
MODE = {'full_uptake': 'frame_uptake', 'rate': 'rate'}
FEAT_NEW = EX2 / 'fitting/jaxENT/_featurise_a101_k109/'
ENSS = ['AF2_MSAss', 'AF2_filtered']
STATES = ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded']
SIGMA_HOMO = C.SIGMA_HOMO
RT = 8.314 * 298 / 1000  # kJ/mol (ranking-invariant to T)
PRIOR = {'AF2_MSAss': 73.3, 'AF2_filtered': 80.8}
CAP = 87.9


def p2(c):
    return c[(c.candidate_kind != 'running_best') & (c.step >= 100)].copy()


def model_label(row_method, edge, k):
    if row_method == 'maxent':
        return 'maxent'
    return f"omc_{'work' if edge == 'work_scale' else 'dihed'}_k{int(k)}"


def add_model(c):
    c = c.copy()
    c['model'] = [model_label(m, e, k) for m, e, k in zip(c.method, c.edge_type, c.graph_k)]
    return c


def load_new(ens):
    z = np.load(FEAT_NEW / f'features_{ens}.npz')
    top = json.load(open(FEAT_NEW / f'topology_{ens}.json'))['topologies']
    res = np.array([t['residues'][0] for t in top])
    return res, np.asarray(z['heavy_contacts'], float), np.asarray(z['acceptor_contacts'], float), np.asarray(z['k_ints'], float)


def new_forward(ens, W, mode):
    """(n,14,15) peptide uptake on new features for weights W (n,500) under averaging mode."""
    res, h, a, k = load_new(ens)
    M = C.pep_map_norm(res, C.SEGS, 2)
    lnpf = C.BC * h + C.BH * a
    if mode == 'frame_uptake':
        F = C.frame_pep_uptake(lnpf, k, M, C.TIMES)  # (500,14,15)
        return np.einsum('nf,fpt->npt', W, F), lnpf
    rates = k[:, None] * np.exp(-lnpf)  # (R,500)
    out = []
    for i in range(0, len(W), 2000):
        mr = rates @ W[i:i + 2000].T  # (R,n)
        D = 1 - np.exp(-mr[:, :, None] * C.TIMES[None, None])
        out.append(np.einsum('pr,rnt->npt', M, D))
    return np.concatenate(out), lnpf


def assignments(ens):
    return C.assignments(ens)


def composition(assign, W):
    W = W / W.sum(1, keepdims=True)
    return np.stack([W[:, assign == 0].sum(1), W[:, assign == 1].sum(1), W[:, (assign == 2) | (assign == 5)].sum(1),
                     W[:, assign == 3].sum(1), W[:, assign == 4].sum(1)], 1)  # F,PUF1,PUF2(+PUF2-like),PUF3,U
