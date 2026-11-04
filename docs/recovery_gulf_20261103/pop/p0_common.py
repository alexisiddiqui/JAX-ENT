"""pop/ shared code. Run from repo root with .venv/bin/python.
Reuses gci2/c0_common.py (validated NumPy BV frame_uptake + row-normalised peptide map)."""
import json, sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gci2'))
import c0_common as C  # noqa: E402

OUT = HERE + '/'
FEAT = C.EX + 'fitting/jaxENT/_featurise_a101_k109/'
ENSS = ['AF2_MSAss', 'AF2_filtered']
DEXP = pd.read_csv(C.DATA + '_output/MoPrP_dfrac.dat', sep=r'\s+', comment='#', header=None).to_numpy()
assert np.allclose(DEXP, C.DEXP)
WTS = C.WTS; SEGS = C.SEGS; LAB = C.PEP_LAB; TIMES = C.TIMES
NP, NT = DEXP.shape
SIG_CELL = 1.0 / WTS                                            # weights read as 1/sigma
SIG_POOL = np.sqrt(np.mean(SIG_CELL ** 2, 1, keepdims=True)) * np.ones((1, NT))  # per-peptide pooled RMS sigma
BC0, BH0 = C.BC, C.BH
STATES = ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded', 'PUF2-like']
TGT6 = np.array([C.TARGET[s] for s in STATES]); TGT6 = TGT6 / TGT6.sum()
REGIONS = [(1, 30), (31, 58), (59, 101)]


def load(ens):
    z = np.load(FEAT + f'features_{ens}.npz')
    top = json.load(open(FEAT + f'topology_{ens}.json'))['topologies']
    res = np.array([t['residues'][0] for t in top])
    return res, np.asarray(z['heavy_contacts'], float), np.asarray(z['acceptor_contacts'], float), np.asarray(z['k_ints'], float)


def pmap(res):
    return C.pep_map_norm(res, SEGS, 2)


def assign(ens):
    return C.assignments(ens)


def pfact_vec(res):
    return np.array([C.PFACT.get(int(r), -1.0) for r in res])


# ---- forward passes ------------------------------------------------------------------
def frame_uptake(L, k, M, w):
    """average per-frame uptake curves. L (res,F) lnPF, w (F,) -> (NP,NT)"""
    rates = k[:, None] * np.exp(-L)
    D = 1 - np.exp(-rates[:, :, None] * TIMES[None, None])     # res,F,T
    return np.einsum('pr,rft,f->pt', M, D, w)


def rate_uptake(L, k, M, w):
    """average per-frame exchange rates, then 1-exp(-kbar t) per residue, then peptide map"""
    kbar = (k[:, None] * np.exp(-L)) @ w
    return M @ (1 - np.exp(-kbar[:, None] * TIMES[None]))


PRED = {'frame_uptake': frame_uptake, 'rate': rate_uptake}


def recovery_p6(P6):
    """vectorised recovery = 100(1-sqrt(JSD_base2)) for state-population rows (...,6) in STATES order"""
    P = np.asarray(P6, float); P = P / P.sum(-1, keepdims=True)
    Q = np.broadcast_to(TGT6, P.shape); Mm = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(P > 0, P * np.log2(P / Mm), 0.0).sum(-1)
        b = np.where(Q > 0, Q * np.log2(Q / Mm), 0.0).sum(-1)
    return 100 * (1 - np.sqrt(np.clip(0.5 * (a + b), 0, None)))


def state_pops(a, w):
    w = np.asarray(w, float); w = w / w.sum()
    return np.array([w[a == i].sum() for i in range(6)])


def recovery(a, w):
    return C.recovery(a, w)


def ess(w):
    w = np.asarray(w, float); w = w / w.sum()
    return 1.0 / np.sum(w ** 2) / len(w)


# ---- scores --------------------------------------------------------------------------
def scores(pred):
    """pred (..., NP, NT). Returns dict of score arrays (...)."""
    from scipy.stats import rankdata
    r = DEXP - pred
    sh = pred.shape[:-2]
    rr = r.reshape(sh + (-1,))
    z = (r / SIG_POOL).reshape(sh + (-1,))
    pr = rankdata(pred.reshape(sh + (-1,)), axis=-1)
    er = rankdata(DEXP.ravel())
    pr = pr - pr.mean(-1, keepdims=True); er = er - er.mean()
    rho = (pr * er).sum(-1) / np.sqrt((pr ** 2).sum(-1) * (er ** 2).sum())
    return dict(mse=np.mean(rr ** 2, -1), wr2=np.mean((WTS.ravel() * rr ** 2), -1),
                chi2_pool=np.mean(z ** 2, -1), maxz=np.abs(z).max(-1), spearman=rho,
                mean_res=rr.mean(-1), frac_exp_gt_pred=(rr > 0).mean(-1))


SCORE_DIR = dict(mse=1, wr2=1, chi2_pool=1, maxz=1, spearman=-1)   # +1 minimise, -1 maximise
