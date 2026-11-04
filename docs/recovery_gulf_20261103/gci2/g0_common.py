"""GCI shared code, recreated from gci/g0_common.py of the previous pass (scratch wiped). Run from repo root."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import c0_common as C  # noqa

OUT = '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2/'
FEAT = C.EX + 'fitting/jaxENT/_featurise_a101_k109/'
ENSS = ['AF2_MSAss', 'AF2_filtered']
DEXP = pd.read_csv(C.DATA + '_output/MoPrP_dfrac.dat', sep=r'\s+', comment='#', header=None).to_numpy()
assert np.allclose(DEXP, C.DEXP)
WTS = C.WTS; SEGS = C.SEGS; LAB = C.PEP_LAB; TIMES = C.TIMES
TLAB = ['5s', '20s', '40s', '1m', '5m', '10m', '20m', '30m', '45m', '1h', '2.7h', '4h', '6.5h', '12.5h', '24h']
NP, NT = DEXP.shape
N = NP * NT
PEP_OF = np.repeat(np.arange(NP), NT)
FOLD_A = [0, 1, 2, 3, 4, 5]
FOLD_B = list(range(6, 14))
PF = C.PFACT
BC0, BH0 = C.BC, C.BH


def load(ens):
    z = np.load(FEAT + f'features_{ens}.npz')
    top = json.load(open(FEAT + f'topology_{ens}.json'))['topologies']
    res = np.array([t['residues'][0] for t in top])
    return res, np.asarray(z['heavy_contacts'], float), np.asarray(z['acceptor_contacts'], float), np.asarray(z['k_ints'], float)


def pmap(res):
    return C.pep_map_norm(res, SEGS, 2)


def frame_uptake(lnpf, kint, M):
    return C.frame_pep_uptake(lnpf, kint, M, TIMES)


def assign(ens):
    return C.assignments(ens)


def recovery(a, w):
    return C.recovery(a, w)


ALPHA_EACH = 1 - 0.95 ** (1 / 3)
X0 = norm.ppf(1 - ALPHA_EACH / 2)
Y0 = norm.ppf(1 - ALPHA_EACH)


def t_n(n):
    return norm.ppf(0.5 + 0.5 * (1 - ALPHA_EACH) ** (1 / n))


def sig_stats(Sig):
    n = len(Sig); sd = np.sqrt(np.diag(Sig)); R = Sig / np.outer(sd, sd); ev = np.linalg.eigvalsh(R)
    a = R.sum() / n; b = np.sum(ev ** 2) / n
    return dict(sd=sd, a_n=a, b_n=b, spec=ev.max() ** 2 / np.sum(ev ** 2), t_n=t_n(n), thrS=np.sqrt(a) * X0, thrQ=np.sqrt(b) * Y0)


def triple(r, st):
    z = r / st['sd']; n = len(z)
    Tn = np.abs(z).max(); S = z.sum() / np.sqrt(n); Q = (np.sum(z ** 2) - n) / np.sqrt(2 * n)
    pT, pS, pQ = Tn <= st['t_n'], abs(S) <= st['thrS'], Q <= st['thrQ']
    return dict(T=Tn, S=S, Q=Q, passT=bool(pT), passS=bool(pS), passQ=bool(pQ), passAll=bool(pT and pS and pQ),
                overfitQ=bool(Q < -st['thrQ']), nflag=int(np.sum(np.abs(z) > st['t_n'])), flagged=flagged(z, st['t_n']), z=z)


def gci(r, Sig):
    st = sig_stats(Sig); d = triple(r, st); d.update({k: st[k] for k in ['a_n', 'b_n', 'spec', 't_n', 'thrS', 'thrQ']}); return d


def flagged(z, tn):
    idx = np.where(np.abs(z) > tn)[0]
    return ';'.join(f'{LAB[i // NT]}@{TLAB[i % NT]}({z[i]:+.1f})' for i in idx)


def code(d):
    """pass code: P or failing stats (t,s,q) + 'o' overfit + nflag"""
    c = ('' if d['passT'] else 't') + ('' if d['passS'] else 's') + ('' if d['passQ'] else 'q')
    return (c or 'P') + ('o' if d['overfitQ'] else '') + (str(d['nflag']) if d['nflag'] else '')


def sigma_variants():
    s_cell = 1.0 / WTS
    pooled = np.sqrt(np.mean(s_cell ** 2, 1, keepdims=True)) * np.ones((1, NT))
    return {'cell': s_cell.ravel(), 'pep_rms': pooled.ravel(), 'homo': np.full(N, C.SIGMA_HOMO),
            'pep_rms_floor': np.maximum(pooled.ravel(), C.SIGMA_HOMO)}


def lam_moment(r, Cm):
    P = np.outer(r, r); off = ~np.eye(len(r), dtype=bool)
    return float(np.sum(P[off] * Cm[off]) / np.sum(Cm[off] ** 2))


# corrections (re-made by g1_corrections.py -> corrections.npz)
_CACHE = {}


def corr_table():
    if 'tab' not in _CACHE:
        _CACHE['tab'] = dict(np.load(OUT + 'corrections.npz', allow_pickle=True))
    return _CACHE['tab']


def lnpf(ens, corr, s=1.0):
    """per-frame lnPF (res, F) for correction corr at strength s."""
    res, h, a, k = load(ens)
    if corr.startswith('bcbh'):
        bc, bh = corr_table()[f'{ens}|{corr}']
        return ((1 - s) * BC0 + s * bc) * h + ((1 - s) * BH0 + s * bh) * a
    L = BC0 * h + BH0 * a
    if corr != 'none':
        L = L + s * corr_table()[f'{ens}|{corr}'][:, None]
    return L


def get_F(ens, corr, s=1.0):
    res, h, a, k = load(ens)
    return frame_uptake(lnpf(ens, corr, s), k, pmap(res))


def cov_ens(F):
    return np.cov(F.reshape(F.shape[0], -1), rowvar=False)
