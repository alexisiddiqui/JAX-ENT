"""dg/ shared code (read-only analysis). Run from repo root with .venv/bin/python.
Numbering: construct seq position = mouse residue - 122 (verified by residue identity: M128->6, I138->16, Y154->32,
C178->56, C213->91, H186->64; key_residues.MD 'PDB + 122 = original'). The '-104' offset in the brief is wrong.
Reuses gci2/c0_common.py (validated NumPy BV frame_uptake + row-normalised peptide map)."""
import json, sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gci2'))
import c0_common as C  # noqa: E402

OUT = HERE + '/'
OFF = 122
RT = 0.0019872 * 298.15          # 0.5925 kcal/mol
FEAT = C.EX + 'fitting/jaxENT/_featurise_a101_k109/'
CLUS = C.CLUS
ENSS = ['AF2_MSAss', 'AF2_filtered']
SEQ = open(C.DATA + 'moprp.seq').read().strip()
DEXP = C.DEXP; WTS = C.WTS; SEGS = C.SEGS; LAB = C.PEP_LAB; TIMES = C.TIMES
NP, NT = DEXP.shape
SIG_CELL = 1.0 / WTS
SIG_POOL = np.sqrt(np.mean(SIG_CELL ** 2, 1, keepdims=True)) * np.ones((1, NT))
BC0, BH0 = C.BC, C.BH
STATES = ['Folded', 'PUF1', 'PUF2', 'PUF3', 'unfolded', 'PUF2-like']
TGT6 = np.array([C.TARGET[s] for s in STATES]); TGT6 = TGT6 / TGT6.sum()

# ---- Moulick, Das & Udgaonkar JBC 2015 Table 1 (pD 4, 25 C; k in s^-1) -------------------------------
# (mouse lo, hi, label, n_exch, k_int_paper, [(n_slow, amp%, k_slow)], usable)
TABLE1 = [
    (127, 132, 'b1', 4, 4e-3, [(1, 49.0, 1.5e-4)]),
    (133, 143, 'b1-a1 loop', 7, 9e-4, [(1, 37.9, 1.9e-4)]),
    (133, 148, 'a1', 5, 5.8e-3, [(3, 54.4, 1.1e-4)]),
    (149, 153, 'a1 C', 2, 1e-2, [(2, 88.8, 7.2e-5)]),
    (154, 167, 'a1-a2 loop + b2', 9, 3.9e-3, [(5, 66.8, 2.1e-5)]),
    (155, 162, 'a1-a2 loop + b2 (inner)', 5, 3.8e-3, [(3, 76.8, 2.1e-5)]),
    (182, 196, 'a2 C + a2-a3 loop', 13, 5.5e-3, [(1, 23.3, 2.3e-5)]),
    (190, 197, 'a2-a3 loop', 6, 8.4e-3, []),
    (197, 201, 'a2-a3 loop', 3, 9.4e-3, []),
    (197, 204, 'a3 N', 6, 3.1e-3, [(3, 69.4, 1.1e-4)]),
    (197, 206, 'a3 N (long)', 8, 3.1e-3, [(3, 70.2, 7.2e-5)]),
    (204, 224, 'a3', 18, 8.1e-3, [(2, 20.9, 3.3e-4), (5, 48.1, 1.4e-5)]),
    (205, 212, 'a3 core (disulfide)', 6, 6.8e-3, [(3, 75.1, 2.1e-6)]),
    (217, 223, 'a3 C', 5, 5.3e-3, [(1, 37.0, 1.1e-4)]),
]


def dG(kint, kslow):
    return RT * np.log(kint / kslow)


def seg_table(kint_133_143=9e-4):
    """one row per (segment, slow phase). kint for 133-143 resolved by d1: 9e-4 (as printed) is correct."""
    rows = []
    for lo, hi, lab, nex, ki, phases in TABLE1:
        if lo == 133 and hi == 143: ki = kint_133_143
        for j, (ns, amp, ks) in enumerate(phases):
            g = dG(ki, ks); K = np.exp(-g / RT)
            name = f'{lo}-{hi}' + (f'.s{j+1}' if len(phases) > 1 else '')
            rows.append(dict(seg=name, mouse=f'{lo}-{hi}', seq_lo=lo - OFF, seq_hi=min(hi, 223) - OFF, label=lab,
                             n_slow=ns, kint=ki, kslow=ks, dG=g, K=K, p_open=K / (1 + K)))
    return pd.DataFrame(rows)


def load(ens):
    z = np.load(FEAT + f'features_{ens}.npz')
    top = json.load(open(FEAT + f'topology_{ens}.json'))['topologies']
    res = np.array([t['residues'][0] for t in top])
    return res, np.asarray(z['heavy_contacts'], float), np.asarray(z['acceptor_contacts'], float), np.asarray(z['k_ints'], float)


def frame_features(ens):
    e = C.ENS[ens]
    df = pd.read_csv(CLUS + f'{e}/{e}_all_frames_features.csv').sort_values('local_frame_index').reset_index(drop=True)
    assert (df.cluster_label.values == C.assignments(ens)).all()
    return df


def pmap(res):
    return C.pep_map_norm(res, SEGS, 2)


def assign(ens):
    return C.assignments(ens)


def pfact_vec(res):
    return np.array([C.PFACT.get(int(r), -1.0) for r in res])


# ---- scaled BV model (same definition as pop/p1 'scale|matched'; recomputed here and checked) ----------
def scaled_bv(ens):
    res, h, a, k = load(ens)
    pf = pfact_vec(res); ok = pf > 0
    Lm = (BC0 * h + BH0 * a).mean(1)
    s = np.median(pf[ok]) / np.median(Lm[ok])
    return dict(s=s, bc=s * BC0, bh=s * BH0)


def shifted_bv_delta(ens):
    res, h, a, k = load(ens)
    pf = pfact_vec(res); ok = pf > 0
    return float(np.median(pf[ok] - (BC0 * h + BH0 * a).mean(1)[ok]))


def lnpf(ens, model='scale'):
    res, h, a, k = load(ens)
    if model == 'scale':
        m = scaled_bv(ens); return m['bc'] * h + m['bh'] * a
    if model == 'shift':
        return BC0 * h + BH0 * a + shifted_bv_delta(ens)
    return BC0 * h + BH0 * a


def frame_uptake(L, k, M, w):
    D = 1 - np.exp(-(k[:, None] * np.exp(-L))[:, :, None] * TIMES[None, None])
    return np.einsum('pr,rft,f->pt', M, D, w)


def frame_uptake_F(L, k, M):
    """per-frame peptide uptake (F, NP, NT) -- linear in w"""
    D = 1 - np.exp(-(k[:, None] * np.exp(-L))[:, :, None] * TIMES[None, None])
    return np.einsum('pr,rft->fpt', M, D)


def rate_uptake(L, k, M, w):
    kbar = (k[:, None] * np.exp(-L)) @ w
    return M @ (1 - np.exp(-kbar[:, None] * TIMES[None]))


PRED = {'frame_uptake': frame_uptake, 'rate': rate_uptake}


def scores(pred):
    r = DEXP - pred
    return dict(mse=float(np.mean(r ** 2)), wr2=float(np.mean(WTS * r ** 2)), chi2_pool=float(np.mean((r / SIG_POOL) ** 2)),
                maxz=float(np.abs(r / SIG_POOL).max()), mean_res=float(r.mean()))


def state_pops(a, w):
    w = np.asarray(w, float); w = w / w.sum()
    return np.array([w[a == i].sum() for i in range(6)])


def recovery_vs(P6, T6):
    """100(1-sqrt(JSD base 2)) of state populations P6 vs target T6 (same as repo calculate_recovery_percentage)."""
    P = np.asarray(P6, float); P = P / P.sum(); Q = np.asarray(T6, float); Q = Q / Q.sum(); Mm = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(P > 0, P * np.log2(P / Mm), 0).sum(); b = np.where(Q > 0, Q * np.log2(Q / Mm), 0).sum()
    return float(100 * (1 - np.sqrt(max(0.5 * (a + b), 0))))


def recovery(a, w):
    return C.recovery(a, w)


def ess(w):
    w = np.asarray(w, float); w = w / w.sum()
    return float(1.0 / np.sum(w ** 2) / len(w))

# ---- Table 2 (HDX-NMR slow-exchanging residues, mouse numbering) -> seq = mouse - 122 ----------------
SLOW_RES_MOUSE = {
    '127-132': [128], '133-143': [138], '133-148': [138, 147, 148], '149-153': [149, 150, 151, 152, 153],
    '154-167': [154, 156, 159, 160, 161, 163], '155-162': [159, 160, 161], '182-196': [185],
    '197-204': [202, 203, 204], '197-206': [202, 203, 204], '204-224': [207, 209, 211, 213, 216, 218, 219],
    '205-212': [207, 209, 211], '217-223': [219, 222, 223],   # Tyr224 is beyond the construct (seq 102)
}
SLOW_RES = {k: [m - OFF for m in v] for k, v in SLOW_RES_MOUSE.items()}


def read_kint(path):
    d = np.loadtxt(path, comments='#'); return dict(zip(d[:, 0].astype(int), d[:, 1]))

# ---- restraint segment sets and alternative global targets -----------------------------------------
S9 = ['127-132', '133-143', '133-148', '149-153', '154-167', '155-162', '182-196', '197-204', '197-206']   # brief's usable set
S11 = S9 + ['205-212', '217-223']        # + in-construct segments the brief excluded on the wrong (-104) numbering
PUF1_TYPE = ['127-132', '133-148', '197-204', '197-206', '217-223']   # PUF1 / m_op~0.4 (beta1, R147-Y148, alpha3 N- and C-term)
PUF2_TYPE = ['149-153', '154-167', '155-162']                          # PUF2: alpha1 C-term + beta2 detached
GLOBAL_SEG = '205-212'                                                 # paper: dG_op = global unfolding


def alt_targets():
    T = seg_table().set_index('seg')
    k1 = T.loc[PUF1_TYPE, 'K'].mean(); k2 = T.loc[PUF2_TYPE, 'K'].mean(); kU = T.loc[GLOBAL_SEG, 'K']
    alt1 = np.array([1 - k1 - k2, k1, k2, 0, 0, 0])
    alt2 = np.array([1 - (k1 - k2) - (k2 - kU) - kU, k1 - k2, k2 - kU, 0, kU, 0])   # hierarchical: PUF1-seg open in PUF1+PUF2+U
    return {'ORIG': TGT6, 'ALT1': alt1, 'ALT2': alt2}
