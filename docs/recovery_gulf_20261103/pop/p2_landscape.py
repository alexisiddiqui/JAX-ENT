"""T2: state-level population landscape (uniform weights within state) under median-scaled BV.
-> p2_grid_{ens}_{model}_{mode}.npz, p2_best.csv, p2_ref.csv (target/uniform), p2_localopt.csv, p2.log
Decoy states: MSAss PUF3+unfolded; Filtered has no PUF3/unfolded frames -> its only decoy is PUF2-like (1 frame)."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import minimize
import p0_common as P

MODELS = json.load(open(P.OUT + 'p1_models.json'))
USE = ['scale|matched', 'shift|matched', 'shift_clamp0|matched', 'affine_iqr|matched', 'none|-']
CORE = ['Folded', 'PUF1', 'PUF2']
SC = ['mse', 'wr2', 'chi2_pool', 'maxz', 'spearman']
CONSTR = [None, 0.0, 0.01, 0.02, 0.05]


def lnpf(key, h, a):
    m = MODELS[key]; L0 = P.BC0 * h + P.BH0 * a
    return {'none': lambda: L0, 'scale': lambda: m['bc'] * h + m['bh'] * a, 'shift': lambda: L0 + m['delta'], 'shift_clamp0': lambda: np.maximum(L0 + m['delta'], 0.0),
            'affine': lambda: m['medpf'] + m['a'] * (L0 - m['medL'])}[m['kind']]()


def comps(n, d):
    """all non-negative integer vectors of length d summing to n"""
    out = []
    for c in itertools.combinations(range(n + d - 1), d - 1):
        b = np.diff(np.concatenate([[-1], c, [n + d - 1]])) - 1
        out.append(b)
    return np.array(out)


def build_grid(d, ndec):
    g = [comps(20, d) / 20.0]                                   # 5% full simplex
    nc = d - ndec
    core = {m: comps(m, nc) for m in range(95, 101)}
    dec = comps(100, 1) if ndec == 0 else None
    blocks = []
    for j in range(0, 6):                                       # 1% grid, decoy sum <= 5%
        dv = comps(j, ndec) if ndec > 0 else np.zeros((1, 0), int)
        cv = core[100 - j]
        blocks.append(np.hstack([np.repeat(cv, len(dv), 0), np.tile(dv, (len(cv), 1))]) / 100.0)
    g += blocks
    hi = comps(20, d).astype(float); hi[:, 0] += 80; g.append(hi / 100.0)   # 1%, F >= 80%
    f = comps(40, nc).astype(float); f[:, 0] += 360                          # 0.25% near target (F>=90%, decoys 0)
    g.append(np.hstack([f / 400.0, np.zeros((len(f), ndec))]))
    G = np.unique(np.round(np.vstack(g), 6), axis=0)
    return G


def state_tables(ens, key):
    res, h, a, k = P.load(ens); M = P.pmap(res); asg = P.assign(ens)
    L = lnpf(key, h, a)
    present = [s for s in range(6) if (asg == s).any()]
    names = [P.STATES[s] for s in present]
    order = [names.index(c) for c in CORE] + [i for i, n in enumerate(names) if n not in CORE]
    present = [present[i] for i in order]; names = [names[i] for i in order]
    rates = k[:, None] * np.exp(-L)
    D = 1 - np.exp(-rates[:, :, None] * P.TIMES[None, None])
    Gs = np.stack([np.einsum('pr,rft->pt', M, D[:, asg == s]) / (asg == s).sum() for s in present])   # S,NP,NT
    Ks = np.stack([rates[:, asg == s].mean(1) for s in present])                                       # S,res
    return dict(M=M, Gs=Gs, Ks=Ks, present=present, names=names, asg=asg)


def predict(tab, Pm, mode):
    if mode == 'frame_uptake':
        return np.einsum('ns,spt->npt', Pm, tab['Gs'])
    kbar = Pm @ tab['Ks']
    return np.einsum('pr,nrt->npt', tab['M'], 1 - np.exp(-kbar[:, :, None] * P.TIMES[None, None]))


def to6(tab, Pm):
    out = np.zeros((len(Pm), 6)); out[:, tab['present']] = Pm; return out


def evaluate(tab, Pm, mode, chunk=20000):
    acc = {k: [] for k in SC + ['mean_res', 'frac_exp_gt_pred']}
    for i in range(0, len(Pm), chunk):
        s = P.scores(predict(tab, Pm[i:i + chunk], mode))
        for k in acc: acc[k].append(s[k])
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out['recovery'] = P.recovery_p6(to6(tab, Pm))
    return out


def sval(tab, p, mode, sc):
    return float(P.scores(predict(tab, p[None], mode))[sc][0])


def refine(tab, p0, mode, sc, c, ndec):
    """continuous SLSQP polish of smooth scores on the simplex, decoy mass <= c"""
    d = len(p0); sgn = P.SCORE_DIR[sc]
    cons = [{'type': 'eq', 'fun': lambda p: p.sum() - 1}]
    if c is not None and ndec > 0:
        cons.append({'type': 'ineq', 'fun': lambda p: c - p[3:].sum()})
    r = minimize(lambda p: sgn * sval(tab, np.clip(p, 0, None), mode, sc), p0, method='SLSQP',
                 bounds=[(0, 1)] * d, constraints=cons, options=dict(ftol=1e-12, maxiter=300))
    p = np.clip(r.x, 0, None); p /= p.sum()
    if c is not None and ndec > 0 and p[3:].sum() > c + 1e-9: return p0
    return p if sgn * sval(tab, p, mode, sc) <= sgn * sval(tab, p0, mode, sc) else p0


def local_opt(tab, p, mode, sc, allow_decoy, eps=0.0025):
    """is p a local optimum? try moving eps of mass between every ordered pair of states"""
    sgn = P.SCORE_DIR[sc]; f0 = sgn * sval(tab, p, mode, sc); d = len(p); best = (0.0, None)
    for i in range(d):
        for j in range(d):
            if i == j or p[j] < eps: continue
            if not allow_decoy and i >= 3: continue
            q = p.copy(); q[i] += eps; q[j] -= eps
            df = sgn * sval(tab, q, mode, sc) - f0
            if df < best[0]: best = (df, f'{tab["names"][j]}->{tab["names"][i]}')
    return best[1] is None, best[1], -best[0]


if __name__ == '__main__':
    best_rows, ref_rows, lo_rows = [], [], []
    for ens in P.ENSS:
        asg = P.assign(ens)
        for key in USE:
            tab = state_tables(ens, f'{ens}|{key}')
            names = tab['names']; d = len(names); ndec = d - 3
            G = build_grid(d, ndec)
            cnt = np.array([(asg == s).sum() for s in tab['present']]); uni = cnt / cnt.sum()
            tgt = np.array([P.C.TARGET[n] for n in names]); tgt /= tgt.sum()
            for mode in P.PRED:
                ev = evaluate(tab, G, mode)
                np.savez_compressed(P.OUT + f'p2_grid_{ens}_{key.split("|")[0]}_{mode}.npz', P=G, names=names, **ev)
                ref = evaluate(tab, np.vstack([tgt, uni]), mode)
                for i, lab in enumerate(['target', 'uniform']):
                    row = dict(ensemble=ens, model=key, mode=mode, point=lab, **{n: round(v, 4) for n, v in zip(names, [tgt, uni][i])})
                    row.update({k: float(ref[k][i]) for k in SC + ['recovery', 'mean_res']})
                    for sc in SC:   # percentile of the point among unconstrained grid and among decoy=0 grid
                        sg = P.SCORE_DIR[sc]
                        row[f'pct_{sc}'] = float(np.mean(sg * ev[sc] < sg * ref[sc][i]) * 100)
                        m0 = G[:, 3:].sum(1) <= 1e-9
                        row[f'pct0_{sc}'] = float(np.mean(sg * ev[sc][m0] < sg * ref[sc][i]) * 100)
                    ref_rows.append(row)
                for sc in SC:
                    for allow in [True, False]:
                        is_lo, mv, gain = local_opt(tab, tgt, mode, sc, allow)
                        lo_rows.append(dict(ensemble=ens, model=key, mode=mode, score=sc, decoys_allowed=allow,
                                            target_is_local_opt=is_lo, best_move=mv, gain=gain))
                for sc in SC:
                    sg = P.SCORE_DIR[sc]
                    for c in CONSTR:
                        m = np.ones(len(G), bool) if c is None else G[:, 3:].sum(1) <= c + 1e-9
                        idx = np.where(m)[0][np.argmin(sg * ev[sc][m])]
                        p = G[idx].copy()
                        pr = refine(tab, p, mode, sc, c, ndec) if sc in ('mse', 'wr2', 'chi2_pool') else p
                        evr = evaluate(tab, pr[None], mode)
                        for lab, pp, e in [('grid', p, {k: v[idx] for k, v in ev.items()}), ('refined', pr, {k: v[0] for k, v in evr.items()})]:
                            row = dict(ensemble=ens, model=key, mode=mode, score=sc, constraint='none' if c is None else f'decoy<={c}',
                                       how=lab, **{n: round(float(v), 4) for n, v in zip(names, pp)})
                            row.update({k: float(e[k]) for k in SC + ['recovery', 'mean_res']})
                            row['folded_majority_puf12_pos'] = bool(pp[0] > 0.5 and pp[1] > 1e-3 and pp[2] > 1e-3)
                            best_rows.append(row)
                print(ens, key, mode, 'grid', len(G), 'done', flush=True)
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 2000)
    B = pd.DataFrame(best_rows); B.to_csv(P.OUT + 'p2_best.csv', index=False)
    R = pd.DataFrame(ref_rows); R.to_csv(P.OUT + 'p2_ref.csv', index=False)
    Lo = pd.DataFrame(lo_rows); Lo.to_csv(P.OUT + 'p2_localopt.csv', index=False)
    print(R.round(4).to_string())
    print(Lo.to_string())
    print(B[B.how == 'refined'].round(4).to_string())
