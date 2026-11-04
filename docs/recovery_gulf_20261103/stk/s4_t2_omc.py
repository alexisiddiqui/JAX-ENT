"""[s4] T2: OMC regulariser with the self-tuning Laplacian L_st (s1 primary kernel: k=20, mean, union, alpha=1/2).
Same data / model / floored sigma / splits / S grid as s2 (= selgci q1).
 'kl'   : min chi2_train + KL(w||p0)/S + gamma_abs w^T L_st w, gamma in {0.1, 0.3, 1, 3} (gamma = 0 is the s2 sweep);
          lap/l0.fit_omc (500-logit L-BFGS, warm-started along S). gamma_abs = gamma * c, c = median over folds of
          (KL/S)/(w^T L_st w) at the gamma = 0 rule-D pick under the T1 two-sided diag z at lambda_geo (upt), S > 0 picks only
          (fallback: the S ~ 0.1 grid point) -> gamma = 1 makes the OMC term equal KL/S at the typical pick (lap/l3 convention).
 'pomc' : no KL: min chi2_train + (c/S) w^T L_st w, each S cold-started from logits = log p0 (p0 = starting point only).
 'pomcu': same, cold-started from uniform logits (p0 plays no role in the fit; KL(w||p0) still enters rule B/D's F).
 S = 0 row = p0 in every variant (the reference). Scored at lam in {0, lambda_geo} (upt C_shape), diag flags, F_diag, F_full.
-> s4_sweep.csv, s4_gamma_scale.csv, s4_weights.npz, s4.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from multiprocessing import Pool
import s0_common as S0
Q, D, L0 = S0.Q, S0.D, S0.L0

PR = dict(np.load(S0.OUT + 's1_primary.npz'))
GAMMAS = [0.1, 0.3, 1.0, 3.0]


def gamma_scale():
    sel = pd.read_csv(S0.OUT + 's3_selected.csv'); W = dict(np.load(S0.OUT + 's2_weights.npz'))
    out = {}
    for ens in D.ENSS:
        Lst = PR[f'{ens}|Lst']
        d = sel[(sel.ensemble == ens) & (sel['mode'] == 'upt') & (sel.lam_key == 'lg') & (sel.version == 'diag') & (sel.rule == 'D_flags_then_F')]
        ratios = []
        for _, r in d.iterrows():
            if r.S > 0:
                w = W[f'{ens}|{r.split}|{r.heldout}|{r.S:.6g}']; ratios.append(r.KL / r.S / float(w @ Lst @ w))
        if ratios: c = float(np.median(ratios)); how = f'D picks ({len(ratios)}/6 folds with S>0)'
        else:
            Sg = S0.SGRID[1:][np.argmin(np.abs(S0.SGRID[1:] - 0.1))]
            sw = pd.read_csv(S0.OUT + 's2_sweep.csv'); sw = sw[(sw.ensemble == ens) & np.isclose(sw.S, Sg) & (sw['mode'] == 'upt') & (sw.lam_key == '0')]
            c = float(np.median([r.KL / r.S / float(W[f'{ens}|{r.split}|{r.heldout}|{r.S:.6g}'] @ Lst @ W[f'{ens}|{r.split}|{r.heldout}|{r.S:.6g}']) for _, r in sw.iterrows()]))
            how = 'fallback S~0.1'
        out[ens] = (c, how)
    return out


def run(job):
    ens, var, gamma, c = job
    logp0, _ = Q.moprp_p0(ens); m = Q.MoPrP(ens, floor=True); Lst = PR[f'{ens}|Lst']
    lg = float(PR[f'{ens}|lam_geo|upt']); CS = PR[f'{ens}|Cw|upt'] / lg
    rows, W = [], {}
    for split, held, tr, va in Q.moprp_splits():
        t0 = time.time(); sc = S0.Scorer(m, va, CS, [0.0, lg]); eta = None
        for S in S0.SGRID:
            if S == 0:
                w, lw, inf = np.exp(logp0), logp0.copy(), dict(conv=True, nit=0, stat=0.0)
            elif var == 'kl':
                w, lw, inf = L0.fit_omc(m, tr, logp0, S, Lst, gamma * c, eta0=eta); eta = inf['x']
            else:
                eta0 = logp0.copy() if var == 'pomc' else np.zeros(m.nF)
                w, lw, inf = S0.fit_pomc(m, tr, eta0, S, Lst, c)
            KL = Q.kl(w, lw, logp0); P = m.pops(w); wLw = float(w @ Lst @ w); ctr = m.chi2_grad(w, tr)[0]
            base = dict(ensemble=ens, variant=var, gamma=gamma, c=c, split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL, wLw=wLw,
                        chi2_train=ctr, conv=inf['conv'], nit=inf['nit'], stat=inf['stat'], rec=m.recovery(w),
                        **{f'pop_{s}': P[i] for i, s in enumerate(Q.MO_STATES)})
            o = sc.score(m, w, KL)
            for lk, lam in (('0', 0.0), ('lg', lg)):
                rows.append(dict(**base, lam_key=lk, lam=lam, n_flags=o[lam]['n_flags'], F_diag=o[lam]['F_diag'], F_full=o[lam]['F_full'],
                                 chi2_cell=o[lam]['chi2_cell']))
            W[f'{ens}|{var}|{gamma:g}|{split}|{held}|{S:.6g}'] = w
        g = pd.DataFrame(rows[-2 * len(S0.SGRID)::2])
        print(f'{ens} {var} gamma={gamma:g} (abs {gamma * c:.3g}) {split} held={held}: ESS {g.ESS.iloc[1]:.2f}->{g.ESS.min():.3f}, '
              f'rec {g.rec.iloc[1]:.1f}..{g.rec.max():.1f} (min {g.rec.min():.1f}), max stat {g.stat.max():.1e}, t={time.time()-t0:.0f}s', flush=True)
    return rows, W


if __name__ == '__main__':
    gs = gamma_scale()
    pd.DataFrame([dict(ensemble=e, c=c, how=h) for e, (c, h) in gs.items()]).to_csv(S0.OUT + 's4_gamma_scale.csv', index=False)
    print(gs, flush=True)
    jobs = [(e, 'pomc', 1.0, gs[e][0]) for e in D.ENSS] + [(e, 'pomcu', 1.0, gs[e][0]) for e in D.ENSS] + \
           [(e, 'kl', g, gs[e][0]) for e in D.ENSS for g in GAMMAS]
    with Pool(5) as pool:
        res = pool.map(run, jobs, chunksize=1)
    rows = [r for rr, _ in res for r in rr]; W = {k: v for _, ww in res for k, v in ww.items()}
    pd.DataFrame(rows).to_csv(S0.OUT + 's4_sweep.csv', index=False); np.savez_compressed(S0.OUT + 's4_weights.npz', **W)
