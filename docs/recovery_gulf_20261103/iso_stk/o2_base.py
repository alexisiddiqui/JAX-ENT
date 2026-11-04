"""[o2] gamma = 0 (MaxEnt / KL-only) baseline on ISO, = stk s5 / selgci q2 sweep: self-consistent ISO_BI / ISO_TRI, residue
target + noise sigma in {0.01, 0.02} x seeds {0,1,2}, uniform p0, exact dual (selgci q0.fit_dual_linear) warm-started along
ISO_SGRID, stop at ESS < 5%, 3 contiguous residue-block 2-fold splits. Scoring: lam = 0 (Sigma = sigma^2 I; selgci) and
lam_geo diag (stk union C_shape; secondary, for T1). Weights saved (float32) for the gamma scale c (o3/o4).
Reproduction check: lam = 0 rule picks vs stk s5_selected (lam_key '0', diag).
-> o2_sweep.csv, o2_weights.npz, o2.log"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from multiprocessing import Pool
import i0_common as C
Q, S0 = C.Q, C.S0

PR = dict(np.load(os.path.join(C.RG, 'stk', 's1_primary.npz')))


def run(job):
    which, sigma, seed = job
    m = Q.ISO(which, sigma, seed); N = m.nF; logp0 = np.full(N, -np.log(N))
    lg = float(PR[f'ISO_{which}|lam_geo']); CS = PR[f'ISO_{which}|Cw'] / lg
    rows, W = [], {}
    for split, held, tr, va in Q.iso_splits(len(m.res)):
        t0 = time.time(); sc = S0.Scorer(m, va, CS, [lg]); lam = None
        for S in S0.ISO_SGRID:
            w, lw, inf, lam_new = Q.fit_dual_linear(m, tr, logp0, S, lam0=lam, gtol=1e-12)
            if S > 0: lam = lam_new
            KL = Q.kl(w, lw, logp0); P = m.pops(w); nf, F, chi = C.heldout_diag(m, w, va, KL); o = sc.score(m, w, KL)[lg]
            rows.append(dict(system=f'ISO_{which.upper()}', sigma=sigma, seed=seed, split=split, heldout=held, S=S, ESS=Q.ess(w), KL=KL,
                             stat=inf['stat'], rec=m.recovery(w), pop_open=P[0], pop_closed=P[1], pop_unassigned=P[2],
                             n_flags=nf, F=F, chi2_cell=chi, n_flags_lg=o['n_flags'], F_lg=o['F_diag']))
            W[f'{which}|{sigma}|{seed}|{split}|{held}|{S:.6g}'] = w.astype(np.float32)
            if Q.ess(w) < 0.05: break
        print(f'ISO_{which} sigma={sigma} seed={seed} {split} {held}: t={time.time()-t0:.0f}s', flush=True)
    return rows, W


if __name__ == '__main__':
    jobs = [(w, s, sd) for w in ('tri', 'bi') for s in (0.01, 0.02) for sd in range(3)]
    with Pool(5) as pool:
        res = pool.map(run, jobs, chunksize=1)
    df = pd.DataFrame([r for rr, _ in res for r in rr]); df.to_csv(C.OUT + 'o2_sweep.csv', index=False)
    np.savez_compressed(C.OUT + 'o2_weights.npz', **{k: v for _, ww in res for k, v in ww.items()})
    sel = []
    for key, g in df.groupby(['system', 'sigma', 'seed', 'split', 'heldout'], sort=False):
        for r in C.select(g): sel.append(r)
    S = pd.DataFrame(sel)
    s5 = pd.read_csv(os.path.join(C.RG, 'stk', 's5_selected.csv'))
    s5 = s5[(s5.lam_key.astype(str) == '0') & (s5.version == 'diag')]
    mg = s5.merge(S, on=['system', 'sigma', 'seed', 'split', 'heldout', 'rule'], suffixes=('_s5', ''))
    print(f'[check] gamma=0 lam=0 picks vs stk s5: {len(mg)} picks, S identical {np.isclose(mg.S_s5, mg.S).sum()}/{len(mg)}, '
          f'max |drec| {np.abs(mg.rec_s5 - mg.rec).max():.2e}', flush=True)
    d = S[S.rule == 'D_flags_then_F']
    for (sy, sg), g in d.groupby(['system', 'sigma']):
        print(f'{sy} sigma={sg}: rule D rec {g.rec.mean():.2f} ± {g.rec.std():.2f} (min {g.rec.min():.1f}) ESS {100 * g.ESS.mean():.0f}% '
              f'#>=95 {int((g.rec >= 95).sum())}/{len(g)}  [seed 0 only: {g[g.seed == 0].rec.mean():.2f}, #>=95 {int((g[g.seed == 0].rec >= 95).sum())}/6]', flush=True)
