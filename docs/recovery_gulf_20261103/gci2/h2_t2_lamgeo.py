"""T2: lambda_geo = 1/N_eff from ensemble geometry in predicted-uptake space.
F (500 x 210) per-frame uptake; d_ij Euclidean; sigma_i = mean dist to k NN (k in 10,20), scaled by c in {0.5,1,2};
K = exp(-d^2/(c^2 sigma_i sigma_j)); alpha=1/2: Kt = K/(q_i q_j)^0.5; P0 = row-normalised Kt; lazy P=(I+P0)/2.
Symmetric form S = Dd^-1/2 Kt Dd^-1/2 (eig mu_k, phi_k); G = I + sum_{k>=2, slowest m} (g(mu_k)-1) phi_k phi_k^T,
g = (1+mu')/(1-mu') with mu' = (1+mu)/2 (lazy) [also non-lazy g=(1+mu)/(1-mu) reported].
C_mean = F~^T G F~ / N^2 ; inflation = tr(C_mean)/tr(C_ens/N); lambda_geo = inflation/N; N_eff = N/inflation.
Also along top eigvec of C_ens. NB: lazy walk on a complete graph (mu=0) gives g=3 -> 'lazy floor' inflation 3.
Structural brackets: cluster-robust (sandwich) variance of the mean with rule-based labels and k-medoids labels on F.
-> h2_t2.csv, h2_struct.csv"""
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import json
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
import g0_common as G

T1P = json.load(open(G.OUT + 'h1_params.json'))


def Fmat(ens, which):
    res, h, a, k = G.load(ens); M = G.pmap(res)
    if which == 'default':
        L = G.BC0 * h + G.BH0 * a
    else:  # T1 hi-conf params
        bc, bh = T1P[f'{ens}|{which}']; L = bc * h + bh * a
    return G.frame_uptake(L, k, M).reshape(500, -1)


def walk_inflation(F, k, c, ms=(10, 50, None)):
    N = len(F); Ft = F - F.mean(0)
    Dm = cdist(F, F); srt = np.sort(Dm, 1)
    sig = srt[:, 1:k + 1].mean(1) * c
    K = np.exp(-Dm ** 2 / np.outer(sig, sig))
    q = K.sum(1); Kt = K / np.sqrt(np.outer(q, q))
    d = Kt.sum(1); S = Kt / np.sqrt(np.outer(d, d))
    mu, phi = np.linalg.eigh(S); order = np.argsort(-mu); mu, phi = mu[order], phi[:, order]
    Cens = np.cov(F, rowvar=False); ev, V = np.linalg.eigh(Cens); v1 = V[:, -1]
    base_tr = np.trace(Cens) / N; base_v1 = v1 @ Cens @ v1 / N
    # projections of centred F onto modes (k>=2)
    Y = phi.T @ Ft  # (N, 210)
    out = []
    for lazy in (True, False):
        mul = (1 + mu) / 2 if lazy else mu
        g = (1 + mul) / np.clip(1 - mul, 1e-12, None)
        for m in ms:
            sel = np.arange(1, N) if m is None else np.arange(1, m + 1)
            gg = np.ones(N); gg[sel] = g[sel]; gg[0] = 1.0
            # C_mean = Ft^T (I + sum (g-1) phi phi^T) Ft / N^2 = Y^T diag(gg) Y / N^2 (phi orthonormal, complete)
            Cm = (Y * gg[:, None]).T @ Y / N ** 2
            infl = np.trace(Cm) / base_tr; infl_v1 = (v1 @ Cm @ v1) / base_v1
            out.append(dict(k=k, c=c, lazy=lazy, m='all' if m is None else m, mu2=mu[1], mu3=mu[2], mu11=mu[10],
                            n_mu_gt_0p9=int(np.sum(mu[1:] > 0.9)), infl_tr=infl, infl_v1=infl_v1,
                            N_eff=N / infl, lam_geo=infl / N, N_eff_v1=N / infl_v1, lam_geo_v1=infl_v1 / N,
                            frac_var_top_mode_slow10=float(np.sum(Y[1:11] ** 2) / np.sum(Y[1:] ** 2))))
    return out


def sandwich(F, lab):
    N = len(F); Ft = F - F.mean(0); Cens = np.cov(F, rowvar=False)
    ev, V = np.linalg.eigh(Cens); v1 = V[:, -1]
    Cm = np.zeros_like(Cens)
    for c in np.unique(lab):
        s = Ft[lab == c].sum(0); Cm += np.outer(s, s)
    Cm /= N ** 2
    infl = np.trace(Cm) / (np.trace(Cens) / N); infl1 = (v1 @ Cm @ v1) / (v1 @ Cens @ v1 / N)
    return infl, infl1


def kmedoids(Dm, k, rng, n_init=10, it=100):
    best = None
    for _ in range(n_init):
        med = [rng.integers(len(Dm))]
        for _ in range(k - 1):
            dmin = Dm[:, med].min(1); med.append(rng.choice(len(Dm), p=dmin ** 2 / np.sum(dmin ** 2)))
        med = np.array(med)
        for _ in range(it):
            lab = np.argmin(Dm[:, med], 1)
            new = np.array([np.where(lab == j)[0][np.argmin(Dm[np.ix_(lab == j, lab == j)].sum(1))] for j in range(k)])
            if set(new) == set(med): break
            med = new
        lab = np.argmin(Dm[:, med], 1); cost = Dm[np.arange(len(Dm)), med[lab]].sum()
        if best is None or cost < best[0]: best = (cost, lab)
    return best[1]


rows, srows = [], []
rng = np.random.default_rng(0)
for ens in G.ENSS:
    for which in ('default', f'90|wr2', f'85|wr2'):
        F = Fmat(ens, which)
        for k in (10, 20):
            for c in (0.5, 1.0, 2.0):
                for r in walk_inflation(F, k, c):
                    rows.append(dict(ensemble=ens, params=which, **r))
        if which == 'default':
            a = G.assign(ens)
            infl, infl1 = sandwich(F, a)
            srows.append(dict(ensemble=ens, labels='rule-based', n_clusters=len(np.unique(a)), infl_tr=infl, infl_v1=infl1,
                              N_eff=500 / infl, lam=infl / 500, N_eff_v1=500 / infl1))
            Dm = cdist(F, F)
            for kk in (3, 5, 8, 12, 20):
                lab = kmedoids(Dm, kk, rng)
                infl, infl1 = sandwich(F, lab)
                srows.append(dict(ensemble=ens, labels=f'kmedoids', n_clusters=kk, sizes=','.join(map(str, sorted(np.bincount(lab))[::-1])),
                                  infl_tr=infl, infl_v1=infl1, N_eff=500 / infl, lam=infl / 500, N_eff_v1=500 / infl1))
        print(ens, which, flush=True)
df = pd.DataFrame(rows); df.to_csv(G.OUT + 'h2_t2.csv', index=False)
sd = pd.DataFrame(srows); sd.to_csv(G.OUT + 'h2_struct.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df[df.lazy].round(4).to_string())
print(df[~df.lazy & (df.m == 'all')].round(4).to_string())
print(sd.round(4).to_string())
