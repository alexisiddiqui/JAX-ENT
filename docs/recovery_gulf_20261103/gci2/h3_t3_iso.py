"""T3: representation-error term from ISO_BI (re-uses gci/g6 machinery: 874 fit frames, residue-level self-consistent
frame_uptake target, 40/60 open/closed truth). Residuals: r_true = target - F w_true ; r_nnls = target - F w_NNLS (unit weights).
(i) multiplier on C_ens: lam_tr = |r|^2/tr(C), lam_mom (off-diag moment), lam_size = smallest lam on grid with triple
rejection <= 5% (1000 noise draws, sigma in .005,.01,.02,.05), each expressed x N.
(ii) diagonal floor sigma_rep^2 = mean r^2 ; size of Sigma = (sigma^2+sigma_rep^2) I + C/N.
-> h3_t3.csv, h3_size.csv"""
import sys, json; sys.path.insert(0, '/tmp/claude-1000/-home-alexi-Documents-JAX-ENT/a5e22bb9-240b-4720-9dcd-ac74d472f964/scratchpad/gci2')
import numpy as np, pandas as pd
import g0_common as G
B = 'jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_self_consistent_iso/'
z = np.load(B + 'fit_features/features_iso_bi.npz')
res = np.array([t['residues'][0] for t in json.load(open(B + 'fit_features/topology_iso_bi.json'))['topologies']])
segs = np.loadtxt(B + 'target_frame_uptake/residue/target_segs.txt', dtype=int)
tgt = np.loadtxt(B + 'target_frame_uptake/residue/target_dfrac.dat')
tms = np.array(open(B + 'target_frame_uptake/residue/target_dfrac.dat').readline().split()[1:6], float)
M = G.C.pep_map_norm(res, segs, 1)
L = 0.35 * np.asarray(z['heavy_contacts'], float) + 2.0 * np.asarray(z['acceptor_contacts'], float)
F = G.C.frame_pep_uptake(L, np.asarray(z['k_ints'], float), M, tms)
lab = pd.read_csv('jaxent/examples/1_IsoValidation_OMass/data/_clustering_results/cluster_assignments_ISO_BI.csv')
lab = lab[lab.ensemble == 'ISO_BI'].cluster_assignment.values
w_true = np.where(lab == 0, 0.4 / np.sum(lab == 0), 0.6 / np.sum(lab == 1))
Nf = F.shape[0]; y0 = tgt.ravel(); n = len(y0)
w_nn = G.C.nnls_simplex(F, tgt.reshape(F.shape[1:]), np.ones(F.shape[1:]))
Ff = F.reshape(Nf, -1); Cm = np.cov(Ff, rowvar=False)
print('NNLS open fraction', w_nn[lab == 0].sum(), 'ESS', 1 / np.sum(w_nn ** 2))
rng = np.random.default_rng(1)
LAMS = np.array([0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]) / Nf
rows, srows = [], []


def rej(r, Sig, sig, nd=1000):
    st = G.sig_stats(Sig); eps = rng.standard_normal((nd, n)) * sig
    Z = (r[None] + eps) / st['sd']
    Tn = np.abs(Z).max(1); S = Z.sum(1) / np.sqrt(n); Q = ((Z ** 2).sum(1) - n) / np.sqrt(2 * n)
    return float(((Tn > st['t_n']) | (np.abs(S) > st['thrS']) | (Q > st['thrQ'])).mean())


for name, w in (('true_40_60', w_true), ('nnls_best', w_nn)):
    r = y0 - Ff.T @ w
    lam_tr = r @ r / np.trace(Cm); lam_mom = G.lam_moment(r, Cm)
    v = np.linalg.eigh(Cm)[1][:, -5:]; frac_top5 = np.sum((v.T @ r) ** 2) / (r @ r)
    s_rep = np.sqrt(np.mean(r ** 2))
    row = dict(residual=name, rms=s_rep, maxabs=np.abs(r).max(), mean=r.mean(), lam_tr_xN=lam_tr * Nf, lam_mom_xN=lam_mom * Nf,
               frac_r2_in_top5_Cmodes=frac_top5, sigma_rep=s_rep, sigma_rep_p95=np.quantile(np.abs(r), 0.95),
               open_frac=w[lab == 0].sum(), recovery_open=w[lab == 0].sum())
    for sig in (0.005, 0.01, 0.02, 0.05):
        rr = [rej(r, sig ** 2 * np.eye(n) + l * Cm, sig) for l in LAMS]
        ok = [l * Nf for l, x in zip(LAMS, rr) if x <= 0.05]
        row[f'lam_size_xN_s{sig}'] = min(ok) if ok else np.nan
        rf = rej(r, (sig ** 2 + s_rep ** 2) * np.eye(n) + Cm / Nf, sig)
        rf2 = rej(r, (sig ** 2 + 2 * s_rep ** 2) * np.eye(n) + Cm / Nf, sig)
        row[f'rej_floor_s{sig}'] = rf; row[f'rej_2floor_s{sig}'] = rf2
        for l, x in zip(LAMS, rr): srows.append(dict(residual=name, sigma=sig, lam_xN=l * Nf, rej_any=x))
        print(name, sig, dict(zip(np.round(LAMS * Nf, 1), rr)), 'floor', rf, rf2, flush=True)
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv(G.OUT + 'h3_t3.csv', index=False); pd.DataFrame(srows).to_csv(G.OUT + 'h3_size.csv', index=False)
pd.set_option('display.width', 250); print(df.T.to_string())
