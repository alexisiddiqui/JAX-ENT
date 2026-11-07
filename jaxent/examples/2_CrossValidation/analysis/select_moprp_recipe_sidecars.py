#!/usr/bin/env python3
"""Selection + parity analysis of the MoPrP recipe campaign (fitting/jaxENT/run_moprp_recipe_sidecars.py).

Scores every saved state of every pipeline history (final, convergence checkpoints, running_best) on the
campaign's own inputs and applies the selgci rules per (ensemble, arm, fold):
  candidate set   = prior (S = 0) + pool states, restricted to ESS >= 10 % (selgci q3 convention)
  rule D          = fewest Sidak-flagged held-out |z| (alpha 0.05, n = held-out cells), ties by
                    F = 1/2 sum_heldout z^2 + KL(w || p0_arm)  (p0_arm = uniform | dG p0)
  rule C          = argmin held-out chi2/cell
Pools: 'main' = final + all convergence checkpoints (running_best excluded); 'final' = final states only;
'converged' = final + checkpoints at step >= 100; 'running_best' = running_best states only (reported
separately, the old sidecar candidate); 'numpy_ref' = exact optimum of the SAME objective (pipeline model:
k_int*exp(-delta), no lnPF floor; sigma floor; arm KL) by L-BFGS on the 500 logits (selgci q0 style), i.e. what
the pipeline should reach; 'published' = selgci q3 picks (clamped b' model, dG p0, KL(w||p0)), arm kl_dg_p0 only.
z uses the per-peptide pooled sigma (floor 0.0586); GCI T_n/S_n/Q_n on held-out z (diag Sigma, alpha/3 each).
Recovery = calculate_recovery_percentage vs the MoPrP target (Folded 97.1 / PUF1 2.4 / PUF2 0.5).

Per-run parity (final state vs numpy_ref at the same S): Delta ESS, Delta recovery, KL(w_pipe || w_ref),
relative objective gap (J(w_pipe) - J(w_ref)) / J(w_ref), J = chi2_train + KL_arm / S, and a replay of the
pipeline's saved train/val/total losses from the scored weights (checks the S <-> maxent_scaling mapping).

Usage (repo root, .venv):  python jaxent/examples/2_CrossValidation/analysis/select_moprp_recipe_sidecars.py
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FIT = EXPERIMENT/"fitting"/"jaxENT"
sys.path.insert(0, str(FIT))
import run_moprp_recipe_sidecars as R  # noqa: E402

DEFAULT_OUT = HERE/"_moprp_recipe_sidecars_20261106"
CLUSTERS = EXPERIMENT/"analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters"
ALPHA = 0.05
ALPHA_EACH = 1-0.95**(1/3)
ESS_MIN = 0.10
FMS = 1000.0  # template forward_model_scaling
STATES = ["Folded", "PUF1", "PUF2", "PUF3", "unfolded", "PUF2-like"]


def t_n(n, alpha=ALPHA):
    return float(norm.ppf(0.5+0.5*(1-alpha)**(1.0/n)))


def gci_triple(z):
    z = np.ravel(z)
    n = len(z)
    tT = float(norm.ppf(0.5+0.5*(1-ALPHA_EACH)**(1/n)))
    thrS, thrQ = float(norm.ppf(1-ALPHA_EACH/2)), float(norm.ppf(1-ALPHA_EACH))
    T, S, Q = float(np.abs(z).max()), float(z.sum()/np.sqrt(n)), float((np.sum(z**2)-n)/np.sqrt(2*n))
    code = ("" if T <= tT else "t")+("" if abs(S) <= thrS else "s")+("" if Q <= thrQ else "q")
    return dict(T_n=T, S_n=S, Q_n=Q, gci_code=(code or "P")+("o" if Q < -thrQ else ""))


def targets_and_mapping():
    from jaxent.examples.common.config import ExperimentConfig
    config = ExperimentConfig.from_yaml(str(EXPERIMENT/"config.yaml"))
    pops = json.loads((EXPERIMENT/"analysis/state_ratios.json").read_text())["fractional_populations"]
    target = {n: pops.get(k, {}).get("fraction", 0.) for n, k in
              [("Folded", "folded"), ("PUF1", "PUF1"), ("PUF2", "PUF2"), ("PUF3", "PUF3"), ("unfolded", "unfolded")]}
    for name in config.scoring.state_mapping.values():
        target.setdefault(name, 0.)
    return config, target


class Scorer:
    """Campaign inputs for one ensemble; NumPy rate forward verified against the pipeline forward pass."""

    def __init__(self, campaign, ens):
        import jax.numpy as jnp
        from jaxent.examples.common import loading
        from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
        from jaxent.examples.common.analysis.rescore_iso_coordinate_sigma import make_predict
        from jaxent.examples.common.optimization import create_data_loaders
        from jaxent.src.custom_types.key import m_key
        self.jnp, self._rec = jnp, calculate_recovery_percentage
        spec = R.build_specs(campaign, ensembles=(ens,), arms=("kl_dg_p0",), splits=(0,))[0]
        self.features, topology, self.model, _, _, full = R.context(spec)
        names = loading.load_sequence_residue_names(str(R.SEQUENCE))
        with contextlib.redirect_stdout(io.StringIO()):
            loader = create_data_loaders(full, full, full, self.features, topology, normalize_sparse_map_rows=True,
                                         residue_names=names)
            config, self.target = targets_and_mapping()
            clusters = loading.load_clustering_results(str(CLUSTERS), config.scoring.ensemble_clustering_map)
        self.state_mapping = config.scoring.state_mapping
        self.assign = np.asarray(clusters[ens]["cluster_assignments"])
        self.M = np.asarray(loader.train.residue_feature_ouput_mapping.todense(), float)
        self.y = np.asarray(loader.train.y_true, float)[..., 0]
        if [int(d.top.fragment_index) for d in full] != list(range(len(full))):
            raise ValueError("full dataset must be ordered by fragment_index")
        dexp, wts, segs, self.labels, self.T = R.source_data()
        np.testing.assert_allclose(self.y, dexp, atol=1e-6)
        self.sig_pep = R.sigma_pep(wts)
        self.sig = self.sig_pep[:, None]*np.ones((1, self.y.shape[1]))
        self.folds = R.block_folds(self.labels)
        h = np.asarray(self.features.heavy_contacts, float)
        a = np.asarray(self.features.acceptor_contacts, float)
        k = np.asarray(self.features.k_ints, float)  # already scaled by exp(-delta)
        self.Kr = k[:, None]*np.exp(-(R.BC*h+R.BH*a))
        self.F = self.Kr.shape[1]
        self.logp0 = np.asarray(np.load(spec.p0_path)["logp0"], float)
        self.p0 = np.exp(self.logp0)
        self.predict = make_predict(self.model.forward[m_key("HDX_peptide")], self.features)
        self.ens = ens
        for w in (np.full(self.F, 1/self.F), self.p0):
            gap = np.abs(self.pred(w)-self.pred_pipeline(w)).max()
            if gap > 1e-5:
                raise ValueError(f"NumPy and pipeline forward disagree by {gap}")

    def pred_pipeline(self, w):
        u = np.asarray(self.predict(self.jnp.asarray(w, dtype=self.jnp.float32), self.model.params), float)
        return self.M@u.T

    def pred(self, w):
        kbar = self.Kr@w
        return self.M@(1-np.exp(-kbar[:, None]*self.T[None]))

    def chi2_grad(self, w, rows):
        kbar = self.Kr@w
        U = 1-np.exp(-kbar[:, None]*self.T[None])
        P = self.M[rows]@U
        r = self.y[rows]-P
        iv = 1/self.sig[rows]**2
        N = r.size
        dP = -2*iv*r/N
        E = self.T[None]*np.exp(-kbar[:, None]*self.T[None])
        return float(np.sum(iv*r**2)/N), ((self.M[rows].T@dP)*E).sum(1)@self.Kr

    def recovery(self, w):
        return float(self._rec(self.assign, w/w.sum(), self.target, self.state_mapping))

    def pops(self, w):
        return {f"pop_{self.state_mapping.get(i, i)}": float(w[self.assign == i].sum()) for i in range(6)}

    def kls(self, w):
        lw = np.log(np.maximum(w, 1e-300))
        nz = w > 0
        return dict(KL_w_p0=float(np.sum(w[nz]*(lw[nz]-self.logp0[nz]))),
                    KL_w_u=float(np.sum(w[nz]*(lw[nz]+np.log(self.F)))),
                    KL_u_w=float(np.mean(-np.log(self.F)-lw)))

    def sigma_loss(self, P, rows):
        """hdx_uptake_sigma_MSE_loss on rows (trace-normalised diagonal precision)."""
        prec = 1/self.sig_pep[rows]**2
        W = prec*len(rows)/prec.sum()
        r = P[rows]-self.y[rows]
        return float(0.5*np.sum(W[:, None]*r**2)/r.size)

    def score(self, w, fold, arm, S=0.0, S_p=None):
        split, held, tr, va = self.folds[fold]
        P = self.pred(w)
        z = (self.y-P)/self.sig
        zv = z[va]
        tn = t_n(zv.size)
        fl = np.abs(zv) > tn
        kl = self.kls(w)
        kl_reg = kl["KL_w_p0"] if is_dg(arm) else kl["KL_u_w"]
        kl_sel = kl["KL_w_p0"] if is_dg(arm) else kl["KL_w_u"]
        chi2_tr = float(np.mean(z[tr]**2))
        out = dict(ESS=float(1/np.sum(w**2)/self.F), chi2_train=chi2_tr, ho_chi2=float(np.mean(zv**2)),
                   ho_n_cells=int(zv.size), ho_t_n=tn, ho_n_flags=int(fl.sum()),
                   ho_flagged=";".join(f"{self.labels[va[i]]}({int(fl[i].sum())})" for i in range(len(va)) if fl[i].any()),
                   F=float(0.5*np.sum(zv**2)+kl_sel), KL_reg=kl_reg, KL_sel=kl_sel,
                   J=float(chi2_tr+kl_reg/S) if S > 0 else np.nan, rec=self.recovery(w),
                   loss_train_replay=self.sigma_loss(P, tr), loss_val_replay=self.sigma_loss(P, va),
                   **kl, **self.pops(w), **gci_triple(zv))
        if S_p:
            # slot weights [1, 1/S_p] are renormalised to sum 1 by Simulation.initialise (models/core.py:83)
            kl_pipe = kl_reg if is_dg(arm) else kl_reg_smoothed(w, self.F)
            out["total_replay"] = FMS*(S_p*out["loss_train_replay"]+kl_pipe)/(1+S_p)
        return out


def is_dg(arm):
    """kl_dg_p0 and its diagnostic optimiser variants share the objective KL(w || p0_dG)."""
    return arm.startswith("kl_dg_p0")


def objective_arm(arm):
    return "kl_dg_p0" if is_dg(arm) else arm


def kl_reg_smoothed(w, F):
    """maxent_convexKL_loss as implemented: epsilon = 1e-10/F added to both, renormalised, KL(prior || w)."""
    eps = 1e-10/F
    q = (np.abs(w)+eps)/np.sum(np.abs(w)+eps)
    p = np.full(F, 1/F)+eps
    p /= p.sum()
    return float(np.sum(p*(np.log(p)-np.log(q))+(q-p)))


# ------------------------------------------------------------------ NumPy reference optimum (same objective)
def fit_ref(sc, rows, arm, S, x0):
    ref = sc.logp0 if arm == "kl_dg_p0" else np.full(sc.F, -np.log(sc.F))
    u = np.exp(ref)

    def f(eta):
        lw = eta-logsumexp(eta)
        w = np.exp(lw)
        c, g = sc.chi2_grad(w, rows)
        if arm == "kl_dg_p0":
            G = g+(lw-ref+1)/S
            return c+np.sum(w*(lw-ref))/S, w*(G-w@G)
        return c+np.sum(u*(ref-lw))/S, w*(g-w@g)+(w-u)/S
    r = minimize(f, x0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=50000, maxfun=100000, ftol=1e-15, gtol=1e-9, maxcor=30))
    lw = r.x-logsumexp(r.x)
    return np.exp(lw), r


def reference_sweeps(sc, arms, grid):
    out = {}
    for arm in arms:
        for k, (split, held, tr, va) in enumerate(sc.folds):
            x = sc.logp0.copy() if arm == "kl_dg_p0" else np.zeros(sc.F)
            x = np.maximum(x, -700.0)
            for i, S in enumerate(grid[arm]):
                w, r = fit_ref(sc, tr, arm, S, x)
                x = r.x
                out[(arm, k, i)] = (w, dict(ref_nit=int(r.nit), ref_fun=float(r.fun), ref_success=bool(r.success)))
    return out


def weights_of(state):
    from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
    w = np.asarray(validated_frame_weight_simplex(state.params.frame_weight_simplex), dtype=float)
    return w/w.sum()


def history_candidates(history):
    entries = [("final", None, history.states[-1])]
    entries += [("convergence", thr, s) for thr, s in history.iter_labeled_convergence_states()]
    if history.best_state is not None:
        entries.append(("running_best", None, history.best_state))
    seen = set()
    for kind, thr, state in entries:
        w = weights_of(state)
        key = (kind, int(state.step), hashlib.sha256(w.tobytes()).hexdigest())
        if key not in seen:
            seen.add(key)
            yield kind, thr, state, w


def select(g):
    """selgci q3 rules on one (ensemble, arm, fold, pool) candidate table (prior included)."""
    c = g[g.ESS >= ESS_MIN]
    mf = c.ho_n_flags.min()
    fs = c[c.ho_n_flags == mf]
    return {"D": fs.F.idxmin(), "C": c.ho_chi2.idxmin()}, int(mf), len(fs), len(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", type=Path, default=R.DEFAULT_OUTPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ensembles", default=",".join(R.ENSEMBLES))
    ap.add_argument("--tables-only", action="store_true", help="re-run selection/tables/figures from saved CSVs")
    ap.add_argument("--reuse-refs", action="store_true",
                    help="load the NumPy reference optima from a previous run's numpy_ref_weights.npz (float32)")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        manifest = json.loads((args.campaign/"manifest.json").read_text())
        rd = lambda n: pd.read_csv(out/n) if (out/n).stat().st_size > 1 else pd.DataFrame()
        finish(out, rd("candidates.csv"), rd("numpy_reference_sweep.csv"), rd("parity_final_vs_numpy_ref.csv"),
               rd("conv_check_20000.csv"), rd("timing.csv"), manifest, time.time())
        return
    from jaxent.src.utils.hdf import load_optimization_history_from_file
    manifest = json.loads((args.campaign/"manifest.json").read_text())
    grid = {arm: [float(s) for s in R.S_GRID[arm]] for arm in R.ARMS}
    t_start = time.time()
    rows, refrows, parity, timing, conv = [], [], [], [], []
    WREF = {}
    for ens in args.ensembles.split(","):
        sc = Scorer(args.campaign, ens)
        t0 = time.time()
        base_arms = sorted({objective_arm(a) for a in R.ARMS})
        if args.reuse_refs:
            old = np.load(out/"numpy_ref_weights.npz")
            oldcsv = pd.read_csv(out/"numpy_reference_sweep.csv")
            oldcsv = oldcsv[(oldcsv.ensemble == ens) & (oldcsv.kind == "numpy_ref")].set_index(["arm", "split_idx", "s_index"])
            refs = {}
            for a in base_arms:
                for k in range(6):
                    for i in range(len(grid[a])):
                        w = np.asarray(old[f"{ens}|{a}|{k}|{i}"], float)
                        o = oldcsv.loc[(a, k, i)]
                        refs[(a, k, i)] = (w/w.sum(), dict(ref_nit=int(o.ref_nit), ref_fun=float(o.ref_fun),
                                                          ref_success=bool(o.ref_success)))
        else:
            refs = reference_sweeps(sc, base_arms, grid)
        print(f"[{ens}] NumPy reference sweeps ({len(refs)} optima) in {time.time()-t0:.0f}s", flush=True)
        specs = R.build_specs(args.campaign, ensembles=(ens,))
        conv_specs = R.build_specs(args.campaign, ensembles=(ens,), splits=(0,), n_steps=R.CONV_CHECK_STEPS,
                                   s_subset=R.conv_check_subset())
        fpgap = 0.0
        for arm in R.ARMS:
            prior = sc.p0 if is_dg(arm) else np.full(sc.F, 1/sc.F)
            for k in range(6):
                base = dict(ensemble=ens, arm=arm, split_idx=k, split=sc.folds[k][0], heldout=sc.folds[k][1])
                r = sc.score(prior, k, arm)
                rows.append(dict(base, **r, S=0.0, s_index=-1, kind="prior", step=0, threshold=np.nan, run_id="prior"))
                refrows.append(dict(base, **r, S=0.0, s_index=-1, kind="prior"))
                for i, S in enumerate(grid[arm]):
                    w_ref, info = refs[(objective_arm(arm), k, i)]
                    rr = sc.score(w_ref, k, arm, S)
                    refrows.append(dict(base, **rr, **info, S=S, s_index=i, kind="numpy_ref"))
                    WREF[f"{ens}|{arm}|{k}|{i}"] = w_ref.astype(np.float32)
            for spec in [s for s in specs if s.arm == arm]+[s for s in conv_specs if s.arm == arm]:
                if not R.completed(spec):
                    print(f"  missing {spec.run_id}", flush=True)
                    continue
                cfg = json.loads(spec.config_path.read_text())
                timing.append(dict(ensemble=ens, arm=arm, n_steps=spec.n_steps, run_id=spec.run_id,
                                   fit_seconds=cfg["moprp_recipe"]["fit_seconds"],
                                   wall_time_seconds=cfg["runtime"]["wall_time_seconds"]))
                hist = load_optimization_history_from_file(str(spec.history_path))
                w_ref = refs[(objective_arm(arm), spec.split_idx, spec.s_index)][0]
                base = dict(ensemble=ens, arm=arm, split_idx=spec.split_idx, split=sc.folds[spec.split_idx][0],
                            heldout=sc.folds[spec.split_idx][1], S=spec.S, s_index=spec.s_index,
                            maxent_scaling=spec.maxent_scaling, n_steps=spec.n_steps, run_id=spec.run_id,
                            final_step=int(hist.states[-1].step))
                rref = sc.score(w_ref, spec.split_idx, arm, spec.S)
                for kind, thr, state, w in history_candidates(hist):
                    r = sc.score(w, spec.split_idx, arm, spec.S, spec.maxent_scaling)
                    saved = state.losses
                    r.update(saved_train=float(np.asarray(saved.train_losses[0])), saved_val=float(np.asarray(saved.val_losses[0])),
                             saved_reg=float(np.asarray(saved.train_losses[1])), saved_total=float(np.asarray(saved.total_train_loss)))
                    r["replay_rel_err"] = max(abs(r["loss_train_replay"]-r["saved_train"])/abs(r["saved_train"]),
                                              abs(r["loss_val_replay"]-r["saved_val"])/abs(r["saved_val"]),
                                              abs(r["total_replay"]-r["saved_total"])/abs(r["saved_total"]))
                    lw, lr_ = np.log(np.maximum(w, 1e-300)), np.log(np.maximum(w_ref, 1e-300))
                    r.update(KL_pipe_ref=float(np.sum(w*(lw-lr_))), maxabs_dw=float(np.abs(w-w_ref).max()),
                             J_ref=rref["J"], J_rel_gap=(r["J"]-rref["J"])/abs(rref["J"]), dESS=r["ESS"]-rref["ESS"],
                             drec=r["rec"]-rref["rec"], ref_ESS=rref["ESS"], ref_rec=rref["rec"])
                    if kind == "final" and spec.split_idx == 0 and spec.s_index % 8 == 0:
                        fpgap = max(fpgap, float(np.abs(sc.pred(w)-sc.pred_pipeline(w)).max()))
                    row = dict(base, **r, kind=kind, step=int(state.step), threshold=thr if thr is not None else np.nan)
                    if spec.n_steps == 5000:
                        rows.append(row)
                    else:
                        conv.append(row)
                    if kind == "final" and spec.n_steps == 5000:
                        parity.append(row)
        print(f"[{ens}] scored; max |NumPy - pipeline forward| on checked finals {fpgap:.1e}", flush=True)
    cand = pd.DataFrame(rows)
    refs_df = pd.DataFrame(refrows)
    cand.to_csv(out/"candidates.csv", index=False)
    refs_df.to_csv(out/"numpy_reference_sweep.csv", index=False)
    pd.DataFrame(parity).to_csv(out/"parity_final_vs_numpy_ref.csv", index=False)
    pd.DataFrame(conv).to_csv(out/"conv_check_20000.csv", index=False)
    pd.DataFrame(timing).to_csv(out/"timing.csv", index=False)
    np.savez_compressed(out/"numpy_ref_weights.npz", **WREF)
    finish(out, cand, refs_df, pd.DataFrame(parity), pd.DataFrame(conv), pd.DataFrame(timing), manifest, t_start)


def finish(out, cand, refs_df, parity, conv, timing, manifest, t_start):
    # ------------------------------------------------------------------ selection
    pools = {"main": lambda g: g.kind.isin(["prior", "final", "convergence"]),
             "final": lambda g: g.kind.isin(["prior", "final"]),
             "converged": lambda g: (g.kind.isin(["prior", "final"]) | ((g.kind == "convergence") & (g.step >= 100))),
             "running_best": lambda g: g.kind.isin(["prior", "running_best"])}
    sel = []
    for (ens, arm, k), g in cand.groupby(["ensemble", "arm", "split_idx"]):
        for pool, fn in pools.items():
            gg = g[fn(g)]
            picks, mf, nmf, nc = select(gg)
            for rule, i in picks.items():
                sel.append(dict(gg.loc[i].to_dict(), pool=pool, rule=rule, min_flags=mf, n_minflag=nmf, n_cand=nc))
        gr = refs_df[(refs_df.ensemble == ens) & (refs_df.arm == arm) & (refs_df.split_idx == k)]
        picks, mf, nmf, nc = select(gr)
        for rule, i in picks.items():
            sel.append(dict(gr.loc[i].to_dict(), pool="numpy_ref", rule=rule, min_flags=mf, n_minflag=nmf, n_cand=nc))
        gp = g[g.kind == "prior"].iloc[0].to_dict()
        sel.append(dict(gp, pool="prior", rule="prior"))
    pub = pd.read_csv(R.STUDY/"selgci"/"q3_selected.csv")
    pub = pub[(pub.system == "MoPrP") & (pub.sigma == "floor")]
    rmap = {"D_flags_then_F": "D", "C_minchi2": "C"}
    for _, r in pub[pub.rule.isin(list(rmap)) & pub.ensemble.isin(cand.ensemble.unique())].iterrows():
        k = [i for i, f in enumerate(R.block_folds(R.source_data()[3])) if f[0] == r.split and f[1] == r.heldout][0]
        sel.append(dict(ensemble=r.ensemble, arm="kl_dg_p0", split_idx=k, split=r.split, heldout=r.heldout, S=r.S,
                        ESS=r.ESS, rec=r.rec, ho_chi2=r.ho_chi2_cell, ho_n_flags=r.ho_n_flags, ho_n_cells=r.ho_n_cells,
                        F=r.F, chi2_train=r.chi2_train, pool="published_clamped", rule=rmap[r.rule], kind="study"))
    sel = pd.DataFrame(sel)
    sel.to_csv(out/"selection.csv", index=False)
    summ = (sel.groupby(["ensemble", "arm", "pool", "rule"])
            .agg(rec_mean=("rec", "mean"), rec_sd=("rec", "std"), rec_min=("rec", "min"), ESS_mean=("ESS", "mean"),
                 ESS_sd=("ESS", "std"), S_median=("S", "median"), ho_chi2_mean=("ho_chi2", "mean"),
                 flags_mean=("ho_n_flags", "mean"), n=("rec", "size"),
                 n_rec_ge95=("rec", lambda x: int((x >= 95).sum())), n_ESS_gt50=("ESS", lambda x: int((x > .5).sum())))
            .reset_index())
    summ.to_csv(out/"summary.csv", index=False)
    write_tables(out, cand, refs_df, parity, conv, sel, summ, timing, manifest, time.time()-t_start)
    make_figures(out, cand, refs_df, sel)
    print(f"analysis wall {time.time()-t_start:.0f}s -> {out}", flush=True)


def fmt(x, f=".1f"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, f)


def write_tables(out, cand, refs, par, conv, sel, summ, timing, manifest, wall):
    L = []
    L += ["## Mean ± SD recovery over the 6 block folds", "",
          "| ensemble | arm | pool | rule | recovery % (min) | ESS % | S median | held-out chi2/cell | flags | rec ≥ 95 | ESS > 50 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["prior", "main", "final", "converged", "running_best", "numpy_ref", "published_clamped"]
    for (ens, arm), g in summ.groupby(["ensemble", "arm"]):
        for pool in order:
            for rule in ("D", "C", "prior"):
                r = g[(g.pool == pool) & (g.rule == rule)]
                if r.empty:
                    continue
                r = r.iloc[0]
                L.append(f"| {ens} | {arm} | {pool} | {rule} | {r.rec_mean:.1f} ± {fmt(r.rec_sd)} ({r.rec_min:.1f}) | "
                         f"{100*r.ESS_mean:.0f} ± {fmt(100*r.ESS_sd, '.0f')} | {r.S_median:.3g} | {r.ho_chi2_mean:.2f} | "
                         f"{r.flags_mean:.1f} | {int(r.n_rec_ge95)}/{int(r.n)} | {int(r.n_ESS_gt50)}/{int(r.n)} |")
    L += ["", "## Per-fold picks (rule D; pool main unless stated)", "",
          "Cells: S · kind@step · ESS % · rec % · held-out chi2/cell · flags/n · GCI T/S/Q (code).", "",
          "| ensemble | arm | fold (held-out) | prior rec | D main | C main | D final-only | running_best D | numpy_ref D | published D |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    def cell(r, gci=True):
        if r is None:
            return "–"
        s = f"{r.S:.3g} · {r.get('kind', '')}@{fmt(r.get('step', np.nan), '.0f')} · {100*r.ESS:.0f} · {r.rec:.1f} · {r.ho_chi2:.2f} · {int(r.ho_n_flags)}/{int(r.ho_n_cells)}"
        if gci and isinstance(r.get("gci_code", None), str):
            s += f" · {r.T_n:.1f}/{r.S_n:+.1f}/{r.Q_n:.1f} ({r.gci_code})"
        return s
    for (ens, arm, k), g in sel.groupby(["ensemble", "arm", "split_idx"]):
        get = lambda pool, rule: (g[(g.pool == pool) & (g.rule == rule)].iloc[0] if not g[(g.pool == pool) & (g.rule == rule)].empty else None)
        pr = get("prior", "prior")
        L.append(f"| {ens} | {arm} | {pr.split} ({pr.heldout}) | {pr.rec:.1f} | {cell(get('main', 'D'))} | {cell(get('main', 'C'), False)} | "
                 f"{cell(get('final', 'D'), False)} | {cell(get('running_best', 'D'), False)} | {cell(get('numpy_ref', 'D'), False)} | "
                 f"{cell(get('published_clamped', 'D'), False)} |")
    if not par.empty:
        L += ["", "## Parity: pipeline final state (5000 steps) vs NumPy optimum of the same objective, per S", "",
              "| ensemble | arm | ESS band (ref) | runs | median rel J gap | max rel J gap | median KL(w_pipe‖w_ref) | median |ΔESS| | max |ΔESS| | median |Δrec| | max |Δrec| | runs within (|Δrec|<2, |ΔESS|<0.10, gap<1 %) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        par = par.copy()
        par["band"] = pd.cut(par.ref_ESS, [0, .1, .2, .5, .8, 1.01], labels=["<10%", "10-20%", "20-50%", "50-80%", ">80%"])
        for (ens, arm, band), g in par.groupby(["ensemble", "arm", "band"], observed=True):
            ok = ((g.drec.abs() < 2) & (g.dESS.abs() < .1) & (g.J_rel_gap < .01)).sum()
            L.append(f"| {ens} | {arm} | {band} | {len(g)} | {g.J_rel_gap.median():.2e} | {g.J_rel_gap.max():.2e} | "
                     f"{g.KL_pipe_ref.median():.2e} | {g.dESS.abs().median():.3f} | {g.dESS.abs().max():.3f} | "
                     f"{g.drec.abs().median():.2f} | {g.drec.abs().max():.2f} | {ok}/{len(g)} |")
        L += ["", f"Loss replay (all scored states): max relative error train/val/total = {cand.replay_rel_err.max():.1e} "
              f"(train/val sigma loss and 1000*(S_p L + KL)/(1 + S_p) recomputed from the weights; slot weights renormalised "
              f"by Simulation.initialise).", ""]
    if not conv.empty:
        L += ["## Convergence check: 20000 vs 5000 steps (split 0, final states)", "",
              "| ensemble | arm | S | ESS ref | ESS 5k | ESS 20k | rec ref | rec 5k | rec 20k | rel J gap 5k | rel J gap 20k |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        c20 = conv[conv.kind == "final"]
        for _, r in c20.iterrows():
            r5 = par[(par.ensemble == r.ensemble) & (par.arm == r.arm) & (par.split_idx == r.split_idx) & (par.s_index == r.s_index)]
            r5 = r5.iloc[0] if not r5.empty else None
            L.append(f"| {r.ensemble} | {r.arm} | {r.S:.3g} | {100*r.ref_ESS:.1f} | {fmt(100*r5.ESS if r5 is not None else np.nan)} | "
                     f"{100*r.ESS:.1f} | {r.ref_rec:.1f} | {fmt(r5.rec if r5 is not None else np.nan)} | {r.rec:.1f} | "
                     f"{fmt(r5.J_rel_gap if r5 is not None else np.nan, '.2e')} | {r.J_rel_gap:.2e} |")
    if not timing.empty:
        t = timing.groupby(["ensemble", "arm", "n_steps"]).agg(runs=("fit_seconds", "size"), median=("fit_seconds", "median"),
                                                                total=("fit_seconds", "sum")).reset_index()
        L += ["", "## Compute time (run_optimization fit seconds, 2 CPU threads per run)", "",
              "| ensemble | arm | n_steps | runs | median s | total s |", "|---|---|---|---|---|---|"]
        L += [f"| {r.ensemble} | {r.arm} | {r.n_steps} | {r.runs} | {r['median']:.1f} | {r.total:.0f} |" for _, r in t.iterrows()]
        walls = manifest.get("fit_wall_seconds", [])
        L += ["", f"Campaign fit-phase wall time(s): {[round(w['seconds']) for w in walls]} s; analysis wall {wall:.0f} s.", ""]
    (out/"results_tables.md").write_text("\n".join(L)+"\n")


def make_figures(out, cand, refs, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ens in cand.ensemble.unique():
        fig, ax = plt.subplots(2, len(R.ARMS), figsize=(6.5*len(R.ARMS), 8), sharex=True)
        for j, arm in enumerate(R.ARMS):
            for k in range(6):
                col = plt.cm.tab10(k)
                f = cand[(cand.ensemble == ens) & (cand.arm == arm) & (cand.split_idx == k) & (cand.kind == "final")].sort_values("S")
                r = refs[(refs.ensemble == ens) & (refs.arm == arm) & (refs.split_idx == k) & (refs.kind == "numpy_ref")].sort_values("S")
                for i, m in enumerate(("rec", "ESS")):
                    sc = 100 if m == "ESS" else 1
                    ax[i, j].plot(r.S, sc*r[m], "-", color=col, lw=1, label=f"fold {k} NumPy ref" if i == 0 else None)
                    ax[i, j].plot(f.S, sc*f[m], "o", color=col, ms=3, label=f"fold {k} pipeline final" if i == 0 else None)
                d = sel[(sel.ensemble == ens) & (sel.arm == arm) & (sel.split_idx == k) & (sel.pool == "main") & (sel.rule == "D")]
                if not d.empty:
                    ax[0, j].plot(d.S, d.rec, "*", color=col, ms=12, mec="k")
            ax[0, j].set_title(f"{ens} · {arm}")
            ax[1, j].set_xlabel("S (study scale: chi2/cell + KL/S)")
            for i in range(2):
                ax[i, j].set_xscale("log")
            ax[0, j].set_ylabel("recovery %")
            ax[1, j].set_ylabel("ESS %")
        ax[0, 0].legend(fontsize=6, ncol=2)
        fig.suptitle("lines: NumPy optimum of the pipeline objective; dots: pipeline final (5000 steps); ★ rule-D pick (main pool)")
        fig.tight_layout()
        fig.savefig(out/f"F_{ens}_sweep.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
