#!/usr/bin/env python3
"""MoPrP recipe campaign through the native jaxent pipeline (both ensembles, block folds, S grid, two KL arms).

Purpose: run the validated MoPrP recipe (docs/recovery_gulf_20261103/selgci; docs/averaging_mode_soft_regression.md
"Recovery-gulf diagnosis and validated MoPrP recipe") through first-party jaxent fitting, so the study's NumPy/SciPy
results can be reproduced -- or the pipeline gap isolated -- with the pipeline's own optimiser, histories and
checkpoints.  Modelled on run_moprp_policy_sidecars.py (same template opt_config, worker/taskset execution,
row-normalised maps) and 4_aSyn/fitting/run_asyn_uptake_sidecars.py (sigma loss, block folds, k_int scaling).
The policy-sidecar runner cannot express this campaign unchanged: it hard-codes eye-MSE (no sigma), the
sequence_cluster/spatial splits, uniform-start MaxEnt only and the unscaled features.

Pipeline pieces used unchanged: common.optimization.run_optimization (compiled), the BV_model standard uptake
forward with frame_averaging_mode="rate" and bc 0.35 / bh 2.0 FIXED, loss "hdx_uptake_sigma_MSE_loss" with a
diagonal per-peptide precision passed via OptimizationConfig.covariance_matrix_path (trace-normalised by the
loader), the rate-sidecar template opt_config (Adam, lr 1.0, EMA 0.5, forward_model_scaling 1000, convergence
thresholds 1e-1..1e-8), HDF5 histories, normalize_sparse_map_rows=True with the proline check.

Recipe inputs (prepare phase, frozen in <output>/):
  * features: _featurise_a101_k109 (97 residues, A101 included, 109-construct k_int in min^-1) copied per ensemble
    with k_int scaled by exp(-delta), delta = pop/p1_models.json 'shift_clamp0|matched' (-3.108 MSAss,
    -3.174 Filtered).  NOTE: this is lnPF + delta WITHOUT the study's floor max(lnPF + delta, 0) (21-22 % of
    residue x frame cells are clamped in the study); the analysis quantifies the effect with NumPy references.
  * data: all 14 peptides x 15 timepoints (moprp.dexp), peptide_trim 2, fragment_index = dexp column order.
  * sigma: per-peptide pooled sigma_p = max(sqrt(mean_t (1/weights)^2), 0.0586) (selgci q0 MoPrP floor=True);
    precision diag(1/sigma_p^2) over the 14 fragments.
  * folds: the study's three contiguous 2-fold block splits (q0_common.moprp_splits, 6 folds):
    cut45|60, cut31|32, cut75|75; train on one block, validate on the other.
  * p0 (arm kl_dg_p0): dg T2 'A|S11|exact' MaxEnt reference, log space (q0_common.moprp_p0), saved as npz.

Arms (frame regulariser slot):
  * maxent_uniform: the pipeline's standard maxent_convexKL_loss = KL(p0 || w) with p0 = uniform (the initial
    weights), start uniform.
  * kl_dg_p0: KL(w || p0) with the dG-derived p0 via run_optimization(frame_regularizer=(name,
    common.losses.maxent_reverse_KL_logp0_loss, log_p0, 1/S_p)), start at p0 (initial_frame_weights=p0).
  * kl_dg_p0_adam0.1 (DIAGNOSTIC, added after the first pass): identical to kl_dg_p0 except that the inner Adam
    step is 0.1 instead of the hard-coded learning_rate=1.0 of OptaxOptimizer (jaxent/src/opt/optimiser.py:131;
    the configured learning rate only pre-scales the gradient, to which Adam is invariant).  Emulated without
    touching jaxent/src by wrapping optax.adam inside the worker process (see _scale_inner_adam).

Strength mapping (derivation in _parity_20261106/REPORT.md): the pipeline minimises
  1000 * [ L_sigma(w) + KL/S_p ],  L_sigma = 1/2 * H * chi2_train(w),  H = n_train / sum_train 1/sigma_p^2,
and Simulation.initialise (jaxent/src/models/core.py:83) renormalises the slot weights [1, 1/S_p] to sum 1, so the
total is 1000/(1 + S_p) * [S_p L_sigma + KL] = 500 H S_p/(1 + S_p) * [chi2_train + KL / S] with S = H S_p / 2
<=>  S_p = 2 S / H (per fold): same minimiser as the study objective, positive constant factor only.
Runs are indexed by the study scale S; maxent_scaling = S_p is stored in each spec.

Usage (repo root, .venv):
  python jaxent/examples/2_CrossValidation/fitting/jaxENT/run_moprp_recipe_sidecars.py --phase prepare
  python .../run_moprp_recipe_sidecars.py --phase fit --jobs 4
  python .../run_moprp_recipe_sidecars.py --phase fit --conv-check --jobs 4     # 20000-step subset
Selection/analysis: jaxent/examples/2_CrossValidation/analysis/select_moprp_recipe_sidecars.py
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
REPO = EXPERIMENT.parents[2]
DATA = EXPERIMENT/"data"/"_MoPrP"
SEQUENCE = DATA/"moprp.seq"
FEATURES_SRC = HERE/"_featurise_a101_k109"
STUDY = REPO/"artifacts"/"recovery_gulf_20261103"          # read-only study code/inputs (gitignored)
TEMPLATE = (HERE/"_moprp_policy_sidecars_rate_fresh_20261023/sequence_cluster/"
            "MoPrP_rate_AF2_filtered_MSE_maxent_sequence_cluster_split000_maxent1000/"
            "MoPrP_rate_AF2_filtered_MSE_maxent_sequence_cluster_split000_maxent1000_config.json")
DEFAULT_OUTPUT = HERE/"_moprp_recipe_sidecars_20261106"
ENSEMBLES = ("AF2_MSAss", "AF2_filtered")
ARMS = ("maxent_uniform", "kl_dg_p0", "kl_dg_p0_adam0.1")
ADAM_DIAGNOSTIC_SCALE = 0.1
BC, BH = 0.35, 2.0
PEPTIDE_TRIM = 2
SIGMA_FLOOR = 0.0586
SPLIT_TYPE = "blocks"
# study scale S grid (chi2/cell + KL/S): the selgci q1 grid logspace(-3.5, 0.5, 33) for both arms (S = 0 is the
# prior itself and is scored without a fit).  maxent_uniform's KL(u||w) lets ESS fall faster (~1 % at S = 1).
S_GRID = {arm: tuple(float(x) for x in np.logspace(-3.5, 0.5, 33)) for arm in ARMS}
CONV_CHECK_STEPS = 20000


def block_folds(labels):
    """q0_common.moprp_splits(): (split, held-out label, train rows, val rows), 6 folds."""
    end = [int(lab.split("-")[1]) for lab in labels]
    blocks = {"cut45|60": ([i for i, e in enumerate(end) if e <= 45], "N 5-45", "C 60-101"),
              "cut31|32": ([i for i, e in enumerate(end) if e <= 31], "N 5-31", "C 32-101"),
              "cut75|75": ([i for i, e in enumerate(end) if e <= 75], "N 5-75", "C 75-101")}
    out = []
    for name, (a, la, lb) in blocks.items():
        b = [i for i in range(len(labels)) if i not in a]
        out.append((name, la, b, a))
        out.append((name, lb, a, b))
    return out


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def fitting_code_sha256():
    paths = list((REPO/"jaxent/src").rglob("*.py"))
    common = REPO/"jaxent/examples/common"
    paths += [common/n for n in ("optimization.py", "config.py", "loading.py", "losses.py", "uptake_models.py")]
    paths += [Path(__file__)]
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(set(paths))}


def git_head():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
        return dict(head=head, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(head="unknown", dirty=None)


def source_data():
    """dexp (14, 15), weights (14, 15) [1/sigma per cell], segments, labels, times (min)."""
    dexp = np.loadtxt(DATA/"moprp.dexp")[:, 1:].T
    wts = np.loadtxt(DATA/"moprp.weights")[:, 1:].T
    segs = np.loadtxt(DATA/"_output"/"MoPrP_segments.txt", dtype=int)
    labels = [f"{s}-{e}" for s, e in segs]
    times = np.loadtxt(DATA/"moprp.times")[1:]*60.0
    return dexp, wts, segs, labels, times


def sigma_pep(wts):
    return np.maximum(np.sqrt(np.mean((1.0/wts)**2, 1)), SIGMA_FLOOR)


def strength_for(S, train_rows, sig):
    """pipeline maxent_scaling S_p for study scale S on a fold (S_p = 2 S / H, H = n / sum 1/sigma^2)."""
    H = len(train_rows)/np.sum(1.0/sig[train_rows]**2)
    return 2.0*S/H


# ============================================================================ specs
@dataclass(frozen=True)
class RunSpec:
    ensemble: str
    arm: str
    split_idx: int
    s_index: int
    S: float              # study scale (chi2/cell + KL/S)
    maxent_scaling: float  # pipeline S_p (KL weight 1/S_p)
    n_steps: int
    output_dir: str
    frame_averaging_mode: str = "rate"
    normalize_sparse_map_rows: bool = True

    @property
    def run_id(self):
        tag = "" if self.n_steps == 5000 else f"_n{self.n_steps}"
        return (f"MoPrP_rate_{self.ensemble}_sigMSE_{self.arm}_{SPLIT_TYPE}_split{self.split_idx:03d}"
                f"_S{self.s_index:02d}_{self.S:.4g}{tag}")

    @property
    def run_dir(self):
        return Path(self.output_dir)/"runs"/self.ensemble/self.arm/f"split{self.split_idx:03d}"/self.run_id

    @property
    def history_path(self):
        return self.run_dir/f"{self.run_id}_results.hdf5"

    @property
    def config_path(self):
        return self.run_dir/f"{self.run_id}_config.json"

    @property
    def features_dir(self):
        return Path(self.output_dir)/"features"

    @property
    def datasplit_dir(self):
        return Path(self.output_dir)/"datasplits"

    @property
    def precision_path(self):
        return self.datasplit_dir/"precision_sigma_floor.npz"

    @property
    def p0_path(self):
        return Path(self.output_dir)/"p0"/f"logp0_{self.ensemble}.npz"


def build_specs(output_dir, ensembles=ENSEMBLES, arms=ARMS, splits=range(6), n_steps=5000, s_subset=None):
    dexp, wts, segs, labels, _ = source_data()
    sig = sigma_pep(wts)
    folds = block_folds(labels)
    specs = []
    for ens in ensembles:
        for arm in arms:
            for k in splits:
                tr = folds[k][2]
                for i, S in enumerate(S_GRID[arm]):
                    if s_subset is not None and (arm, i) not in s_subset:
                        continue
                    specs.append(RunSpec(ensemble=ens, arm=arm, split_idx=k, s_index=i, S=S,
                                         maxent_scaling=float(strength_for(S, tr, sig)), n_steps=n_steps,
                                         output_dir=str(output_dir)))
    if len({s.run_id for s in specs}) != len(specs):
        raise ValueError("duplicate run ids")
    return specs


# ============================================================================ prepare
def prepare(out):
    import dataclasses
    import jaxent.src.interfaces.topology as pt
    from jaxent.src.custom_types.datapoint import ExpD_Datapoint
    from jaxent.src.custom_types.HDX import HDX_peptide
    from jaxent.src.models.HDX.BV.features import BV_input_features
    out.mkdir(parents=True, exist_ok=True)
    dexp, wts, segs, labels, times = source_data()
    sig = sigma_pep(wts)
    folds = block_folds(labels)
    # --- datasplits (full + 6 block folds), fragment_index = dexp column
    ds = out/"datasplits"
    data = []
    for i, (s, e) in enumerate(segs):
        top = pt.TopologyFactory.from_range(chain="A", start=int(s), end=int(e), fragment_index=i, peptide=True,
                                            peptide_trim=PEPTIDE_TRIM, fragment_name="MoPrPCrossVal")
        if sorted(top.peptide_residues) != list(range(s+PEPTIDE_TRIM, e+1)):
            raise ValueError(f"peptide_trim mismatch for {s}-{e}")
        data.append(HDX_peptide(dfrac=[float(v) for v in dexp[i]], top=top))
    save = lambda items, stem: ExpD_Datapoint.save_list_to_files(items, json_path=str(stem)+"_topology.json",
                                                                  csv_path=str(stem)+"_dfrac.csv")
    ds.mkdir(parents=True, exist_ok=True)
    save(data, ds/"full_dataset")
    for k, (split, held, tr, va) in enumerate(folds):
        sd = ds/SPLIT_TYPE/f"split_{k:03d}"
        sd.mkdir(parents=True, exist_ok=True)
        save([data[i] for i in tr], sd/"train")
        save([data[i] for i in va], sd/"val")
    np.savez(ds/"precision_sigma_floor.npz", precision=np.diag(1.0/sig**2))
    write_json(ds/"folds.json", [dict(split_idx=k, split=s, heldout=h, train=[labels[i] for i in tr],
                                      val=[labels[i] for i in va], train_rows=tr, val_rows=va,
                                      H=float(len(tr)/np.sum(1/sig[tr]**2)))
                                 for k, (s, h, tr, va) in enumerate(folds)])
    # --- study cross-checks (read-only import of the selgci q0 code)
    sys.path.insert(0, str(STUDY/"selgci"))
    import q0_common as Q
    if [(a, b, list(c), list(d)) for a, b, c, d in Q.moprp_splits()] != [(a, b, list(c), list(d)) for a, b, c, d in folds]:
        raise ValueError("block folds differ from q0_common.moprp_splits")
    np.testing.assert_allclose(dexp, Q.D.DEXP, atol=0)
    np.testing.assert_allclose(times, Q.D.TIMES, rtol=1e-12)
    models = json.loads((STUDY/"pop"/"p1_models.json").read_text())
    infos = {}
    fdir = out/"features"
    fdir.mkdir(parents=True, exist_ok=True)
    (out/"p0").mkdir(parents=True, exist_ok=True)
    for ens in ENSEMBLES:
        delta = float(models[f"{ens}|shift_clamp0|matched"]["delta"])
        m = Q.MoPrP(ens, floor=True)
        np.testing.assert_allclose(sig, m.sig_pep, atol=0)
        feat = BV_input_features.load(str(FEATURES_SRC/f"features_{ens}.npz"))
        k = np.asarray(feat.k_ints, float)
        scaled = dataclasses.replace(feat, k_ints=np.asarray(k*np.exp(-delta), dtype=np.asarray(feat.k_ints).dtype))
        scaled.save(str(fdir/f"features_{ens}.npz"))
        (fdir/f"topology_{ens}.json").write_bytes((FEATURES_SRC/f"topology_{ens}.json").read_bytes())
        logp0, info = Q.moprp_p0(ens)
        np.savez(out/"p0"/f"logp0_{ens}.npz", logp0=logp0, p0=np.exp(logp0))
        h = np.asarray(feat.heavy_contacts, float); a = np.asarray(feat.acceptor_contacts, float)
        L0 = BC*h+BH*a
        infos[ens] = dict(delta=delta, frac_cells_clamped_in_study=float(np.mean(L0+delta < 0)),
                          p0_recovery=float(m.recovery(np.exp(logp0))), p0_ESS=float(Q.ess(np.exp(logp0))),
                          uniform_recovery=float(m.recovery(np.full(m.nF, 1/m.nF))),
                          p0_constraints=info["constraints"], p0_dropped=info["dropped"],
                          p0_n_zero=int(np.sum(np.exp(logp0) == 0)))
        print(f"[prepare] {ens}: delta {delta:.4f}, clamped cells in study {infos[ens]['frac_cells_clamped_in_study']:.3f}, "
              f"p0 rec {infos[ens]['p0_recovery']:.1f} ESS {infos[ens]['p0_ESS']:.2f}, uniform rec "
              f"{infos[ens]['uniform_recovery']:.1f}", flush=True)
    # --- sparse-map proline check on the copied features
    from jaxent.examples.common import loading
    from jaxent.src.data.splitting.sparse_map import check_unrepresented_residues
    names = loading.load_sequence_residue_names(str(SEQUENCE))
    missing = {}
    for ens in ENSEMBLES:
        _, topology = loading.load_features_and_topology(str(fdir), ens)
        full = HDX_peptide.load_list_from_files(json_path=str(ds/"full_dataset_topology.json"),
                                                csv_path=str(ds/"full_dataset_dfrac.csv"))
        miss = check_unrepresented_residues(topology, full, residue_names=names, allow_missing_residues=False)
        missing[ens] = {f"{full[i].top.residue_start}-{full[i].top.residue_end}": r for i, r in miss.items()}
    print(f"[prepare] peptide residues absent from features (must be prolines): {missing}", flush=True)
    return infos, missing, sig


# ============================================================================ fit
def configured_model():
    from jaxent.examples.common import loading
    from jaxent.examples.common.uptake_models import build_uptake_model
    from jaxent.src.custom_types.key import m_key
    times = loading.load_hdx_timepoints_minutes(DATA/"moprp.times")
    model = build_uptake_model("standard", times, kint_unit="min^-1")
    model.forward[m_key("HDX_peptide")].frame_averaging_mode = "rate"
    p = model.params
    if not (np.isclose(np.asarray(p.bv_bc).item(), BC) and np.isclose(np.asarray(p.bv_bh).item(), BH)):
        raise ValueError("BV defaults are not 0.35 / 2.0")
    return model


def context(spec):
    from jaxent.examples.common import loading
    with contextlib.redirect_stdout(io.StringIO()):
        features, topology = loading.load_features_and_topology(str(spec.features_dir), spec.ensemble)
        train, val, full, _ = loading.load_experimental_data("", str(spec.datasplit_dir), SPLIT_TYPE, spec.split_idx)
    model = configured_model()
    for d in (train, val, full):
        loading.validate_hdx_timepoint_count(d, np.asarray(model.params.timepoints), label=spec.ensemble)
    return features, topology, model, train, val, full


def _scale_inner_adam(scale):
    """Process-local: make optax.adam(learning_rate) use learning_rate*scale (emulates a configurable inner
    Adam step in OptaxOptimizer, whose chains call inject_hyperparams(optax.adam)(learning_rate=1.0))."""
    import optax
    base = optax.adam

    def adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, nesterov=False):
        return base(learning_rate*scale, b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype, nesterov=nesterov)
    optax.adam = adam


def run_fit(spec):
    import jax.numpy as jnp
    from jaxent.examples.common import loading
    from jaxent.examples.common.config import LossConfig, OptimizationConfig
    from jaxent.examples.common.losses import maxent_reverse_KL_logp0_loss
    from jaxent.examples.common.optimization import create_data_loaders, run_optimization
    from jaxent.src.custom_types.key import m_key
    features, topology, model, train, val, _ = context(spec)
    template = json.loads(TEMPLATE.read_text())
    if template["loss_config"]["optimize_bv_params"]:
        raise ValueError("template must have fixed BV parameters")
    options = OptimizationConfig(**{**template["opt_config"], "n_steps": spec.n_steps,
                                    "covariance_matrix_path": str(spec.precision_path)})
    loss = LossConfig(primary_loss="hdx_uptake_sigma_MSE_loss", maxent_scaling=spec.maxent_scaling,
                      optimize_bv_params=False)
    extra = {}
    if spec.arm == "kl_dg_p0_adam0.1":
        _scale_inner_adam(ADAM_DIAGNOSTIC_SCALE)
    if spec.arm.startswith("kl_dg_p0"):
        z = np.load(spec.p0_path)
        logp0 = np.asarray(z["logp0"], dtype=np.float64)
        p0 = np.exp(logp0 - np.max(logp0)); p0 /= p0.sum()
        extra = dict(frame_regularizer=("maxent_reverse_KL_logp0_loss", maxent_reverse_KL_logp0_loss,
                                        jnp.asarray(logp0, dtype=jnp.float32), 1.0/spec.maxent_scaling),
                     initial_frame_weights=jnp.asarray(p0, dtype=jnp.float32)/jnp.sum(jnp.asarray(p0, dtype=jnp.float32)))
    names = loading.load_sequence_residue_names(str(SEQUENCE))
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        loader = create_data_loaders(train+val, train, val, features, topology,
                                     normalize_sparse_map_rows=True, residue_names=names)
        run_optimization(train_data=train, val_data=val, prior_data=loader, features=features, feature_top=topology,
                         forward_model=model, model_parameters=model.params, convergence=options.convergence_rates,
                         loss_config=loss, opt_config=options, name=spec.run_id, output_dir=str(spec.run_dir),
                         execution_mode="compiled", normalize_sparse_map_rows=True, residue_names=names, **extra)
    config = json.loads(spec.config_path.read_text())
    config["effective_settings"]["frame_averaging_mode"] = model.forward[m_key("HDX_peptide")].frame_averaging_mode
    config["moprp_recipe"] = dict(spec=asdict(spec), training_loss="hdx_uptake_sigma_MSE_loss (diag per-peptide "
                                  "precision 1/sigma_p^2, trace-normalised by the loader)", kint_unit="min^-1",
                                  time_unit="min", bv_fixed=[BC, BH], regulariser_weight=1/spec.maxent_scaling,
                                  regulariser=("KL(w||p0_dG) maxent_reverse_KL_logp0_loss, start p0" if spec.arm.startswith("kl_dg_p0")
                                               else "maxent_convexKL_loss KL(uniform||w), start uniform"),
                                  inner_adam_step=ADAM_DIAGNOSTIC_SCALE if spec.arm == "kl_dg_p0_adam0.1" else 1.0,
                                  fit_seconds=time.time()-t0, code_sha256=digest(Path(__file__)))
    write_json(spec.config_path, config)
    print(f"Completed {spec.run_id} in {time.time()-t0:.0f}s", flush=True)


def completed(spec):
    if not spec.history_path.exists() or not spec.config_path.exists():
        return False
    try:
        stored = json.loads(spec.config_path.read_text()).get("moprp_recipe", {}).get("spec")
        return stored is not None and RunSpec(**stored) == spec
    except (OSError, ValueError, TypeError):
        return False


def execute(specs, args):
    pending = [s for s in specs if not completed(s)]
    cores = sorted(os.sched_getaffinity(0))
    if args.cores:
        cores = [int(c) for c in args.cores.split(",")]
    for name in ("logs", "specs"):
        (args.output_dir/name).mkdir(exist_ok=True)

    def queue(slot):
        failures = []
        for spec in pending[slot::args.jobs]:
            path = args.output_dir/"specs"/f"{spec.run_id}.json"
            write_json(path, asdict(spec))
            with (args.output_dir/"logs"/f"{spec.run_id}.log").open("w") as log:
                cmd = ["taskset", "-c", ",".join(map(str, cores[2*slot:2*slot+2])), sys.executable,
                       str(Path(__file__).resolve()), "--worker-spec", str(path)]
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}).returncode
            print(f"Fit {spec.run_id}: exit {code}", flush=True)
            if code:
                failures.append(spec.run_id)
        return failures
    print(f"Fitting {len(pending)} of {len(specs)} runs with {args.jobs} jobs", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = [n for g in pool.map(queue, range(args.jobs)) for n in g]
    print(f"Fit phase wall time {time.time()-t0:.0f}s", flush=True)
    if failures:
        raise RuntimeError(f"failed fits: {failures}")


def conv_check_subset():
    """20000-step convergence check: S indices near ESS ~90/50/20 % per arm (fixed after the 5000-step pass)."""
    return {("kl_dg_p0", 20), ("kl_dg_p0", 25), ("kl_dg_p0", 28),
            ("maxent_uniform", 19), ("maxent_uniform", 21), ("maxent_uniform", 22)}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("prepare", "fit"), default="prepare")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cores", help="explicit comma-separated CPU list (2 per job)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--ensembles", default=",".join(ENSEMBLES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--splits", default="0,1,2,3,4,5")
    ap.add_argument("--conv-check", action="store_true", help=f"{CONV_CHECK_STEPS}-step runs on split 0, 3 S per arm")
    ap.add_argument("--limit", type=int, help="fit only the first N pending specs (timing tests)")
    ap.add_argument("--worker-spec", type=Path)
    args = ap.parse_args()
    if args.worker_spec:
        spec = RunSpec(**json.loads(args.worker_spec.read_text()))
        if not completed(spec):
            run_fit(spec)
        return
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpath = args.output_dir/"manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    if args.phase == "prepare":
        infos, missing, sig = prepare(args.output_dir)
        inputs = [DATA/n for n in ("moprp.dexp", "moprp.weights", "moprp.times", "moprp.seq")]
        inputs += [DATA/"_output"/"MoPrP_segments.txt", TEMPLATE, STUDY/"pop"/"p1_models.json",
                   STUDY/"dg"/"d2_indicators.npz", STUDY/"selgci"/"q0_common.py"]
        inputs += [FEATURES_SRC/f"{p}_{e}.{x}" for e in ENSEMBLES for p, x in (("features", "npz"), ("topology", "json"))]
        manifest.update(created_utc=datetime.now(timezone.utc).isoformat(), git=git_head(),
                        optimization_code_sha256=fitting_code_sha256(), input_sha256={str(p): digest(p) for p in inputs},
                        ensembles=infos, sparse_map_missing_residues=missing, sigma_pep=sig.tolist(),
                        settings=dict(S_grid=S_GRID, arms=list(ARMS), split_type=SPLIT_TYPE, folds="q0_common.moprp_splits",
                                      bv_fixed=dict(bc=BC, bh=BH), frame_averaging_mode="rate", peptide_trim=PEPTIDE_TRIM,
                                      normalize_sparse_map_rows=True, primary_loss="hdx_uptake_sigma_MSE_loss",
                                      sigma="per-peptide pooled RMS of 1/weights, floor 0.0586",
                                      delta="k_int * exp(-delta) (no lnPF floor)", template_opt_config=str(TEMPLATE),
                                      strength_mapping="maxent_scaling S_p = 2 S / H, H = n_train / sum_train 1/sigma^2"))
        write_json(mpath, manifest)
        return
    if not manifest:
        ap.error("run --phase prepare first")
    if manifest["optimization_code_sha256"] != fitting_code_sha256():
        changed = [k for k, v in fitting_code_sha256().items() if manifest["optimization_code_sha256"].get(k) != v]
        print(f"WARNING: fitting code changed since prepare: {changed}", flush=True)
    if args.conv_check:
        specs = build_specs(args.output_dir, ensembles=args.ensembles.split(","), arms=args.arms.split(","),
                            splits=(0,), n_steps=CONV_CHECK_STEPS, s_subset=conv_check_subset())
    else:
        specs = build_specs(args.output_dir, ensembles=args.ensembles.split(","), arms=args.arms.split(","),
                            splits=[int(x) for x in args.splits.split(",")])
    if args.limit:
        specs = [s for s in specs if not completed(s)][:args.limit]
    manifest.setdefault("fit_phases", []).append(dict(utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                                                      git=git_head(), n_specs=len(specs)))
    key = "conv_check_specs" if args.conv_check else "specs"
    known = {s["run_id"]: s for s in manifest.get(key, [])}
    known.update({s.run_id: dict(asdict(s), run_id=s.run_id) for s in specs})
    manifest[key] = sorted(known.values(), key=lambda s: s["run_id"])
    write_json(mpath, manifest)
    t0 = time.time()
    execute(specs, args)
    manifest = json.loads(mpath.read_text())
    manifest.setdefault("fit_wall_seconds", []).append(dict(conv_check=args.conv_check, n=len(specs),
                                                            seconds=time.time()-t0, jobs=args.jobs))
    write_json(mpath, manifest)


if __name__ == "__main__":
    main()
