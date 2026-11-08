#!/usr/bin/env python3
"""aSyn re-fit PILOT on the raw S11 peptide uptake curves, through the native jaxent pipeline.

Modelled on the MoPrP policy sidecars (2_CrossValidation/fitting/jaxENT/run_moprp_policy_sidecars.py and
3_CrossValidationBV/.../run_moprp_bv_policy_sidecars.py) and the validated MoPrP recipe
(docs/recovery_gulf_20261103/selgci: rate averaging, scaled BV, KL to uniform, rule-D selection, GCI).

Pipeline pieces used unchanged:
  common.optimization.run_optimization / create_data_loaders, BV_model (standard uptake forward,
  frame_averaging_mode="rate", bc 0.35 / bh 2.0 FIXED), loss "hdx_uptake_sigma_MSE_loss" with the precision
  matrix passed through OptimizationConfig.covariance_matrix_path, maxent_convexKL_loss via maxent_scaling,
  the optimiser/convergence checkpoints of the MoPrP rate template (Adam, lr 1.0, EMA 0.5, scaling 1000),
  HDF5 histories, normalize_sparse_map_rows=True with the proline check.

aSyn specifics (prepare phase):
  * k_int: calculate_HDXrate(T=293 K, pD = pH as read: 7.4 Tris/Extra, 7.2 Intra, unit s^-1) on the aSyn PDB;
    global ln PF shift delta applied as k_int,i * exp(-delta) in a per-cell features copy
    (identical to ln PF + delta under rate averaging). delta = median over the 21 training peptides of
    [ln(mean_i k_int,i / k_fit,p) - ln PF_prior,p], ln PF_prior,p = -ln(mean_i <exp(-L_i)>_uniform),
    k_fit = stretched exponential (plateau 1) on the 8-point grid. Frozen across S and folds.
  * time grid [0.05, 0.1, 0.25, 0.5, 1, 10, 30] s with kint in s^-1 (the standard BV uptake pass multiplies
    k*t directly; kint_unit="s^-1" is recorded, time is in s).
  * targets: fraction u/q; peptides start..end with peptide_trim=1 (residues start+1..end; prolines absent from
    the features pass the proline check).
  * sigma_pt = max(sd/q, floor), floor = pooled median of sd/q over the condition.  The pipeline has no per-point
    sigma path (see README): training uses a diagonal per-peptide precision 1/sigma_p^2 with sigma_p the RMS of
    sigma_pt over timepoints (the MoPrP sig_pep convention); the loader trace-normalises it.  Variant 'floorBV'
    adds sigma_BV^2 (1.4826 MAD of prior training residuals after delta).  Held-out z for selection use sigma_pt
    (and sigma_pt^2 + sigma_BV^2 for floorBV).
  * folds: 4 contiguous disjoint peptide blocks, leave-one-block-out; by default tail peptides (start >= 114) are
    excluded from train and val, kept in an 'excluded' list and predicted; with --include-tail they form a fifth
    block E_114-140 fitted and held out like the others (2026-11-08).
  * --calibration tris freezes the ensemble's Tris_only training-median delta for every condition (one shift,
    "fit the Tris median"); --bradykinin-chem multiplies k_int of the other conditions by the paper's bradykinin
    chemistry factor net of pH (Extracellular 0.848, Intracellular 0.920).

Regulariser arms (--regulariser; default convexKL reproduces the 2026-11-06 pilot):
  * convexKL : maxent_convexKL_loss via maxent_scaling = KL(uniform || w) (the pipeline default; README gap 4).
  * reverseKL: the MoPrP recipe direction KL(w || uniform) through run_optimization(frame_regularizer=
    ("maxent_reverse_KL_logp0_loss", common.losses.maxent_reverse_KL_logp0_loss, log p0 = -ln F, 1/S)); the
    objective is then 1000/(1+S) [S L_sigma + KL(w||u)], the same regulariser that rule-D F scores.
  * --inner-adam-step x != 1 emulates a configurable inner Adam step (OptaxOptimizer hard-codes
    learning_rate=1.0, jaxent/src/opt/optimiser.py:131) by wrapping optax.adam inside the worker process, as in
    run_moprp_recipe_sidecars._scale_inner_adam; recorded in the spec and run id.

Selection (analyze phase): candidates = convergence checkpoints with step >= 100 plus the final state of every
S run, plus the prior (uniform weights, S = 0); running_best and intermediate trajectory states are scored for
diagnostics but excluded.  Rule D = fewest Sidak-flagged held-out |z| (alpha 0.05), ties by
F = 1/2 sum z^2 + KL(w||uniform); rule C = min held-out chi2/N.  GCI T_n/S_n/Q_n on held-out z.

Usage (repo root, .venv):
  python jaxent/examples/4_aSyn/fitting/run_asyn_uptake_sidecars.py --phase prepare
  python jaxent/examples/4_aSyn/fitting/run_asyn_uptake_sidecars.py --smoke --phase all
  python jaxent/examples/4_aSyn/fitting/run_asyn_uptake_sidecars.py --phase fit --jobs 9
  python jaxent/examples/4_aSyn/fitting/run_asyn_uptake_sidecars.py --phase analyze
  python jaxent/examples/4_aSyn/fitting/run_asyn_uptake_sidecars.py --phase render
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
EX = HERE.parent
REPO = EX.parents[2]
DATA = EX/"data"/"_aSyn"
UPTAKE = DATA/"uptake"
SOURCE_FEATURES = {"TrisMD": DATA/"tris_MD"/"features", "ControlMD": DATA/"control_MD"/"features"}
KINT_PDB = DATA/"aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb"
TEMPLATE = (REPO/"jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_policy_sidecars_rate_fresh_20261023/"
            "sequence_cluster/MoPrP_rate_AF2_filtered_MSE_maxent_sequence_cluster_split000_maxent1000/"
            "MoPrP_rate_AF2_filtered_MSE_maxent_sequence_cluster_split000_maxent1000_config.json")
CELLS = ("TrisMD__Tris_only", "ControlMD__Tris_only", "TrisMD__Extracellular", "TrisMD__Intracellular")
PD = {"Tris_only": 7.4, "Extracellular": 7.4, "Intracellular": 7.2}
CALIBRATION_COND = "Tris_only"  # --calibration tris: delta from <ensemble>__Tris_only, frozen for the other conditions
TEMP_K = 293.0
BC, BH = 0.35, 2.0
TIMES_S = (0.05, 0.1, 0.25, 0.5, 1.0, 10.0, 30.0)  # t_index 1..7 of the 8-point grid (t = 0 is identically 0)
T_INDEX = (1, 2, 3, 4, 5, 6, 7)
PEPTIDE_TRIM = 1
TAIL_START = 114
VARIANTS = ("floor", "floorBV")
REGULARISERS = ("convexKL", "reverseKL")
REGULARISER_DESCRIPTION = {
    "convexKL": "maxent_convexKL_loss KL(uniform||w), weight 1/maxent_scaling, p0 uniform",
    "reverseKL": "maxent_reverse_KL_logp0_loss KL(w||uniform) via frame_regularizer, weight 1/S, log p0 = -ln F",
}
MAIN_VARIANT = "floor"
STRENGTHS = (2., 5., 10., 15., 20., 30., 60., 100., 125.)  # spans ESS ~100% -> <10% in every fold (README)
MAIN_EXTRA_STRENGTHS = (80., 150.)  # first-pass grid points kept for the main cell
MIN_CANDIDATE_STEP = 100
ALPHA = 0.05
ALPHA_EACH = 1-0.95**(1/3)
SPLIT_TYPE = "blocks"
# contiguous disjoint peptide blocks (end <= 38 | 39-69 | 55-94 not in B | 90-113); see README for the choice
BASE_BLOCKS = {
    "A_1-38": lambda s, e: e <= 38,
    "B_39-69": lambda s, e: s >= 39 and e <= 69,
    "C_55-94": lambda s, e: s >= 55 and e <= 94 and not (s >= 39 and e <= 69),
    "D_90-113": lambda s, e: 90 <= s < TAIL_START,
}
TAIL_BLOCK = {"E_114-140": lambda s, e: s >= TAIL_START}
INCLUDE_TAIL = False
BLOCKS = dict(BASE_BLOCKS)
FOLDS = tuple(BLOCKS)


def set_tail_mode(include_tail):
    """--include-tail: the acidic tail peptides (start >= 114) form a fifth block fitted and held out like the
    others (user 2026-11-07: excluding the C-tail from the fit was unfounded).  Default keeps the 2026-11-06
    pilot behaviour (tail excluded from train/val, predicted only)."""
    global INCLUDE_TAIL, BLOCKS, FOLDS
    INCLUDE_TAIL = bool(include_tail)
    BLOCKS = {**BASE_BLOCKS, **(TAIL_BLOCK if INCLUDE_TAIL else {})}
    FOLDS = tuple(BLOCKS)


def default_output():
    return HERE/f"_pilot_uptake_{date.today():%Y%m%d}"


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


# ============================================================================ specs
@dataclass(frozen=True)
class RunSpec:
    cell: str
    variant: str
    split_idx: int
    strength: float
    n_steps: int
    output_dir: str
    frame_averaging_mode: str = "rate"
    normalize_sparse_map_rows: bool = True
    regulariser: str = "convexKL"  # convexKL = maxent_convexKL_loss KL(u||w); reverseKL = KL(w||u) (recipe)
    inner_adam_step: float = 1.0  # 1.0 = OptaxOptimizer as shipped; otherwise emulated (see _scale_inner_adam)
    include_tail: bool = False  # tail peptides (start >= 114) as a fifth fitted block (see set_tail_mode)

    @property
    def ensemble(self):
        return self.cell.split("__")[0]

    @property
    def condition(self):
        return self.cell.split("__")[1]

    @property
    def fold(self):
        return FOLDS[self.split_idx]

    @property
    def run_id(self):
        tag = "maxent" if self.regulariser == "convexKL" else self.regulariser
        if self.inner_adam_step != 1.0:
            tag += f"_adam{self.inner_adam_step:g}"
        if self.include_tail:
            tag += "_tailin"
        return f"aSyn_rate_{self.cell}_{self.variant}_sigMSE_{tag}_{SPLIT_TYPE}_split{self.split_idx:03d}_maxent{self.strength:g}"

    @property
    def run_dir(self):
        return Path(self.output_dir)/"runs"/self.cell/self.variant/SPLIT_TYPE/self.run_id

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
        return Path(self.output_dir)/"datasplits"/self.condition

    @property
    def precision_path(self):
        return self.datasplit_dir/f"precision_{self.cell}_{self.variant}.npz"


def build_specs(args):
    specs = []
    for cell in args.cells:
        for variant in args.variants:
            if variant != MAIN_VARIANT and cell not in args.floorbv_cells:
                continue
            for split_idx in args.split_indices:
                extra = MAIN_EXTRA_STRENGTHS if (cell == CELLS[0] and not args.smoke
                                                 and args.regulariser == "convexKL") else ()
                for strength in sorted(set(args.strengths) | set(extra)):
                    specs.append(RunSpec(cell=cell, variant=variant, split_idx=split_idx, strength=float(strength),
                                         n_steps=args.n_steps, output_dir=str(args.output_dir),
                                         regulariser=args.regulariser, inner_adam_step=args.inner_adam_step,
                                         include_tail=args.include_tail))
    if len({s.run_id for s in specs}) != len(specs):
        raise ValueError("duplicate run ids")
    return specs


# ============================================================================ prepare
def residue_names_from_pdb():
    import MDAnalysis as mda
    from MDAnalysis.lib.util import convert_aa_code
    u = mda.Universe(str(KINT_PDB))
    res = u.select_atoms("protein").residues
    names = {}
    for r in res:
        rn = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS"}.get(r.resname, r.resname)
        names[int(r.resid)] = convert_aa_code(rn)
    return names, res


def intrinsic_rates(condition, residues):
    from jaxent.src.models.func.uptake import calculate_HDXrate
    kd = calculate_HDXrate(residues, temperature=TEMP_K, pD=PD[condition], unit="s^-1")
    return {int(r.resid): float(v) for r, v in kd.items()}


def stretched_k(y8):
    t = np.r_[0.0, TIMES_S]
    m = np.isfinite(y8)
    f = lambda x: (1-np.exp(-(np.exp(x[0])*t[m])**x[1]))-y8[m]
    x = least_squares(f, [0.0, 0.7], bounds=([-20, 0.05], [20, 3])).x
    return float(np.exp(x[0])), float(x[1])


def peptide_table(condition):
    t = pd.read_csv(UPTAKE/f"{condition}_peptide_uptake.csv")
    t = t[t.t_index.isin((0,)+T_INDEX)]
    peps = t.drop_duplicates("peptide_index").sort_values("peptide_index").reset_index(drop=True)
    piv = lambda col, idx: t.pivot(index="peptide_index", columns="t_index", values=col)[list(idx)].to_numpy(float)
    peps["residue_list"] = peps.residues.map(lambda s: [int(x) for x in str(s).split()])
    blocks = []
    for _, p in peps.iterrows():
        if p.start >= TAIL_START and not INCLUDE_TAIL:
            blocks.append("tail")
            continue
        hit = [b for b, fn in BLOCKS.items() if fn(p.start, p.end)]
        if len(hit) != 1:
            raise ValueError(f"peptide {p.label} matches blocks {hit}")
        blocks.append(hit[0])
    peps["block"] = blocks
    return peps, piv("fraction", T_INDEX), piv("fraction_sd", T_INDEX), piv("fraction", (0,)+T_INDEX)


def write_datasplits(condition, peps, y, out):
    """_datasplits layout read by common.loading.load_experimental_data (plus an 'excluded' tail list)."""
    import jaxent.src.interfaces.topology as pt
    from jaxent.src.custom_types.datapoint import ExpD_Datapoint
    from jaxent.src.custom_types.HDX import HDX_peptide
    out.mkdir(parents=True, exist_ok=True)
    data = []
    for i, p in peps.iterrows():
        top = pt.TopologyFactory.from_range(chain="A", start=int(p.start), end=int(p.end), fragment_index=int(i),
                                            peptide=True, peptide_trim=PEPTIDE_TRIM, fragment_name=f"aSyn_{condition}",
                                            fragment_sequence=p.sequence)
        if sorted(top.peptide_residues) != list(range(p.start+1, p.end+1)):
            raise ValueError(f"peptide_trim mismatch for {p.label}")
        data.append(HDX_peptide(dfrac=[float(v) for v in y[i]], top=top))
    save = lambda items, stem: ExpD_Datapoint.save_list_to_files(items, json_path=str(stem)+"_topology.json",
                                                                  csv_path=str(stem)+"_dfrac.csv")
    save(data, out/"full_dataset")
    if any(b == "tail" for b in peps.block):
        save([d for d, b in zip(data, peps.block) if b == "tail"], out/"excluded")
    for k, fold in enumerate(FOLDS):
        sd = out/SPLIT_TYPE/f"split_{k:03d}"
        sd.mkdir(parents=True, exist_ok=True)
        save([d for d, b in zip(data, peps.block) if b not in ("tail", fold)], sd/"train")
        save([d for d, b in zip(data, peps.block) if b == fold], sd/"val")
    return data


def bradykinin_chem_factor(cond):
    """Residual (non-pH) chemistry factor from the paper's bradykinin calibration (SI Figure S5):
    k_brady(cond)/k_brady(Tris) = 1/norm_factor, divided by the pH-only expectation 10^(pH_cond - pH_Tris)
    that the Bai k_int at the condition pD already carries.  Tris 1.0, Extracellular 0.848, Intracellular 0.920."""
    norm = json.loads((UPTAKE/"S11_time_grids.json").read_text())["norm_factor"][cond]
    return float((1.0/norm)/10**(PD[cond]-PD[CALIBRATION_COND]))


def prepare_cell(cell, out, names, residues_mda, delta_override=None, chem_factor=1.0):
    """delta_override: ln PF shift taken from another cell (e.g. the ensemble's Tris_only median) instead of this
    cell's own training median; chem_factor multiplies k_int (bradykinin chemistry net of pH)."""
    from jaxent.src.models.HDX.BV.features import BV_input_features
    import jaxent.src.interfaces.topology as pt
    ens, cond = cell.split("__")
    feat = BV_input_features.load(SOURCE_FEATURES[ens]/"features.npz")
    top = pt.PTSerialiser.load_list_from_json(SOURCE_FEATURES[ens]/"topology.json")
    res = np.array([int(t.residues[0]) for t in top])
    h = np.asarray(feat.heavy_contacts, float)
    a = np.asarray(feat.acceptor_contacts, float)
    E0 = np.exp(-(BC*h+BH*a)).mean(1)
    del h, a
    kd = intrinsic_rates(cond, residues_mda)
    kint = np.array([kd[r] for r in res])*chem_factor
    peps, y, sd, y8 = peptide_table(cond)
    ridx = {r: i for i, r in enumerate(res)}
    rows = [[ridx[r] for r in rl] for rl in peps.residue_list]
    train = np.where(peps.block != "tail")[0]
    kfit = np.array([stretched_k(v) for v in y8])
    mean_kint = np.array([kint[r].mean() for r in rows])
    lnpf_exp = np.log(mean_kint/kfit[:, 0])
    lnpf_prior = -np.log(np.array([E0[r].mean() for r in rows]))
    delta_cell = float(np.median((lnpf_exp-lnpf_prior)[train]))
    delta = delta_cell if delta_override is None else float(delta_override)
    M = np.zeros((len(peps), len(res)))
    for p, r in enumerate(rows):
        M[p, r] = 1/len(r)
    P0 = M@(1-np.exp(-(kint*np.exp(-delta)*E0)[:, None]*np.asarray(TIMES_S)[None]))
    floor = float(np.nanmedian(sd))
    sig = {"floor": np.maximum(sd, floor)}
    r0 = (P0-y)[train].ravel()
    sigma_bv = float(1.4826*np.median(np.abs(r0-np.median(r0))))
    sig["floorBV"] = np.sqrt(sig["floor"]**2+sigma_bv**2)
    # per-cell features copy: k_int (s^-1, condition pD) scaled by exp(-delta)
    fdir = out/"features"
    fdir.mkdir(parents=True, exist_ok=True)
    scaled = dataclasses.replace(feat, k_ints=np.asarray(kint*np.exp(-delta), dtype=np.asarray(feat.k_ints).dtype))
    scaled.save(str(fdir/f"features_{cell}.npz"))
    pt.PTSerialiser.save_list_to_json(top, fdir/f"topology_{cell}.json")
    ds = out/"datasplits"/cond
    write_datasplits(cond, peps, y, ds)
    for v in VARIANTS:
        sp = np.sqrt(np.mean(sig[v]**2, 1))  # per-peptide pooled sigma (MoPrP sig_pep)
        np.savez(ds/f"precision_{cell}_{v}.npz", precision=np.diag(1/sp**2))
        pd.DataFrame(sig[v], columns=[f"{t:g}" for t in TIMES_S]).to_csv(ds/f"sigma_pt_{cell}_{v}.csv", index=False)
    cd = out/cell
    cd.mkdir(parents=True, exist_ok=True)
    tab = peps[["peptide_index", "label", "start", "end", "q", "block"]].copy()
    raw = pd.read_csv(UPTAKE/"S11_fragment_uptake_raw.csv")
    raw = raw[(raw.condition == cond) & (raw.fragment == "peptide")].sort_values(["start", "end"]).reset_index(drop=True)
    norm_f = json.loads((UPTAKE/"S11_time_grids.json").read_text())["norm_factor"][cond]
    tab = tab.assign(k_fit_s=kfit[:, 0], beta_fit=kfit[:, 1], kfit_over_reported=kfit[:, 0]/(1000*raw.k_exp/norm_f),
                     mean_kint=mean_kint, lnPF_exp=lnpf_exp, lnPF_prior_unshifted=lnpf_prior,
                     lnPF_exp_minus_prior=lnpf_exp-lnpf_prior, prior_mean_resid=(P0-y).mean(1),
                     prior_chi2_floor=np.mean(((P0-y)/sig["floor"])**2, 1))
    tab.to_csv(cd/"prep_peptides.csv", index=False)
    np.savez(cd/"prep_arrays.npz", res=res, kint=kint, E0=E0, y=y, sd=sd, sig_floor=sig["floor"],
             sig_floorBV=sig["floorBV"], blocks=peps.block.to_numpy(), labels=peps.label.to_numpy(),
             starts=peps.start.to_numpy(), ends=peps.end.to_numpy(), lnpf_exp=lnpf_exp, delta=delta, M=M,
             times=np.asarray(TIMES_S))
    info = dict(cell=cell, n_frames=int(feat.features_shape[1]), n_residues=len(res), delta=delta, floor=floor,
                delta_cell_median=delta_cell, delta_source="own training median" if delta_override is None else
                f"{ens}__{CALIBRATION_COND} training median (frozen)", kint_chem_factor=chem_factor,
                sigma_bv=sigma_bv, n_train=int(len(train)), n_tail=int((peps.block == "tail").sum()),
                blocks={b: peps.label[peps.block == b].tolist() for b in FOLDS+("tail",)},
                prior_chi2_train_floor=float(np.mean(((P0-y)/sig["floor"])[train]**2)),
                prior_chi2_train_floorBV=float(np.mean(((P0-y)/sig["floorBV"])[train]**2)),
                kfit_over_reported_median=float(np.median(tab.kfit_over_reported)), pD=PD[cond], temperature_K=TEMP_K,
                kint_unit="s^-1", time_unit="s", times_s=list(TIMES_S))
    write_json(cd/"prep_info.json", info)
    print(f"[prepare] {cell}: F={info['n_frames']} delta={delta:.3f} floor={floor:.4f} sigma_BV={sigma_bv:.4f} "
          f"prior chi2/N train {info['prior_chi2_train_floor']:.1f} (+BV {info['prior_chi2_train_floorBV']:.2f})", flush=True)
    return info


def validate_sparse_maps(out, cells, names):
    from jaxent.examples.common import loading
    from jaxent.src.custom_types.HDX import HDX_peptide
    from jaxent.src.data.splitting.sparse_map import check_unrepresented_residues
    report = {}
    for cell in cells:
        cond = cell.split("__")[1]
        _, topology = loading.load_features_and_topology(str(out/"features"), cell)
        data = HDX_peptide.load_list_from_files(json_path=str(out/"datasplits"/cond/"full_dataset_topology.json"),
                                                csv_path=str(out/"datasplits"/cond/"full_dataset_dfrac.csv"))
        missing = check_unrepresented_residues(topology, data, residue_names=names, allow_missing_residues=False)
        report[cell] = {f"{data[i].top.residue_start}-{data[i].top.residue_end}": r for i, r in missing.items()}
    return report


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names, residues_mda = residue_names_from_pdb()
    (out/"datasplits").mkdir(parents=True, exist_ok=True)
    (out/"datasplits"/"asyn.seq").write_text("".join(names[i] for i in sorted(names))+"\n")
    infos = {}
    for cell in sorted(args.cells, key=lambda c: c.split("__")[1] != CALIBRATION_COND):  # Tris_only cells first
        ens, cond = cell.split("__")
        override = None
        if args.calibration == "tris" and cond != CALIBRATION_COND:
            src = f"{ens}__{CALIBRATION_COND}"
            if src not in infos:
                raise ValueError(f"--calibration tris needs {src} among the cells to calibrate {cell}")
            override = infos[src]["delta"]
        chem = bradykinin_chem_factor(cond) if args.bradykinin_chem else 1.0
        infos[cell] = prepare_cell(cell, out, names, residues_mda, delta_override=override, chem_factor=chem)
    missing = validate_sparse_maps(out, args.cells, names)
    print(f"[prepare] peptide residues absent from features (must be prolines): {missing}", flush=True)
    return infos, missing


def residue_names(out):
    from jaxent.examples.common import loading
    return loading.load_sequence_residue_names(str(Path(out)/"datasplits"/"asyn.seq"))


# ============================================================================ fit
def configured_model():
    from jaxent.examples.common.uptake_models import build_uptake_model
    from jaxent.src.custom_types.key import m_key
    model = build_uptake_model("standard", np.asarray(TIMES_S), kint_unit="s^-1")
    model.forward[m_key("HDX_peptide")].frame_averaging_mode = "rate"
    p = model.params
    if not (np.isclose(np.asarray(p.bv_bc).item(), BC) and np.isclose(np.asarray(p.bv_bh).item(), BH)):
        raise ValueError("BV defaults are not 0.35 / 2.0")
    if model.config.kint_unit != "s^-1":  # standard BV uptake multiplies k_int * t directly: t must be in s
        raise ValueError("kint unit must be s^-1")
    np.testing.assert_allclose(np.asarray(p.timepoints), TIMES_S, rtol=0, atol=1e-6)
    return model


def context(spec):
    from jaxent.examples.common import loading
    with contextlib.redirect_stdout(io.StringIO()):
        features, topology = loading.load_features_and_topology(str(spec.features_dir), spec.cell)
        train, val, full, _ = loading.load_experimental_data("", str(spec.datasplit_dir), SPLIT_TYPE, spec.split_idx)
    model = configured_model()
    for d in (train, val, full):
        loading.validate_hdx_timepoint_count(d, np.asarray(TIMES_S), label=spec.cell)
    return features, topology, model, train, val, full


def _scale_inner_adam(scale):
    """Process-local: make optax.adam(learning_rate) use learning_rate*scale (emulates a configurable inner
    Adam step in OptaxOptimizer, whose chains call inject_hyperparams(optax.adam)(learning_rate=1.0))."""
    import optax
    base = optax.adam

    def adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, nesterov=False):
        return base(learning_rate*scale, b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype, nesterov=nesterov)
    optax.adam = adam


def regulariser_kwargs(spec, n_frames):
    """Extra run_optimization kwargs for the regulariser arm (empty for the pipeline default)."""
    if spec.regulariser == "convexKL":
        return {}
    if spec.regulariser != "reverseKL":
        raise ValueError(f"unknown regulariser {spec.regulariser}")
    import jax.numpy as jnp
    from jaxent.examples.common.losses import maxent_reverse_KL_logp0_loss
    log_p0 = jnp.full((n_frames,), -np.log(n_frames), dtype=jnp.float32)  # uniform p0 in log space
    return dict(frame_regularizer=("maxent_reverse_KL_logp0_loss", maxent_reverse_KL_logp0_loss, log_p0,
                                   1.0/spec.strength))


def run_fit(spec):
    from jaxent.examples.common.config import LossConfig, OptimizationConfig
    from jaxent.examples.common.optimization import create_data_loaders, run_optimization
    from jaxent.src.custom_types.key import m_key
    features, topology, model, train, val, _ = context(spec)
    if spec.inner_adam_step != 1.0:
        _scale_inner_adam(spec.inner_adam_step)
    extra = regulariser_kwargs(spec, int(features.features_shape[1]))
    template = json.loads(TEMPLATE.read_text())
    if template["loss_config"]["optimize_bv_params"]:
        raise ValueError("template must have fixed BV parameters")
    options = OptimizationConfig(**{**template["opt_config"], "n_steps": spec.n_steps,
                                    "covariance_matrix_path": str(spec.precision_path)})
    loss = LossConfig(primary_loss="hdx_uptake_sigma_MSE_loss", maxent_scaling=spec.strength, optimize_bv_params=False)
    names = residue_names(spec.output_dir)
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
    config["asyn_pilot"] = dict(spec=asdict(spec), training_loss="hdx_uptake_sigma_MSE_loss (diag per-peptide precision, "
                                "trace-normalised by the loader)", kint_unit="s^-1", time_unit="s",
                                bv_fixed=[BC, BH], coefficient=1/spec.strength, fit_seconds=time.time()-t0,
                                regulariser=REGULARISER_DESCRIPTION[spec.regulariser], inner_adam_step=spec.inner_adam_step,
                                code_sha256=digest(Path(__file__)))
    write_json(spec.config_path, config)
    print(f"Completed {spec.run_id} in {time.time()-t0:.0f}s", flush=True)


def completed(spec):
    if not spec.history_path.exists() or not spec.config_path.exists():
        return False
    try:
        stored = json.loads(spec.config_path.read_text()).get("asyn_pilot", {}).get("spec")
        return stored is not None and RunSpec(**stored) == spec
    except (OSError, ValueError, TypeError):
        return False


def execute(specs, args):
    pending = [s for s in specs if not completed(s)]
    cores = sorted(os.sched_getaffinity(0))
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


# ============================================================================ analyze (selection)
def t_n(n, alpha=ALPHA):
    """Sidak per-cell |z| threshold for n cells at family-wise alpha (selgci q0_common.t_n)."""
    return float(norm.ppf(0.5+0.5*(1-alpha)**(1.0/n)))


def gci_triple(z):
    """GCI T_n/S_n/Q_n with diagonal Sigma (a_n = b_n = 1), alpha split three ways (gci2 g0_common.triple)."""
    z = np.ravel(z)
    n = len(z)
    tT = float(norm.ppf(0.5+0.5*(1-ALPHA_EACH)**(1/n)))
    thrS, thrQ = float(norm.ppf(1-ALPHA_EACH/2)), float(norm.ppf(1-ALPHA_EACH))
    T, S, Q = float(np.abs(z).max()), float(z.sum()/np.sqrt(n)), float((np.sum(z**2)-n)/np.sqrt(2*n))
    code = ("" if T <= tT else "t")+("" if abs(S) <= thrS else "s")+("" if Q <= thrQ else "q")
    return dict(T_n=T, S_n=S, Q_n=Q, gci_code=(code or "P")+("o" if Q < -thrQ else ""), gci_tT=tT, gci_thrS=thrS,
                gci_thrQ=thrQ)


def saved_candidates(history):
    """Ex3 saved_candidates with kinds: convergence checkpoints (labelled), trajectory, final, running_best."""
    from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
    entries = []
    for i, s in enumerate(history.states):
        entries.append(("final" if i == len(history.states)-1 else "trajectory", None, s))
    for thr, s in history.iter_labeled_convergence_states():
        entries.append(("convergence", thr, s))
    if history.best_state is not None:
        entries.append(("running_best", None, history.best_state))
    seen = set()
    for kind, thr, state in entries:
        w = np.asarray(validated_frame_weight_simplex(state.params.frame_weight_simplex), dtype=float)
        w /= w.sum()
        key = (kind, int(state.step), hashlib.sha256(w.tobytes()).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        yield kind, thr, state, w


class CellScorer:
    """Native forward pass (rate, fixed BV, scaled k_int) on all 30 peptides; held-out z with sigma_pt."""

    def __init__(self, out, cell, variant):
        import jax.numpy as jnp
        from jaxent.examples.common import loading
        from jaxent.examples.common.optimization import create_data_loaders
        from jaxent.examples.common.analysis.rescore_iso_coordinate_sigma import make_predict
        from jaxent.src.custom_types.key import m_key
        self.jnp = jnp
        spec = RunSpec(cell=cell, variant=variant, split_idx=0, strength=1., n_steps=1, output_dir=str(out))
        self.features, topology, self.model, _, _, full = context(spec)
        names = residue_names(out)
        with contextlib.redirect_stdout(io.StringIO()):
            loader = create_data_loaders(full, full, full, self.features, topology, normalize_sparse_map_rows=True,
                                         residue_names=names)
        self.Mfull = np.asarray(loader.train.residue_feature_ouput_mapping.todense())
        self.y = np.asarray(loader.train.y_true)[..., 0]
        self.idx = [int(d.top.fragment_index) for d in full]
        if self.idx != list(range(len(full))):
            raise ValueError("full dataset must be ordered by fragment_index")
        self.prep = dict(np.load(Path(out)/cell/"prep_arrays.npz", allow_pickle=True))
        np.testing.assert_allclose(self.y, self.prep["y"], atol=1e-9)
        np.testing.assert_allclose(self.Mfull, self.prep["M"], atol=1e-9)
        self.sig = self.prep[f"sig_{variant}"]
        self.prec = np.diag(np.load(spec.precision_path)["precision"])
        self.blocks = self.prep["blocks"]
        self.labels = list(self.prep["labels"])
        self.predict = make_predict(self.model.forward[m_key("HDX_peptide")], self.features)
        self.F = int(self.features.features_shape[1])
        self.Kr = None

    def pred(self, w):
        u = np.asarray(self.predict(self.jnp.asarray(w), self.model.params))  # (T, R)
        return self.Mfull@u.T                                                   # (30, T)

    def lnpf(self, w):
        """rate-averaged per-residue effective ln PF (incl. delta): -ln(sum_f w_f exp(-L_if)) + delta."""
        if self.Kr is None:
            h = np.asarray(self.features.heavy_contacts, float)
            a = np.asarray(self.features.acceptor_contacts, float)
            self.Kr = np.exp(-(BC*h+BH*a))
        return -np.log(self.Kr@w)+float(self.prep["delta"])

    def replay_loss(self, P, rows):
        """hdx_uptake_sigma_MSE_loss on the given rows with the trace-normalised diagonal precision."""
        W = self.prec[rows]*len(rows)/self.prec[rows].sum()
        r = P[rows]-self.y[rows]
        return float(0.5*np.sum(W[:, None]*r**2)/r.size)

    def score(self, w, fold):
        P = self.pred(w)
        z = (self.y-P)/self.sig
        va = np.where(self.blocks == fold)[0]
        tr = np.where((self.blocks != fold) & (self.blocks != "tail"))[0]
        tail = np.where(self.blocks == "tail")[0]
        zv = z[va]
        tn = t_n(zv.size)
        fl = np.abs(zv) > tn
        nz = w > 0
        KL = float(np.sum(w[nz]*np.log(w[nz]*self.F)))
        row = dict(ESS=float(1/np.sum(w**2)/self.F), KL=KL, chi2_train=float(np.mean(z[tr]**2)),
                   ho_chi2=float(np.mean(zv**2)), ho_n_cells=int(zv.size), ho_t_n=tn, ho_n_flags=int(fl.sum()),
                   ho_flagged=";".join(f"{self.labels[va[i]]}({int(fl[i].sum())})" for i in range(len(va)) if fl[i].any()),
                   ho_maxabsz=float(np.abs(zv).max()), F=float(0.5*np.sum(zv**2)+KL),
                   chi2_tail=float(np.mean(z[tail]**2)) if len(tail) else float("nan"), mean_resid_train=float((P-self.y)[tr].mean()),
                   loss_train_replay=self.replay_loss(P, tr), loss_val_replay=self.replay_loss(P, va), **gci_triple(zv))
        return row, P, z


def analyze(specs, args):
    from jaxent.src.utils.hdf import load_optimization_history_from_file
    out = args.output_dir
    summary, timing = [], []
    groups = {}
    for s in specs:
        groups.setdefault((s.cell, s.variant), []).append(s)
    for (cell, variant), group in groups.items():
        sc = CellScorer(out, cell, variant)
        vd = out/cell/variant
        vd.mkdir(parents=True, exist_ok=True)
        rows, W, PRED = [], {}, {}
        max_replay = 0.0
        uniform = np.full(sc.F, 1/sc.F)
        for split_idx in sorted({s.split_idx for s in group}):
            fold = FOLDS[split_idx]
            r, P, _ = sc.score(uniform, fold)
            r.update(cell=cell, variant=variant, fold=fold, split_idx=split_idx, strength=0.0, kind="prior",
                     convergence_threshold=np.nan, step=0, in_pool=True, run_id="prior", wall_s=0.0)
            rows.append(r)
            key = f"{fold}|prior"
            W[key], PRED[key] = uniform, P
            for spec in sorted([s for s in group if s.split_idx == split_idx], key=lambda s: s.strength):
                if not completed(spec):
                    raise ValueError(f"incomplete fit {spec.run_id}")
                cfg = json.loads(spec.config_path.read_text())
                hist = load_optimization_history_from_file(str(spec.history_path))
                if int(hist.states[-1].step) != spec.n_steps:
                    print(f"  note: {spec.run_id} final step {int(hist.states[-1].step)} != n_steps {spec.n_steps}")
                timing.append(dict(run_id=spec.run_id, cell=cell, variant=variant, fold=fold, strength=spec.strength,
                                   fit_seconds=cfg["asyn_pilot"]["fit_seconds"],
                                   wall_time_seconds=cfg["runtime"]["wall_time_seconds"]))
                for ci, (kind, thr, state, w) in enumerate(saved_candidates(hist)):
                    r, P, _ = sc.score(w, fold)
                    saved_tr = float(np.asarray(state.losses.train_losses[0]))
                    saved_va = float(np.asarray(state.losses.val_losses[0]))
                    err = max(abs(r["loss_train_replay"]-saved_tr)/max(abs(saved_tr), 1e-12),
                              abs(r["loss_val_replay"]-saved_va)/max(abs(saved_va), 1e-12))
                    max_replay = max(max_replay, err)
                    step = int(state.step)
                    in_pool = (kind == "final") or (kind == "convergence" and step >= MIN_CANDIDATE_STEP)
                    r.update(cell=cell, variant=variant, fold=fold, split_idx=split_idx, strength=spec.strength,
                             kind=kind, convergence_threshold=thr if thr is not None else np.nan, step=step,
                             in_pool=in_pool, run_id=spec.run_id, candidate_index=ci, saved_loss_train=saved_tr,
                             saved_loss_val=saved_va, replay_rel_err=err, wall_s=cfg["runtime"]["wall_time_seconds"])
                    rows.append(r)
                    if in_pool:
                        key = f"{fold}|{spec.strength:g}|{kind}|{step}"
                        W[key], PRED[key] = w, P
        cand = pd.DataFrame(rows)
        cand.to_csv(vd/"candidates.csv", index=False)
        sel_rows, Wd, Wc, Pd, Pc, Lnd = [], [], [], [], [], []
        P_prior = None
        for fold, g in cand[cand.in_pool].groupby("fold", sort=False):
            g = g.reset_index(drop=True)
            mf = g.ho_n_flags.min()
            fs = g[g.ho_n_flags == mf]
            pick = {"D": int(fs.F.idxmin()), "C": int(g.ho_chi2.idxmin()), "prior": int(g.index[g.kind == "prior"][0])}
            for rule, i in pick.items():
                r = g.loc[i].to_dict()
                r.update(rule=rule, min_flags=int(mf), n_minflag=len(fs), n_pool=len(g))
                sel_rows.append(r)
            keyof = lambda r: f"{fold}|prior" if r.kind == "prior" else f"{fold}|{r.strength:g}|{r.kind}|{int(r.step)}"
            rd, rc = g.loc[pick["D"]], g.loc[pick["C"]]
            Wd.append(W[keyof(rd)]); Wc.append(W[keyof(rc)])
            Pd.append(PRED[keyof(rd)]); Pc.append(PRED[keyof(rc)])
            Lnd.append(sc.lnpf(W[keyof(rd)]))
            P_prior = PRED[f"{fold}|prior"]
        sel = pd.DataFrame(sel_rows)
        sel.to_csv(vd/"fold_table.csv", index=False)
        folds_present = list(sel[sel.rule == "D"].fold)
        Wd = np.array(Wd)
        np.savez_compressed(vd/"pilot_selected.npz", weights_ruleD=Wd, weights_ruleC=np.array(Wc),
                            pred_ruleD=np.array(Pd), pred_ruleC=np.array(Pc), pred_prior=P_prior,
                            lnpf_ruleD=np.array(Lnd), lnpf_prior=sc.lnpf(uniform), folds=np.array(folds_present),
                            S_ruleD=sel[sel.rule == "D"].strength.to_numpy(), sigma=sc.sig, y=sc.y,
                            blocks=sc.blocks, labels=np.array(sc.labels), delta=float(sc.prep["delta"]))
        cond = cell.split("__")[1]
        np.savez(vd/f"{cond}_uptake_pilot_{cell.split('__')[0]}_{variant}_ruleD_selected.npz", frame_weights=Wd,
                 pred_ln_pf=np.array(Lnd), bv_bc=np.full(len(Wd), BC), bv_bh=np.full(len(Wd), BH))
        for _, r in sel.iterrows():
            summary.append({k: r[k] for k in ("cell", "variant", "fold", "rule", "strength", "kind", "step", "ESS", "KL",
                                                "chi2_train", "ho_chi2", "ho_n_flags", "ho_n_cells", "T_n", "S_n", "Q_n",
                                                "gci_code", "chi2_tail", "F", "ho_flagged", "n_pool", "min_flags")})
        print(f"[analyze] {cell} {variant}: rule-D S {sel[sel.rule == 'D'].strength.tolist()} "
              f"ESS {np.round(sel[sel.rule == 'D'].ESS, 3).tolist()} flags {sel[sel.rule == 'D'].ho_n_flags.tolist()} "
              f"(prior {sel[sel.rule == 'prior'].ho_n_flags.tolist()}); max loss replay rel err {max_replay:.1e}", flush=True)
    sm = pd.DataFrame(summary)
    tag = "_smoke" if args.smoke else ""
    sm.to_csv(out/f"selection_summary{tag}.csv", index=False)
    pd.DataFrame(timing).to_csv(out/f"timing{tag}.csv", index=False)
    write_results_table(out, sm, pd.DataFrame(timing), tag)
    return sm


def write_results_table(out, sm, timing, tag=""):
    lines = []
    for variant in VARIANTS:
        v = sm[sm.variant == variant]
        if v.empty:
            continue
        lines += [f"### Noise variant `{variant}`", "",
                  "| cell | delta | floor | sigma_BV | fold | S (D) | kind@step | ESS % | train chi2/N | held-out chi2/N "
                  "(prior) | flags/n (prior) | T_n / S_n / Q_n (code) | prior T_n / S_n / Q_n (code) | S (C) | C held-out chi2/N | tail chi2/N (prior) |",
                  "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|"]
        for cell in CELLS:
            c = v[v.cell == cell]
            if not (out/cell/"prep_info.json").exists():
                continue
            if c.empty:
                continue
            info = json.loads((out/cell/"prep_info.json").read_text())
            for fold in FOLDS:
                f = c[c.fold == fold]
                if f.empty:
                    continue
                d, p, cc = (f[f.rule == x].iloc[0] for x in ("D", "prior", "C"))
                lines.append(
                    f"| {cell} | {info['delta']:.2f} | {info['floor']:.4f} | {info['sigma_bv']:.3f} | {fold} | {d.strength:g} | "
                    f"{d.kind}@{int(d.step)} | {100*d.ESS:.1f} | {d.chi2_train:.1f} | {d.ho_chi2:.1f} ({p.ho_chi2:.1f}) | "
                    f"{int(d.ho_n_flags)}/{int(d.ho_n_cells)} ({int(p.ho_n_flags)}) | {d.T_n:.1f} / {d.S_n:+.1f} / {d.Q_n:.0f} ({d.gci_code}) | "
                    f"{p.T_n:.1f} / {p.S_n:+.1f} / {p.Q_n:.0f} ({p.gci_code}) | {cc.strength:g} | {cc.ho_chi2:.1f} | "
                    f"{d.chi2_tail:.0f} ({p.chi2_tail:.0f}) |")
        lines.append("")
    if not timing.empty:
        t = timing.groupby(["cell", "variant"]).wall_time_seconds.agg(["count", "median", "max", "sum"])
        lines += ["### Compute time per run (s, run_optimization wall time)", "", "| cell | variant | runs | median | max | total |",
                  "|---|---|---:|---:|---:|---:|"]
        lines += [f"| {c} | {v} | {int(r['count'])} | {r['median']:.0f} | {r['max']:.0f} | {r['sum']:.0f} |" for (c, v), r in t.iterrows()]
        lines.append("")
    (out/f"results_table{tag}.md").write_text("\n".join(lines))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("prepare", "fit", "analyze", "render", "all"), default="prepare")
    ap.add_argument("--smoke", action="store_true", help="1 split, 2 S values, few steps, main cell/variant only")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=None, help="default 5000 (200 with --smoke)")
    ap.add_argument("--strengths", help="comma-separated maxent_scaling grid (KL weight = 1/S)")
    ap.add_argument("--cells", help="comma-separated subset of " + ",".join(CELLS))
    ap.add_argument("--variants", help="comma-separated subset of " + ",".join(VARIANTS))
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--floorbv-cells", default=CELLS[0], help="cells that also get the floorBV noise variant")
    ap.add_argument("--regulariser", choices=REGULARISERS, default="convexKL",
                    help="convexKL = pipeline default KL(u||w); reverseKL = recipe KL(w||u) (frame_regularizer hook)")
    ap.add_argument("--inner-adam-step", type=float, default=1.0,
                    help="inner Adam step (1.0 = OptaxOptimizer as shipped; e.g. 0.1 = MoPrP diagnostic, emulated)")
    ap.add_argument("--splits", help="comma-separated split indices (default all four folds; 0 with --smoke)")
    ap.add_argument("--calibration", choices=("cell", "tris"), default="cell",
                    help="ln PF shift delta: 'cell' = each cell's own training median (2026-11-06 pilot); "
                         "'tris' = the ensemble's Tris_only training median, frozen for every condition")
    ap.add_argument("--bradykinin-chem", action="store_true",
                    help="multiply k_int of the non-Tris conditions by the paper's bradykinin chemistry factor net of pH "
                         "(Extracellular 0.848, Intracellular 0.920; SI Figure S5)")
    ap.add_argument("--include-tail", action="store_true",
                    help="fit the tail peptides (start >= 114) as a fifth block E_114-140 instead of excluding them")
    ap.add_argument("--worker-spec", type=Path)
    args = ap.parse_args()
    if args.worker_spec:
        spec = RunSpec(**json.loads(args.worker_spec.read_text()))
        set_tail_mode(spec.include_tail)
        if not completed(spec):
            run_fit(spec)
        return
    set_tail_mode(args.include_tail)
    args.output_dir = (args.output_dir or default_output()).resolve()
    if args.smoke:
        args.output_dir = args.output_dir.with_name(args.output_dir.name+"_smoke")
    args.cells = tuple(args.cells.split(",")) if args.cells else ((CELLS[0],) if args.smoke else CELLS)
    args.variants = tuple(args.variants.split(",")) if args.variants else ((MAIN_VARIANT,) if args.smoke else VARIANTS)
    args.strengths = sorted(float(x) for x in args.strengths.split(",")) if args.strengths else (
        [10., 1000.] if args.smoke else list(STRENGTHS))
    args.floorbv_cells = tuple(args.floorbv_cells.split(","))
    args.split_indices = tuple(int(x) for x in args.splits.split(",")) if args.splits else (
        (0,) if args.smoke else tuple(range(len(FOLDS))))
    if args.n_steps is None:
        args.n_steps = 200 if args.smoke else 5000
    if not 1 <= args.jobs <= len(os.sched_getaffinity(0))//2:
        ap.error("need two CPUs per job")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpath = args.output_dir/"manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    phases = ("prepare", "fit", "analyze", "render") if args.phase == "all" else (args.phase,)
    specs = build_specs(args)
    if "prepare" in phases:
        infos, missing = prepare(args)
        inputs = [UPTAKE/f"{c}_peptide_uptake.csv" for c in PD]+[UPTAKE/"S11_fragment_uptake_raw.csv", KINT_PDB,
                  TEMPLATE]+[SOURCE_FEATURES[e]/n for e in SOURCE_FEATURES for n in ("features.npz", "topology.json")]
        manifest.update(created_utc=datetime.now(timezone.utc).isoformat(), git=git_head(),
                        optimization_code_sha256=fitting_code_sha256(), input_sha256={str(p): digest(p) for p in inputs},
                        cells=infos, sparse_map_missing_residues=missing)
    manifest.setdefault("phases", []).append(dict(phases=list(phases), utc=datetime.now(timezone.utc).isoformat(),
                                                  argv=sys.argv, git=git_head()))
    manifest["settings"] = dict(smoke=args.smoke, n_steps=args.n_steps, strengths=args.strengths, cells=list(args.cells),
                                main_extra_strengths=list(MAIN_EXTRA_STRENGTHS),
                                variants=list(args.variants), floorbv_cells=list(args.floorbv_cells), folds=list(FOLDS), split_type=SPLIT_TYPE,
                                times_s=list(TIMES_S), kint_unit="s^-1", time_unit="s", pD=PD, temperature_K=TEMP_K,
                                bv_fixed=dict(bc=BC, bh=BH), frame_averaging_mode="rate", peptide_trim=PEPTIDE_TRIM,
                                normalize_sparse_map_rows=True, primary_loss="hdx_uptake_sigma_MSE_loss",
                                regulariser=args.regulariser, maxent=REGULARISER_DESCRIPTION[args.regulariser],
                                inner_adam_step=args.inner_adam_step, calibration=args.calibration,
                                bradykinin_chem=args.bradykinin_chem, include_tail=args.include_tail,
                                template_opt_config=str(TEMPLATE), min_candidate_step=MIN_CANDIDATE_STEP,
                                tail_start=TAIL_START, alpha=ALPHA)
    manifest["specs"] = [asdict(s) for s in specs]
    write_json(mpath, manifest)
    if "fit" in phases:
        if manifest.get("optimization_code_sha256") and manifest["optimization_code_sha256"] != fitting_code_sha256():
            changed = [k for k, v in fitting_code_sha256().items() if manifest["optimization_code_sha256"].get(k) != v]
            print(f"WARNING: fitting code changed since prepare: {changed}", flush=True)
        execute(specs, args)
    if "analyze" in phases:
        analyze(specs, args)
    if "render" in phases:
        sys.path.insert(0, str(EX/"analysis"))
        import render_asyn_pilot as R
        for cell in args.cells:
            for variant in args.variants:
                if (args.output_dir/cell/variant/"pilot_selected.npz").exists():
                    R.render_cell(args.output_dir/cell, variant)


if __name__ == "__main__":
    main()
