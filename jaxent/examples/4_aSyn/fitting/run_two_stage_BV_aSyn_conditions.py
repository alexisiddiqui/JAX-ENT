#!/usr/bin/env python3
"""Two-stage aSyn fit: calibrate the BV parameters, then reweight with them held constant.

Stage 1 (``<out>_stage1/``): per condition, BV-regulariser arm (0.5, 1.0; L1 to the defaults, as in the
manuscript sweep) and sequence-cluster split, optimise bc/bh only with the frame weights frozen at uniform
(``optimise_aSyn_conditions_BV.py --freeze-frame-weights``).  The final bc/bh of each run are collected into
``<out>_stage1/fitted_bv_params.json`` (+ .csv).

Stage 2 (``<out>/``, standard layout for run_comprehensive_analysis.sh): the manuscript MaxEnt sweep
(20 values x 2 arms x 3 splits x 4 conditions, 5000 steps, Adam lr 1.0, EMA 0.5, scaling 1000) with the BV
parameters pinned to the stage-1 values of the same condition / arm / split and no BV regulariser
(``--fixed-bv-params``).  Then ``run_comprehensive_analysis.sh <out>`` as the original runner does.

Usage (repo root, .venv):  python jaxent/examples/4_aSyn/fitting/run_two_stage_BV_aSyn_conditions.py [--jobs 10]
                           [--smoke] [--out DIR] [--skip-analysis]
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EX = HERE.parent
OPTIMISE = HERE/"optimise_aSyn_conditions_BV.py"
ANALYSIS_SH = EX/"run_comprehensive_analysis.sh"
CONDITIONS = ("Tris_only", "Extracellular", "Intracellular", "Lysosomal")
MAXENT = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000)
BVREG = (0.5, 1.0)
SPLIT_TYPE = "sequence_cluster"
N_SPLITS = 3
COMMON = ["--loss-function", "MSE", "--bv-reg-function", "L1", "--split-types", SPLIT_TYPE,
          "--initial-steps", "0", "--initial-learning-rate", "1.0", "--learning-rate", "1.0",
          "--ema-alpha", "0.5", "--forward-model-scaling", "1000.0", "--model-parameters-lr-scale", "1.0"]


def run_jobs(cmds, jobs, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(EX.parents[2])}

    def one(item):
        name, cmd = item
        with (log_dir/f"{name}.log").open("w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE)).returncode
        print(f"  {name}: exit {code}", flush=True)
        return name, code
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, cmds))
    failed = [n for n, c in results if c]
    print(f"{len(cmds)} jobs in {time.time()-t0:.0f}s; failed: {failed}", flush=True)
    if failed:
        raise RuntimeError(f"failed jobs: {failed}")


def collect_stage1(stage1_dir: Path, conditions, n_steps):
    from jaxent.src.utils.hdf import load_optimization_history_from_file
    import numpy as np
    params, rows = {}, []
    for cond in conditions:
        params[cond] = {}
        for r in BVREG:
            params[cond][f"{r:g}"] = {}
            for k in range(N_SPLITS):
                run = f"{cond}_MSE_{SPLIT_TYPE}_split{k:03d}_maxent1.0_bvreg{r:.1f}_bvregfnL1"
                h = load_optimization_history_from_file(str(stage1_dir/SPLIT_TYPE/f"{run}_results.hdf5"))
                st = h.states[-1]
                mp = st.params.model_parameters[0]
                bc, bh = float(np.asarray(mp.bv_bc).reshape(-1)[0]), float(np.asarray(mp.bv_bh).reshape(-1)[0])
                tr, va = float(np.asarray(st.losses.train_losses[0])), float(np.asarray(st.losses.val_losses[0]))
                params[cond][f"{r:g}"][f"{k:03d}"] = dict(bv_bc=bc, bv_bh=bh, step=int(st.step), train_loss=tr,
                                                          val_loss=va)
                rows.append(dict(condition=cond, bv_reg=r, split=k, bv_bc=bc, bv_bh=bh, step=int(st.step),
                                 train_loss=tr, val_loss=va, n_steps=n_steps))
    (stage1_dir/"fitted_bv_params.json").write_text(json.dumps(params, indent=2))
    import pandas as pd
    pd.DataFrame(rows).to_csv(stage1_dir/"fitted_bv_params.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--out", type=Path, help="stage-2 results dir (default fitting/_optimise_aSyn_BV_FIXEDBV_<steps>_<ts>)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--smoke", action="store_true", help="1 condition, 2 maxent values, 50 steps")
    ap.add_argument("--skip-stage1", action="store_true", help="reuse <out>_stage1/fitted_bv_params.json")
    ap.add_argument("--skip-analysis", action="store_true")
    a = ap.parse_args()
    conditions = tuple(a.conditions.split(","))
    maxent = MAXENT
    if a.smoke:
        conditions, maxent, a.n_steps = conditions[:1], (1, 1000), 50
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (a.out or HERE/f"_optimise_aSyn_BV_FIXEDBV_{a.n_steps}{'_smoke' if a.smoke else ''}_{ts}").resolve()
    stage1 = out.with_name(out.name+"_stage1")
    py = sys.executable
    print(f"stage 1 -> {stage1}\nstage 2 -> {out}", flush=True)

    if not a.skip_stage1:
        cmds = []
        for cond in conditions:
            for r in BVREG:
                cmds.append((f"stage1_{cond}_bvreg{r:g}",
                             [py, str(OPTIMISE), "--condition", cond, "--maxent-range", "1,1",
                              "--bvreg-range", f"{r},{r}", "--n-steps", str(a.n_steps), "--freeze-frame-weights",
                              "--output-dir", str(stage1), *COMMON]))
        print(f"Stage 1: {len(cmds)} parameter fits (frames frozen), {N_SPLITS} splits each", flush=True)
        run_jobs(cmds, min(a.jobs, len(cmds)), stage1/"logs")
    params = collect_stage1(stage1, conditions, a.n_steps)
    pjson = stage1/"fitted_bv_params.json"

    cmds = []
    for cond in conditions:
        for m in maxent:
            for r in BVREG:
                cmds.append((f"{cond}_maxent{m}_bvreg{r:g}",
                             [py, str(OPTIMISE), "--condition", cond, "--maxent-range", f"{m},{m}",
                              "--bvreg-range", f"{r},{r}", "--n-steps", str(a.n_steps), "--fixed-bv-params",
                              str(pjson), "--output-dir", str(out), *COMMON]))
    print(f"Stage 2: {len(cmds)} sweep jobs x {N_SPLITS} splits, BV fixed", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out/"two_stage_manifest.json").write_text(json.dumps(dict(
        created=ts, n_steps=a.n_steps, maxent=list(maxent), bv_reg=list(BVREG), conditions=list(conditions),
        stage1_dir=str(stage1), fitted_bv_params=params, optimiser_settings=COMMON,
        note="stage 1: bc/bh fitted with frame weights frozen (L1 to defaults, weight = bv_reg); "
             "stage 2: frame weights fitted with bc/bh pinned to stage-1 values, no BV regulariser"), indent=2))
    run_jobs(cmds, a.jobs, out/"logs")
    if not a.skip_analysis:
        print("Running comprehensive analysis", flush=True)
        with (out/"logs"/"comprehensive_analysis_launcher.log").open("w") as log:
            code = subprocess.run(["bash", str(ANALYSIS_SH), str(out)], stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONPATH": str(EX.parents[2])}).returncode
        print(f"analysis exit {code}", flush=True)
    print("TWO_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
