#!/usr/bin/env python3
"""Durable sequential controller for the three MoPrP sidecar experiments.

Run under a user systemd service. A filesystem lock prevents duplicate queues;
stage markers and native per-fit completion records allow restart/resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
STAGES = ("maxent", "all_pairs", "knn20")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, status):
    status["updated_utc"] = now()
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
    temporary.replace(path)


def run_campaign(output, analysis, jobs, runner=None):
    runner = Path(runner or HERE/"run_moprp_policy_sidecars.py").resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    status_path = output/"background_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    manifest = output/"manifest.json"
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    campaign_settings = json.loads(manifest.read_text()).get("settings", {})
    if previous and previous.get("manifest_sha256") != fingerprint:
        raise ValueError("Campaign manifest changed since the background run")
    if previous.get("runner") not in (None, str(runner)):
        raise ValueError("Campaign runner changed since the background run")
    status = dict(previous, manifest_sha256=fingerprint, jobs=jobs, stage_order=list(STAGES),
        output_dir=str(output), analysis_dir=str(analysis), runner=str(runner), status="running")
    combined_done = previous.get("status") == "complete" and (analysis/"audit.json").exists()
    status.setdefault("started_utc", now())
    status.setdefault("completed_stages", [])
    status.pop("error", None)
    status.pop("failed_stage", None)
    base = [sys.executable, str(runner), "--jobs", str(jobs),
        "--output-dir", str(output), "--analysis-dir", str(analysis)]
    if campaign_settings.get("smoke"):
        base += ["--smoke"]
    if "frame_averaging_mode" in campaign_settings:
        base += ["--frame-averaging-mode", campaign_settings["frame_averaging_mode"]]
    if "bandwidths" in campaign_settings:
        base += ["--bandwidths", ",".join(map(str, campaign_settings["bandwidths"]))]
    for key, option in (("strengths", "--strengths"), ("bv_reg_values", "--bv-reg-values")):
        if key in campaign_settings:
            base += [option, ",".join(map(str, campaign_settings[key]))]
    try:
        for stage in STAGES:
            audit = analysis/f"audit_{stage}.json"
            report = analysis/f"report_{stage}.html"
            if stage in status["completed_stages"] and audit.exists() and report.exists():
                print(f"Reuse completed experiment: {stage}", flush=True)
                continue
            status.update(current_stage=stage, stage_started_utc=now())
            write_status(status_path, status)
            print(f"Start experiment {stage}, {jobs} workers", flush=True)
            command = base+["--phase", "all", "--experiment", stage]
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise RuntimeError(f"Experiment {stage} exited {result.returncode}; subsequent experiments were not started")
            verified = json.loads(audit.read_text())
            if verified["completed_fits"] != verified["expected_fits"]:
                raise ValueError(f"Incomplete experiment audit: {stage}")
            if stage not in status["completed_stages"]:
                status["completed_stages"].append(stage)
            write_status(status_path, status)
            print(f"Finished experiment {stage}: {verified['completed_fits']} audited histories", flush=True)
        if not combined_done:
            status["current_stage"] = "combined_analysis"
            write_status(status_path, status)
            result = subprocess.run(base+["--phase", "analyze", "--experiment", "all"], check=False)
            if result.returncode:
                raise RuntimeError(f"Combined analysis exited {result.returncode}")
            sigma_plot = runner.parents[2]/"analysis/plot_moprp_bv_coordinate_sigma.py"
            sigma_requested = (analysis/"audit_profiled_coordinate_sigma.json").exists() or (
                campaign_settings.get("frame_averaging_mode") in {"rate", "linear_uptake"}
                and "bv_reg_values" in campaign_settings)
            if sigma_requested and sigma_plot.exists():
                status["current_stage"] = "coordinate_sigma_plot"
                write_status(status_path, status)
                result = subprocess.run([sys.executable, str(sigma_plot), "--analysis-dir", str(analysis),
                    "--campaign-dir", str(output)], check=False)
                if result.returncode:
                    raise RuntimeError(f"Coordinate Sigma-MSE plot update exited {result.returncode}")
        status.update(status="complete", current_stage="complete", finished_utc=now())
        write_status(status_path, status)
        print("All MoPrP experiments and combined analysis complete", flush=True)
    except Exception as error:
        status.update(status="failed", failed_stage=status.get("current_stage"), error=str(error))
        write_status(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"_moprp_policy_sidecars")
    parser.add_argument("--analysis-dir", type=Path, default=EXPERIMENT/"analysis/_moprp_policy_sidecars")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--runner", type=Path, default=HERE/"run_moprp_policy_sidecars.py")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    output, analysis = args.output_dir.resolve(), args.analysis_dir.resolve()
    if args.status:
        path = output/"background_status.json"
        status = json.loads(path.read_text()) if path.exists() else dict(status="not_submitted")
        counts = {"all_pairs": 0, "knn20": 0}
        manifest = output/"manifest.json"
        def identity(spec):
            return json.dumps(dict(spec, frame_averaging_mode=spec.get("frame_averaging_mode", "frame_uptake")), sort_keys=True)
        active = {identity(s) for s in json.loads(manifest.read_text()).get("specs", [])} if manifest.exists() else None
        inactive = 0
        for path in output.glob("*/*/*_config.json"):
            try:
                config = json.loads(path.read_text()).get("moprp_sidecar", {}).get("spec", {})
            except (OSError, ValueError):
                continue  # A native worker may currently be writing its configuration.
            if config.get("method") == "omc":
                if active is not None and identity(config) not in active:
                    inactive += 1
                    continue
                stage = "knn20" if config.get("graph_k") == 20 else "all_pairs"
                counts[stage] += 1
        status["completed_new_fits"] = counts
        status["inactive_completed_omc_fits"] = inactive
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if args.jobs < 1:
        parser.error("Workers must be positive")
    if not args.runner.is_file():
        parser.error("Campaign runner must be an existing script")
    if not (output/"manifest.json").exists():
        parser.error("Prepare the campaign with run_moprp_policy_sidecars.py --phase prepare first")
    with (output/"background.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another MoPrP background controller is running") from error
        run_campaign(output, analysis, args.jobs, args.runner)


if __name__ == "__main__":
    main()
