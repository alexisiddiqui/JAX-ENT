import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[3]/"examples/2_CrossValidation/fitting/jaxENT"
sys.path.insert(0, str(HERE))
import run_moprp_sidecars_background as background


def test_experiments_run_sequentially_and_finished_queue_can_be_reopened(tmp_path, monkeypatch):
    output, analysis = tmp_path/"fits", tmp_path/"analysis"
    output.mkdir()
    (output/"manifest.json").write_text(json.dumps(dict(settings=dict(strengths=[.1,1.,5.], bandwidths=[.001, .01, .1, 1., 10., 100., 1000.]))))
    calls = []
    def run(command, check):
        assert not check
        stage = command[command.index("--experiment")+1]
        assert command[command.index("--jobs")+1] == "10"
        assert command[command.index("--bandwidths")+1] == "0.001,0.01,0.1,1.0,10.0,100.0,1000.0"
        assert command[command.index("--strengths")+1] == "0.1,1.0,5.0"
        calls.append(stage)
        suffix = "" if stage == "all" else f"_{stage}"
        (analysis/f"audit{suffix}.json").write_text(json.dumps(dict(completed_fits=84, expected_fits=84)))
        (analysis/f"report{suffix}.html").write_text("verified")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(background.subprocess, "run", run)
    background.run_campaign(output, analysis, 10)
    assert calls == ["maxent", "all_pairs", "knn20", "all"]
    status = json.loads((output/"background_status.json").read_text())
    assert status["status"] == "complete"
    calls.clear()
    background.run_campaign(output, analysis, 10)
    assert not calls


def test_failed_experiment_stops_queue_and_resume_preserves_completed_stage(tmp_path, monkeypatch):
    output, analysis = tmp_path/"fits", tmp_path/"analysis"
    output.mkdir()
    (output/"manifest.json").write_text("{}")
    calls = []
    def run(command, check):
        stage = command[command.index("--experiment")+1]
        calls.append(stage)
        if stage == "maxent":
            (analysis/"audit_maxent.json").write_text(json.dumps(dict(completed_fits=84, expected_fits=84)))
            (analysis/"report_maxent.html").write_text("verified")
        return subprocess.CompletedProcess(command, 1 if stage == "all_pairs" else 0)
    monkeypatch.setattr(background.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="subsequent experiments were not started"):
        background.run_campaign(output, analysis, 10)
    assert calls == ["maxent", "all_pairs"]
    status = json.loads((output/"background_status.json").read_text())
    assert status["completed_stages"] == ["maxent"]
    assert status["failed_stage"] == "all_pairs"
    calls.clear()
    def resume(command, check):
        stage = command[command.index("--experiment")+1]
        calls.append(stage)
        suffix = "" if stage == "all" else f"_{stage}"
        (analysis/f"audit{suffix}.json").write_text(json.dumps(dict(completed_fits=420, expected_fits=420)))
        (analysis/f"report{suffix}.html").write_text("verified")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(background.subprocess, "run", resume)
    background.run_campaign(output, analysis, 10)
    assert calls == ["all_pairs", "knn20", "all"]
    assert json.loads((output/"background_status.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("mode", ["rate", "linear_uptake"])
def test_explicit_bv_runner_and_smoke_policy_survive_controller_restart(tmp_path, monkeypatch, mode):
    output, analysis = tmp_path/"fits", tmp_path/"analysis"
    output.mkdir()
    (output/"manifest.json").write_text(json.dumps(dict(settings=dict(smoke=True, frame_averaging_mode=mode, bandwidths=[.001, 1., 1000.]))))
    runner = tmp_path/"bv_runner.py"
    runner.write_text("# fixture")
    calls = []
    def run(command, check):
        assert command[1] == str(runner.resolve())
        assert "--smoke" in command
        assert command[command.index("--frame-averaging-mode")+1] == mode
        calls.append(command)
        stage = command[command.index("--experiment")+1]
        suffix = "" if stage == "all" else f"_{stage}"
        (analysis/f"audit{suffix}.json").write_text(json.dumps(dict(completed_fits=56, expected_fits=56)))
        (analysis/f"report{suffix}.html").write_text("smoke")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(background.subprocess, "run", run)
    background.run_campaign(output, analysis, 10, runner)
    assert len(calls) == 4
    calls.clear()
    background.run_campaign(output, analysis, 10, runner)
    assert not calls
    with pytest.raises(ValueError, match="runner changed"):
        background.run_campaign(output, analysis, 10)
