from pathlib import Path
from types import SimpleNamespace

import run_project
from run_project import main


def test_quality_inference_entrypoint_exists():
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "classification"
        / "inference"
        / "run_quality_inference.py"
    )
    assert entrypoint.exists(), "Missing classification inference entrypoint"


def test_run_project_accepts_all_flag(monkeypatch):
    calls = []

    def fake_run_stage(stage):
        calls.append(stage)
        return 0

    monkeypatch.setattr("run_project.run_stage", fake_run_stage)
    exit_code = main(["--all"])

    assert exit_code == 0
    assert calls == ["all"]


def test_all_stage_runs_full_pipeline_in_order(monkeypatch):
    commands = []

    def fake_subprocess_run(command, cwd):
        commands.append(Path(command[1]).name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_project.subprocess, "run", fake_subprocess_run)

    assert run_project.run_stage("all") == 0
    assert commands == [
        "run_quality_inference.py",
        "severity_mapper.py",
        "generate_gradcam.py",
        "temporal_analysis.py",
        "validate_outputs.py",
    ]
