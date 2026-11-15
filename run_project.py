#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STAGE_SCRIPTS = {
    "classification": ("classification/inference/run_quality_inference.py", ["--all"]),
    "severity": ("severity/severity_mapper.py", []),
    "gradcam": ("explainability/gradcam/generate_gradcam.py", ["--all"]),
    "temporal": ("temporal/temporal_analysis.py", []),
    "validation": ("reports/validate_outputs.py", []),
}
STAGES = ["classification", "severity", "gradcam", "temporal", "validation"]


def run_stage(stage: str):
    if stage == "all":
        for item in STAGES:
            exit_code = run_stage(item)
            if exit_code != 0:
                return exit_code
        return 0

    if stage not in STAGE_SCRIPTS:
        raise ValueError(f"Unsupported stage: {stage}")

    script, extra_args = STAGE_SCRIPTS[stage]
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        print(f"Missing stage script: {script_path}", file=sys.stderr)
        return 1

    command = [sys.executable, str(script_path), *extra_args]
    print(f"\n=== {stage}: {Path(script).name} {' '.join(extra_args)} ===".rstrip(), flush=True)
    return subprocess.run(command, cwd=str(PROJECT_ROOT)).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Master runner for the satellite water-quality project.")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default=None)
    parser.add_argument("--all", dest="run_all", action="store_true", help="Run all project stages")
    args = parser.parse_args(argv)

    stage = "all" if args.run_all or args.stage is None else args.stage
    return run_stage(stage)


if __name__ == "__main__":
    raise SystemExit(main())
