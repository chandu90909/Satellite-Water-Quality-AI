#!/usr/bin/env python3
"""Run the whole pipeline for any lake and time period.

Example::

    python pipeline/run_lake.py --lake "Kolleru Lake" --start 2025-01-01 --end 2025-03-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.lake_pipeline import STAGES, PipelineError, run_lake


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel-2 imagery for a lake and run the full analysis.")
    parser.add_argument("--lake", required=True, help="Lake name, e.g. \"Kolleru Lake\"")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cadence", choices=("monthly", "scene"), default="monthly",
                        help="Monthly median composite (default) or every clear acquisition")
    parser.add_argument("--max-cloud", type=float, default=40, help="Maximum scene cloud cover percentage")
    parser.add_argument("--boundary", help="Optional GeoJSON lake boundary")
    parser.add_argument("--project", help="Google Cloud project ID for Earth Engine")
    parser.add_argument("--stages", default=None,
                        help=f"Comma separated subset of: {', '.join(STAGES)}")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse imagery already in data/raw for this lake")
    args = parser.parse_args(argv)

    stages = None
    if args.stages:
        stages = [name.strip() for name in args.stages.split(",") if name.strip()]
        unknown = [name for name in stages if name not in STAGES]
        if unknown:
            parser.error(f"Unknown stage(s): {', '.join(unknown)}")

    try:
        result = run_lake(
            args.lake,
            args.start,
            args.end,
            cadence=args.cadence,
            max_cloud=args.max_cloud,
            boundary=args.boundary,
            project=args.project,
            stages=stages,
            skip_download=args.skip_download,
        )
    except PipelineError as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # download or model failures
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    summary = result.summary()
    print("\nPipeline summary")
    print(f"  lake: {summary['lake']} ({summary['slug']})")
    print(f"  acquisitions: {summary['acquisitions']}")
    print(f"  classified patches: {summary['classified_patches']}")
    print(f"  grad-cam images: {summary['gradcam_images']}")
    print(f"  stages: {', '.join(summary['completed_stages'])}")
    if summary["validation_issues"]:
        print(f"  validation issues: {len(summary['validation_issues'])}")
        for issue in summary["validation_issues"][:10]:
            print(f"    - {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
