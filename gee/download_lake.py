"""Download Sentinel-2 acquisitions for any lake and time period.

Examples::

    python gee/download_lake.py --lake "Kolleru Lake" --start 2025-01-01 --end 2025-03-31
    python gee/download_lake.py --lake "Hussain Sagar" --start 2025-01-01 --end 2025-03-31 --cadence scene
    python gee/download_lake.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gee.boundary import BoundaryError, LakeBoundary, lake_slug, resolve_boundary
from gee.config import EarthEngineError, check_connection, initialize
from gee.sentinel2 import (
    DEFAULT_MAX_CLOUD,
    DEFAULT_SCALE,
    Period,
    download_period,
    download_scenes,
    month_periods,
    parse_date,
)

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_NAME = "acquisition_manifest.json"


@dataclass
class DownloadResult:
    lake: str
    slug: str
    boundary: LakeBoundary
    cadence: str
    paths: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    empty_periods: list[str] = field(default_factory=list)

    def image_names(self) -> list[str]:
        return [path.name for path in self.paths]


def _write_manifest(lake_dir: Path, boundary: LakeBoundary, cadence: str, records: list[dict]) -> Path:
    path = lake_dir / MANIFEST_NAME
    payload = {"lake": lake_dir.name, "acquisitions": []}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {"lake": lake_dir.name, "acquisitions": []}
    existing = {
        record.get("label"): record
        for record in payload.get("acquisitions", [])
        if isinstance(record, dict) and record.get("label")
    }
    for record in records:
        existing[record["label"]] = record
    payload.update({
        "lake": lake_dir.name,
        "slug": boundary.slug,
        "boundary": {
            "source": boundary.source,
            "bbox": list(boundary.bbox),
            "display_name": boundary.display_name,
        },
        "cadence": cadence,
        "bands": ["B3", "B4", "B5", "B8", "B11", "SCL"],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acquisitions": [existing[key] for key in sorted(existing)],
    })
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def download_lake(
    lake: str,
    start: str | None = None,
    end: str | None = None,
    *,
    cadence: str = "monthly",
    max_cloud: float = DEFAULT_MAX_CLOUD,
    scale: int = DEFAULT_SCALE,
    boundary_path: str | Path | None = None,
    project: str | None = None,
    refresh_boundary: bool = False,
    overwrite: bool = False,
    log=print,
) -> DownloadResult:
    """Resolve the lake boundary and download Sentinel-2 imagery for the period."""
    if cadence not in {"monthly", "scene"}:
        raise ValueError("cadence must be 'monthly' or 'scene'")
    if start is None or end is None:
        raise ValueError("A start and end date are required to download imagery")
    start_date = parse_date(start)
    end_date = parse_date(end)
    if end_date < start_date:
        raise ValueError("End date must be on or after the start date")

    log(f"Resolving boundary for '{lake}' ...")
    boundary = resolve_boundary(lake, boundary_path, refresh=refresh_boundary)
    log(f"Boundary: {boundary.slug} ({boundary.display_name}) via {boundary.source}")

    initialize(project)
    lake_dir = RAW_ROOT / boundary.slug
    lake_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    result = DownloadResult(lake=lake, slug=boundary.slug, boundary=boundary, cadence=cadence)

    if cadence == "monthly":
        periods = month_periods(start_date, end_date)
        log(f"Monthly composites: {len(periods)} period(s) at {scale} m")
        for period in periods:
            path = lake_dir / f"{boundary.slug}_{period.label}.tif"
            if path.exists() and path.stat().st_size > 0 and not overwrite:
                log(f"  {period.label}: already downloaded ({path.name})")
                result.paths.append(path)
                result.skipped.append(period.label)
            else:
                log(f"  {period.label}: exporting {period.describe()} ...")
                downloaded = download_period(
                    boundary, period, path, scale=scale, max_cloud=max_cloud, project=project
                )
                if downloaded is None:
                    log(f"  {period.label}: no clear Sentinel-2 scene, skipped")
                    result.empty_periods.append(period.label)
                    continue
                result.paths.append(downloaded)
            records.append({
                "label": period.label,
                "image": path.name,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "cadence": "monthly",
                "max_cloud": max_cloud,
                "scale": scale,
            })
    else:
        log(f"Clear acquisitions: exporting scenes at {scale} m ...")
        paths = download_scenes(
            boundary, start_date, end_date, lake_dir,
            scale=scale, max_cloud=max_cloud, project=project,
        )
        for path in paths:
            label = path.stem.split("_")[-1]
            result.paths.append(path)
            records.append({
                "label": label,
                "image": path.name,
                "cadence": "scene",
                "max_cloud": max_cloud,
                "scale": scale,
            })
        if not paths:
            log("No clear Sentinel-2 scenes matched the period and cloud filter")

    manifest = _write_manifest(lake_dir, boundary, cadence, records)
    log(f"Manifest: {manifest}")
    log(f"Downloaded/available acquisitions: {len(result.paths)}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel-2 imagery for a lake from Google Earth Engine.")
    parser.add_argument("--lake", help="Lake name, e.g. \"Kolleru Lake\"")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cadence", choices=("monthly", "scene"), default="monthly",
                        help="Monthly median composite (default) or every clear acquisition")
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD,
                        help="Maximum scene cloud cover percentage (default 40)")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Export resolution in metres")
    parser.add_argument("--boundary", help="Optional GeoJSON lake boundary")
    parser.add_argument("--refresh-boundary", action="store_true", help="Ignore the cached boundary")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing acquisitions")
    parser.add_argument("--project", help="Google Cloud project ID for Earth Engine")
    parser.add_argument("--check", action="store_true", help="Verify Earth Engine credentials and exit")
    args = parser.parse_args(argv)

    if args.check:
        try:
            status = check_connection(args.project)
        except EarthEngineError as exc:
            print(f"Earth Engine not ready.\n{exc}", file=sys.stderr)
            return 1
        print(json.dumps(status, indent=2))
        return 0

    if not args.lake:
        parser.error("--lake is required")
    if not args.start or not args.end:
        parser.error("--start and --end are required")

    try:
        result = download_lake(
            args.lake,
            args.start,
            args.end,
            cadence=args.cadence,
            max_cloud=args.max_cloud,
            scale=args.scale,
            boundary_path=args.boundary,
            project=args.project,
            refresh_boundary=args.refresh_boundary,
            overwrite=args.overwrite,
        )
    except (BoundaryError, EarthEngineError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # network / Earth Engine failures
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"\n{len(result.paths)} acquisition(s) ready in data/raw/{result.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
