"""Run the complete water-quality workflow for a single lake.

Steps, in order: Earth Engine download, water masks, EfficientNet classification,
severity summaries, Grad-CAM, temporal analysis and output validation. Stages are
skipped when they are not requested, which lets the dashboard reuse imagery that
is already on disk.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "quality_predictions"
SEVERITY_ROOT = PROJECT_ROOT / "outputs" / "severity"
TEMPORAL_ROOT = PROJECT_ROOT / "outputs" / "temporal"

STAGES = ("download", "water_masks", "classification", "severity", "gradcam", "temporal", "validation")
STAGE_LABELS = {
    "download": "Downloading Sentinel-2 imagery from Earth Engine",
    "water_masks": "Generating water masks",
    "classification": "Classifying water patches with EfficientNet-B0",
    "severity": "Aggregating severity and quality percentages",
    "gradcam": "Generating Grad-CAM explanations",
    "temporal": "Building temporal trend summaries",
    "validation": "Validating generated outputs",
}


class PipelineError(RuntimeError):
    """Raised when a required pipeline stage cannot produce usable output."""


@dataclass
class PipelineResult:
    lake: str
    slug: str
    completed: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    classified_patches: int = 0
    gradcam_images: int = 0
    validation_errors: list[str] = field(default_factory=list)
    download_skipped: list[str] = field(default_factory=list)
    empty_periods: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "lake": self.lake,
            "slug": self.slug,
            "acquisitions": len(self.images),
            "classified_patches": self.classified_patches,
            "gradcam_images": self.gradcam_images,
            "completed_stages": list(self.completed),
            "validation_issues": list(self.validation_errors),
        }


def run_lake(
    lake: str,
    start: str | None = None,
    end: str | None = None,
    *,
    cadence: str = "monthly",
    max_cloud: float = 40,
    boundary: str | Path | None = None,
    project: str | None = None,
    stages=None,
    skip_download: bool = False,
    log=print,
    on_stage=None,
) -> PipelineResult:
    """Download (optionally) and analyse one lake end to end."""
    from gee.boundary import lake_slug

    requested = [stage for stage in STAGES if stage in (stages or STAGES)]
    selected = set(requested)
    if "download" not in selected:
        skip_download = True
    if not skip_download and (not start or not end):
        raise PipelineError("A start and end date are required to download imagery")

    slug = lake_slug(lake)
    result = PipelineResult(lake=lake, slug=slug)
    log(f"Analysing lake '{lake}' (folder: {slug})")

    def mark(stage: str):
        log(f"[{stage}] {STAGE_LABELS[stage]}")
        if on_stage is not None:
            on_stage(stage)

    if not skip_download:
        mark("download")
        from gee.boundary import BoundaryError
        from gee.config import EarthEngineError
        from gee.download_lake import download_lake

        try:
            download = download_lake(
                lake,
                start,
                end,
                cadence=cadence,
                max_cloud=max_cloud,
                boundary_path=boundary,
                project=project,
                log=log,
            )
        except (BoundaryError, EarthEngineError) as exc:
            raise PipelineError(str(exc)) from exc
        result.download_skipped = download.skipped
        result.empty_periods = download.empty_periods
        if not download.paths:
            raise PipelineError(
                "No Sentinel-2 acquisitions were available for that lake and period. "
                "Try a longer period, a higher cloud limit, or provide a boundary file."
            )
        result.completed.append("download")

    lake_dir = RAW_ROOT / slug
    if not lake_dir.is_dir():
        raise PipelineError(f"No imagery found for '{lake}' in {lake_dir}")
    from common.ordering import sort_acquisitions
    images = sort_acquisitions([path.name for path in lake_dir.glob("*.tif")])
    if not images:
        raise PipelineError(f"No GeoTIFF acquisitions found for '{lake}'")
    result.images = images
    log(f"Acquisitions on disk: {len(images)}")

    if "water_masks" in selected:
        mark("water_masks")
        from segmentation.masks.water_masks import generate_for_lake

        masks = generate_for_lake(slug, images)
        log(f"Water masks generated: {len(masks)}")
        result.completed.append("water_masks")

    if "classification" in selected:
        mark("classification")
        from classification.inference.quality_inference import process_lake

        summaries = process_lake(slug)
        classified = int(sum(len(frame) for frame in summaries))
        if classified == 0:
            raise PipelineError("No water patches were classified; check the water mask coverage")
        result.classified_patches = classified
        log(f"Classified {classified} water patches across {len(summaries)} acquisition(s)")
        result.completed.append("classification")

    if "severity" in selected:
        mark("severity")
        from severity.severity_mapper import summarize_directory

        summary = summarize_directory(PREDICTION_ROOT, SEVERITY_ROOT)
        log(f"Severity summary refreshed: {len(summary)} acquisition row(s)")
        result.completed.append("severity")

    if "gradcam" in selected:
        mark("gradcam")
        from explainability.gradcam.generate_gradcam import generate_for_lake as generate_gradcam_for_lake

        paths = generate_gradcam_for_lake(slug, images, log=log)
        result.gradcam_images = len(paths)
        log(f"Grad-CAM overlays generated: {len(paths)}")
        result.completed.append("gradcam")

    if "temporal" in selected:
        mark("temporal")
        from temporal import temporal_analysis

        frame = temporal_analysis.summarize_lake_quality(slug)
        if frame.empty:
            raise PipelineError("No classified acquisitions are available for temporal analysis")
        TEMPORAL_ROOT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(TEMPORAL_ROOT / f"{slug}_temporal_summary.csv", index=False)
        temporal_analysis.create_temporal_plot(slug, frame)
        log(f"Temporal summary: {len(frame)} acquisition(s)")
        result.completed.append("temporal")

    if "validation" in selected:
        mark("validation")
        from reports.validate_outputs import validate_outputs

        errors, image_count = validate_outputs(PROJECT_ROOT)
        result.validation_errors = list(errors)
        if errors:
            log(f"Validation reported {len(errors)} issue(s) across {image_count} source image(s)")
            for error in errors[:10]:
                log(f"  - {error}")
        else:
            log(f"Validation passed for {image_count} source image(s)")
        result.completed.append("validation")

    log(f"Finished '{lake}': {', '.join(result.completed)}")
    return result
