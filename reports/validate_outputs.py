import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


CLASS_CODES = {"Good": 1, "Moderate": 2, "Poor": 3}
SEVERITY_NAMES = {"Good": "Low", "Moderate": "Medium", "Poor": "High"}
PATCH_SIZE = 128


def validate_outputs(project_root: Path):
    raw_root = project_root / "data" / "raw"
    prediction_root = project_root / "outputs" / "quality_predictions"
    quality_root = project_root / "outputs" / "gis" / "quality_maps"
    severity_root = project_root / "outputs" / "gis" / "severity_maps"
    errors = []
    images = [
        (lake_dir.name, image_path)
        for lake_dir in sorted(raw_root.iterdir())
        if lake_dir.is_dir()
        for image_path in sorted(lake_dir.glob("*.tif"))
    ] if raw_root.exists() else []

    if not images:
        return [f"No source TIFFs found under {raw_root}"], 0

    for lake_name, image_path in images:
        image_name = image_path.stem
        csv_path = prediction_root / lake_name / f"{image_name}_quality_predictions.csv"
        quality_path = quality_root / f"{image_name}_quality.tif"
        severity_path = severity_root / f"{image_name}_severity.tif"
        mask_path = project_root / "data" / "masks" / "water" / lake_name / f"{image_name}_water_mask.tif"

        if not csv_path.exists():
            errors.append(f"Missing predictions CSV: {csv_path}")
            continue
        if not quality_path.exists():
            errors.append(f"Missing quality map: {quality_path}")
            continue
        if not severity_path.exists():
            errors.append(f"Missing severity map: {severity_path}")
            continue
        if not mask_path.exists():
            errors.append(f"Missing water mask: {mask_path}")
            continue

        try:
            predictions = pd.read_csv(csv_path)
        except Exception as exc:
            errors.append(f"Cannot read predictions CSV {csv_path}: {exc}")
            continue

        required_columns = {"lake", "image", "row", "col", "prediction", "confidence", "severity"}
        missing_columns = required_columns.difference(predictions.columns)
        if missing_columns:
            errors.append(f"{csv_path} is missing columns: {', '.join(sorted(missing_columns))}")
            continue

        invalid_classes = set(predictions["prediction"].dropna()) - set(CLASS_CODES)
        if invalid_classes:
            errors.append(f"{csv_path} contains unknown classes: {', '.join(sorted(invalid_classes))}")
        gradcam_dir = project_root / "outputs" / "gradcam" / lake_name
        for class_name in set(predictions["prediction"].dropna()).intersection(CLASS_CODES):
            gradcam_path = gradcam_dir / f"{image_name}_{class_name.lower()}_gradcam.png"
            if not gradcam_path.is_file() or gradcam_path.stat().st_size == 0:
                errors.append(f"Missing Grad-CAM image for {image_name} class {class_name}")

        expected_severity = predictions["prediction"].map(SEVERITY_NAMES)
        if not predictions["severity"].equals(expected_severity):
            errors.append(f"{csv_path} has severity labels inconsistent with predictions")

        confidence = pd.to_numeric(predictions["confidence"], errors="coerce")
        if confidence.isna().any() or not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            errors.append(f"{csv_path} has invalid confidence values")

        try:
            with rasterio.open(image_path) as source, rasterio.open(quality_path) as quality_src, rasterio.open(severity_path) as severity_src, rasterio.open(mask_path) as mask_src:
                quality = quality_src.read(1)
                severity = severity_src.read(1)
                water_mask = mask_src.read(1)
                expected_shape = (source.height, source.width)
                for label, dataset in (("quality", quality_src), ("severity", severity_src)):
                    if (dataset.height, dataset.width) != expected_shape:
                        errors.append(f"{label} map dimensions do not match source image: {image_name}")
                    if dataset.crs != source.crs or dataset.transform != source.transform:
                        errors.append(f"{label} map georeferencing does not match source image: {image_name}")

                if quality.shape != expected_shape or severity.shape != expected_shape or water_mask.shape != expected_shape:
                    errors.append(f"Raster dimensions are inconsistent for {image_name}")
                    continue
                if not np.isin(quality, [0, 1, 2, 3]).all():
                    errors.append(f"Quality map contains invalid class codes: {image_name}")
                if not np.isin(severity, [0, 1, 2, 3]).all():
                    errors.append(f"Severity map contains invalid class codes: {image_name}")
                if not np.array_equal(quality, severity):
                    errors.append(f"Quality and severity class rasters differ: {image_name}")
                if np.any(quality[water_mask == 0] != 0):
                    errors.append(f"Quality map classifies pixels outside the water mask: {image_name}")

                for row in predictions.itertuples(index=False):
                    try:
                        patch_row, patch_col = int(row.row), int(row.col)
                    except (TypeError, ValueError):
                        errors.append(f"{csv_path} has invalid patch coordinates")
                        break
                    if patch_row < 0 or patch_col < 0 or patch_row >= source.height or patch_col >= source.width:
                        errors.append(f"{csv_path} has out-of-bounds patch coordinates")
                        break
                    expected_code = CLASS_CODES.get(row.prediction)
                    if expected_code is None:
                        continue
                    row_end = min(patch_row + PATCH_SIZE, source.height)
                    col_end = min(patch_col + PATCH_SIZE, source.width)
                    patch_water = water_mask[patch_row:row_end, patch_col:col_end] > 0
                    patch_quality = quality[patch_row:row_end, patch_col:col_end]
                    if not patch_water.any() or not np.all(patch_quality[patch_water] == expected_code):
                        errors.append(f"Prediction CSV and quality map disagree for {image_name} at ({patch_row}, {patch_col})")
                        break
        except Exception as exc:
            errors.append(f"Cannot validate rasters for {image_name}: {exc}")

    summary_path = project_root / "outputs" / "severity" / "severity_summary.csv"
    if not summary_path.exists():
        errors.append(f"Missing severity summary: {summary_path}")
    else:
        try:
            summary = pd.read_csv(summary_path)
            if len(summary) != len(images):
                errors.append(f"Severity summary has {len(summary)} rows for {len(images)} source images")
        except Exception as exc:
            errors.append(f"Cannot read severity summary {summary_path}: {exc}")

    temporal_root = project_root / "outputs" / "temporal"
    for lake_name in sorted({lake_name for lake_name, _ in images}):
        for suffix in ("_temporal_summary.csv", "_temporal.png"):
            temporal_path = temporal_root / f"{lake_name}{suffix}"
            if not temporal_path.exists():
                errors.append(f"Missing temporal output: {temporal_path}")

    return errors, len(images)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate generated water-quality prediction and GIS outputs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    errors, image_count = validate_outputs(args.project_root.resolve())

    if errors:
        print(f"Validation failed: {len(errors)} issue(s) across {image_count} source images.")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validation passed for {image_count} source images and their prediction, quality, severity, Grad-CAM, and temporal outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())