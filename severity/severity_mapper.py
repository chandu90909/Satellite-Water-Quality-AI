import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.ordering import acquisition_sort_key, sort_acquisitions

CLASS_NAMES = ["Good", "Moderate", "Poor"]
SEVERITY_MAP = {"Good": "Low", "Moderate": "Medium", "Poor": "High"}
SUMMARY_COLUMNS = [
    "lake", "image", "classified_patches",
    "good_percentage", "moderate_percentage", "poor_percentage",
    "low_percentage", "medium_percentage", "high_percentage",
    "dominant_quality", "dominant_severity",
]


def summarize_predictions(prediction_csv: Path):
    df = pd.read_csv(prediction_csv)
    if df.empty:
        return {
            "lake": None,
            "image": None,
            "classified_patches": 0,
            "good_percentage": 0.0,
            "moderate_percentage": 0.0,
            "poor_percentage": 0.0,
            "low_percentage": 0.0,
            "medium_percentage": 0.0,
            "high_percentage": 0.0,
            "dominant_quality": "Good",
            "dominant_severity": "Low",
        }

    counts = df["prediction"].value_counts()
    total = float(len(df))
    quality = {cls: float(counts.get(cls, 0)) / total * 100.0 for cls in CLASS_NAMES}
    severity = {
        "Low": quality.get("Good", 0.0),
        "Medium": quality.get("Moderate", 0.0),
        "High": quality.get("Poor", 0.0),
    }
    dominant_quality = max(CLASS_NAMES, key=lambda cls: quality.get(cls, 0.0))
    dominant_severity = SEVERITY_MAP[dominant_quality]
    return {
        "lake": df["lake"].iloc[0] if "lake" in df.columns else None,
        "image": df["image"].iloc[0] if "image" in df.columns else None,
        "classified_patches": int(len(df)),
        "good_percentage": quality.get("Good", 0.0),
        "moderate_percentage": quality.get("Moderate", 0.0),
        "poor_percentage": quality.get("Poor", 0.0),
        "low_percentage": severity["Low"],
        "medium_percentage": severity["Medium"],
        "high_percentage": severity["High"],
        "dominant_quality": dominant_quality,
        "dominant_severity": dominant_severity,
    }


def generate_severity_map_from_quality(quality_path: Path, out_path: Path):
    with rasterio.open(str(quality_path)) as src:
        quality = src.read(1)
        transform = src.transform
        crs = src.crs
    severity = np.zeros_like(quality, dtype=np.uint8)
    severity[quality == 1] = 1
    severity[quality == 2] = 2
    severity[quality == 3] = 3
    severity[quality == 0] = 0
    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=severity.shape[0],
        width=severity.shape[1],
        count=1,
        dtype=severity.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(severity, 1)
    return severity


def summarize_directory(input_dir: Path, output_dir: Path):
    rows = []
    for csv_path in sort_acquisitions(input_dir.rglob("*_quality_predictions.csv")):
        summary = summarize_predictions(csv_path)
        summary["lake"] = csv_path.parent.name if summary["lake"] is None else summary["lake"]
        summary["image"] = csv_path.name.replace("_quality_predictions.csv", "")
        rows.append(summary)

        quality_map_path = PROJECT_ROOT / "outputs" / "gis" / "quality_maps" / f"{summary['image']}_quality.tif"
        if quality_map_path.exists():
            severity_map_path = PROJECT_ROOT / "outputs" / "gis" / "severity_maps" / f"{summary['image']}_severity.tif"
            severity_map_path.parent.mkdir(parents=True, exist_ok=True)
            generate_severity_map_from_quality(quality_map_path, severity_map_path)

    rows.sort(key=lambda row: (row["lake"] or "", acquisition_sort_key(row["image"] or "")))

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "severity_summary.csv", index=False)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Summarize quality classification outputs into severity percentages.")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "outputs" / "quality_predictions"), help="Directory containing quality prediction CSV outputs")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "outputs" / "severity"), help="Directory to save severity summary CSV")
    args = parser.parse_args()
    summary_df = summarize_directory(Path(args.input), Path(args.output))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
