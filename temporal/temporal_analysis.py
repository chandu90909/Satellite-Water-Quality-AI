#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.ordering import acquisition_sort_key, sort_acquisitions

PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "quality_predictions"
TEMPORAL_ROOT = PROJECT_ROOT / "outputs" / "temporal"


def available_lakes():
    if not PREDICTION_ROOT.exists():
        return []
    return sorted(path.name for path in PREDICTION_ROOT.iterdir() if path.is_dir())


def summarize_lake_quality(lake_name: str):
    prediction_dir = PREDICTION_ROOT / lake_name
    rows = []
    for csv_path in sort_acquisitions(prediction_dir.glob("*_quality_predictions.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        counts = df["prediction"].value_counts(normalize=True).mul(100)
        rows.append({
            "lake": lake_name,
            "image": csv_path.name.replace("_quality_predictions.csv", ""),
            "good_pct": float(counts.get("Good", 0.0)),
            "moderate_pct": float(counts.get("Moderate", 0.0)),
            "poor_pct": float(counts.get("Poor", 0.0)),
        })
    frame = pd.DataFrame(rows, columns=["lake", "image", "good_pct", "moderate_pct", "poor_pct"])
    if not frame.empty:
        frame = frame.sort_values(by="image", key=lambda column: column.map(acquisition_sort_key)).reset_index(drop=True)
    return frame


def create_temporal_plot(lake_name: str, frame: pd.DataFrame):
    if frame.empty:
        return None
    TEMPORAL_ROOT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(frame))
    ax.plot(x, frame["good_pct"], label="Good", marker="o")
    ax.plot(x, frame["moderate_pct"], label="Moderate", marker="o")
    ax.plot(x, frame["poor_pct"], label="Poor", marker="o")
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["image"], rotation=30, ha="right")
    ax.set_title(f"{lake_name} proxy-quality temporal trend")
    ax.set_ylabel("Percentage of classified water patches (%)")
    ax.set_xlabel("Acquisition (numeric order)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = TEMPORAL_ROOT / f"{lake_name}_temporal.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot the temporal quality trend for each lake.")
    parser.add_argument("--lake", type=str, default=None, help="Optional lake name for single-lake analysis")
    args = parser.parse_args()

    lakes = [args.lake] if args.lake else available_lakes()
    if not lakes:
        print(f"No prediction outputs found under {PREDICTION_ROOT}. Run the classification stage first.")
        return 1

    TEMPORAL_ROOT.mkdir(parents=True, exist_ok=True)
    for lake_name in lakes:
        frame = summarize_lake_quality(lake_name)
        out_csv = TEMPORAL_ROOT / f"{lake_name}_temporal_summary.csv"
        frame.to_csv(out_csv, index=False)
        if frame.empty:
            print(f"No classified acquisitions for {lake_name}; wrote an empty summary and skipped the chart.")
            continue
        create_temporal_plot(lake_name, frame)
        print(f"Saved temporal summary for {lake_name}: {out_csv} ({len(frame)} acquisitions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
