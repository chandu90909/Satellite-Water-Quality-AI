#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.inference.quality_inference import process_all, process_lake


def main():
    parser = argparse.ArgumentParser(description="Run EfficientNet quality inference on Sentinel-2 water pixels.")
    parser.add_argument("--lake", type=str, default=None, help="Lake name, e.g. Hussain_Sagar")
    parser.add_argument("--image", type=str, default=None, help="Specific image name, e.g. Hussain_Sagar_1.tif")
    parser.add_argument("--all", action="store_true", help="Run all lakes")
    args = parser.parse_args()

    if args.all:
        summaries = process_all()
        print(f"Processed {len(summaries)} lake-image summaries.")
        return 0

    if args.lake is None:
        raise ValueError("Provide --lake when not using --all")

    if args.image is not None:
        from classification.inference.quality_inference import process_image
        process_image(args.lake, args.image)
        print(f"Processed {args.lake}/{args.image}.")
        return 0

    process_lake(args.lake)
    print(f"Processed all images for {args.lake}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
