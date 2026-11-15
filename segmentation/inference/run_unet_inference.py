#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MASK_ROOT = PROJECT_ROOT / "data" / "masks" / "water"
SUMMARY_PATH = MASK_ROOT / "unet_full_inference_summary.csv"


def ensure_masks_exist():
    if SUMMARY_PATH.exists():
        return True

    lake_dirs = [p for p in (PROJECT_ROOT / "data" / "raw").iterdir() if p.is_dir()]
    if not lake_dirs:
        return False

    generator = PROJECT_ROOT / "segmentation" / "dataset" / "generate_water_masks.py"
    if generator.exists():
        subprocess.run([sys.executable, str(generator)], cwd=str(PROJECT_ROOT), check=False)
    return SUMMARY_PATH.exists()


def main():
    parser = argparse.ArgumentParser(description="Validate or regenerate the existing U-Net water-mask outputs.")
    parser.add_argument("--lake", type=str, default=None, help="Optional lake name, e.g. Hussain_Sagar")
    parser.add_argument("--image", type=str, default=None, help="Optional image name")
    parser.add_argument("--refresh", action="store_true", help="Force regeneration of water masks from the project scripts")
    args = parser.parse_args()

    if args.refresh:
        generator = PROJECT_ROOT / "segmentation" / "dataset" / "generate_water_masks.py"
        if generator.exists():
            subprocess.run([sys.executable, str(generator)], cwd=str(PROJECT_ROOT), check=True)

    if not ensure_masks_exist():
        raise FileNotFoundError("No U-Net water-mask outputs were found and none could be generated. Check data/raw and segmentation/dataset/generate_water_masks.py.")

    lake_filter = args.lake
    image_filter = args.image
    lake_dirs = [p for p in MASK_ROOT.iterdir() if p.is_dir()] if MASK_ROOT.exists() else []
    files = []
    for lake_dir in lake_dirs:
        if lake_filter and lake_dir.name != lake_filter:
            continue
        for tif_path in sorted(lake_dir.glob("*_water_mask.tif")):
            if image_filter and tif_path.name != f"{Path(image_filter).stem}_water_mask.tif":
                continue
            files.append(tif_path)

    print(f"Validated water masks: {len(files)}")
    if files:
        print(files[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
