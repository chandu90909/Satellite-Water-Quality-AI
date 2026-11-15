from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "unet"
)

SPLITS = [
    "train",
    "val",
    "test"
]

IGNORE_VALUE = 255


for split in SPLITS:

    image_dir = DATASET_DIR / split / "images"
    mask_dir = DATASET_DIR / split / "masks"
    valid_dir = DATASET_DIR / split / "valid"

    image_files = sorted(image_dir.glob("*.npy"))

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATASET")
    print("=" * 70)

    print(f"Image patches : {len(image_files)}")

    total_water = 0
    total_non_water = 0
    total_ignore = 0

    water_ratios = []

    for image_file in image_files:

        mask_file = mask_dir / image_file.name
        valid_file = valid_dir / image_file.name

        mask = np.load(mask_file)
        valid = np.load(valid_file)

        water = np.sum(mask == 1)
        non_water = np.sum(mask == 0)
        ignored = np.sum(mask == IGNORE_VALUE)

        total_water += water
        total_non_water += non_water
        total_ignore += ignored

        valid_pixels = water + non_water

        if valid_pixels > 0:
            ratio = water / valid_pixels
            water_ratios.append(ratio)

    total_valid = (
        total_water + total_non_water
    )

    total_pixels = (
        total_valid + total_ignore
    )

    if total_valid > 0:

        water_percentage = (
            total_water
            / total_valid
            * 100
        )

        non_water_percentage = (
            total_non_water
            / total_valid
            * 100
        )

    else:

        water_percentage = 0
        non_water_percentage = 0

    ignore_percentage = (
        total_ignore
        / total_pixels
        * 100
        if total_pixels > 0
        else 0
    )

    print(
        f"Water pixels     : {total_water:,} "
        f"({water_percentage:.2f}% of valid)"
    )

    print(
        f"Non-water pixels : {total_non_water:,} "
        f"({non_water_percentage:.2f}% of valid)"
    )

    print(
        f"Ignore pixels    : {total_ignore:,} "
        f"({ignore_percentage:.2f}% of all)"
    )

    if water_ratios:

        print(
            f"Minimum water ratio : "
            f"{min(water_ratios) * 100:.2f}%"
        )

        print(
            f"Maximum water ratio : "
            f"{max(water_ratios) * 100:.2f}%"
        )

        print(
            f"Average water ratio : "
            f"{np.mean(water_ratios) * 100:.2f}%"
        )


print("\n" + "=" * 70)
print("DATASET VERIFICATION COMPLETED")
print("=" * 70)