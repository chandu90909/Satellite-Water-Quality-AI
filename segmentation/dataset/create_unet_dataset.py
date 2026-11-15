import sys
import random
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.utils.band_reader import read_required_bands


RAW_DIR = PROJECT_ROOT / "data" / "raw"

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "water_scl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "unet"
)

PATCH_SIZE = 256

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42

IGNORE_VALUE = 255

MIN_VALID_RATIO = 0.80
MIN_WATER_RATIO = 0.01

# Keep some patches containing no water
MAX_EMPTY_PATCH_RATIO = 0.30

LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]

random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# Collect image/label pairs
# ---------------------------------------------------------

image_pairs = []

for lake in LAKES:

    raw_dir = RAW_DIR / lake
    label_dir = LABEL_DIR / lake

    label_files = sorted(
        label_dir.glob("*_scl_label.tif")
    )

    for label_file in label_files:

        image_name = (
            label_file.name
            .replace("_scl_label.tif", ".tif")
        )

        image_file = raw_dir / image_name

        if image_file.exists():

            image_pairs.append(
                (
                    lake,
                    image_file,
                    label_file
                )
            )


print("=" * 80)
print("U-NET DATASET CREATION")
print("=" * 80)

print(
    f"Total labelled images: {len(image_pairs)}"
)


# ---------------------------------------------------------
# Split by image
# ---------------------------------------------------------

random.shuffle(image_pairs)

total = len(image_pairs)

train_end = int(total * TRAIN_RATIO)

val_end = (
    train_end
    + int(total * VAL_RATIO)
)

train_pairs = image_pairs[:train_end]

val_pairs = image_pairs[
    train_end:val_end
]

test_pairs = image_pairs[
    val_end:
]


print("\nIMAGE SPLIT")
print("-" * 60)

print(f"Train images      : {len(train_pairs)}")
print(f"Validation images : {len(val_pairs)}")
print(f"Test images       : {len(test_pairs)}")


# ---------------------------------------------------------
# Create directories
# ---------------------------------------------------------

for split in ["train", "val", "test"]:

    (
        OUTPUT_DIR / split / "images"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_DIR / split / "masks"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_DIR / split / "valid"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Process one split
# ---------------------------------------------------------

def create_patches(pairs, split):

    candidates = []

    print("\n" + "=" * 80)
    print(f"PREPARING {split.upper()} PATCHES")
    print("=" * 80)

    for lake, image_file, label_file in pairs:

        print(
            f"Processing image: "
            f"{lake}/{image_file.name}"
        )

        with rasterio.open(image_file) as src:

            bands = read_required_bands(src)

            image = np.stack(
                [
                    bands["B3"],
                    bands["B4"],
                    bands["B5"],
                    bands["B8"],
                    bands["B11"]
                ],
                axis=0
            ).astype(np.float32)

        with rasterio.open(label_file) as src:

            mask = src.read(1)

        height = min(
            image.shape[1],
            mask.shape[0]
        )

        width = min(
            image.shape[2],
            mask.shape[1]
        )

        image = image[:, :height, :width]
        mask = mask[:height, :width]

        # -------------------------------------------------
        # Normalize each channel
        # -------------------------------------------------

        for channel in range(image.shape[0]):

            band = image[channel]

            valid = np.isfinite(band)

            if np.any(valid):

                low = np.percentile(
                    band[valid], 2
                )

                high = np.percentile(
                    band[valid], 98
                )

                if high > low:

                    band = (
                        band - low
                    ) / (
                        high - low
                    )

                    band = np.clip(
                        band,
                        0,
                        1
                    )

                    image[channel] = band

        # -------------------------------------------------
        # Extract candidate patches
        # -------------------------------------------------

        for row in range(
            0,
            height - PATCH_SIZE + 1,
            PATCH_SIZE
        ):

            for col in range(
                0,
                width - PATCH_SIZE + 1,
                PATCH_SIZE
            ):

                image_patch = image[
                    :,
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]

                mask_patch = mask[
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]

                valid_pixels = np.sum(
                    mask_patch != IGNORE_VALUE
                )

                valid_ratio = (
                    valid_pixels
                    / mask_patch.size
                )

                if valid_ratio < MIN_VALID_RATIO:
                    continue

                water_pixels = np.sum(
                    mask_patch == 1
                )

                non_water_pixels = np.sum(
                    mask_patch == 0
                )

                valid_total = (
                    water_pixels
                    + non_water_pixels
                )

                if valid_total == 0:
                    continue

                water_ratio = (
                    water_pixels
                    / valid_total
                )

                candidates.append(
                    {
                        "lake": lake,
                        "image": image_file.stem,
                        "row": row,
                        "col": col,
                        "image_data": image_patch,
                        "mask_data": mask_patch,
                        "water_ratio": water_ratio
                    }
                )

    # -----------------------------------------------------
    # Separate water and empty patches
    # -----------------------------------------------------

    water_candidates = [
        x for x in candidates
        if x["water_ratio"] >= MIN_WATER_RATIO
    ]

    empty_candidates = [
        x for x in candidates
        if x["water_ratio"] < MIN_WATER_RATIO
    ]

    random.shuffle(empty_candidates)

    max_empty = int(
        len(water_candidates)
        * MAX_EMPTY_PATCH_RATIO
        / (1 - MAX_EMPTY_PATCH_RATIO)
    )

    empty_candidates = empty_candidates[
        :max_empty
    ]

    selected = (
        water_candidates
        + empty_candidates
    )

    random.shuffle(selected)

    # -----------------------------------------------------
    # Save patches
    # -----------------------------------------------------

    patch_count = 0

    for item in selected:

        base_name = (
            f"{item['lake']}_"
            f"{item['image']}_"
            f"r{item['row']}_"
            f"c{item['col']}"
        )

        image_output = (
            OUTPUT_DIR
            / split
            / "images"
            / f"{base_name}.npy"
        )

        mask_output = (
            OUTPUT_DIR
            / split
            / "masks"
            / f"{base_name}.npy"
        )

        valid_output = (
            OUTPUT_DIR
            / split
            / "valid"
            / f"{base_name}.npy"
        )

        # Keep 255 in the mask
        np.save(
            mask_output,
            item["mask_data"].astype(np.uint8)
        )

        # Valid pixels = 1
        valid_mask = (
            item["mask_data"] != IGNORE_VALUE
        ).astype(np.uint8)

        np.save(
            valid_output,
            valid_mask
        )

        np.save(
            image_output,
            item["image_data"].astype(
                np.float32
            )
        )

        patch_count += 1

    print(
        f"\n{split.upper()} PATCHES: "
        f"{patch_count}"
    )

    print(
        f"  Water-containing candidates: "
        f"{len(water_candidates)}"
    )

    print(
        f"  Empty candidates retained: "
        f"{len(empty_candidates)}"
    )

    return patch_count


# ---------------------------------------------------------
# Create all splits
# ---------------------------------------------------------

train_count = create_patches(
    train_pairs,
    "train"
)

val_count = create_patches(
    val_pairs,
    "val"
)

test_count = create_patches(
    test_pairs,
    "test"
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("U-NET DATASET CREATION COMPLETED")
print("=" * 80)

print(f"Train patches      : {train_count}")
print(f"Validation patches : {val_count}")
print(f"Test patches       : {test_count}")

print(
    f"Total patches      : "
    f"{train_count + val_count + test_count}"
)

print("\nEach patch contains:")
print("  Image : 5 spectral channels")
print("  Mask  : 0=non-water, 1=water, 255=ignore")
print("  Valid : 1=valid pixel, 0=ignored pixel")