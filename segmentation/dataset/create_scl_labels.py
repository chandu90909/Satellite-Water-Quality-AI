import sys
from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "water_scl"
)


LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]


# ---------------------------------------------------------
# SCL classes
#
# Output:
#   0   = non-water
#   1   = water
#   255 = ignore / uncertain
#
# Sentinel-2 SCL:
#   4 = Vegetation
#   5 = Not-vegetated
#   6 = Water
#
# We use only clear land classes 4 and 5 as negatives.
# Other classes are ignored to avoid forcing uncertain
# pixels into water/non-water.
# ---------------------------------------------------------

WATER_CLASS = 6

NON_WATER_CLASSES = {
    4,  # Vegetation
    5   # Not-vegetated
}

IGNORE_VALUE = 255


total_images = 0
scl_images = 0
skipped_images = 0


# ---------------------------------------------------------
# Process lakes
# ---------------------------------------------------------

for lake in LAKES:

    input_dir = RAW_DIR / lake
    output_dir = OUTPUT_DIR / lake

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tif_files = sorted(
        input_dir.glob("*.tif")
    )

    print("\n" + "=" * 70)
    print(f"LAKE: {lake}")
    print("=" * 70)

    for tif_file in tif_files:

        total_images += 1

        print(f"\nChecking: {tif_file.name}")

        with rasterio.open(tif_file) as src:

            # -------------------------------------------------
            # Find SCL band from its description
            # -------------------------------------------------

            scl_band = None

            for band_number, description in enumerate(
                src.descriptions,
                start=1
            ):

                if description == "SCL":
                    scl_band = band_number
                    break

            # -------------------------------------------------
            # Skip images without SCL
            # -------------------------------------------------

            if scl_band is None:

                print("  SCL: NOT AVAILABLE")
                print("  Status: SKIPPED")

                skipped_images += 1

                continue

            scl_images += 1

            print(
                f"  SCL band: {scl_band}"
            )

            scl = src.read(scl_band)

            # -------------------------------------------------
            # Start with ignore everywhere
            # -------------------------------------------------

            label = np.full(
                scl.shape,
                IGNORE_VALUE,
                dtype=np.uint8
            )

            # -------------------------------------------------
            # Water
            # -------------------------------------------------

            label[scl == WATER_CLASS] = 1

            # -------------------------------------------------
            # Clear non-water
            # -------------------------------------------------

            for class_value in NON_WATER_CLASSES:

                label[scl == class_value] = 0

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            water_pixels = np.sum(
                label == 1
            )

            non_water_pixels = np.sum(
                label == 0
            )

            ignored_pixels = np.sum(
                label == IGNORE_VALUE
            )

            total_pixels = label.size

            print(
                f"  Water pixels     : "
                f"{water_pixels:,} "
                f"({water_pixels / total_pixels * 100:.2f}%)"
            )

            print(
                f"  Non-water pixels : "
                f"{non_water_pixels:,} "
                f"({non_water_pixels / total_pixels * 100:.2f}%)"
            )

            print(
                f"  Ignored pixels   : "
                f"{ignored_pixels:,} "
                f"({ignored_pixels / total_pixels * 100:.2f}%)"
            )

            # -------------------------------------------------
            # Save GeoTIFF
            # -------------------------------------------------

            output_file = (
                output_dir
                / f"{tif_file.stem}_scl_label.tif"
            )

            profile = src.profile.copy()

            profile.update(
                dtype="uint8",
                count=1,
                compress="lzw",
                nodata=IGNORE_VALUE
            )

            with rasterio.open(
                output_file,
                "w",
                **profile
            ) as dst:

                dst.write(
                    label,
                    1
                )

            print(
                f"  Saved: {output_file}"
            )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SCL LABEL GENERATION COMPLETED")
print("=" * 70)

print(
    f"Total raw images checked : {total_images}"
)

print(
    f"SCL-labelled images      : {scl_images}"
)

print(
    f"Images without SCL       : {skipped_images}"
)

print(
    f"Expected total images    : 36"
)

print(
    "\nLabel values:"
)

print(
    "  0   = Non-water"
)

print(
    "  1   = Water"
)

print(
    "  255 = Ignore / uncertain"
)