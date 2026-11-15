import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.utils.band_reader import (
    read_required_bands,
    read_scl,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data" / "raw"

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "water_v2"
)


LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]


# ---------------------------------------------------------
# Thresholds for 6-band images
# ---------------------------------------------------------

NDWI_THRESHOLD = 0.05
MNDWI_THRESHOLD = 0.10
NDVI_MAX = 0.20


# ---------------------------------------------------------
# Create mask for 6-band images
# ---------------------------------------------------------

def create_spectral_water_mask(b3, b4, b8, b11):

    ndwi = (
        (b3 - b8)
        / (b3 + b8 + 1e-10)
    )

    mndwi = (
        (b3 - b11)
        / (b3 + b11 + 1e-10)
    )

    ndvi = (
        (b8 - b4)
        / (b8 + b4 + 1e-10)
    )

    water = (
        (ndwi > NDWI_THRESHOLD)
        &
        (mndwi > MNDWI_THRESHOLD)
        &
        (ndvi < NDVI_MAX)
    )

    return water.astype(np.uint8)


# ---------------------------------------------------------
# Create mask from SCL
# ---------------------------------------------------------

def create_scl_water_mask(scl):

    # Sentinel-2 SCL class 6 = Water

    water = scl == 6

    return water.astype(np.uint8)


# ---------------------------------------------------------
# Process all images
# ---------------------------------------------------------

total = 0

scl_count = 0
spectral_count = 0


for lake in LAKES:

    input_dir = RAW_DIR / lake
    output_dir = MASK_DIR / lake

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tif_files = sorted(
        input_dir.glob("*.tif")
    )

    print("\n" + "=" * 70)
    print(f"LAKE: {lake}")
    print(f"IMAGES: {len(tif_files)}")
    print("=" * 70)

    for tif_file in tif_files:

        print(
            f"\nProcessing: {tif_file.name}"
        )

        with rasterio.open(tif_file) as src:

            bands = read_required_bands(src)

            scl = read_scl(src)

            # -------------------------------------------------
            # 26-band image → SCL
            # -------------------------------------------------

            if scl is not None:

                mask = create_scl_water_mask(scl)

                method = "SCL"

                scl_count += 1

            # -------------------------------------------------
            # 6-band image → spectral rule
            # -------------------------------------------------

            else:

                mask = create_spectral_water_mask(
                    bands["B3"],
                    bands["B4"],
                    bands["B8"],
                    bands["B11"]
                )

                method = "SPECTRAL"

                spectral_count += 1

            # -------------------------------------------------
            # Save GeoTIFF
            # -------------------------------------------------

            output_tif = (
                output_dir
                / f"{tif_file.stem}_water_mask.tif"
            )

            profile = src.profile.copy()

            profile.update(
                dtype="uint8",
                count=1,
                compress="lzw"
            )

            with rasterio.open(
                output_tif,
                "w",
                **profile
            ) as dst:

                dst.write(mask, 1)

            # -------------------------------------------------
            # Save PNG
            # -------------------------------------------------

            output_png = (
                output_dir
                / f"{tif_file.stem}_water_mask.png"
            )

            png_mask = (
                mask * 255
            ).astype(np.uint8)

            Image.fromarray(
                png_mask
            ).save(output_png)

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            total_pixels = mask.size

            water_pixels = np.sum(
                mask == 1
            )

            water_percentage = (
                water_pixels
                / total_pixels
                * 100
            )

            print(
                f"  Method: {method}"
            )

            print(
                f"  Water pixels: "
                f"{water_pixels:,} / "
                f"{total_pixels:,} "
                f"({water_percentage:.2f}%)"
            )

            print(
                f"  Saved: {output_tif}"
            )

            print(
                f"  Saved: {output_png}"
            )

            total += 1


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("WATER MASK V2 GENERATION COMPLETED")
print("=" * 70)

print(
    f"Total masks generated : {total}"
)

print(
    f"SCL-based masks       : {scl_count}"
)

print(
    f"Spectral-based masks  : {spectral_count}"
)

print(
    f"Expected masks        : 36"
)