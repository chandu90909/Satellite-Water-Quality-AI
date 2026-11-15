from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]


def save_index(index_array, src, lake, index_name, image_name):
    """
    Save a calculated water-quality index as:
    1. GeoTIFF
    2. PNG preview
    """

    # ---------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------

    output_dir = (
        BASE_DIR
        / "data"
        / "processed"
        / lake
        / index_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Clean invalid values
    # ---------------------------------------------------------

    index_array = index_array.astype("float32")

    index_array = np.where(
        np.isfinite(index_array),
        index_array,
        np.nan
    )

    # ---------------------------------------------------------
    # Save GeoTIFF
    # ---------------------------------------------------------

    output_tif = output_dir / f"{image_name}_{index_name}.tif"

    profile = src.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="lzw"
    )

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(index_array, 1)

    # ---------------------------------------------------------
    # Save PNG preview
    # ---------------------------------------------------------

    output_png = output_dir / f"{image_name}_{index_name}.png"

    plt.figure(figsize=(8, 6))

    plt.imshow(
        index_array,
        cmap="viridis"
    )

    plt.colorbar(label=index_name.upper())

    plt.title(f"{lake} - {image_name} - {index_name.upper()}")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_png,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {output_tif}")
    print(f"  Saved: {output_png}")