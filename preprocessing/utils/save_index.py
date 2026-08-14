import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio


def save_index(index_array,
               src,
               image_path,
               index_name,
               cmap="viridis"):

    # ---------------------------------------
    # Create Output Folders
    # ---------------------------------------

    png_folder = os.path.join(
        "preprocessing",
        "outputs",
        index_name,
        "png"
    )

    tif_folder = os.path.join(
        "preprocessing",
        "outputs",
        index_name,
        "tif"
    )

    os.makedirs(png_folder, exist_ok=True)
    os.makedirs(tif_folder, exist_ok=True)

    filename = os.path.basename(image_path).replace(".tif", "")

    png_file = os.path.join(
        png_folder,
        filename + "_" + index_name + ".png"
    )

    tif_file = os.path.join(
        tif_folder,
        filename + "_" + index_name + ".tif"
    )

    # ---------------------------------------
    # Save GeoTIFF
    # ---------------------------------------

    profile = src.profile.copy()

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw"
    )

    with rasterio.open(
        tif_file,
        "w",
        **profile
    ) as dst:

        dst.write(index_array.astype(np.float32), 1)

    # ---------------------------------------
    # Save PNG
    # ---------------------------------------

    plt.figure(figsize=(8, 8))

    plt.imshow(index_array, cmap=cmap)

    plt.colorbar(label=index_name.upper())

    plt.title(filename)

    plt.axis("off")

    plt.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------
    # Statistics
    # ---------------------------------------

    valid = index_array[np.isfinite(index_array)]

    print("=" * 60)

    print(filename)

    print("Minimum :", np.min(valid))

    print("Maximum :", np.max(valid))

    print("Mean    :", np.mean(valid))

    print("Saved PNG :", png_file)

    print("Saved TIF :", tif_file)