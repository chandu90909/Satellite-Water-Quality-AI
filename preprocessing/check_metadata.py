import rasterio
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# -------------------------------------------------------
# Read all GeoTIFF files
# -------------------------------------------------------
files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

plt.figure(figsize=(18, 12))

for i, file in enumerate(files):

    with rasterio.open(file) as src:

        # Read Sentinel-2 RGB bands
        red = src.read(4).astype(np.float32)
        green = src.read(3).astype(np.float32)
        blue = src.read(2).astype(np.float32)

    # Stack into RGB image
    rgb = np.dstack((red, green, blue))

    # Replace NoData (0) with NaN
    rgb[rgb == 0] = np.nan

    # Calculate percentiles ignoring NaN values
    p2 = np.nanpercentile(rgb, 2)
    p98 = np.nanpercentile(rgb, 98)

    print(f"\n{os.path.basename(file)}")
    print(f"2% = {p2:.2f}")
    print(f"98% = {p98:.2f}")

    # Contrast stretch
    rgb = (rgb - p2) / (p98 - p2)

    # Clip values
    rgb = np.clip(rgb, 0, 1)

    # Replace NaN with black
    rgb = np.nan_to_num(rgb)

    # Display
    plt.subplot(3, 4, i + 1)
    plt.imshow(rgb)
    plt.title(os.path.basename(file).replace(".tif", ""))
    plt.axis("off")

plt.tight_layout()
plt.show()