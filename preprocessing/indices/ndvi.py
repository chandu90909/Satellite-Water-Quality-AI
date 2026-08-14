import os
import sys
import glob
import numpy as np
import rasterio

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.utils.save_index import save_index

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

print("=" * 60)
print(f"Found {len(files)} Sentinel-2 Images")
print("=" * 60)

for image in files:

    print("\nProcessing:", os.path.basename(image))

    with rasterio.open(image) as src:

        nir = src.read(8).astype(np.float32)
        red = src.read(4).astype(np.float32)

        ndvi = (nir - red) / (nir + red + 1e-10)

        ndvi[red == 0] = np.nan

        save_index(
            index_array=ndvi,
            src=src,
            image_path=image,
            index_name="ndvi",
            cmap="RdYlGn"
        )

print("\n" + "=" * 60)
print("NDVI Processing Completed Successfully")
print("=" * 60)