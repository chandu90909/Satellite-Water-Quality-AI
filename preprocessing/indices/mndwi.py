import os
import sys
import glob
import numpy as np
import rasterio

# --------------------------------------------------
# Add Project Root to Python Path
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.utils.save_index import save_index

# --------------------------------------------------
# Read All Sentinel-2 Images
# --------------------------------------------------

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

print("=" * 60)
print(f"Found {len(files)} Sentinel-2 Images")
print("=" * 60)

# --------------------------------------------------
# Process Every Image
# --------------------------------------------------

for image in files:

    print("\nProcessing:", os.path.basename(image))

    with rasterio.open(image) as src:

        green = src.read(3).astype(np.float32)
        swir = src.read(11).astype(np.float32)

        mndwi = (green - swir) / (green + swir + 1e-10)

        mndwi[green == 0] = np.nan

        save_index(
            index_array=mndwi,
            src=src,
            image_path=image,
            index_name="mndwi",
            cmap="Blues"
        )

print("\n" + "=" * 60)
print("MNDWI Processing Completed Successfully")
print("=" * 60)