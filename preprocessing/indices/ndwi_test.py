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

        # Read Bands
        green = src.read(3).astype(np.float32)
        nir = src.read(8).astype(np.float32)

        # NDWI Calculation
        ndwi = (green - nir) / (green + nir + 1e-10)

        # Remove NoData Pixels
        ndwi[green == 0] = np.nan

        # Save Outputs
        save_index(
            index_array=ndwi,
            src=src,
            image_path=image,
            index_name="ndwi",
            cmap="RdYlBu"
        )

print("\n" + "=" * 60)
print("NDWI Processing Completed Successfully")
print("=" * 60)