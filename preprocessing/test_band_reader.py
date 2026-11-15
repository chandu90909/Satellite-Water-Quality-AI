import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import rasterio

from preprocessing.utils.band_reader import read_required_bands


BASE_DIR = PROJECT_ROOT

TEST_FILES = [
    BASE_DIR / "data/raw/Hussain_Sagar/Hussain_Sagar_1.tif",
    BASE_DIR / "data/raw/Kolleru_Lake/Kolleru_1.tif",
    BASE_DIR / "data/raw/Pulicat_Lake/Pulicat_1.tif",
    BASE_DIR / "data/raw/Pulicat_Lake/Pulicat_7.tif",
]


for file in TEST_FILES:

    print("\n" + "=" * 60)
    print("Testing:", file.name)

    with rasterio.open(file) as src:

        bands = read_required_bands(src)

        for name, array in bands.items():
            print(
                f"{name}: shape={array.shape}, "
                f"dtype={array.dtype}"
            )


print("\nAll band-reader tests passed.")