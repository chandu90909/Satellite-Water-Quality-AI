from pathlib import Path
import rasterio

from preprocessing.utils.band_reader import read_required_bands
from preprocessing.utils.save_index import save_index


BASE_DIR = Path(__file__).resolve().parents[2]

LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]


def calculate_ndwi(b3, b8):
    return (b3 - b8) / (b3 + b8 + 1e-10)


for lake in LAKES:

    input_dir = BASE_DIR / "data" / "raw" / lake

    for tif_file in sorted(input_dir.glob("*.tif")):

        print(f"Processing NDWI: {lake} / {tif_file.name}")

        with rasterio.open(tif_file) as src:

            bands = read_required_bands(src)

            ndwi = calculate_ndwi(
                bands["B3"],
                bands["B8"]
            )

            save_index(
                ndwi,
                src,
                lake,
                "ndwi",
                tif_file.stem
            )

print("\nNDWI processing completed.")