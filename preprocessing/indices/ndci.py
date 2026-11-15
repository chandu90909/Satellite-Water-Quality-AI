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


def calculate_ndci(b5, b4):
    return (b5 - b4) / (b5 + b4 + 1e-10)


for lake in LAKES:

    input_dir = BASE_DIR / "data" / "raw" / lake

    for tif_file in sorted(input_dir.glob("*.tif")):

        print(f"Processing NDCI: {lake} / {tif_file.name}")

        with rasterio.open(tif_file) as src:

            bands = read_required_bands(src)

            ndci = calculate_ndci(
                bands["B5"],
                bands["B4"]
            )

            save_index(
                ndci,
                src,
                lake,
                "ndci",
                tif_file.stem
            )

print("\nNDCI processing completed.")