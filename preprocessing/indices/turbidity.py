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


def calculate_turbidity(b4, b3):
    return b4 / (b3 + 1e-10)


for lake in LAKES:

    input_dir = BASE_DIR / "data" / "raw" / lake

    for tif_file in sorted(input_dir.glob("*.tif")):

        print(f"Processing Turbidity: {lake} / {tif_file.name}")

        with rasterio.open(tif_file) as src:

            bands = read_required_bands(src)

            turbidity = calculate_turbidity(
                bands["B4"],
                bands["B3"]
            )

            save_index(
                turbidity,
                src,
                lake,
                "turbidity",
                tif_file.stem
            )

print("\nTurbidity processing completed.")