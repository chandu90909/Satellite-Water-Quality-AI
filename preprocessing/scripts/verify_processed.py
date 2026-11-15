from pathlib import Path
import rasterio
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]

INDICES = [
    "ndwi",
    "mndwi",
    "ndvi",
    "ndci",
    "turbidity",
    "chlorophyll"
]


total_expected = 36 * 6
total_found = 0

print("=" * 80)
print("PROCESSED DATA VERIFICATION")
print("=" * 80)


for lake in LAKES:

    print(f"\n{'=' * 70}")
    print(f"LAKE: {lake}")
    print(f"{'=' * 70}")

    for index in INDICES:

        folder = BASE_DIR / "data" / "processed" / lake / index

        tif_files = sorted(folder.glob("*.tif"))

        print(f"\n{index.upper()}: {len(tif_files)} files")

        for tif_file in tif_files:

            total_found += 1

            try:

                with rasterio.open(tif_file) as src:

                    data = src.read(1)

                    finite = np.isfinite(data)

                    valid_count = np.sum(finite)

                    if valid_count > 0:

                        valid_data = data[finite]

                        print(
                            f"  {tif_file.name}: "
                            f"valid={valid_count}, "
                            f"min={valid_data.min():.4f}, "
                            f"max={valid_data.max():.4f}, "
                            f"mean={valid_data.mean():.4f}"
                        )

                    else:

                        print(
                            f"  WARNING: {tif_file.name} "
                            f"contains no valid pixels"
                        )

            except Exception as e:

                print(
                    f"  ERROR: {tif_file.name} -> {e}"
                )


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"Expected GeoTIFFs : {total_expected}")
print(f"Found GeoTIFFs    : {total_found}")

if total_found == total_expected:
    print("STATUS: ALL 216 PROCESSED TIFFs FOUND")
else:
    print("STATUS: FILE COUNT DOES NOT MATCH EXPECTED COUNT")