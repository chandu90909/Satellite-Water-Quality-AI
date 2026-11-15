from pathlib import Path
import rasterio
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASK_DIR = PROJECT_ROOT / "data" / "masks" / "water"

LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]


expected_total = 36
total = 0

print("=" * 80)
print("WATER MASK VERIFICATION")
print("=" * 80)


for lake in LAKES:

    lake_dir = MASK_DIR / lake

    mask_files = sorted(lake_dir.glob("*_water_mask.tif"))

    print(f"\n{lake}")
    print("-" * 60)

    print(f"Mask files: {len(mask_files)}")

    for mask_file in mask_files:

        with rasterio.open(mask_file) as src:

            mask = src.read(1)

            unique_values = np.unique(mask)

            water_pixels = np.sum(mask == 1)
            total_pixels = mask.size

            water_percentage = (
                water_pixels / total_pixels * 100
            )

            print(
                f"{mask_file.name}: "
                f"shape={mask.shape}, "
                f"values={unique_values.tolist()}, "
                f"water={water_percentage:.2f}%"
            )

        total += 1


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"Expected masks : {expected_total}")
print(f"Found masks    : {total}")

if total == expected_total:
    print("STATUS: ALL 36 MASKS FOUND")
else:
    print("STATUS: MASK COUNT ERROR")