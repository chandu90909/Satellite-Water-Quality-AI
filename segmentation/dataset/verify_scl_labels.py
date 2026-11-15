import sys
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "water_scl"
)


LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]


expected = 22
found = 0

print("=" * 80)
print("SCL LABEL VERIFICATION")
print("=" * 80)


for lake in LAKES:

    lake_dir = LABEL_DIR / lake

    files = sorted(
        lake_dir.glob("*_scl_label.tif")
    )

    print(f"\n{lake}")
    print("-" * 60)

    print(f"Files: {len(files)}")

    for file in files:

        with rasterio.open(file) as src:

            label = src.read(1)

            unique_values = np.unique(label)

            water = np.sum(label == 1)
            non_water = np.sum(label == 0)
            ignored = np.sum(label == 255)

            total = label.size

            valid = water + non_water

            water_pct = (
                water / valid * 100
                if valid > 0
                else 0
            )

            ignored_pct = (
                ignored / total * 100
            )

            print(
                f"{file.name}: "
                f"values={unique_values.tolist()}, "
                f"water={water_pct:.2f}%, "
                f"ignored={ignored_pct:.2f}%"
            )

            if valid == 0:
                print(
                    "  WARNING: no valid training pixels"
                )

            if water == 0:
                print(
                    "  WARNING: no water pixels"
                )

            if non_water == 0:
                print(
                    "  WARNING: no non-water pixels"
                )

        found += 1


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"Expected SCL labels : {expected}")
print(f"Found SCL labels    : {found}")

if found == expected:
    print("STATUS: ALL 22 SCL LABELS FOUND")
else:
    print("STATUS: LABEL COUNT ERROR")