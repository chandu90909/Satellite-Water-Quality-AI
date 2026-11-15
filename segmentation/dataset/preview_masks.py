import sys
from pathlib import Path

import rasterio
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MASK_DIR = PROJECT_ROOT / "data" / "masks" / "water"

TEST_FILES = [
    MASK_DIR / "Hussain_Sagar" / "Hussain_Sagar_1_water_mask.tif",
    MASK_DIR / "Hussain_Sagar" / "Hussain_Sagar_10_water_mask.tif",

    MASK_DIR / "Kolleru_Lake" / "Kolleru_1_water_mask.tif",
    MASK_DIR / "Kolleru_Lake" / "Kolleru_7_water_mask.tif",

    MASK_DIR / "Pulicat_Lake" / "Pulicat_1_water_mask.tif",
    MASK_DIR / "Pulicat_Lake" / "Pulicat_7_water_mask.tif",
]


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "water_masks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for mask_file in TEST_FILES:

    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    plt.figure(figsize=(8, 6))

    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)

    plt.title(mask_file.stem)

    plt.axis("off")

    output_file = OUTPUT_DIR / f"{mask_file.stem}_preview.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


print("\nPreview generation completed.")