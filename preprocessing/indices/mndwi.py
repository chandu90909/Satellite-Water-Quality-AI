import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ---------------------------------------
# Input Images
# ---------------------------------------

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

# ---------------------------------------
# Output Folder
# ---------------------------------------

output_folder = "preprocessing/outputs/mndwi"

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------
# Process Every Image
# ---------------------------------------

for file in files:

    with rasterio.open(file) as src:

        green = src.read(3).astype(np.float32)
        swir = src.read(11).astype(np.float32)

    mndwi = (green - swir) / (green + swir + 1e-10)

    print("=" * 50)
    print(os.path.basename(file))
    print("Minimum :", np.min(mndwi))
    print("Maximum :", np.max(mndwi))
    print("Mean    :", np.mean(mndwi))

    plt.figure(figsize=(8,8))

    plt.imshow(mndwi, cmap="Blues")

    plt.colorbar(label="MNDWI")

    plt.title(os.path.basename(file))

    plt.axis("off")

    output = os.path.join(
        output_folder,
        os.path.basename(file).replace(".tif", "_mndwi.png")
    )

    plt.savefig(output, dpi=300)

    plt.show()

    plt.close()

print("\nFinished.")