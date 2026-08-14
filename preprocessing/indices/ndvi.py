import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

output_folder = "preprocessing/outputs/ndvi"

os.makedirs(output_folder, exist_ok=True)

for file in files:

    with rasterio.open(file) as src:

        nir = src.read(8).astype(np.float32)
        red = src.read(4).astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-10)

    print("="*50)
    print(os.path.basename(file))
    print("Minimum :", np.min(ndvi))
    print("Maximum :", np.max(ndvi))
    print("Mean    :", np.mean(ndvi))

    plt.figure(figsize=(8,8))

    plt.imshow(ndvi,cmap="RdYlGn")

    plt.colorbar(label="NDVI")

    plt.axis("off")

    plt.title(os.path.basename(file))

    output = os.path.join(
        output_folder,
        os.path.basename(file).replace(".tif","_ndvi.png")
    )

    plt.savefig(output,dpi=300)

    plt.show()

    plt.close()

print("\nFinished.")