import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

output_folder = "preprocessing/outputs/ndci"

os.makedirs(output_folder, exist_ok=True)

for file in files:

    with rasterio.open(file) as src:

        red = src.read(4).astype(np.float32)
        red_edge = src.read(5).astype(np.float32)

    ndci = (red_edge-red)/(red_edge+red+1e-10)

    plt.figure(figsize=(8,8))

    plt.imshow(ndci,cmap="viridis")

    plt.colorbar(label="NDCI")

    plt.title(os.path.basename(file))

    plt.axis("off")

    output=os.path.join(
        output_folder,
        os.path.basename(file).replace(".tif","_ndci.png")
    )

    plt.savefig(output,dpi=300)

    plt.show()

    plt.close()

print("Finished.")