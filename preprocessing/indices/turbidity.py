import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

output_folder = "preprocessing/outputs/turbidity"

os.makedirs(output_folder, exist_ok=True)

for file in files:

    with rasterio.open(file) as src:

        red = src.read(4).astype(np.float32)
        green = src.read(3).astype(np.float32)

    turbidity = red/(green+1e-10)

    plt.figure(figsize=(8,8))

    plt.imshow(turbidity,cmap="inferno")

    plt.colorbar(label="Turbidity")

    plt.axis("off")

    plt.title(os.path.basename(file))

    output=os.path.join(
        output_folder,
        os.path.basename(file).replace(".tif","_turbidity.png")
    )

    plt.savefig(output,dpi=300)

    plt.show()

    plt.close()

print("Finished.")