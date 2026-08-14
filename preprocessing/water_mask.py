import rasterio
import matplotlib.pyplot as plt
import numpy as np

image = "data/raw/Hussain_Sagar/Hussain_Sagar_1.tif"

with rasterio.open(image) as src:

    green = src.read(3).astype(np.float32)
    nir = src.read(8).astype(np.float32)

ndwi = (green - nir) / (green + nir + 1e-10)

mask = ndwi > 0

plt.figure(figsize=(8,8))

plt.imshow(mask, cmap="gray")

plt.title("Water Mask")

plt.axis("off")

plt.show()

print("Water Pixels :", np.sum(mask))
print("Land Pixels  :", np.sum(~mask))