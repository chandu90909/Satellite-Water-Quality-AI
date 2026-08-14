import rasterio
import matplotlib.pyplot as plt
import numpy as np

img1 = "data/raw/Hussain_Sagar/Hussain_Sagar_1.tif"
img2 = "data/raw/Hussain_Sagar/Hussain_Sagar_6.tif"

with rasterio.open(img1) as src:

    green1 = src.read(3).astype(np.float32)
    nir1 = src.read(8).astype(np.float32)

with rasterio.open(img2) as src:

    green2 = src.read(3).astype(np.float32)
    nir2 = src.read(8).astype(np.float32)

ndwi1 = (green1 - nir1)/(green1 + nir1 + 1e-10)
ndwi2 = (green2 - nir2)/(green2 + nir2 + 1e-10)

difference = ndwi2 - ndwi1

difference[np.isnan(difference)] = 0

plt.figure(figsize=(8,8))

plt.imshow(difference, cmap="RdBu")

plt.colorbar(label="NDWI Difference")

plt.title("Temporal Difference (June - January)")

plt.axis("off")

plt.show()

print("Difference Statistics")
print("----------------------")
print("Minimum :", np.min(difference))
print("Maximum :", np.max(difference))
print("Mean    :", np.mean(difference))