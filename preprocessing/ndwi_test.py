import rasterio
import matplotlib.pyplot as plt
import numpy as np

image = "data/raw/Hussain_Sagar/Hussain_Sagar_1.tif"

with rasterio.open(image) as src:

    green = src.read(3).astype(np.float32)
    nir = src.read(8).astype(np.float32)

ndwi = (green - nir) / (green + nir + 1e-10)

ndwi[green == 0] = np.nan

plt.figure(figsize=(8,8))

plt.imshow(ndwi, cmap="RdYlBu")

plt.colorbar(label="NDWI")

plt.title("NDWI - Hussain Sagar")

plt.axis("off")

plt.show()

print("NDWI Statistics")
print("--------------------")
print("Minimum :", np.nanmin(ndwi))
print("Maximum :", np.nanmax(ndwi))
print("Mean    :", np.nanmean(ndwi))