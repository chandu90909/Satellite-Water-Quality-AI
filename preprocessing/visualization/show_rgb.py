import rasterio
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# Path to GeoTIFF
# ----------------------------------------------------
image = "data/raw/Hussain_Sagar/Hussain_Sagar_1.tif"

# ----------------------------------------------------
# Read Image
# ----------------------------------------------------
with rasterio.open(image) as src:

    print("=" * 60)
    print("CRS           :", src.crs)
    print("Bands         :", src.count)
    print("Width         :", src.width)
    print("Height        :", src.height)
    print("Resolution    :", src.res)
    print("Data Type     :", src.dtypes)
    print("NoData Value  :", src.nodata)
    print("=" * 60)

    # Sentinel-2 RGB Bands
    red = src.read(4).astype(np.float32)
    green = src.read(3).astype(np.float32)
    blue = src.read(2).astype(np.float32)

# ----------------------------------------------------
# Print Band Statistics
# ----------------------------------------------------
print("Red   : Min =", red.min(), " Max =", red.max())
print("Green : Min =", green.min(), " Max =", green.max())
print("Blue  : Min =", blue.min(), " Max =", blue.max())

# ----------------------------------------------------
# Stack RGB
# ----------------------------------------------------
rgb = np.dstack((red, green, blue))

# ----------------------------------------------------
# Ignore NoData / Zero Pixels
# ----------------------------------------------------
valid_pixels = rgb[rgb > 0]

if len(valid_pixels) == 0:
    print("No valid pixels found!")
    exit()

# ----------------------------------------------------
# Contrast Stretch (2% - 98%)
# ----------------------------------------------------
p2 = np.percentile(valid_pixels, 2)
p98 = np.percentile(valid_pixels, 98)

print("2nd Percentile  :", p2)
print("98th Percentile :", p98)

rgb = (rgb - p2) / (p98 - p2)
rgb = np.clip(rgb, 0, 1)

# ----------------------------------------------------
# Display Image
# ----------------------------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("Hussain Sagar RGB Image")
plt.axis("off")
plt.show()