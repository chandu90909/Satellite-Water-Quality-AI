import rasterio
import matplotlib.pyplot as plt
import numpy as np
import glob

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

plt.figure(figsize=(18,12))

for i,file in enumerate(files):

    with rasterio.open(file) as src:

        red=src.read(4)
        green=src.read(3)
        blue=src.read(2)

    rgb=np.dstack((red,green,blue))

    rgb=rgb/np.percentile(rgb,98)
    rgb=np.clip(rgb,0,1)

    plt.subplot(3,4,i+1)
    plt.imshow(rgb)
    plt.title(file.split("/")[-1][:7])
    plt.axis("off")

plt.tight_layout()
plt.show()