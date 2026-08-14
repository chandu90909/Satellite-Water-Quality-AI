import rasterio
import numpy as np
import glob
import os

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

for file in files:

    print("\n"+"="*60)
    print(os.path.basename(file))

    with rasterio.open(file) as src:

        for band in range(1, src.count+1):

            img = src.read(band).astype(np.float32)

            img[img==0] = np.nan

            print(f"Band {band}")

            print(" Min :", np.nanmin(img))
            print(" Max :", np.nanmax(img))
            print(" Mean:", np.nanmean(img))
            print(" Std :", np.nanstd(img))