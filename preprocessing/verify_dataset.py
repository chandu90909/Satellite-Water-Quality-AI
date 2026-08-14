import rasterio
import glob
import os

files = sorted(glob.glob("data/raw/Hussain_Sagar/*.tif"))

print("="*60)
print("DATASET VERIFICATION")
print("="*60)

for file in files:

    try:

        with rasterio.open(file) as src:

            print(f"\n{os.path.basename(file)}")

            print("Bands :", src.count)

            print("CRS :", src.crs)

            print("Resolution :", src.res)

            print("Width :", src.width)

            print("Height :", src.height)

            print("Status : OK")

    except Exception as e:

        print(file)

        print(e)