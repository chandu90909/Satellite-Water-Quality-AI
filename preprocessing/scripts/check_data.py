import os
import glob
import rasterio


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]


def inspect_image(path):

    with rasterio.open(path) as src:

        print("\n" + "-" * 70)
        print("FILE:", os.path.basename(path))
        print("-" * 70)

        print("Dimensions :", src.width, "x", src.height)
        print("Bands      :", src.count)
        print("CRS        :", src.crs)
        print("Data type  :", src.dtypes[0])
        print("NoData     :", src.nodata)

        print("\nBand availability:")

        required_bands = {
            3: "Green (B3)",
            4: "Red (B4)",
            5: "Red Edge (B5)",
            8: "NIR (B8)",
            11: "SWIR (B11)",
            15: "SCL"
        }

        for band_number, name in required_bands.items():

            if band_number <= src.count:
                print(
                    f"  Band {band_number:2d} : {name} -> AVAILABLE"
                )
            else:
                print(
                    f"  Band {band_number:2d} : {name} -> MISSING"
                )


def main():

    print("=" * 70)
    print("SATELLITE DATA VERIFICATION")
    print("=" * 70)

    total_images = 0

    for lake in LAKES:

        lake_dir = os.path.join(
            RAW_DIR,
            lake
        )

        files = sorted(
            glob.glob(
                os.path.join(
                    lake_dir,
                    "*.tif"
                )
            )
        )

        print("\n")
        print("=" * 70)
        print(f"LAKE: {lake}")
        print(f"Images found: {len(files)}")
        print("=" * 70)

        total_images += len(files)

        for image in files:

            inspect_image(image)

    print("\n")
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print("Total TIFF images:", total_images)
    print("Expected images  :", 36)

    if total_images == 36:
        print("STATUS: Image count OK")
    else:
        print("STATUS: Image count DOES NOT MATCH expected 36")


if __name__ == "__main__":
    main()