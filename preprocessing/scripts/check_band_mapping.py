import rasterio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

LAKES = [
    "Hussain_Sagar",
    "Kolleru_Lake",
    "Pulicat_Lake"
]

for lake in LAKES:

    lake_dir = BASE_DIR / "data" / "raw" / lake
    files = sorted(lake_dir.glob("*.tif"))

    print("\n" + "#" * 80)
    print("LAKE:", lake)
    print("NUMBER OF TIFF FILES:", len(files))
    print("#" * 80)

    # Only inspect the first TIFF initially
    if not files:
        print("No TIFF files found.")
        continue

    file = files[0]

    print("\nFILE:", file.name)
    print("PATH:", file)

    try:
        with rasterio.open(file) as src:

            print("\nBand count:", src.count)
            print("Dimensions:", src.width, "x", src.height)
            print("CRS:", src.crs)
            print("Data type:", src.dtypes[0])

            print("\nBand descriptions:")

            for i, desc in enumerate(src.descriptions, start=1):
                print(f"  Band {i}: {desc}")

            print("\nDataset tags:")

            tags = src.tags()

            if tags:
                for key, value in tags.items():
                    print(f"  {key}: {value}")
            else:
                print("  No dataset tags.")

    except Exception as e:
        print("ERROR:", e)