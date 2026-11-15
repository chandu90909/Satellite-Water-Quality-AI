import rasterio


def get_band_map(src):
    """
    Returns a dictionary mapping Sentinel-2 band names
    to their actual raster band numbers.
    """

    band_map = {}

    for i, description in enumerate(src.descriptions, start=1):

        if description is not None:
            band_map[description] = i

    return band_map


def read_required_bands(src):
    """
    Reads the spectral bands required by the project.

    Required:
        B3  - Green
        B4  - Red
        B5  - Red Edge
        B8  - NIR
        B11 - SWIR
    """

    band_map = get_band_map(src)

    required = ["B3", "B4", "B5", "B8", "B11"]

    missing = [band for band in required if band not in band_map]

    if missing:
        raise ValueError(
            f"Missing required bands: {missing}"
        )

    bands = {
        "B3": src.read(band_map["B3"]).astype("float32"),
        "B4": src.read(band_map["B4"]).astype("float32"),
        "B5": src.read(band_map["B5"]).astype("float32"),
        "B8": src.read(band_map["B8"]).astype("float32"),
        "B11": src.read(band_map["B11"]).astype("float32"),
    }

    return bands


def read_scl(src):
    """
    Reads Sentinel-2 Scene Classification Layer if available.
    """

    band_map = get_band_map(src)

    if "SCL" not in band_map:
        return None

    return src.read(band_map["SCL"])