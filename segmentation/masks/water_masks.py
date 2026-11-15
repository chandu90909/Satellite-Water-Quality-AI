"""Spectral water masks for downloaded acquisitions.

The thresholds match the masks that already ship with the project
(``segmentation/dataset/generate_water_masks.py``), so imagery acquired through
Earth Engine is processed under the same rules as the original lakes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from preprocessing.utils.band_reader import read_required_bands

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MASK_ROOT = PROJECT_ROOT / "data" / "masks" / "water"

NDWI_THRESHOLD = 0.0
MNDWI_THRESHOLD = 0.0


def create_water_mask(b3: np.ndarray, b8: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Water (1) where both NDWI and MNDWI are positive, otherwise 0."""
    b3 = np.asarray(b3, dtype=np.float32)
    b8 = np.asarray(b8, dtype=np.float32)
    b11 = np.asarray(b11, dtype=np.float32)
    valid = np.isfinite(b3) & np.isfinite(b8) & np.isfinite(b11)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (b3 - b8) / (b3 + b8 + 1e-10)
        mndwi = (b3 - b11) / (b3 + b11 + 1e-10)
    water = valid & (ndwi > NDWI_THRESHOLD) & (mndwi > MNDWI_THRESHOLD)
    return water.astype(np.uint8)


def generate_for_image(lake_name: str, image_name: str) -> Path:
    image_path = RAW_ROOT / lake_name / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with rasterio.open(image_path) as source:
        bands = read_required_bands(source)
        mask = create_water_mask(bands["B3"], bands["B8"], bands["B11"])
        profile = source.profile.copy()

    profile.update(dtype="uint8", count=1, compress="lzw", nodata=None)
    output_path = MASK_ROOT / lake_name / f"{Path(image_name).stem}_water_mask.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", **profile) as destination:
        destination.write(mask, 1)
        destination.set_band_description(1, "water_mask")
    return output_path


def generate_for_lake(lake_name: str, image_names=None) -> list[Path]:
    lake_dir = RAW_ROOT / lake_name
    if not lake_dir.is_dir():
        raise FileNotFoundError(f"No imagery found for lake '{lake_name}' in {lake_dir}")
    if image_names is None:
        image_names = [path.name for path in sorted(lake_dir.glob("*.tif"))]
    return [generate_for_image(lake_name, name) for name in image_names]
