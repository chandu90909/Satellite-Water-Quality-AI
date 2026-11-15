"""Sentinel-2 L2A acquisition from Earth Engine.

Exports the five bands the classifier was trained on (B3, B4, B5, B8, B11) plus
the scene classification layer (SCL) as a single GeoTIFF in EPSG:4326 at 10 m,
matching the contract of the imagery already in ``data/raw``.

Clouds, shadows, cirrus, snow and defective pixels are removed with the SCL band
before compositing. Monthly periods use a per-band median composite; scene mode
downloads every clear acquisition individually.
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as merge_rasters

from gee.boundary import LakeBoundary
from gee.config import initialize

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = ("B3", "B4", "B5", "B8", "B11")
SCL_BAND = "SCL"
EXPORT_BANDS = BANDS + (SCL_BAND,)

# SCL classes that must not contribute to a water-quality composite.
CLOUD_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)

DEFAULT_SCALE = 10
DEFAULT_MAX_CLOUD = 40
TILE_PIXELS = 1600
MAX_SPLIT_DEPTH = 2


class DownloadError(RuntimeError):
    """Raised when an Earth Engine export cannot be downloaded."""


@dataclass
class Period:
    label: str
    start: date
    end: date

    def describe(self) -> str:
        return f"{self.start.isoformat()} to {self.end.isoformat()}"


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Dates must use YYYY-MM-DD, got '{value}'") from exc


def month_periods(start: date, end: date) -> list[Period]:
    """Split ``start..end`` into calendar-month periods (inclusive)."""
    if end < start:
        raise ValueError("End date must be on or after the start date")
    periods: list[Period] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = date(cursor.year + 1, 1, 1) if cursor.month == 12 else date(cursor.year, cursor.month + 1, 1)
        period_start = max(cursor, start)
        period_end = min(next_month - timedelta(days=1), end)
        if period_start <= period_end:
            periods.append(Period(f"{cursor.year}{cursor.month:02d}", period_start, period_end))
        cursor = next_month
    return periods


def _mask_clouds(image):
    scl = image.select(SCL_BAND)
    clear = None
    for value in CLOUD_SCL_CLASSES:
        condition = scl.neq(value)
        clear = condition if clear is None else clear.And(condition)
    return image.updateMask(clear)


def _composite(collection):
    """Median composite of clear observations plus the modal SCL class."""
    import ee

    masked = collection.map(_mask_clouds)
    bands = masked.select(list(BANDS)).median()
    scl = masked.select(SCL_BAND).reduce(ee.Reducer.mode()).rename(SCL_BAND)
    return bands.addBands(scl).select(list(EXPORT_BANDS))


def build_collection(
    boundary: LakeBoundary,
    start: date,
    end: date,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    project: str | None = None,
):
    ee = initialize(project)
    geometry = ee.Geometry(boundary.geometry)
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(max_cloud)))
    )


def _tile_boxes(bounds: tuple[float, float, float, float], scale: int, max_pixels: int = TILE_PIXELS):
    min_lon, min_lat, max_lon, max_lat = bounds
    center_lat = (min_lat + max_lat) / 2.0
    metres_per_degree_lat = 110_574.0
    metres_per_degree_lon = max(111_320.0 * math.cos(math.radians(center_lat)), 1.0)
    degrees_lat = max_pixels * scale / metres_per_degree_lat
    degrees_lon = max_pixels * scale / metres_per_degree_lon

    rows = max(1, math.ceil((max_lat - min_lat) / degrees_lat))
    columns = max(1, math.ceil((max_lon - min_lon) / degrees_lon))
    boxes = []
    for row in range(rows):
        for column in range(columns):
            box_min_lat = min_lat + row * degrees_lat
            box_min_lon = min_lon + column * degrees_lon
            boxes.append((
                box_min_lon,
                box_min_lat,
                min(box_min_lon + degrees_lon, max_lon),
                min(box_min_lat + degrees_lat, max_lat),
            ))
    return boxes


def _fetch_tiles(image, box, scale: int, crs: str, timeout: float = 300.0, depth: int = 0) -> list[bytes]:
    """Download one region, splitting it when Earth Engine rejects the size."""
    import ee
    import requests

    region = ee.Geometry.Rectangle(list(box), crs, False)
    message = ""
    try:
        url = image.getDownloadURL({
            "name": "sentinel2",
            "bands": list(EXPORT_BANDS),
            "scale": scale,
            "crs": crs,
            "region": region,
            "format": "GEO_TIFF",
            "filePerBand": False,
        })
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200 and response.content:
            return [response.content]
        message = f"HTTP {response.status_code}: {response.text[:300]}"
    except Exception as exc:  # network or Earth Engine size rejection
        message = str(exc)

    if depth < MAX_SPLIT_DEPTH:
        min_lon, min_lat, max_lon, max_lat = box
        mid_lon = (min_lon + max_lon) / 2.0
        mid_lat = (min_lat + max_lat) / 2.0
        quadrants = [
            (min_lon, min_lat, mid_lon, mid_lat),
            (mid_lon, min_lat, max_lon, mid_lat),
            (min_lon, mid_lat, mid_lon, max_lat),
            (mid_lon, mid_lat, max_lon, max_lat),
        ]
        blobs: list[bytes] = []
        for quadrant in quadrants:
            blobs.extend(_fetch_tiles(image, quadrant, scale, crs, timeout, depth + 1))
        return blobs

    raise DownloadError(f"Earth Engine export failed for {box}: {message}")


def download_image(image, output_path: Path, bounds, scale: int = DEFAULT_SCALE, crs: str = "EPSG:4326") -> Path:
    """Download ``image`` over ``bounds`` and write the project GeoTIFF layout."""
    blobs: list[bytes] = []
    for box in _tile_boxes(bounds, scale):
        blobs.extend(_fetch_tiles(image, box, scale, crs))
    if not blobs:
        raise DownloadError("Earth Engine returned no raster data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        datasets = []
        try:
            for index, blob in enumerate(blobs):
                tile_path = Path(tmp) / f"tile_{index}.tif"
                tile_path.write_bytes(blob)
                datasets.append(rasterio.open(tile_path))
            mosaic, mosaic_transform = merge_rasters(datasets, nodata=np.nan)
        finally:
            for dataset in datasets:
                dataset.close()

    profile = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": mosaic_transform,
        "compress": "lzw",
        "nodata": float("nan"),
    }
    with rasterio.open(str(output_path), "w", **profile) as destination:
        destination.write(mosaic.astype(np.float32))
        for index, name in enumerate(EXPORT_BANDS, start=1):
            destination.set_band_description(index, name)
        destination.update_tags(
            source=S2_COLLECTION,
            exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    return output_path


def download_period(
    boundary: LakeBoundary,
    period: Period,
    output_path: Path,
    *,
    scale: int = DEFAULT_SCALE,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    project: str | None = None,
) -> Path | None:
    """Build and download one monthly composite; ``None`` when no clear scene exists."""
    collection = build_collection(boundary, period.start, period.end, max_cloud, project)
    if int(collection.size().getInfo()) == 0:
        return None
    return download_image(_composite(collection), output_path, boundary.bounds(), scale=scale)


def download_scenes(
    boundary: LakeBoundary,
    start: date,
    end: date,
    output_dir: Path,
    *,
    scale: int = DEFAULT_SCALE,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    project: str | None = None,
    limit: int = 60,
) -> list[Path]:
    """Download individual clear acquisitions, one GeoTIFF per acquisition date."""
    import ee

    collection = build_collection(boundary, start, end, max_cloud, project)
    collection = collection.sort("system:time_start").limit(limit)
    if int(collection.size().getInfo()) == 0:
        return []

    indexes = collection.aggregate_array("system:index").getInfo()
    stamps = collection.aggregate_array("system:time_start").getInfo()
    seen_dates: set[date] = set()
    saved: list[Path] = []
    for system_index, timestamp in zip(indexes, stamps):
        scene_date = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc).date()
        if scene_date in seen_dates:
            continue
        seen_dates.add(scene_date)
        path = output_dir / f"{boundary.slug}_{scene_date.strftime('%Y%m%d')}.tif"
        if path.exists() and path.stat().st_size > 0:
            saved.append(path)
            continue
        image = _composite(collection.filter(ee.Filter.eq("system:index", system_index)))
        saved.append(download_image(image, path, boundary.bounds(), scale=scale))
    return saved
