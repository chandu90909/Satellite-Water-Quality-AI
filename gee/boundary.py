"""Resolve a lake name to a geographic boundary for Earth Engine queries.

Resolution order:
    1. An explicit GeoJSON file (``--boundary``).
    2. A cached boundary in ``data/boundaries/<slug>.geojson``.
    3. The extent of imagery already downloaded for that lake.
    4. OpenStreetMap Nominatim, then Photon, restricted to water features.

Boundaries are polygons in longitude/latitude (WGS84).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
BOUNDARY_ROOT = PROJECT_ROOT / "data" / "boundaries"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
USER_AGENT = "SatelliteWaterQualityResearch/1.0 (local research dashboard)"

WATER_FEATURE_TYPES = {
    "water",
    "lake",
    "reservoir",
    "lagoon",
    "bay",
    "pond",
    "basin",
    "wetland",
    "river",
    "canal",
    "strait",
}

# Nominatim sometimes returns a small pond or a single building footprint for a
# lake name. Anything below roughly 220 m across cannot produce usable 10 m
# satellite data, so it is rejected instead of silently downloading a handful
# of pixels. A verified boundary can still be supplied with --boundary.
MIN_BOUNDARY_SPAN_DEG = 0.002


class BoundaryError(RuntimeError):
    """Raised when no usable lake boundary can be resolved."""


@dataclass
class LakeBoundary:
    slug: str
    display_name: str
    geometry: dict
    bbox: tuple[float, float, float, float]
    source: str
    properties: dict = field(default_factory=dict)

    def bounds(self) -> tuple[float, float, float, float]:
        return self.bbox


def lake_slug(name: str) -> str:
    """Normalize a user supplied lake name to the project folder convention.

    ``"Kolleru Lake"`` -> ``"Kolleru_Lake"``; an existing lake folder is reused
    with its current spelling so new acquisitions land beside old ones.
    """
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", str(name or "")).strip()
    if not cleaned:
        raise ValueError("Lake name must contain at least one letter or digit")
    tokens = [token[:1].upper() + token[1:] for token in cleaned.split()]
    slug = "_".join(tokens)
    if RAW_ROOT.exists():
        for existing in RAW_ROOT.iterdir():
            if existing.is_dir() and existing.name.lower() == slug.lower():
                return existing.name
    return slug


def _count_points(coordinates) -> int:
    if isinstance(coordinates, (list, tuple)):
        if len(coordinates) == 2 and all(isinstance(item, (int, float)) for item in coordinates):
            return 1
        return sum(_count_points(item) for item in coordinates)
    return 0


def _bbox_from_geometry(geometry: dict) -> tuple[float, float, float, float] | None:
    if not isinstance(geometry, dict):
        return None
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind not in {"Polygon", "MultiPolygon"} or not coordinates:
        return None
    xs: list[float] = []
    ys: list[float] = []

    def walk(value):
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
                xs.append(float(value[0]))
                ys.append(float(value[1]))
            else:
                for item in value:
                    walk(item)

    walk(coordinates)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _rectangle_geometry(bbox: tuple[float, float, float, float]) -> dict:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }


def _normalize_geometry(geometry: dict) -> dict | None:
    """Keep polygonal geometries that GEE and rasterio can both consume."""
    if not geometry:
        return None
    kind = geometry.get("type")
    if kind in {"Polygon", "MultiPolygon"}:
        return geometry
    if kind == "GeometryCollection":
        polygons = [
            _normalize_geometry(item)
            for item in geometry.get("geometries", [])
        ]
        polygons = [item for item in polygons if item]
        return polygons[0] if polygons else None
    return None


def boundary_from_imagery(slug: str) -> LakeBoundary | None:
    """Use the extent of already downloaded acquisitions as the lake boundary."""
    import rasterio

    lake_dir = RAW_ROOT / slug
    if not lake_dir.is_dir():
        return None
    bounds = []
    for path in sorted(lake_dir.glob("*.tif")):
        try:
            with rasterio.open(path) as source:
                if source.crs is None:
                    continue
                bounds.append(source.bounds)
        except Exception:
            continue
    if not bounds:
        return None
    bbox = (
        min(item.left for item in bounds),
        min(item.bottom for item in bounds),
        max(item.right for item in bounds),
        max(item.top for item in bounds),
    )
    if (
        bbox[2] - bbox[0] < MIN_BOUNDARY_SPAN_DEG
        or bbox[3] - bbox[1] < MIN_BOUNDARY_SPAN_DEG
    ):
        return None
    return LakeBoundary(
        slug=slug,
        display_name=slug.replace("_", " "),
        geometry=_rectangle_geometry(bbox),
        bbox=bbox,
        source="existing-imagery",
    )


def _score_candidate(feature_class: str, feature_type: str, bbox, area: float) -> float:
    score = math.log1p(max(area, 0.0))
    kind = (feature_type or "").lower()
    category = (feature_class or "").lower()
    if kind in WATER_FEATURE_TYPES:
        score += 6.0
    if category in {"natural", "waterway", "water", "landuse"}:
        score += 2.0
    if category in {"place", "boundary", "administrative", "highway"}:
        score -= 8.0
    return score


def _pick_candidate(candidates: list[dict]) -> dict | None:
    scored = []
    for candidate in candidates:
        geometry = _normalize_geometry(candidate.get("geometry") or {})
        if not geometry:
            continue
        bbox = _bbox_from_geometry(geometry)
        if bbox is None:
            continue
        span_x = bbox[2] - bbox[0]
        span_y = bbox[3] - bbox[1]
        if span_x < MIN_BOUNDARY_SPAN_DEG or span_y < MIN_BOUNDARY_SPAN_DEG:
            continue
        candidate = {**candidate, "geometry": geometry, "bbox": bbox}
        candidate["score"] = _score_candidate(
            candidate.get("feature_class", ""),
            candidate.get("feature_type", ""),
            bbox,
            span_x * span_y,
        )
        scored.append(candidate)
    if not scored:
        return None
    return max(scored, key=lambda item: item["score"])


def _geocode_nominatim(name: str, timeout: float = 30.0) -> dict | None:
    params = {
        "q": name,
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 8,
        "addressdetails": 0,
        "accept-language": "en",
    }
    response = requests.get(
        f"{NOMINATIM_URL}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    candidates = []
    for item in response.json() or []:
        candidates.append({
            "display_name": item.get("display_name") or name,
            "geometry": item.get("geojson") or {},
            # jsonv2 exposes "category"; older formats use "class".
            "feature_class": item.get("category") or item.get("class", ""),
            "feature_type": item.get("type", ""),
            "source": "nominatim",
        })
    return _pick_candidate(candidates)


def _photon_extent_to_geometry(extent) -> dict | None:
    if not extent or len(extent) != 4:
        return None
    try:
        min_lon, max_lat, max_lon, min_lat = (float(value) for value in extent)
    except (TypeError, ValueError):
        return None
    if max_lon <= min_lon or max_lat <= min_lat:
        return None
    return _rectangle_geometry((min_lon, min_lat, max_lon, max_lat))


def _geocode_photon(name: str, timeout: float = 30.0) -> dict | None:
    params = {"q": name, "limit": 8, "lang": "en"}
    response = requests.get(
        f"{PHOTON_URL}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    candidates = []
    for feature in (response.json() or {}).get("features", []):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            geometry = _photon_extent_to_geometry(properties.get("extent")) or geometry
        candidates.append({
            "display_name": " ".join(
                str(part) for part in (
                    properties.get("name"),
                    properties.get("city"),
                    properties.get("state"),
                    properties.get("country"),
                ) if part
            ) or name,
            "geometry": geometry,
            "feature_class": properties.get("osm_key", ""),
            "feature_type": properties.get("osm_value", ""),
            "source": "photon",
        })
    return _pick_candidate(candidates)


def _load_geojson_boundary(path: Path, slug: str, name: str) -> LakeBoundary:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if not features:
            raise BoundaryError(f"{path} contains no features")
        geometry = features[0].get("geometry") or {}
        properties = features[0].get("properties") or {}
    elif payload.get("type") == "Feature":
        geometry = payload.get("geometry") or {}
        properties = payload.get("properties") or {}
    else:
        geometry = payload
        properties = {}
    geometry = _normalize_geometry(geometry)
    if not geometry:
        raise BoundaryError(f"{path} does not contain a Polygon or MultiPolygon geometry")
    bbox = _bbox_from_geometry(geometry)
    if bbox is None:
        raise BoundaryError(f"{path} has invalid coordinates")
    return LakeBoundary(
        slug=slug,
        display_name=properties.get("name") or name or slug,
        geometry=geometry,
        bbox=bbox,
        source=str(path),
        properties=properties,
    )


def _cache_path(slug: str) -> Path:
    return BOUNDARY_ROOT / f"{slug}.geojson"


def write_cached_boundary(boundary: LakeBoundary) -> Path:
    BOUNDARY_ROOT.mkdir(parents=True, exist_ok=True)
    path = _cache_path(boundary.slug)
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": boundary.display_name,
                "slug": boundary.slug,
                "source": boundary.source,
                "bbox": list(boundary.bbox),
                "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "geometry": boundary.geometry,
        }],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def _candidate_boundary(name: str, slug: str) -> LakeBoundary | None:
    cached = _cache_path(slug)
    if cached.exists():
        boundary = _load_geojson_boundary(cached, slug, name)
        boundary.source = f"cache:{cached.name}"
        return boundary

    from_imagery = boundary_from_imagery(slug)
    if from_imagery:
        return from_imagery

    for resolver in (_geocode_nominatim, _geocode_photon):
        try:
            candidate = resolver(name)
        except Exception:
            continue
        if not candidate:
            continue
        return LakeBoundary(
            slug=slug,
            display_name=candidate.get("display_name") or name,
            geometry=candidate["geometry"],
            bbox=tuple(candidate["bbox"]),
            source=candidate.get("source", "geocoder"),
        )
    return None


def resolve_boundary(
    lake_name: str,
    boundary_path: str | Path | None = None,
    *,
    refresh: bool = False,
    cache: bool = True,
) -> LakeBoundary:
    """Resolve ``lake_name`` to a cached, explicit or geocoded polygon."""
    slug = lake_slug(lake_name)

    if boundary_path:
        path = Path(boundary_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise BoundaryError(f"Boundary file not found: {path}")
        boundary = _load_geojson_boundary(path, slug, lake_name)
        boundary.display_name = lake_name or boundary.display_name
        if cache:
            write_cached_boundary(boundary)
        return boundary

    if refresh:
        cached = _cache_path(slug)
        if cached.exists():
            cached.unlink()

    boundary = _candidate_boundary(lake_name, slug)
    if boundary is None:
        raise BoundaryError(
            f"Could not find a water boundary for '{lake_name}'. "
            "Pass an explicit polygon with --boundary path/to/lake.geojson."
        )
    boundary.display_name = lake_name or boundary.display_name
    if cache and boundary.source not in {"existing-imagery"} and not boundary.source.startswith("cache:"):
        write_cached_boundary(boundary)
    return boundary
