import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from backend import jobs
from common.normalization import normalize_bands, valid_pixel_mask
from gee.boundary import boundary_from_imagery, lake_slug
from gee.sentinel2 import EXPORT_BANDS, _tile_boxes, download_image, month_periods
from segmentation.masks.water_masks import create_water_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_lake_slug_reuses_existing_folder_names():
    assert lake_slug("Hussain Sagar") == "Hussain_Sagar"
    assert lake_slug("Kolleru Lake") == "Kolleru_Lake"
    assert lake_slug("  pulicat   lake ") == "Pulicat_Lake"
    assert lake_slug("Lake 10") == "Lake_10"
    with pytest.raises(ValueError):
        lake_slug("   ")


def test_month_periods_cover_partial_months_in_order():
    periods = month_periods(date(2025, 1, 15), date(2025, 3, 2))

    assert [period.label for period in periods] == ["202501", "202502", "202503"]
    assert periods[0].start == date(2025, 1, 15)
    assert periods[0].end == date(2025, 1, 31)
    assert periods[-1].end == date(2025, 3, 2)
    assert all(period.start <= period.end for period in periods)
    with pytest.raises(ValueError):
        month_periods(date(2025, 3, 2), date(2025, 1, 15))


def test_tile_boxes_cover_the_requested_extent():
    bounds = (78.0, 17.0, 78.02, 17.02)
    boxes = _tile_boxes(bounds, scale=10, max_pixels=100)

    assert len(boxes) > 1
    assert min(box[0] for box in boxes) == pytest.approx(bounds[0])
    assert min(box[1] for box in boxes) == pytest.approx(bounds[1])
    assert max(box[2] for box in boxes) == pytest.approx(bounds[2])
    assert max(box[3] for box in boxes) == pytest.approx(bounds[3])


def test_normalization_maps_reflectance_to_training_range():
    gradient = np.linspace(0, 10000, 400, dtype=np.float32).reshape(20, 20)
    stack = np.stack([gradient + channel * 200 for channel in range(5)], axis=-1)

    normalized = normalize_bands(stack)

    assert normalized.shape == stack.shape
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    # A uniform 2/98 percentile stretch leaves the median close to 0.5.
    assert float(np.median(normalized[..., 0])) == pytest.approx(0.5, abs=0.06)


def test_normalization_ignores_pixels_without_signal():
    stack = np.ones((10, 10, 5), dtype=np.float32)
    stack[:5] = 0.0
    valid = valid_pixel_mask(stack)

    assert not valid[:5].any()
    assert valid[5:].all()
    assert float(normalize_bands(stack)[:5].max()) == 0.0


def test_water_mask_flags_high_ndwi_pixels():
    green = np.array([[0.25, 0.05]], dtype=np.float32)
    nir = np.array([[0.05, 0.30]], dtype=np.float32)
    swir = np.array([[0.04, 0.35]], dtype=np.float32)

    mask = create_water_mask(green, nir, swir)

    assert mask.tolist() == [[1, 0]]


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "raw" / "Hussain_Sagar").is_dir(),
    reason="project imagery is not available",
)
def test_boundary_falls_back_to_existing_imagery():
    boundary = boundary_from_imagery("Hussain_Sagar")

    assert boundary is not None
    assert boundary.geometry["type"] == "Polygon"
    min_lon, min_lat, max_lon, max_lat = boundary.bbox
    assert min_lon < 78.5 < max_lon
    assert min_lat < 17.42 < max_lat


def test_download_image_writes_the_project_geotiff_contract(tmp_path, monkeypatch):
    """The exporter must produce one 6-band EPSG:4326 GeoTIFF with descriptions."""
    import requests

    bounds = (78.0, 17.0, 78.002, 17.002)
    transform = from_bounds(*bounds, 3, 3)
    with rasterio.MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=3, height=3, count=len(EXPORT_BANDS),
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dataset:
            for index in range(1, len(EXPORT_BANDS) + 1):
                dataset.write(np.full((3, 3), index * 1000, dtype=np.float32), index)
        blob = memory.read()

    class FakeResponse:
        status_code = 200
        content = blob
        text = ""

    class FakeImage:
        def getDownloadURL(self, params):
            assert params["bands"] == list(EXPORT_BANDS)
            assert params["format"] == "GEO_TIFF"
            return "https://example.invalid/tile.tif"

    fake_ee = types.SimpleNamespace(
        Geometry=types.SimpleNamespace(Rectangle=lambda coords, crs, geodesic: {"coords": coords})
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(requests, "get", lambda url, timeout: FakeResponse())

    output = download_image(FakeImage(), tmp_path / "lake_202501.tif", bounds)

    with rasterio.open(output) as dataset:
        assert dataset.count == len(EXPORT_BANDS)
        assert dataset.dtypes == ("float32",) * len(EXPORT_BANDS)
        assert dataset.crs.to_string() == "EPSG:4326"
        assert tuple(dataset.descriptions) == EXPORT_BANDS
        assert dataset.read(1)[0, 0] == pytest.approx(1000.0)


def test_validate_request_normalizes_lake_and_dates():
    request = jobs.validate_request({
        "lake": "Kolleru Lake",
        "start": "2025-01-01",
        "end": "2025-03-31",
        "cadence": "monthly",
        "max_cloud": 25,
    })

    assert request["slug"] == "Kolleru_Lake"
    assert request["start"] == "2025-01-01"
    assert request["cadence"] == "monthly"
    assert request["max_cloud"] == 25.0


@pytest.mark.parametrize("payload", [
    {"lake": "", "start": "2025-01-01", "end": "2025-01-02"},
    {"lake": "Lake", "start": "2025-02-01", "end": "2025-01-01"},
    {"lake": "Lake", "start": "01/02/2025", "end": "2025-01-03"},
    {"lake": "Lake", "start": "2025-01-01", "end": "2025-01-02", "cadence": "weekly"},
    {"lake": "Lake", "start": "2025-01-01", "end": "2025-01-02", "max_cloud": 150},
    {"lake": "Lake", "start": "2020-01-01", "end": "2025-01-02"},
])
def test_validate_request_rejects_invalid_input(payload):
    with pytest.raises(jobs.JobRequestError):
        jobs.validate_request(payload)


def test_started_job_reports_pipeline_failures(monkeypatch):
    import time

    captured = {}

    def fake_run_lake(lake, start, end, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("Earth Engine is not configured")

    module = types.ModuleType("pipeline.lake_pipeline")
    module.run_lake = fake_run_lake
    monkeypatch.setitem(sys.modules, "pipeline.lake_pipeline", module)

    job = jobs.start_job(jobs.validate_request({
        "lake": "Hussain Sagar", "start": "2025-01-01", "end": "2025-01-31",
    }))
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = jobs.get_job(job["id"])
        if snapshot["state"] in {"done", "failed"}:
            break
        time.sleep(0.05)

    snapshot = jobs.get_job(job["id"])
    assert snapshot["state"] == "failed"
    assert "Earth Engine is not configured" in snapshot["error"]
    assert captured["boundary"] is None


def test_project_id_prefers_explicit_then_local_config(tmp_path, monkeypatch):
    from gee import config

    path = tmp_path / "earth_engine.json"
    path.write_text('{"project": "demo-project"}', encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    assert config.project_id() == "demo-project"
    assert config.project_id("explicit-project") == "explicit-project"


def test_gcloud_credentials_carry_the_project_quota(monkeypatch):
    from gee import config

    monkeypatch.setattr(config, "_gcloud_access_token", lambda: "test-token")
    credentials = config.gcloud_credentials("demo-project")

    assert credentials is not None
    assert credentials.token == "test-token"
    assert credentials.quota_project_id == "demo-project"


def test_gcloud_credentials_are_skipped_without_a_token(monkeypatch):
    from gee import config

    monkeypatch.setattr(config, "_gcloud_access_token", lambda: None)
    assert config.gcloud_credentials("demo-project") is None


def test_boundary_file_must_stay_inside_the_project():
    with pytest.raises(jobs.JobRequestError):
        jobs.validate_request({
            "lake": "Lake",
            "start": "2025-01-01",
            "end": "2025-01-02",
            "boundary": "../outside.geojson",
        })
