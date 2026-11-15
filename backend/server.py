#!/usr/bin/env python3
import argparse
import csv
import io
import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import jobs
from common.ordering import acquisition_sort_key
from gee.config import client_installed, project_id, service_account_configured

FRONTEND_ROOT = PROJECT_ROOT / "frontend"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "severity" / "severity_summary.csv"
CLASS_NAMES = ("Good", "Moderate", "Poor")
CLASS_COLORS = ("#dce8e2", "#2f8c69", "#e3ad43", "#d85b51")
NUMERIC_FIELDS = (
    "good_percentage", "moderate_percentage", "poor_percentage",
    "low_percentage", "medium_percentage", "high_percentage",
)

_summary_cache = {"key": None, "records": []}
_map_cache = {}
_map_cache_limit = 24
_render_lock = threading.Lock()


def read_records(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def get_summary_records():
    """Read the severity summary once and reuse it until the file changes."""
    if not SUMMARY_PATH.exists():
        return []
    stat = SUMMARY_PATH.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    if _summary_cache["key"] == key:
        return _summary_cache["records"]

    records = read_records(SUMMARY_PATH)
    for record in records:
        for field in NUMERIC_FIELDS:
            try:
                record[field] = float(record.get(field, 0) or 0)
            except (TypeError, ValueError):
                record[field] = 0.0
    _summary_cache["key"] = key
    _summary_cache["records"] = records
    return records


def lake_summaries(records):
    result = []
    for lake_name in sorted({row.get("lake", "") for row in records if row.get("lake")}):
        lake_rows = [row for row in records if row.get("lake") == lake_name]
        count = len(lake_rows)
        result.append({
            "lake": lake_name,
            "image_count": count,
            "good_percentage": sum(row["good_percentage"] for row in lake_rows) / count,
            "moderate_percentage": sum(row["moderate_percentage"] for row in lake_rows) / count,
            "poor_percentage": sum(row["poor_percentage"] for row in lake_rows) / count,
        })
    return result


def resolve_image(lake_name, image_name):
    known_lakes = {row["lake"] for row in get_summary_records()}
    if lake_name not in known_lakes or not image_name or Path(image_name).name != image_name:
        return None
    stem = Path(image_name).stem
    return stem if stem == image_name or image_name.endswith(".tif") else None


def render_quality_map(path):
    """Render a quality map to PNG, reusing a cached render while the file is unchanged."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _map_cache.get(key)
    if cached is not None:
        return cached

    with rasterio.open(path) as source:
        scale = min(1.0, 1000 / max(source.width, source.height))
        shape = (max(1, round(source.height * scale)), max(1, round(source.width * scale)))
        quality = source.read(1, out_shape=shape, resampling=Resampling.nearest)

    # pyplot is not thread-safe and this server handles requests on threads.
    with _render_lock:
        figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        try:
            axis.imshow(quality, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=3, interpolation="nearest")
            axis.set_axis_off()
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=130, facecolor="#f5f7f5")
            body = buffer.getvalue()
        finally:
            plt.close(figure)

    if len(_map_cache) >= _map_cache_limit:
        _map_cache.clear()
    _map_cache[key] = body
    return body


class DashboardHandler(BaseHTTPRequestHandler):
    def send_bytes(self, status, body, content_type, cache_control="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8")

    def not_found(self, message="Not found"):
        self.send_json(404, {"error": message})

    def read_json(self, limit=65536):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("A JSON request body is required")
        if length > limit:
            raise ValueError("Request body is too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def same_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlsplit(origin).netloc == (self.headers.get("Host") or "")

    def do_POST(self):
        request = urlsplit(self.path)
        if not self.same_origin():
            return self.send_json(403, {"error": "Cross-origin analysis requests are not allowed"})
        if request.path != "/api/analyze":
            return self.not_found()
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self.send_json(415, {"error": "Content-Type must be application/json"})
        try:
            payload = self.read_json()
            request_data = jobs.validate_request(payload)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": str(exc)})
        job = jobs.start_job(request_data)
        return self.send_json(202, {"job": job})

    def do_GET(self):
        request = urlsplit(self.path)
        if request.path in ("/", "/index.html"):
            return self.serve_static("index.html")
        if request.path.startswith("/static/"):
            return self.serve_static(request.path.removeprefix("/static/"))
        if not request.path.startswith("/api/"):
            return self.not_found()

        query = parse_qs(request.query)
        records = get_summary_records()
        if request.path == "/api/health":
            return self.send_json(200, {"status": "ok", "image_count": len(records)})
        if request.path == "/api/config":
            return self.send_json(200, {
                "gee_installed": client_installed(),
                "project_configured": bool(project_id()),
                "service_account": service_account_configured(),
            })
        if request.path == "/api/jobs":
            return self.send_json(200, {"jobs": jobs.list_jobs()})
        if request.path.startswith("/api/jobs/"):
            job_id = unquote(request.path[len("/api/jobs/"):]).strip("/")
            job = jobs.get_job(job_id)
            if job is None:
                return self.not_found("Unknown analysis job")
            return self.send_json(200, {"job": job})
        if request.path == "/api/summary":
            return self.send_json(200, {"image_count": len(records), "lakes": lake_summaries(records)})
        if request.path.startswith("/api/lakes/") and request.path.endswith("/images"):
            lake_name = unquote(request.path[len("/api/lakes/"):-len("/images")].strip("/"))
            rows = sorted(
                (row for row in records if row.get("lake") == lake_name),
                key=lambda row: acquisition_sort_key(row.get("image", "")),
            )
            if not rows:
                return self.not_found("Lake has no summary records")
            return self.send_json(200, rows)
        if request.path == "/api/temporal":
            lake_name = query.get("lake", [""])[0]
            if lake_name not in {row.get("lake") for row in records}:
                return self.not_found("Unknown lake")
            path = PROJECT_ROOT / "outputs" / "temporal" / f"{lake_name}_temporal_summary.csv"
            temporal_rows = read_records(path)
            for row in temporal_rows:
                for field in ("good_pct", "moderate_pct", "poor_pct"):
                    row[field] = float(row.get(field, 0) or 0)
            return self.send_json(200, temporal_rows)
        if request.path == "/api/quality-map":
            lake_name, image_name = query.get("lake", [""])[0], query.get("image", [""])[0]
            stem = resolve_image(lake_name, image_name)
            if stem is None:
                return self.not_found("Unknown lake or image")
            map_path = PROJECT_ROOT / "outputs" / "gis" / "quality_maps" / f"{stem}_quality.tif"
            if not map_path.exists():
                return self.not_found("Quality map is not available")
            return self.send_bytes(200, render_quality_map(map_path), "image/png")
        if request.path == "/api/gradcam-classes":
            lake_name, image_name = query.get("lake", [""])[0], query.get("image", [""])[0]
            stem = resolve_image(lake_name, image_name)
            if stem is None:
                return self.not_found("Unknown lake or image")
            output_dir = PROJECT_ROOT / "outputs" / "gradcam" / lake_name
            classes = [
                class_name for class_name in CLASS_NAMES
                if (output_dir / f"{stem}_{class_name.lower()}_gradcam.png").exists()
            ]
            return self.send_json(200, classes)
        if request.path == "/api/gradcam":
            lake_name, image_name = query.get("lake", [""])[0], query.get("image", [""])[0]
            class_name = query.get("class", [""])[0]
            stem = resolve_image(lake_name, image_name)
            if stem is None or class_name not in CLASS_NAMES:
                return self.not_found("Unknown lake, image, or class")
            image_path = PROJECT_ROOT / "outputs" / "gradcam" / lake_name / f"{stem}_{class_name.lower()}_gradcam.png"
            return self.serve_png(image_path)
        return self.not_found()

    def serve_static(self, name):
        allowed = {"index.html", "app.js", "styles.css"}
        if name not in allowed:
            return self.not_found()
        path = FRONTEND_ROOT / name
        if not path.exists():
            return self.not_found()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(200, path.read_bytes(), f"{content_type}; charset=utf-8")

    def serve_png(self, path):
        if not path.is_file():
            return self.not_found("Image output is not available")
        self.send_bytes(200, path.read_bytes(), "image/png")

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local satellite water-quality dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
