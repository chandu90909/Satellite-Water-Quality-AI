"""Background pipeline jobs for the local dashboard.

Only one analysis runs at a time so a dashboard click cannot overload the local
machine or the Earth Engine quota. Job state is kept in memory; restarting the
dashboard clears the history.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_JOB_HISTORY = 20
MAX_LOG_LINES = 300

STAGE_PROGRESS = {
    "download": 10,
    "water_masks": 30,
    "classification": 55,
    "severity": 70,
    "gradcam": 85,
    "temporal": 92,
    "validation": 98,
}

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lake-analysis")


class JobRequestError(ValueError):
    """Raised when an analysis request is malformed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_date(value, field_name: str) -> date:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise JobRequestError(f"{field_name} must be a date in YYYY-MM-DD format") from exc


def validate_request(payload) -> dict:
    """Validate a dashboard analysis request and return normalized arguments."""
    if not isinstance(payload, dict):
        raise JobRequestError("Expected a JSON object")

    lake = str(payload.get("lake") or "").strip()
    if not lake:
        raise JobRequestError("A lake name is required")

    start = _parse_date(payload.get("start"), "Start date")
    end = _parse_date(payload.get("end"), "End date")
    if end < start:
        raise JobRequestError("End date must be on or after the start date")
    if (end - start).days > 365 * 5 + 1:
        raise JobRequestError("The time period is limited to five years")

    cadence = str(payload.get("cadence") or "monthly").strip().lower()
    if cadence not in {"monthly", "scene"}:
        raise JobRequestError("cadence must be 'monthly' or 'scene'")

    try:
        max_cloud = float(payload.get("max_cloud", 40))
    except (TypeError, ValueError) as exc:
        raise JobRequestError("max_cloud must be a number between 0 and 100") from exc
    if not 0 <= max_cloud <= 100:
        raise JobRequestError("max_cloud must be between 0 and 100")

    boundary = payload.get("boundary")
    boundary_path = None
    if boundary:
        candidate = Path(str(boundary))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
            raise JobRequestError("Boundary files must live inside the project folder")
        if not candidate.exists():
            raise JobRequestError(f"Boundary file not found: {boundary}")
        boundary_path = str(candidate)

    from gee.boundary import lake_slug

    return {
        "lake": lake,
        "slug": lake_slug(lake),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cadence": cadence,
        "max_cloud": max_cloud,
        "boundary": boundary_path,
    }


def _snapshot(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        snapshot = dict(job)
        snapshot["log"] = list(job["log"])
        return snapshot


def _update(job_id: str, **changes) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(changes)


def _append(job_id: str, message: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["log"].append(line)
        if len(job["log"]) > MAX_LOG_LINES:
            del job["log"][: len(job["log"]) - MAX_LOG_LINES]


def start_job(request: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    now = _now()
    job = {
        "id": job_id,
        "lake": request["lake"],
        "slug": request["slug"],
        "start": request["start"],
        "end": request["end"],
        "cadence": request["cadence"],
        "max_cloud": request["max_cloud"],
        "boundary": request.get("boundary"),
        "state": "queued",
        "stage": None,
        "progress": 0,
        "message": "Queued",
        "error": None,
        "warnings": [],
        "summary": None,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "log": [f"{datetime.now().strftime('%H:%M:%S')} Queued analysis for {request['lake']} "
                f"({request['start']} to {request['end']})"],
    }
    with _lock:
        _jobs[job_id] = job
        while len(_jobs) > MAX_JOB_HISTORY:
            _jobs.popitem(last=False)
    _executor.submit(_execute, job_id)
    return _snapshot(job_id)


def _execute(job_id: str) -> None:
    """Run a job and guarantee it reaches a terminal state."""
    try:
        _run_pipeline(job_id)
    except Exception as exc:  # never leave a job stuck in "running"
        message = str(exc) or exc.__class__.__name__
        _append(job_id, f"ERROR: {message}")
        _update(
            job_id,
            state="failed",
            error=message,
            message=message,
            finished_at=_now(),
        )


def _run_pipeline(job_id: str) -> None:
    request = _snapshot(job_id)
    if request is None:
        return
    _update(job_id, state="running", started_at=_now(), message="Starting pipeline")
    boundary_mode = "explicit GeoJSON" if request.get("boundary") else "name lookup / cached imagery"
    _append(job_id, f"Boundary mode: {boundary_mode}")

    def log(message: str) -> None:
        _append(job_id, str(message))

    def on_stage(stage: str) -> None:
        _update(
            job_id,
            stage=stage,
            progress=STAGE_PROGRESS.get(stage, 0),
            message=stage.replace("_", " ").capitalize(),
        )

    from pipeline.lake_pipeline import run_lake

    result = run_lake(
        request["lake"],
        request["start"],
        request["end"],
        cadence=request["cadence"],
        max_cloud=request["max_cloud"],
        boundary=request.get("boundary"),
        stages=None,
        log=log,
        on_stage=on_stage,
    )
    summary = result.summary()
    message = (
        f"Completed: {summary['acquisitions']} acquisition(s), "
        f"{summary['classified_patches']} classified patches"
    )
    _update(
        job_id,
        state="done",
        stage=None,
        progress=100,
        message=message,
        warnings=list(result.validation_errors),
        summary=summary,
        finished_at=_now(),
    )
    _append(job_id, message)


def get_job(job_id: str) -> dict | None:
    return _snapshot(job_id)


def list_jobs(limit: int = 10) -> list[dict]:
    with _lock:
        ids = list(_jobs.keys())[-limit:][::-1]
    return [snapshot for snapshot in (_snapshot(job_id) for job_id in ids) if snapshot]


def active_job() -> dict | None:
    with _lock:
        for job in _jobs.values():
            if job["state"] in {"queued", "running"}:
                return dict(job, log=list(job["log"]))
    return None
