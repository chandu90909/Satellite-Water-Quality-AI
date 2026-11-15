"""Earth Engine credentials and initialization.

The client library is imported lazily so the rest of the project (dashboard,
tests, existing inference) keeps working without an Earth Engine account.

Configuration, in order of precedence:
    ``--project`` argument
    ``EE_PROJECT`` environment variable
    ``GOOGLE_CLOUD_PROJECT`` environment variable

Credentials:
    ``EE_SERVICE_ACCOUNT`` + ``EE_PRIVATE_KEY_FILE`` for a service account
    otherwise the credentials written by ``earthengine authenticate``, with an
    automatic fallback to the account signed in with ``gcloud auth login``
    when those stored credentials cannot reach the project.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "earth_engine.json"

SETUP_HELP = """\
Earth Engine is not configured yet. One-time setup:

  1. Create or pick a Google Cloud project with the Earth Engine API enabled
     (https://code.earthengine.google.com/register) and note its project ID.
  2. Authenticate the local client, either
       .venv311\\Scripts\\earthengine.exe authenticate
     or by creating a service account key and setting
       EE_SERVICE_ACCOUNT=name@project.iam.gserviceaccount.com
       EE_PRIVATE_KEY_FILE=path\\to\\key.json
  3. Provide the project ID with --project, the EE_PROJECT environment variable,
     or config/earth_engine.json ({"project": "your-project-id"}).
"""


class EarthEngineError(RuntimeError):
    """Raised when Earth Engine is missing, unauthenticated or misconfigured."""


def _import_ee():
    try:
        import ee  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise EarthEngineError(
            "The earthengine-api package is not installed. "
            "Run: .venv311\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return ee


def client_installed() -> bool:
    """True when the Earth Engine client library is importable."""
    try:
        _import_ee()
    except EarthEngineError:
        return False
    return True


def local_project() -> str | None:
    """Project ID from the optional project-local configuration file."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("project") if isinstance(payload, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def project_id(project: str | None = None) -> str | None:
    """Resolve the Google Cloud project used for Earth Engine requests."""
    return (
        project
        or os.environ.get("EE_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or local_project()
    )


def service_account_configured() -> bool:
    return bool(os.environ.get("EE_SERVICE_ACCOUNT") and os.environ.get("EE_PRIVATE_KEY_FILE"))


def _gcloud_executable() -> str | None:
    for name in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _gcloud_access_token() -> str | None:
    """Access token for the Google account currently signed in with gcloud."""
    executable = _gcloud_executable()
    if not executable:
        return None
    command = f'"{executable}" auth print-access-token'
    try:
        result = subprocess.run(  # noqa: S602 - fixed command, no user input
            command if os.name == "nt" else [executable, "auth", "print-access-token"],
            shell=os.name == "nt",
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


def gcloud_credentials(project: str | None):
    """Build Earth Engine credentials from the active gcloud account."""
    from google.oauth2.credentials import Credentials  # noqa: PLC0415 - optional

    token = _gcloud_access_token()
    if not token:
        return None
    return Credentials(token, quota_project_id=project)


_CREDENTIAL_SOURCE = "earthengine"


def credential_source() -> str:
    """Which credential path the last successful initialize() used."""
    return _CREDENTIAL_SOURCE


def initialize(project: str | None = None):
    """Initialize and return the Earth Engine client, or raise with setup help."""
    global _CREDENTIAL_SOURCE
    ee = _import_ee()
    resolved_project = project_id(project)
    account = os.environ.get("EE_SERVICE_ACCOUNT")
    key_file = os.environ.get("EE_PRIVATE_KEY_FILE")

    def start(credentials=None) -> None:
        kwargs = {"project": resolved_project} if resolved_project else {}
        if credentials is not None:
            kwargs["credentials"] = credentials
        ee.Initialize(**kwargs)
        # Initialize() only builds the client; one call proves account access.
        ee.data.getProjectConfig()

    try:
        if account and key_file:
            start(ee.ServiceAccountCredentials(account, str(Path(key_file))))
            _CREDENTIAL_SOURCE = "service_account"
        else:
            start()
            _CREDENTIAL_SOURCE = "earthengine"
        return ee
    except Exception as exc:  # the client raises a variety of error types
        first_error = exc

    fallback = gcloud_credentials(resolved_project) if resolved_project else None
    if fallback is not None:
        try:
            start(fallback)
            _CREDENTIAL_SOURCE = "gcloud"
            return ee
        except Exception as exc:
            first_error = exc

    raise EarthEngineError(f"{first_error}\n\n{SETUP_HELP}") from first_error


def check_connection(project: str | None = None) -> dict:
    """Round-trip a trivial request so auth problems surface immediately."""
    ee = initialize(project)
    value = ee.Number(1).add(1).getInfo()
    return {
        "ok": value == 2,
        "project": project_id(project),
        "service_account": service_account_configured(),
        "credentials": credential_source(),
    }
