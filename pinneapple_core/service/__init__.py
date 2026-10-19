"""Runtime pieces for app servers (apps/*, the MCP server, self-hosted services), standard library only.

==================  ======================================================================================
``JobQueue``        persistent queue (SQLite): thread or process workers, progress, logs, cancel, retry,
                    max concurrent jobs across server workers, resume after a restart
``APIKeyStore``     API keys: created once, stored hashed, revocable; sent as ``X-API-Key``
``UsageMeter``      usage per app, principal and unit with monthly free limits (``<PREFIX>_FREE_<UNIT>``)
``UploadStore``     uploads with a size limit, allowed suffixes, safe names and SHA-256
``data_dir``        where an app keeps this state (``<PREFIX>_DATA_DIR``, else ``PINNEAPPLE_APP_DATA/<app>``)
==================  ======================================================================================

The FastAPI routes on top of these are in ``apps/_shared/appkit.py`` (``install_runtime``).
"""
from __future__ import annotations

import os
from pathlib import Path

from .apikeys import APIKeyStore
from .db import LocalDB, connect, transaction
from .jobs import FINAL, STATUSES, JobCancelled, JobContext, JobQueue, JobTimeout
from .metering import UsageLimitExceeded, UsageMeter, current_period, limits_from_env
from .uploads import UploadRejected, UploadStore, UploadTooLarge, safe_name


def data_dir(app: str, prefix: str | None = None, environ=None) -> Path:
    """The directory for an app's state: ``<PREFIX>_DATA_DIR``, else ``$PINNEAPPLE_APP_DATA/<app>``, else
    ``/data/<app>`` when ``/data`` is a writable volume, else ``~/.pinneapple/apps/<app>``. Created if missing."""
    env = os.environ if environ is None else environ
    if prefix and env.get(f"{prefix}_DATA_DIR"):
        d = Path(env[f"{prefix}_DATA_DIR"])
    elif env.get("PINNEAPPLE_APP_DATA"):
        d = Path(env["PINNEAPPLE_APP_DATA"]) / app
    elif os.path.isdir("/data") and os.access("/data", os.W_OK):
        d = Path("/data") / app
    else:
        d = Path.home() / ".pinneapple" / "apps" / app
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = ["JobQueue", "JobContext", "JobCancelled", "JobTimeout", "STATUSES", "FINAL", "APIKeyStore",
           "UsageMeter", "UsageLimitExceeded", "limits_from_env", "current_period", "UploadStore",
           "UploadTooLarge", "UploadRejected", "safe_name", "data_dir", "LocalDB", "connect", "transaction"]
